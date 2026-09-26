from __future__ import annotations

from datetime import datetime
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ..local_store import local_store_path
from ..output import (
    OutputFormat,
    OutputMessage,
    normalize_messages,
    render_messages,
    render_source_citations,
)
from ..private_fs import atomic_write_private_text, create_private_text, ensure_private_dir
from ..sdk_client import GpttyClient
from ..session_state import SessionStateError
from ..tui_archive import TUIArchive
from ._client import build_client
from ._session import resolve_attached_conversation

NO_CONVERSATION_ERROR = (
    "gptty export requires a conversation URL/id or an attached conversation. "
    "Run `gptty attach <url-or-id>` first."
)

DEFAULT_EXPORT_DIRECTORY = Path.home() / "Documents" / "gptty-exports"


def run_export(
    args: Any,
    *,
    client_factory: Callable[..., Any] = GpttyClient,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    try:
        conversation_ref = resolve_conversation_ref(args, stderr=stderr)
    except SessionStateError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 1

    if not conversation_ref:
        print(NO_CONVERSATION_ERROR, file=stderr)
        return 2

    client = build_client(client_factory, args)
    options: dict[str, Any] = {}
    last = getattr(args, "last", None)
    if last is not None:
        options["limit"] = int(last)

    try:
        response = client.get_messages(conversation_ref, **options)
    except Exception as exc:
        print(f"gptty: export request failed: {exc}", file=stderr)
        return 1

    output_format: OutputFormat = getattr(args, "format", "markdown")
    messages = normalize_messages(response)
    rendered = render_messages(messages, output_format)
    observations = _local_source_citations(args, conversation_ref)
    if output_format in {"plain", "markdown"}:
        source_block = render_source_citations(observations, output_format)
        if source_block:
            rendered = rendered.rstrip() + "\n\n" + source_block
    output_path = getattr(args, "output", None)
    if output_path:
        return write_export(
            output_path,
            rendered,
            overwrite=bool(getattr(args, "overwrite", False)),
            stderr=stderr,
        )

    print(rendered, file=stdout)
    return 0


def resolve_conversation_ref(
    args: Any,
    *,
    stderr: TextIO = sys.stderr,
) -> str | None:
    return resolve_attached_conversation(
        args,
        explicit=getattr(args, "url_or_id", None),
        stderr=stderr,
    )


def write_export(output_path: str | Path, content: str, *, overwrite: bool, stderr: TextIO) -> int:
    path = Path(output_path)
    payload = content + "\n"

    try:
        if overwrite:
            atomic_write_private_text(path, payload)
        else:
            create_private_text(path, payload)
    except FileExistsError:
        print(f"gptty: output file already exists: {path}. Use --overwrite to replace it.", file=stderr)
        return 1
    except OSError as exc:
        print(f"gptty: failed to write export to {path}: {exc}", file=stderr)
        return 1

    return 0


def save_markdown_export(
    messages: list[OutputMessage],
    *,
    directory: str | Path | None = None,
    title: str | None = None,
    now: datetime | None = None,
    observations: dict[str, Any] | None = None,
) -> Path:
    if directory is None:
        root = ensure_private_dir(DEFAULT_EXPORT_DIRECTORY)
    else:
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M-%S")
    stem = _export_filename_stem(title)
    payload = render_messages(messages, "markdown").rstrip()
    source_block = render_source_citations(observations, "markdown")
    if source_block:
        payload += "\n\n" + source_block
    payload += "\n"
    suffix = 1
    while True:
        label = "" if suffix == 1 else f" ({suffix})"
        candidate = root / f"{timestamp} - {stem}{label}.md"
        try:
            create_private_text(candidate, payload)
        except FileExistsError:
            suffix += 1
            continue
        return candidate.resolve()


def _local_source_citations(
    args: Any,
    conversation_ref: str,
) -> dict[str, list[dict[str, Any]]] | None:
    state_path = getattr(args, "state", None) or "gptty_state.json"
    try:
        archive = TUIArchive(
            db_path=local_store_path(
                profile=getattr(args, "profile", None),
                state_path=state_path,
            ),
            reconcile_pending=False,
        )
        observations = archive.source_citation_observations(conversation_ref)
    except Exception:
        return None
    return observations if observations.get("sources") else None


def _export_filename_stem(title: str | None) -> str:
    value = " ".join((title or "chat").split())
    value = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "-", value).strip(" .-")
    return (value or "chat")[:80].rstrip(" .-") or "chat"
