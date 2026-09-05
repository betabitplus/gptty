from __future__ import annotations

from datetime import datetime
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ..output import OutputFormat, OutputMessage, normalize_messages, render_messages
from ..sdk_client import GpttyClient
from ..state import StateError, load_chat_state

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
        conversation_ref = resolve_conversation_ref(args)
    except StateError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 1

    if not conversation_ref:
        print(NO_CONVERSATION_ERROR, file=stderr)
        return 2

    client = client_factory(
        auth_file=getattr(args, "auth", "auth_data.json"),
        timeout=getattr(args, "timeout", 90),
    )
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
    rendered = render_messages(normalize_messages(response), output_format)
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


def resolve_conversation_ref(args: Any) -> str | None:
    explicit = getattr(args, "url_or_id", None)
    if explicit:
        return str(explicit)

    state = load_chat_state(Path(getattr(args, "state", "gptty_state.json")))
    return state.current_conversation


def write_export(output_path: str | Path, content: str, *, overwrite: bool, stderr: TextIO) -> int:
    path = Path(output_path)
    if path.exists() and not overwrite:
        print(f"gptty: output file already exists: {path}. Use --overwrite to replace it.", file=stderr)
        return 1

    try:
        path.write_text(content + "\n", encoding="utf-8")
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
) -> Path:
    root = Path(directory).expanduser() if directory is not None else DEFAULT_EXPORT_DIRECTORY
    root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M-%S")
    stem = _export_filename_stem(title)
    candidate = root / f"{timestamp} - {stem}.md"
    suffix = 2
    while candidate.exists():
        candidate = root / f"{timestamp} - {stem} ({suffix}).md"
        suffix += 1
    candidate.write_text(render_messages(messages, "markdown") + "\n", encoding="utf-8")
    return candidate.resolve()


def _export_filename_stem(title: str | None) -> str:
    value = " ".join((title or "chat").split())
    value = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "-", value).strip(" .-")
    return (value or "chat")[:80].rstrip(" .-") or "chat"
