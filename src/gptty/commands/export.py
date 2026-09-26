from __future__ import annotations

from datetime import datetime
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ..exporter_bridge import ExporterBridgeError, export_persistent_conversation
from ..output import (
    OutputFormat,
    OutputMessage,
    render_messages,
    render_source_citations,
)
from ..private_fs import create_private_text, ensure_private_dir
from ..sdk_client import GpttyClient
from ..session_state import SessionStateError
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
    # Retained in the signature for test/backward compatibility; persistent export
    # is now delegated to chatgpt-exporter rather than the current-branch SDK reader.
    _ = client_factory
    try:
        conversation_ref = resolve_conversation_ref(args, stderr=stderr)
    except SessionStateError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 1

    if not conversation_ref:
        print(NO_CONVERSATION_ERROR, file=stderr)
        return 2

    if getattr(args, "last", None) is not None:
        print(
            "gptty: persistent export is a complete visible-graph artifact; "
            "use `gptty messages --last N` for a current-branch slice.",
            file=stderr,
        )
        return 2

    output_format: OutputFormat = getattr(args, "format", "markdown")
    if output_format == "plain":
        print(
            "gptty: plain output is a current-branch projection, not a visible-graph "
            "artifact; use `gptty messages --format plain` instead.",
            file=stderr,
        )
        return 2

    requested_output = getattr(args, "output", None)
    if requested_output and output_format != "markdown":
        print(
            "gptty: persistent file export writes a Markdown visible-graph bundle; "
            "use --format json without --output to print its context sidecar.",
            file=stderr,
        )
        return 2

    try:
        if requested_output:
            markdown_path = _persistent_markdown_path(requested_output)
            reserved = False
            if not bool(getattr(args, "overwrite", False)):
                collision = reserve_persistent_bundle(markdown_path)
                if collision is not None:
                    print(
                        f"gptty: export artifact already exists: {collision}. "
                        "Use --overwrite to replace the bundle.",
                        file=stderr,
                    )
                    return 1
                reserved = True
            try:
                export_persistent_conversation(
                    conversation_ref,
                    markdown_path,
                    auth_file=getattr(args, "auth", None),
                    timeout=float(getattr(args, "timeout", 120)),
                )
            except Exception:
                if reserved:
                    cleanup_persistent_bundle(markdown_path)
                raise
            return 0

        with tempfile.TemporaryDirectory(prefix="gptty-export-") as temporary:
            markdown_path = Path(temporary) / "conversation.md"
            artifact = export_persistent_conversation(
                conversation_ref,
                markdown_path,
                auth_file=getattr(args, "auth", None),
                timeout=float(getattr(args, "timeout", 120)),
            )
            selected = (
                artifact.markdown_path
                if output_format == "markdown"
                else artifact.context_path
            )
            rendered = selected.read_text(encoding="utf-8").rstrip("\n")
            print(rendered, file=stdout)
            return 0
    except ExporterBridgeError as exc:
        print(f"gptty: export failed: {exc}", file=stderr)
        return 1
    except OSError as exc:
        print(f"gptty: failed to read/write export artifact: {exc}", file=stderr)
        return 1


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


def _persistent_markdown_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".md":
        path = path.with_suffix(".md")
    return path


def _bundle_paths(markdown_path: Path) -> tuple[Path, Path, Path]:
    return (
        markdown_path,
        markdown_path.with_suffix(".context.json"),
        markdown_path.with_suffix(".manifest.json"),
    )


def reserve_persistent_bundle(markdown_path: Path) -> Path | None:
    created: list[Path] = []
    for candidate in _bundle_paths(markdown_path):
        try:
            create_private_text(candidate, "")
        except FileExistsError:
            for reserved in created:
                reserved.unlink(missing_ok=True)
            return candidate
        created.append(candidate)
    return None


def cleanup_persistent_bundle(markdown_path: Path) -> None:
    for candidate in _bundle_paths(markdown_path):
        candidate.unlink(missing_ok=True)


def reserve_default_persistent_export_path(
    *,
    title: str | None = None,
    now: datetime | None = None,
    directory: str | Path | None = None,
) -> Path:
    root = ensure_private_dir(directory or DEFAULT_EXPORT_DIRECTORY)
    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M-%S")
    stem = _export_filename_stem(title)
    suffix = 1
    while True:
        label = "" if suffix == 1 else f" ({suffix})"
        candidate = root / f"{timestamp} - {stem}{label}.md"
        if reserve_persistent_bundle(candidate) is None:
            return candidate.resolve()
        suffix += 1


def save_markdown_export(
    messages: list[OutputMessage],
    *,
    directory: str | Path | None = None,
    title: str | None = None,
    now: datetime | None = None,
    observations: dict[str, Any] | None = None,
    artifact_scope: str | None = None,
    artifact_provenance: str | None = None,
) -> Path:
    if directory is None:
        root = ensure_private_dir(DEFAULT_EXPORT_DIRECTORY)
    else:
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d_%H-%M-%S")
    stem = _export_filename_stem(title)
    payload = render_messages(messages, "markdown").rstrip()
    if artifact_scope or artifact_provenance:
        metadata: list[str] = []
        if artifact_scope:
            metadata.append(f"> Export scope: `{artifact_scope}`")
        if artifact_provenance:
            metadata.append(f"> Provenance: `{artifact_provenance}`")
        payload = "\n".join(metadata) + "\n\n" + payload
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


def _export_filename_stem(title: str | None) -> str:
    value = " ".join((title or "chat").split())
    value = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "-", value).strip(" .-")
    return (value or "chat")[:80].rstrip(" .-") or "chat"
