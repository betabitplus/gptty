from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .file_lock import KernelFileLock
from .local_store import DB_FILENAME, LocalEventStore
from .private_fs import PRIVATE_FILE_MODE, PRIVATE_MODES_SUPPORTED, atomic_write_private_text
from .profiles import data_dir

_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,128}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conversation_id(value: str) -> str:
    candidate = str(value or "").strip()
    if "/c/" in candidate:
        candidate = candidate.split("/c/", 1)[1].split("/", 1)[0].split("?", 1)[0]
    if not _CONVERSATION_ID_RE.fullmatch(candidate):
        raise ValueError(f"invalid ChatGPT conversation id: {value!r}")
    return candidate.lower()


def archive_root() -> Path:
    override = os.environ.get("GPTTY_ARCHIVE_HOME")
    if override:
        return Path(override).expanduser()
    return data_dir() / "chat-archive"


class TUIArchive:
    """Transactional TUI observation archive with portable file projections."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        db_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser() if root is not None else archive_root()
        self.pending_dir = self.root / "pending"
        self.conversations_dir = self.root / "conversations"
        self._ensure_directory(self.root)
        self._ensure_directory(self.pending_dir)
        self._ensure_directory(self.conversations_dir)
        self.store = LocalEventStore(
            Path(db_path).expanduser() if db_path is not None else self.root / DB_FILENAME
        )
        self._projection_checked: set[str] = set()

    def record_user(
        self,
        text: str,
        *,
        conversation_ref: str | None,
        model: str | None,
        media_count: int = 0,
    ) -> str:
        turn_id = uuid.uuid4().hex
        event = {
            "schema": 1,
            "event_id": f"{turn_id}:user",
            "turn_id": turn_id,
            "observed_at": _now_iso(),
            "source": "gptty-tui",
            "scope": "tui-observed",
            "role": "user",
            "text": text,
            "model": model,
            "media_count": max(0, int(media_count)),
            "conversation_id": None,
        }
        if conversation_ref:
            conversation_id = _conversation_id(conversation_ref)
            event["conversation_id"] = conversation_id
            self._append_conversation_event(conversation_id, event)
        else:
            self.store.put_pending_tui_event(turn_id, event)
            try:
                self._write_atomic(self.pending_dir / f"{turn_id}.json", event)
            except OSError:
                pass
        return turn_id

    def bind_turn(self, turn_id: str, conversation_ref: str) -> str:
        conversation_id = _conversation_id(conversation_ref)
        self._ensure_conversation_imported(conversation_id)
        pending = self.pending_dir / f"{turn_id}.json"

        with self._projection_guard(conversation_id):
            event, inserted = self.store.bind_pending_tui_event(turn_id, conversation_id)
            if event is not None and inserted:
                self._project_inserted_event(conversation_id, event)
        if event is not None:
            self._remove_projection(pending)
            return conversation_id

        legacy_event = self._read_pending_projection(pending)
        if legacy_event is not None:
            legacy_event["conversation_id"] = conversation_id
            self._append_conversation_event(conversation_id, legacy_event)
        self._remove_projection(pending)
        return conversation_id

    def record_assistant(
        self,
        turn_id: str,
        *,
        conversation_ref: str,
        text: str,
        title: str | None,
        model: str | None,
        status: str,
    ) -> None:
        conversation_id = self.bind_turn(turn_id, conversation_ref)
        event = {
            "schema": 1,
            "event_id": f"{turn_id}:assistant",
            "turn_id": turn_id,
            "observed_at": _now_iso(),
            "source": "gptty-tui",
            "scope": "tui-observed",
            "role": "assistant",
            "text": text,
            "model": model,
            "status": status,
            "conversation_id": conversation_id,
        }
        self._append_conversation_event(conversation_id, event)
        self._write_meta(conversation_id, title=title)

    def record_terminal(
        self,
        turn_id: str,
        *,
        conversation_ref: str,
        label: str,
        status: str,
        text: str,
        source: str | None = None,
    ) -> None:
        conversation_id = self.bind_turn(turn_id, conversation_ref)
        event = {
            "schema": 1,
            "event_id": f"{turn_id}:terminal",
            "turn_id": turn_id,
            "observed_at": _now_iso(),
            "source": "gptty-tui",
            "scope": "tui-observed",
            "role": str(label or "turn").strip() or "turn",
            "text": text,
            "status": status,
            "terminal_source": source,
            "conversation_id": conversation_id,
        }
        self._append_conversation_event(conversation_id, event)

    def record_observed_terminal(
        self,
        *,
        conversation_ref: str,
        label: str,
        status: str,
        text: str,
        source: str,
    ) -> None:
        conversation_id = _conversation_id(conversation_ref)
        normalized_label = str(label or "turn").strip() or "turn"
        normalized_status = str(status or "").strip()
        normalized_text = str(text or "").strip()
        normalized_source = str(source or "").strip()
        if not normalized_status or not normalized_text or not normalized_source:
            return
        self._ensure_conversation_imported(conversation_id)
        turn_id = uuid.uuid4().hex
        event = {
            "schema": 1,
            "event_id": f"{turn_id}:terminal",
            "turn_id": turn_id,
            "observed_at": _now_iso(),
            "source": "gptty-tui",
            "scope": "tui-observed",
            "role": normalized_label,
            "text": normalized_text,
            "status": normalized_status,
            "terminal_source": normalized_source,
            "conversation_id": conversation_id,
        }
        with self._projection_guard(conversation_id):
            if not self.store.insert_tui_terminal_once(conversation_id, event):
                return
            self._project_inserted_event(conversation_id, event)

    def record_chat_terminal_resolution(
        self,
        *,
        conversation_ref: str,
        resolved_status: str,
        source: str,
    ) -> bool:
        """Append proof that newer evidence superseded one chat-level marker."""

        normalized_status = str(resolved_status or "").strip()
        normalized_source = str(source or "").strip()
        if not normalized_status or not normalized_source:
            return False

        conversation_id = _conversation_id(conversation_ref)
        self._ensure_conversation_imported(conversation_id)

        turn_id = uuid.uuid4().hex
        event = {
            "schema": 1,
            "event_id": f"{turn_id}:terminal-resolution",
            "turn_id": turn_id,
            "observed_at": _now_iso(),
            "source": "gptty-tui",
            "scope": "tui-observed",
            "role": "chat",
            "text": (
                "Newer canonical evidence superseded the prior local "
                f"chat-level {normalized_status} marker."
            ),
            "status": "resolved",
            "terminal_source": normalized_source,
            "terminal_resolution": True,
            "resolved_status": normalized_status,
            "conversation_id": conversation_id,
        }
        with self._projection_guard(conversation_id):
            if not self.store.insert_tui_terminal_resolution_if_current(
                conversation_id,
                event,
                resolved_status=normalized_status,
            ):
                return False
            self._project_inserted_event(conversation_id, event)
        return True

    def conversation_terminal_marker(
        self,
        conversation_ref: str,
    ) -> tuple[str, str, str, str | None] | None:
        """Return the latest unresolved persistent chat-level terminal state."""

        conversation_id = _conversation_id(conversation_ref)
        self._ensure_conversation_imported(conversation_id)
        event = self.store.latest_chat_terminal(conversation_id)
        if event is None or event.get("terminal_resolution") is True:
            return None
        status = str(event.get("status") or "").strip()
        text = str(event.get("text") or "").strip()
        if not status or not text:
            return None
        source = str(event.get("terminal_source") or "").strip() or None
        return ("chat", status, text, source)

    def conversation_paths(self, conversation_ref: str) -> dict[str, Path]:
        conversation_id = _conversation_id(conversation_ref)
        directory = self.conversations_dir / conversation_id
        return {
            "directory": directory,
            "events": directory / "events.jsonl",
            "transcript": directory / "transcript.md",
            "meta": directory / "meta.json",
        }

    def _append_conversation_event(
        self,
        conversation_id: str,
        event: dict[str, Any],
    ) -> None:
        self._ensure_conversation_imported(conversation_id)
        with self._projection_guard(conversation_id):
            if not self.store.insert_tui_event(conversation_id, event):
                return
            self._project_inserted_event(conversation_id, event)

    def _ensure_conversation_imported(self, conversation_id: str) -> None:
        if (
            self.store.tui_imported(conversation_id)
            and conversation_id in self._projection_checked
        ):
            return
        with self._projection_guard(conversation_id):
            if not self.store.tui_imported(conversation_id):
                paths = self.conversation_paths(conversation_id)
                events = self._read_legacy_events(paths["events"])
                title = self._read_legacy_title(paths["meta"])
                self.store.import_tui_conversation(
                    conversation_id,
                    events=events,
                    title=title,
                    imported_at=_now_iso(),
                )
            if conversation_id not in self._projection_checked:
                self._rebuild_events_projection(conversation_id)
                self._rebuild_transcript_projection(conversation_id)
                self._projection_checked.add(conversation_id)

    @contextmanager
    def _projection_guard(self, conversation_id: str):
        path = self.conversation_paths(conversation_id)["directory"] / ".projection.lock"
        lock = KernelFileLock(path)
        lock.acquire(timeout=5.0)
        try:
            yield
        finally:
            lock.release()

    def _project_inserted_event(
        self,
        conversation_id: str,
        event: dict[str, Any],
    ) -> None:
        paths = self.conversation_paths(conversation_id)
        self._ensure_directory(paths["directory"])
        try:
            if paths["events"].exists():
                self._append_json_event(paths["events"], event)
            else:
                self._rebuild_events_projection(conversation_id)
        except OSError:
            pass
        try:
            if paths["transcript"].exists():
                self._append_text(
                    paths["transcript"],
                    self._event_markdown(event),
                )
            else:
                self._rebuild_transcript_projection(conversation_id)
        except OSError:
            pass

    def _write_meta(self, conversation_id: str, *, title: str | None) -> None:
        self._ensure_conversation_imported(conversation_id)
        with self._projection_guard(conversation_id):
            previous_title = self.store.tui_title(conversation_id)
            effective_title = self.store.set_tui_title(conversation_id, title)
            paths = self.conversation_paths(conversation_id)
            payload = {
                "schema": 1,
                "conversation_id": conversation_id,
                "web_url": f"https://chatgpt.com/c/{conversation_id}",
                "scope": "tui-observed",
                "title": effective_title,
                "updated_at": _now_iso(),
                "events_path": str(paths["events"]),
                "transcript_path": str(paths["transcript"]),
            }
            try:
                self._write_atomic(paths["meta"], payload)
            except OSError:
                pass
            if effective_title and effective_title != previous_title:
                try:
                    self._rewrite_transcript_title(paths["transcript"], effective_title)
                except OSError:
                    pass

    def _rebuild_events_projection(self, conversation_id: str) -> None:
        paths = self.conversation_paths(conversation_id)
        events = self.store.tui_events(conversation_id)
        text = "".join(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            for event in events
        )
        self._write_text_atomic(paths["events"], text)

    def _rebuild_transcript_projection(self, conversation_id: str) -> None:
        paths = self.conversation_paths(conversation_id)
        title = self.store.tui_title(conversation_id)
        pieces = [self._transcript_header(conversation_id, title)]
        pieces.extend(
            self._event_markdown(event)
            for event in self.store.tui_events(conversation_id)
        )
        self._write_text_atomic(paths["transcript"], "".join(pieces))

    def _rewrite_transcript_title(self, path: Path, title: str) -> None:
        if not path.exists():
            return
        temporary = self._temporary_path(path)
        fd: int | None = None
        try:
            with path.open("r", encoding="utf-8") as source:
                fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    PRIVATE_FILE_MODE,
                )
                if PRIVATE_MODES_SUPPORTED:
                    os.fchmod(fd, PRIVATE_FILE_MODE)
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
                    fd = None
                    source.readline()
                    target.write(f"# {title}\n")
                    for line in source:
                        target.write(line)
                    target.flush()
                    os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            if fd is not None:
                os.close(fd)
            self._remove_projection(temporary)

    @staticmethod
    def _transcript_header(conversation_id: str, title: str | None) -> str:
        return "\n".join(
            [
                f"# {title or 'gptty TUI transcript'}",
                "",
                f"- Conversation ID: `{conversation_id}`",
                f"- Web: https://chatgpt.com/c/{conversation_id}",
                "- Scope: `tui-observed` — only turns actually sent/observed through gptty",
                "- Authority: transactional local observation; file projections may be rebuilt from SQLite",
                "",
            ]
        ) + "\n"

    @staticmethod
    def _event_markdown(event: dict[str, Any]) -> str:
        if event.get("terminal_resolution") is True:
            return ""
        role = str(event.get("role") or "event").upper()
        observed = str(event.get("observed_at") or "")
        status = str(event.get("status") or "").strip()
        suffix = f" — {status}" if status and status != "complete" else ""
        lines = [f"## {role}{suffix}"]
        if observed:
            lines.extend([f"_Observed: {observed}_", ""])
        lines.extend([str(event.get("text") or ""), ""])
        return "\n".join(lines) + "\n"

    @staticmethod
    def _read_legacy_events(path: Path) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        events.append(item)
        except OSError:
            pass
        return events

    @staticmethod
    def _read_legacy_title(path: Path) -> str | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        title = str(raw.get("title") or "").strip()
        return title or None

    @staticmethod
    def _read_pending_projection(path: Path) -> dict[str, Any] | None:
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return event if isinstance(event, dict) else None

    @staticmethod
    def _append_json_event(path: Path, event: dict[str, Any]) -> None:
        payload = (
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        TUIArchive._append_text(path, payload)

    @staticmethod
    def _append_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = text.encode("utf-8")
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            if os.name != "nt":
                os.fchmod(fd, 0o600)
            view = memoryview(payload)
            remaining = len(view)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(
                        f"short archive projection write: {written}/{remaining} bytes"
                    )
                view = view[written:]
                remaining -= written
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        TUIArchive._write_text_atomic(path, text)

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        TUIArchive._ensure_directory(path.parent)
        atomic_write_private_text(path, text, sync=True)

    @staticmethod
    def _temporary_path(path: Path) -> Path:
        return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")

    @staticmethod
    def _remove_projection(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            try:
                path.chmod(0o700)
            except OSError:
                pass
