from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    """Append-only record of turns that actually passed through gptty."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser() if root is not None else archive_root()
        self.pending_dir = self.root / "pending"
        self.conversations_dir = self.root / "conversations"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

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
            self._refresh_projection(conversation_id)
        else:
            self._write_atomic(self.pending_dir / f"{turn_id}.json", event)
        return turn_id

    def bind_turn(self, turn_id: str, conversation_ref: str) -> str:
        conversation_id = _conversation_id(conversation_ref)
        pending = self.pending_dir / f"{turn_id}.json"
        if pending.exists():
            try:
                event = json.loads(pending.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                event = None
            if isinstance(event, dict):
                event["conversation_id"] = conversation_id
                self._append_conversation_event(conversation_id, event)
                self._refresh_projection(conversation_id)
            try:
                pending.unlink()
            except FileNotFoundError:
                pass
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
        self._refresh_projection(conversation_id)

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
        paths = self.conversation_paths(conversation_id)
        directory = paths["directory"]
        directory.mkdir(parents=True, exist_ok=True)
        events_path = paths["events"]
        event_id = str(event.get("event_id") or "")
        if event_id and self._event_exists(events_path, event_id):
            return
        payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(events_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            written = os.write(fd, payload)
            if written != len(payload):
                raise OSError(f"short archive write: {written}/{len(payload)} bytes")
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _event_exists(path: Path, event_id: str) -> bool:
        if not path.exists():
            return False
        needle = f'"event_id":"{event_id}"'
        try:
            with path.open("r", encoding="utf-8") as handle:
                return any(needle in line for line in handle)
        except OSError:
            return False

    def _write_meta(self, conversation_id: str, *, title: str | None) -> None:
        paths = self.conversation_paths(conversation_id)
        previous: dict[str, Any] = {}
        try:
            raw = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                previous = raw
        except (OSError, json.JSONDecodeError):
            pass
        payload = {
            "schema": 1,
            "conversation_id": conversation_id,
            "web_url": f"https://chatgpt.com/c/{conversation_id}",
            "scope": "tui-observed",
            "title": title or previous.get("title"),
            "updated_at": _now_iso(),
            "events_path": str(paths["events"]),
            "transcript_path": str(paths["transcript"]),
        }
        self._write_atomic(paths["meta"], payload)

    def _refresh_projection(self, conversation_id: str) -> None:
        paths = self.conversation_paths(conversation_id)
        events = self._read_events(paths["events"])
        title = None
        try:
            raw = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                title = raw.get("title")
        except (OSError, json.JSONDecodeError):
            pass

        lines = [
            f"# {title or 'gptty TUI transcript'}",
            "",
            f"- Conversation ID: `{conversation_id}`",
            f"- Web: https://chatgpt.com/c/{conversation_id}",
            "- Scope: `tui-observed` — only turns actually sent/observed through gptty",
            "- Authority: append-only local observation; absence from ChatGPT web does not invalidate these events",
            "",
        ]
        for event in events:
            role = str(event.get("role") or "event").upper()
            observed = str(event.get("observed_at") or "")
            status = str(event.get("status") or "").strip()
            suffix = f" — {status}" if status and status != "complete" else ""
            lines.append(f"## {role}{suffix}")
            if observed:
                lines.append(f"_Observed: {observed}_")
                lines.append("")
            text = str(event.get("text") or "")
            lines.append(text)
            lines.append("")

        paths["directory"].mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(paths["transcript"], "\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _read_events(path: Path) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if not path.exists():
            return events
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
        except OSError:
            return []
        return events

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        TUIArchive._write_text_atomic(path, text)

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
