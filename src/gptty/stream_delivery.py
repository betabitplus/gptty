from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .file_lock import KernelFileLock
from .local_store import DB_FILENAME, LocalEventStore

_DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_PROJECTION_TAIL_ROWS = 20_000
_LEGACY_MAX_BACKUPS = 4

_HEALTH_TYPES = {
    "stream_handoff_ws_subscribed",
    "stream_handoff_ws_reconnecting",
    "stream_handoff_delivery_recovered",
    "stream_handoff_server_quiet",
    "stream_handoff_server_stalled",
    "stream_handoff_terminal_status",
    "stream_handoff_server_resumed",
}
_TEXT_TYPES = {
    "assistant_text_snapshot",
    "assistant_text_delta",
    "assistant_text_revision",
}
_IDENTITY_TYPES = {
    "browser_native_write_identity_resolved",
    "browser_native_write_completed",
}


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class StreamDeliveryJournal:
    """Content-safe delivery evidence backed by the shared transactional store."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        db_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_bytes = max(1024 * 1024, int(max_bytes))
        self.store = LocalEventStore(
            Path(db_path).expanduser()
            if db_path is not None
            else self.path.parent / DB_FILENAME
        )
        self._projection_lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._import_legacy_projections()
        self._append(
            {
                "schema": 1,
                "event": "journal_start",
                "observed_at_ms": int(time.time() * 1000),
                "pid": os.getpid(),
            }
        )

    def _import_legacy_projections(self) -> None:
        candidates = [
            self.path.parent / f"{self.path.name}.{index}"
            for index in range(_LEGACY_MAX_BACKUPS, 0, -1)
        ]
        candidates.append(self.path)
        for candidate in candidates:
            self.store.import_delivery_projection(
                candidate,
                self._legacy_records(candidate),
            )

    @staticmethod
    def _legacy_records(path: Path):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        yield item
        except OSError:
            return

    def observe(self, conversation_ref: str | None, event: Any) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return

        record: dict[str, Any] | None = None
        if event_type in _HEALTH_TYPES:
            record = self._health_record(event_type, event)
        elif event_type == "canonical_intermediate_message":
            record = self._canonical_record(event)
        elif event_type in _TEXT_TYPES:
            record = self._text_record(event_type, event)
        elif event_type in _IDENTITY_TYPES:
            record = self._identity_record(event_type, event)

        if record is None:
            return
        normalized_ref = (
            conversation_ref.strip()
            if isinstance(conversation_ref, str) and conversation_ref.strip()
            else None
        )
        record["conversation_ref"] = normalized_ref
        record["observed_at_ms"] = int(time.time() * 1000)
        self._append(record)

    def records(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return [
            {"local_event_id": event_id, **record}
            for event_id, record in self.store.delivery_events(limit=limit)
        ]

    @staticmethod
    def _health_record(event_type: str, event: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"schema": 1, "event": event_type}
        for key in (
            "topic_id",
            "reason",
            "attempt",
            "server_idle_seconds",
            "silent_seconds",
            "catchup_count",
            "last_offset",
            "last_offset_age_seconds",
            "reconnect_count",
            "stream_status",
        ):
            value = event.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                record[key] = value
        return record

    @staticmethod
    def _canonical_record(event: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": 1,
            "event": "canonical_intermediate_message",
        }
        for key in (
            "message_id",
            "message_kind",
            "turn_exchange_id",
            "source_offset",
            "source_time_ms",
            "tool_name",
        ):
            value = event.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                record[key] = value
        text = event.get("text")
        if isinstance(text, str):
            record["payload_chars"] = len(text)
            if len(text) <= 200_000:
                record["payload_sha256"] = _sha256_text(text)
        return record

    @staticmethod
    def _text_record(event_type: str, event: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"schema": 1, "event": event_type}
        for key in ("message_id", "sequence", "source_offset"):
            value = event.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                record[key] = value
        payload = event.get("delta")
        if not isinstance(payload, str):
            payload = event.get("text")
        if isinstance(payload, str):
            record["payload_chars"] = len(payload)
            record["payload_sha256"] = _sha256_text(payload)
        return record

    @staticmethod
    def _identity_record(event_type: str, event: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"schema": 1, "event": event_type}
        for key in ("conversation_id", "submission_id"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                record[key] = value.strip()
        return record

    def _append(self, record: dict[str, Any]) -> None:
        try:
            event_id = self.store.append_delivery_event(record)
        except Exception:
            # Observability must never interfere with the live chat.
            return

        projected = {"local_event_id": event_id, **record}
        try:
            self._project(projected)
        except Exception:
            # SQLite is authoritative; the support projection is best-effort.
            return

    def _project(self, record: dict[str, Any]) -> None:
        lock = KernelFileLock(self._projection_lock_path)
        lock.acquire(timeout=2.0)
        try:
            payload = self._encode_projection(record)
            try:
                current_size = self.path.stat().st_size
            except OSError:
                current_size = 0
            if current_size + len(payload) > self.max_bytes:
                self._compact_projection()
                return
            self._append_projection(payload)
        finally:
            lock.release()

    def _compact_projection(self) -> None:
        rows = self.store.delivery_events(limit=_PROJECTION_TAIL_ROWS)
        target = max(1, self.max_bytes * 3 // 4)
        selected: list[bytes] = []
        total = 0
        for event_id, record in reversed(rows):
            payload = self._encode_projection(
                {"local_event_id": event_id, **record}
            )
            if selected and total + len(payload) > target:
                break
            selected.append(payload)
            total += len(payload)
        selected.reverse()
        self._replace_projection(b"".join(selected))

    @staticmethod
    def _encode_projection(record: dict[str, Any]) -> bytes:
        return (
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")

    def _append_projection(self, payload: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            if os.name != "nt":
                os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(
                        f"short delivery projection write: {written}/{len(view)} bytes"
                    )
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _replace_projection(self, payload: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                if os.name != "nt":
                    os.fchmod(fd, 0o600)
                view = memoryview(payload)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError(
                            f"short delivery projection rewrite: {written}/{len(view)} bytes"
                        )
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, self.path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
