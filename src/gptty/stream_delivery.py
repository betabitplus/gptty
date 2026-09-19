from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_DEFAULT_MAX_BYTES = 50 * 1024 * 1024
_MAX_BACKUPS = 4

_HEALTH_TYPES = {
    "stream_handoff_ws_subscribed",
    "stream_handoff_ws_reconnecting",
    "stream_handoff_delivery_recovered",
    "stream_handoff_server_quiet",
    "stream_handoff_server_stalled",
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
    """Append-only, content-safe delivery evidence for reconnect diagnostics."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_bytes = max(1024 * 1024, int(max_bytes))
        self._lock = threading.Lock()
        self._append(
            {
                "schema": 1,
                "event": "journal_start",
                "observed_at_ms": int(time.time() * 1000),
                "pid": os.getpid(),
            }
        )

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
            # Hash short/medium visible text; for huge tool results the message id,
            # kind, and length are enough to audit delivery without expensive I/O.
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

    def _rotate_if_needed(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            return
        if stat.st_size < self.max_bytes:
            return
        try:
            (self.path.parent / f"{self.path.name}.{_MAX_BACKUPS}").unlink(
                missing_ok=True
            )
        except OSError:
            pass
        for index in range(_MAX_BACKUPS - 1, 0, -1):
            source = self.path.parent / f"{self.path.name}.{index}"
            target = self.path.parent / f"{self.path.name}.{index + 1}"
            try:
                source.replace(target)
            except OSError:
                pass
        try:
            self.path.replace(self.path.parent / f"{self.path.name}.1")
        except OSError:
            pass

    def _append(self, record: dict[str, Any]) -> None:
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                fd = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o600,
                )
                try:
                    os.write(
                        fd,
                        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode(
                            "utf-8"
                        ),
                    )
                finally:
                    os.close(fd)
                os.chmod(self.path, 0o600)
        except OSError:
            # Observability must never interfere with the live chat.
            return
