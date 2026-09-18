from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MATCH_WINDOW_MS = 10_000
DEFAULT_REFRESH_SECONDS = 0.5


@dataclass(frozen=True)
class CodexProActivitySnapshot:
    bound: bool = False
    session_sha256: str | None = None
    last_event_age_seconds: float | None = None
    last_tool: str | None = None
    inflight: bool = False
    inflight_tool: str | None = None
    last_heartbeat_age_seconds: float | None = None
    outcome: str | None = None


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    return str(value)


def canonical_args_sha256(value: Any) -> str:
    payload = json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _offset_ms(value: Any) -> int | None:
    if not isinstance(value, str) or not value.strip():
        return None
    head = value.strip().split("-", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _parse_codexpro_tool_call(event: Any) -> tuple[str, str, int] | None:
    if not isinstance(event, dict):
        return None
    if event.get("type") != "canonical_intermediate_message":
        return None
    if event.get("message_kind") != "tool_call":
        return None
    text = event.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    path = payload.get("path")
    args = payload.get("args")
    if not isinstance(path, str) or "/link_" not in path:
        return None
    segments = [part for part in path.split("/") if part]
    if len(segments) < 3 or "codex" not in segments[0].lower():
        return None
    tool = segments[-1].strip()
    if not tool:
        return None
    source_ms = _offset_ms(event.get("source_offset"))
    if source_ms is None:
        source_time_ms = event.get("source_time_ms")
        if (
            isinstance(source_time_ms, (int, float))
            and not isinstance(source_time_ms, bool)
            and source_time_ms > 0
        ):
            source_ms = int(source_time_ms)
    if source_ms is None:
        return None
    return tool, canonical_args_sha256(args if isinstance(args, dict) else {}), source_ms


class CodexProActivityTracker:
    def __init__(
        self,
        *,
        journal_path: str | Path | None = None,
        mapping_path: str | Path | None = None,
        match_window_ms: int = DEFAULT_MATCH_WINDOW_MS,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    ) -> None:
        configured_journal = os.environ.get("GPTTY_CODEXPRO_ACTIVITY_JOURNAL", "").strip()
        self.journal_path = Path(
            journal_path
            or configured_journal
            or (Path.home() / ".codexpro" / "logs" / "tool-activity.jsonl")
        ).expanduser()
        configured_mapping = os.environ.get("GPTTY_CODEXPRO_SESSION_MAP", "").strip()
        self.mapping_path = (
            Path(mapping_path or configured_mapping).expanduser()
            if (mapping_path or configured_mapping)
            else None
        )
        self.match_window_ms = max(1, int(match_window_ms))
        self.refresh_seconds = max(0.0, float(refresh_seconds))
        self._lock = threading.RLock()
        self._mapping: dict[str, str] = {}
        self._records: list[dict[str, Any]] = []
        self._journal_signature: tuple[int, int] | None = None
        self._last_refresh_at = 0.0
        self._load_mapping()

    def _load_mapping(self) -> None:
        if self.mapping_path is None or not self.mapping_path.exists():
            return
        try:
            payload = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        conversations = payload.get("conversations") if isinstance(payload, dict) else None
        if not isinstance(conversations, dict):
            return
        for conversation_id, value in conversations.items():
            if not isinstance(conversation_id, str):
                continue
            if isinstance(value, str):
                session = value
            elif isinstance(value, dict):
                session = value.get("session_sha256")
            else:
                continue
            if isinstance(session, str) and session.startswith("sha256:"):
                self._mapping[conversation_id] = session

    def _save_mapping(self) -> None:
        if self.mapping_path is None:
            return
        payload = {
            "version": 1,
            "conversations": {
                conversation_id: {"session_sha256": session}
                for conversation_id, session in sorted(self._mapping.items())
            },
        }
        tmp = self.mapping_path.with_name(f".{self.mapping_path.name}.tmp")
        try:
            self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(tmp, 0o600)
            tmp.replace(self.mapping_path)
            os.chmod(self.mapping_path, 0o600)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _session_fingerprint(record: dict[str, Any]) -> str | None:
        headers = record.get("header_fingerprints")
        meta = record.get("meta_fingerprints")
        header_session = (
            headers.get("x-openai-session") if isinstance(headers, dict) else None
        )
        meta_session = meta.get("openai/session") if isinstance(meta, dict) else None
        if isinstance(header_session, str) and isinstance(meta_session, str):
            return header_session if header_session == meta_session else None
        if isinstance(header_session, str):
            return header_session
        if isinstance(meta_session, str):
            return meta_session
        return None

    def _refresh_records(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_refresh_at < self.refresh_seconds:
            return
        self._last_refresh_at = now
        try:
            stat = self.journal_path.stat()
        except OSError:
            self._records = []
            self._journal_signature = None
            return
        signature = (stat.st_mtime_ns, stat.st_size)
        if not force and signature == self._journal_signature:
            return
        try:
            text = self.journal_path.read_text(encoding="utf-8")
        except OSError:
            return
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
        self._records = records
        self._journal_signature = signature

    def observe_tool_call(self, conversation_id: str, event: Any) -> bool:
        parsed = _parse_codexpro_tool_call(event)
        if parsed is None or not conversation_id.strip():
            return False
        tool, args_sha256, source_ms = parsed
        with self._lock:
            self._refresh_records(force=True)
            candidates: list[tuple[int, str]] = []
            for record in self._records:
                if record.get("event") != "tool_start":
                    continue
                if record.get("tool") != tool or record.get("args_sha256") != args_sha256:
                    continue
                observed = record.get("observed_at_ms")
                if not isinstance(observed, (int, float)) or isinstance(observed, bool):
                    continue
                delta = abs(int(observed) - source_ms)
                if delta > self.match_window_ms:
                    continue
                session = self._session_fingerprint(record)
                if session is not None:
                    candidates.append((delta, session))
            if not candidates:
                return False
            sessions = {session for _, session in candidates}
            if len(sessions) != 1:
                return False
            session = next(iter(sessions))
            previous = self._mapping.get(conversation_id)
            self._mapping[conversation_id] = session
            if previous != session:
                self._save_mapping()
            return True

    def session_for(self, conversation_id: str | None) -> str | None:
        if not conversation_id:
            return None
        with self._lock:
            return self._mapping.get(conversation_id)

    def snapshot(
        self,
        conversation_id: str | None,
        *,
        now_ms: int | None = None,
    ) -> CodexProActivitySnapshot:
        if not conversation_id:
            return CodexProActivitySnapshot()
        with self._lock:
            session = self._mapping.get(conversation_id)
            if session is None:
                return CodexProActivitySnapshot()
            self._refresh_records()
            starts: dict[str, dict[str, Any]] = {}
            events_by_activity: dict[str, list[dict[str, Any]]] = {}
            for record in self._records:
                activity_id = record.get("activity_id")
                if not isinstance(activity_id, str) or not activity_id:
                    continue
                if record.get("event") == "tool_start":
                    if self._session_fingerprint(record) != session:
                        continue
                    starts[activity_id] = record
                    events_by_activity.setdefault(activity_id, []).append(record)
                elif activity_id in starts:
                    events_by_activity.setdefault(activity_id, []).append(record)
            if not starts:
                return CodexProActivitySnapshot(bound=True, session_sha256=session)

            current_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
            latest_record: dict[str, Any] | None = None
            latest_tool: str | None = None
            latest_observed = -1
            inflight: list[tuple[int, str, float | None]] = []
            latest_outcome: str | None = None

            for activity_id, start in starts.items():
                events = events_by_activity.get(activity_id, [start])
                finish = next(
                    (item for item in reversed(events) if item.get("event") == "tool_finish"),
                    None,
                )
                heartbeat = next(
                    (
                        item
                        for item in reversed(events)
                        if item.get("event") == "tool_heartbeat"
                    ),
                    None,
                )
                for item in events:
                    observed = item.get("observed_at_ms")
                    if (
                        isinstance(observed, (int, float))
                        and not isinstance(observed, bool)
                        and int(observed) >= latest_observed
                    ):
                        latest_observed = int(observed)
                        latest_record = item
                        tool_value = start.get("tool")
                        latest_tool = tool_value if isinstance(tool_value, str) else None
                        if isinstance(finish, dict):
                            outcome = finish.get("outcome")
                            latest_outcome = outcome if isinstance(outcome, str) else None
                if finish is None:
                    heartbeat_observed = (
                        heartbeat.get("observed_at_ms") if isinstance(heartbeat, dict) else None
                    )
                    heartbeat_age = (
                        max(0.0, (current_ms - int(heartbeat_observed)) / 1000.0)
                        if isinstance(heartbeat_observed, (int, float))
                        and not isinstance(heartbeat_observed, bool)
                        else None
                    )
                    start_observed = start.get("observed_at_ms")
                    sort_value = (
                        int(heartbeat_observed)
                        if isinstance(heartbeat_observed, (int, float))
                        and not isinstance(heartbeat_observed, bool)
                        else int(start_observed)
                        if isinstance(start_observed, (int, float))
                        and not isinstance(start_observed, bool)
                        else 0
                    )
                    tool_value = start.get("tool")
                    inflight.append(
                        (
                            sort_value,
                            tool_value if isinstance(tool_value, str) else "",
                            heartbeat_age,
                        )
                    )

            last_age = (
                max(0.0, (current_ms - latest_observed) / 1000.0)
                if latest_record is not None and latest_observed >= 0
                else None
            )
            inflight.sort(reverse=True)
            active = inflight[0] if inflight else None
            return CodexProActivitySnapshot(
                bound=True,
                session_sha256=session,
                last_event_age_seconds=last_age,
                last_tool=latest_tool,
                inflight=active is not None,
                inflight_tool=active[1] or None if active is not None else None,
                last_heartbeat_age_seconds=active[2] if active is not None else None,
                outcome=latest_outcome,
            )
