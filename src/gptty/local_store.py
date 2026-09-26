from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .file_lock import KernelFileLock
from .profiles import profile_paths

SCHEMA_VERSION = 4
DB_FILENAME = "local-state.sqlite3"


class LocalStoreCompatibilityError(RuntimeError):
    """Raised when local durable state was written by a newer schema."""


class LocalSessionConflictError(RuntimeError):
    """Raised when a stale session writer would overwrite a newer revision."""


def local_store_root(*, profile: str | None, state_path: str | Path) -> Path:
    if profile:
        return profile_paths(profile).profile_dir / "runs"
    return Path(state_path).expanduser().parent / ".gptty_runs"


def local_store_path(*, profile: str | None, state_path: str | Path) -> Path:
    return local_store_root(profile=profile, state_path=state_path) / DB_FILENAME


class LocalEventStore:
    """Transactional authority for local sessions and operational event state."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        bootstrap_lock = KernelFileLock(Path(f"{self.db_path}.bootstrap.lock"))
        bootstrap_lock.acquire(timeout=10.0)
        try:
            self._initialize_schema()
        finally:
            bootstrap_lock.release()

    def create_run(
        self,
        *,
        run_id: str,
        summary: dict[str, Any],
        first_event: dict[str, Any],
    ) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO local_runs(run_id, summary_json, updated_at) VALUES(?, ?, ?)",
                (
                    run_id,
                    self._encode(summary),
                    str(first_event.get("timestamp") or ""),
                ),
            )
            db.execute(
                """
                INSERT INTO local_run_events(run_id, seq, event_type, payload_json, created_at)
                VALUES(?, 1, ?, ?, ?)
                """,
                (
                    run_id,
                    str(first_event.get("type") or ""),
                    self._encode(first_event),
                    str(first_event.get("timestamp") or ""),
                ),
            )
            db.commit()

    def append_run_event(
        self,
        *,
        run_id: str,
        summary: dict[str, Any],
        event: dict[str, Any],
    ) -> int:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                """
                UPDATE local_runs
                SET summary_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    self._encode(summary),
                    str(event.get("timestamp") or ""),
                    run_id,
                ),
            )
            if updated.rowcount != 1:
                raise KeyError(f"unknown run_id: {run_id}")
            row = db.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM local_run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            seq = int(row[0]) + 1
            db.execute(
                """
                INSERT INTO local_run_events(run_id, seq, event_type, payload_json, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    seq,
                    str(event.get("type") or ""),
                    self._encode(event),
                    str(event.get("timestamp") or ""),
                ),
            )
            db.commit()
            return seq

    def replace_run_summary(
        self,
        run_id: str,
        summary: dict[str, Any],
        *,
        updated_at: str,
    ) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                """
                UPDATE local_runs
                SET summary_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (self._encode(summary), updated_at, run_id),
            )
            if updated.rowcount != 1:
                raise KeyError(f"unknown run_id: {run_id}")
            db.commit()

    def run_summary(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT summary_json FROM local_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._decode_dict(row[0]) if row is not None else None

    def run_ids_before(self, cutoff: str) -> list[str]:
        """Return non-running local run ids older than an ISO-8601 cutoff."""

        with self._connect() as db:
            rows = db.execute(
                "SELECT run_id, summary_json FROM local_runs WHERE updated_at < ?",
                (cutoff,),
            ).fetchall()
        result: list[str] = []
        for run_id, summary_json in rows:
            summary = self._decode_dict(summary_json)
            if summary is not None and str(summary.get("status") or "") == "running":
                continue
            result.append(str(run_id))
        return result

    def delete_runs(self, run_ids: list[str]) -> int:
        normalized = list(dict.fromkeys(str(run_id) for run_id in run_ids if str(run_id)))
        if not normalized:
            return 0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            deleted = 0
            for run_id in normalized:
                cursor = db.execute("DELETE FROM local_runs WHERE run_id = ?", (run_id,))
                deleted += max(0, int(cursor.rowcount))
            db.commit()
        return deleted

    def prune_runs_before(self, cutoff: str) -> list[str]:
        run_ids = self.run_ids_before(cutoff)
        self.delete_runs(run_ids)
        return run_ids

    def run_events(
        self,
        run_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as db:
            if limit is None:
                rows = db.execute(
                    """
                    SELECT payload_json
                    FROM local_run_events
                    WHERE run_id = ?
                    ORDER BY seq ASC
                    """,
                    (run_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT payload_json
                    FROM (
                        SELECT seq, payload_json
                        FROM local_run_events
                        WHERE run_id = ?
                        ORDER BY seq DESC
                        LIMIT ?
                    )
                    ORDER BY seq ASC
                    """,
                    (run_id, max(0, int(limit))),
                ).fetchall()
        return [
            item
            for row in rows
            if (item := self._decode_dict(row[0])) is not None
        ]

    def import_delivery_projection(
        self,
        source_path: str | Path,
        records: Iterable[Any],
    ) -> int:
        source_key = str(Path(source_path).expanduser().resolve())
        imported = 0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT 1 FROM delivery_imports WHERE source_path = ?",
                (source_key,),
            ).fetchone()
            if existing is not None:
                db.commit()
                return 0
            for raw in records:
                if not isinstance(raw, dict):
                    continue
                record = dict(raw)
                record.pop("local_event_id", None)
                observed_at_ms = record.get("observed_at_ms")
                if isinstance(observed_at_ms, bool) or not isinstance(
                    observed_at_ms, (int, float)
                ):
                    observed_at_ms = 0
                conversation_ref = record.get("conversation_ref")
                db.execute(
                    """
                    INSERT INTO delivery_events(
                        conversation_ref,
                        event_type,
                        payload_json,
                        observed_at_ms
                    )
                    VALUES(?, ?, ?, ?)
                    """,
                    (
                        str(conversation_ref).strip()
                        if isinstance(conversation_ref, str)
                        and conversation_ref.strip()
                        else None,
                        str(record.get("event") or ""),
                        self._encode(record),
                        int(observed_at_ms),
                    ),
                )
                imported += 1
            db.execute(
                """
                INSERT INTO delivery_imports(source_path, imported_at)
                VALUES(?, CURRENT_TIMESTAMP)
                """,
                (source_key,),
            )
            db.commit()
        return imported

    def append_delivery_event(self, record: dict[str, Any]) -> int:
        observed_at_ms = record.get("observed_at_ms")
        if isinstance(observed_at_ms, bool) or not isinstance(observed_at_ms, (int, float)):
            observed_at_ms = 0
        conversation_ref = record.get("conversation_ref")
        event_type = str(record.get("event") or "")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                """
                INSERT INTO delivery_events(
                    conversation_ref,
                    event_type,
                    payload_json,
                    observed_at_ms
                )
                VALUES(?, ?, ?, ?)
                """,
                (
                    str(conversation_ref).strip()
                    if isinstance(conversation_ref, str) and conversation_ref.strip()
                    else None,
                    event_type,
                    self._encode(record),
                    int(observed_at_ms),
                ),
            )
            event_id = int(cursor.lastrowid)
            db.commit()
        return event_id

    def delivery_events(
        self,
        *,
        limit: int | None = None,
    ) -> list[tuple[int, dict[str, Any]]]:
        with self._connect() as db:
            if limit is None:
                rows = db.execute(
                    """
                    SELECT id, payload_json
                    FROM delivery_events
                    ORDER BY id ASC
                    """
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT id, payload_json
                    FROM (
                        SELECT id, payload_json
                        FROM delivery_events
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (max(0, int(limit)),),
                ).fetchall()
        result: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            item = self._decode_dict(row[1])
            if item is not None:
                result.append((int(row[0]), item))
        return result

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT session_id, revision, kind, discovery_hint,
                       current_conversation, model, reasoning_effort, goal_id,
                       created_at_ms, last_seen_at_ms, imported_from
                FROM local_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return self._session_row(row)

    def session_imported_from(self, source_path: str | Path) -> bool:
        source = str(Path(source_path).expanduser().resolve())
        with self._connect() as db:
            row = db.execute(
                """
                SELECT 1
                FROM local_session_imports
                WHERE source_path = ?
                UNION ALL
                SELECT 1
                FROM local_sessions
                WHERE imported_from = ?
                LIMIT 1
                """,
                (source, source),
            ).fetchone()
        return row is not None

    def create_session_claiming_import(
        self,
        session_id: str,
        *,
        source_path: str | Path,
        kind: str,
        discovery_hint: str | None,
        current_conversation: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        goal_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Create one session while atomically claiming a legacy state source.

        Returns ``(session, imported)``. If another distinct session has already
        claimed the legacy source, ``session`` is ``None`` so the caller can seed
        from the normal default instead. If this same session was concurrently
        created by another process, its existing row is returned with
        ``imported=False``.
        """

        source = str(Path(source_path).expanduser().resolve())
        now_ms = int(time.time() * 1000)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                """
                SELECT session_id, revision, kind, discovery_hint,
                       current_conversation, model, reasoning_effort, goal_id,
                       created_at_ms, last_seen_at_ms, imported_from
                FROM local_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if existing is not None:
                db.commit()
                session = self._session_row(existing)
                if session is None:
                    raise RuntimeError(f"failed to read local session: {session_id}")
                return session, False

            claimed = db.execute(
                """
                SELECT session_id
                FROM local_session_imports
                WHERE source_path = ?
                UNION ALL
                SELECT session_id
                FROM local_sessions
                WHERE imported_from = ?
                LIMIT 1
                """,
                (source, source),
            ).fetchone()
            if claimed is not None:
                db.commit()
                return None, False

            db.execute(
                """
                INSERT INTO local_sessions(
                    session_id, revision, kind, discovery_hint,
                    current_conversation, model, reasoning_effort, goal_id,
                    created_at_ms, last_seen_at_ms, imported_from
                )
                VALUES(?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind,
                    discovery_hint,
                    current_conversation,
                    model,
                    reasoning_effort,
                    goal_id,
                    now_ms,
                    now_ms,
                    source,
                ),
            )
            db.execute(
                """
                INSERT INTO local_session_imports(source_path, session_id, imported_at_ms)
                VALUES(?, ?, ?)
                """,
                (source, session_id, now_ms),
            )
            row = db.execute(
                """
                SELECT session_id, revision, kind, discovery_hint,
                       current_conversation, model, reasoning_effort, goal_id,
                       created_at_ms, last_seen_at_ms, imported_from
                FROM local_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            db.commit()
        session = self._session_row(row)
        if session is None:
            raise RuntimeError(f"failed to create local session: {session_id}")
        return session, True

    def create_session(
        self,
        session_id: str,
        *,
        kind: str,
        discovery_hint: str | None,
        current_conversation: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        goal_id: str | None = None,
        imported_from: str | None = None,
    ) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                INSERT OR IGNORE INTO local_sessions(
                    session_id, revision, kind, discovery_hint,
                    current_conversation, model, reasoning_effort, goal_id,
                    created_at_ms, last_seen_at_ms, imported_from
                )
                VALUES(?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind,
                    discovery_hint,
                    current_conversation,
                    model,
                    reasoning_effort,
                    goal_id,
                    now_ms,
                    now_ms,
                    imported_from,
                ),
            )
            row = db.execute(
                """
                SELECT session_id, revision, kind, discovery_hint,
                       current_conversation, model, reasoning_effort, goal_id,
                       created_at_ms, last_seen_at_ms, imported_from
                FROM local_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            db.commit()
        session = self._session_row(row)
        if session is None:
            raise RuntimeError(f"failed to create local session: {session_id}")
        return session

    def save_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        current_conversation: str | None,
        model: str | None,
        reasoning_effort: str | None = None,
        goal_id: str | None,
        discovery_hint: str | None = None,
    ) -> int:
        now_ms = int(time.time() * 1000)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                """
                UPDATE local_sessions
                SET revision = revision + 1,
                    current_conversation = ?,
                    model = ?,
                    reasoning_effort = ?,
                    goal_id = ?,
                    discovery_hint = COALESCE(?, discovery_hint),
                    last_seen_at_ms = ?
                WHERE session_id = ? AND revision = ?
                """,
                (
                    current_conversation,
                    model,
                    reasoning_effort,
                    goal_id,
                    discovery_hint,
                    now_ms,
                    session_id,
                    int(expected_revision),
                ),
            )
            if cursor.rowcount != 1:
                actual = db.execute(
                    "SELECT revision FROM local_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                db.rollback()
                if actual is None:
                    raise LocalSessionConflictError(
                        f"local session disappeared: {session_id}"
                    )
                raise LocalSessionConflictError(
                    f"local session changed concurrently: {session_id} "
                    f"expected={expected_revision} actual={int(actual[0])}"
                )
            revision = int(expected_revision) + 1
            db.commit()
        return revision

    def touch_session(self, session_id: str, *, discovery_hint: str | None = None) -> None:
        now_ms = int(time.time() * 1000)
        with self._connect() as db:
            db.execute(
                """
                UPDATE local_sessions
                SET last_seen_at_ms = ?,
                    discovery_hint = COALESCE(?, discovery_hint)
                WHERE session_id = ?
                """,
                (now_ms, discovery_hint, session_id),
            )
            db.commit()

    def put_pending_tui_event(self, turn_id: str, event: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                INSERT INTO tui_pending(turn_id, payload_json, created_at)
                VALUES(?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (
                    turn_id,
                    self._encode(event),
                    str(event.get("observed_at") or ""),
                ),
            )
            db.commit()

    def prune_pending_tui_before(self, cutoff: str) -> list[str]:
        """Delete orphan pending TUI prompts older than an ISO-8601 cutoff."""

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT turn_id FROM tui_pending WHERE created_at <> '' AND created_at < ?",
                (cutoff,),
            ).fetchall()
            turn_ids = [str(row[0]) for row in rows]
            if turn_ids:
                db.executemany(
                    "DELETE FROM tui_pending WHERE turn_id = ?",
                    ((turn_id,) for turn_id in turn_ids),
                )
            db.commit()
        return turn_ids

    def pop_pending_tui_event(self, turn_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload_json FROM tui_pending WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is not None:
                db.execute("DELETE FROM tui_pending WHERE turn_id = ?", (turn_id,))
            db.commit()
        return self._decode_dict(row[0]) if row is not None else None

    def bind_pending_tui_event(
        self,
        turn_id: str,
        conversation_id: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload_json FROM tui_pending WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                db.commit()
                return None, False
            event = self._decode_dict(row[0])
            if event is None:
                db.execute("DELETE FROM tui_pending WHERE turn_id = ?", (turn_id,))
                db.commit()
                return None, False
            event["conversation_id"] = conversation_id
            cursor = self._insert_tui_event_row(db, conversation_id, event)
            db.execute("DELETE FROM tui_pending WHERE turn_id = ?", (turn_id,))
            db.commit()
            return event, cursor.rowcount == 1

    def insert_tui_event(
        self,
        conversation_id: str,
        event: dict[str, Any],
    ) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = self._insert_tui_event_row(db, conversation_id, event)
            inserted = cursor.rowcount == 1
            db.commit()
            return inserted

    def import_tui_conversation(
        self,
        conversation_id: str,
        *,
        events: list[dict[str, Any]],
        title: str | None,
        imported_at: str,
    ) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            marker = db.execute(
                "SELECT 1 FROM tui_imports WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if marker is not None:
                db.commit()
                return
            for event in events:
                try:
                    self._insert_tui_event_row(db, conversation_id, event)
                except ValueError:
                    continue
            normalized_title = str(title).strip() if title is not None else ""
            if normalized_title:
                db.execute(
                    """
                    INSERT INTO tui_conversations(conversation_id, title, updated_at)
                    VALUES(?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        title = COALESCE(tui_conversations.title, excluded.title),
                        updated_at = excluded.updated_at
                    """,
                    (conversation_id, normalized_title, imported_at),
                )
            db.execute(
                """
                INSERT INTO tui_imports(conversation_id, imported_at)
                VALUES(?, ?)
                """,
                (conversation_id, imported_at),
            )
            db.commit()

    def tui_imported(self, conversation_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM tui_imports WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return row is not None

    def tui_events(
        self,
        conversation_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as db:
            if limit is None:
                rows = db.execute(
                    """
                    SELECT payload_json
                    FROM tui_events
                    WHERE conversation_id = ?
                    ORDER BY id ASC
                    """,
                    (conversation_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT payload_json
                    FROM (
                        SELECT id, payload_json
                        FROM tui_events
                        WHERE conversation_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (conversation_id, max(0, int(limit))),
                ).fetchall()
        return [
            item
            for row in rows
            if (item := self._decode_dict(row[0])) is not None
        ]

    def tui_terminal_exists(
        self,
        conversation_id: str,
        *,
        role: str,
        status: str,
        text: str,
        source: str,
    ) -> bool:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT 1
                FROM tui_events
                WHERE conversation_id = ?
                  AND role = ?
                  AND status = ?
                  AND text_value = ?
                  AND terminal_source = ?
                LIMIT 1
                """,
                (conversation_id, role, status, text, source),
            ).fetchone()
        return row is not None

    def insert_tui_terminal_once(
        self,
        conversation_id: str,
        event: dict[str, Any],
    ) -> bool:
        role = str(event.get("role") or "")
        status = str(event.get("status") or "")
        text = str(event.get("text") or "")
        source = str(event.get("terminal_source") or "")
        if not status or not text or not source:
            raise ValueError("terminal status, text, and source are required")

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                """
                SELECT status, text_value, terminal_source
                FROM tui_events
                WHERE conversation_id = ?
                  AND role = ?
                  AND status <> ''
                  AND text_value <> ''
                ORDER BY id DESC
                LIMIT 1
                """,
                (conversation_id, role),
            ).fetchone()
            if current is not None and (
                str(current[0] or "") == status
                and str(current[1] or "") == text
                and str(current[2] or "") == source
            ):
                db.commit()
                return False
            cursor = self._insert_tui_event_row(db, conversation_id, event)
            inserted = cursor.rowcount == 1
            db.commit()
            return inserted

    def insert_tui_terminal_resolution_if_current(
        self,
        conversation_id: str,
        event: dict[str, Any],
        *,
        resolved_status: str,
    ) -> bool:
        """Atomically supersede one current chat-level terminal status."""

        expected_status = str(resolved_status or "").strip().lower()
        if not expected_status:
            raise ValueError("resolved_status is required")
        if event.get("terminal_resolution") is not True:
            raise ValueError("terminal resolution event must declare terminal_resolution")

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT status, payload_json
                FROM tui_events
                WHERE conversation_id = ?
                  AND lower(role) = 'chat'
                  AND status <> ''
                  AND text_value <> ''
                ORDER BY id DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            current_payload = self._decode_dict(row[1]) if row is not None else None
            current_status = str(row[0] or "").strip().lower() if row is not None else ""
            if (
                row is None
                or current_payload is None
                or current_payload.get("terminal_resolution") is True
                or current_status != expected_status
            ):
                db.commit()
                return False

            cursor = self._insert_tui_event_row(db, conversation_id, event)
            inserted = cursor.rowcount == 1
            db.commit()
            return inserted

    def latest_chat_terminal(
        self,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT payload_json
                FROM tui_events
                WHERE conversation_id = ?
                  AND lower(role) = 'chat'
                  AND status <> ''
                  AND text_value <> ''
                ORDER BY id DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        return self._decode_dict(row[0]) if row is not None else None

    def tui_conversation_ids_before(self, cutoff: str) -> list[str]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT conversation_id FROM tui_conversations WHERE updated_at < ?",
                (cutoff,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def delete_tui_conversations(self, conversation_ids: list[str]) -> int:
        normalized = list(
            dict.fromkeys(
                str(conversation_id)
                for conversation_id in conversation_ids
                if str(conversation_id)
            )
        )
        if not normalized:
            return 0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            deleted = 0
            for conversation_id in normalized:
                db.execute(
                    "DELETE FROM tui_events WHERE conversation_id = ?",
                    (conversation_id,),
                )
                db.execute(
                    "DELETE FROM tui_imports WHERE conversation_id = ?",
                    (conversation_id,),
                )
                cursor = db.execute(
                    "DELETE FROM tui_conversations WHERE conversation_id = ?",
                    (conversation_id,),
                )
                deleted += max(0, int(cursor.rowcount))
            db.commit()
        return deleted

    def prune_tui_conversations_before(self, cutoff: str) -> list[str]:
        conversation_ids = self.tui_conversation_ids_before(cutoff)
        self.delete_tui_conversations(conversation_ids)
        return conversation_ids

    def privacy_inventory(self) -> dict[str, int]:
        """Return content-free counts for local privacy/lifecycle status."""

        with self._connect() as db:
            return {
                "runs": int(db.execute("SELECT COUNT(*) FROM local_runs").fetchone()[0]),
                "pending_prompts": int(db.execute("SELECT COUNT(*) FROM tui_pending").fetchone()[0]),
                "archived_conversations": int(
                    db.execute(
                        """
                        SELECT COUNT(*)
                        FROM (
                            SELECT conversation_id FROM tui_conversations
                            UNION
                            SELECT conversation_id FROM tui_events
                            UNION
                            SELECT conversation_id FROM tui_imports
                        )
                        """
                    ).fetchone()[0]
                ),
                "delivery_events": int(
                    db.execute("SELECT COUNT(*) FROM delivery_events").fetchone()[0]
                ),
            }

    def set_tui_title(self, conversation_id: str, title: str | None) -> str | None:
        normalized = str(title).strip() if title is not None else ""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT title FROM tui_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            effective = normalized or (
                str(previous[0]) if previous and previous[0] else ""
            )
            db.execute(
                """
                INSERT INTO tui_conversations(conversation_id, title, updated_at)
                VALUES(?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, effective or None),
            )
            db.commit()
        return effective or None

    def tui_title(self, conversation_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT title FROM tui_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        value = str(row[0]).strip()
        return value or None

    def _insert_tui_event_row(
        self,
        db: sqlite3.Connection,
        conversation_id: str,
        event: dict[str, Any],
    ) -> sqlite3.Cursor:
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("TUI event_id is required")
        return db.execute(
            """
            INSERT OR IGNORE INTO tui_events(
                conversation_id,
                event_id,
                role,
                status,
                terminal_source,
                text_value,
                payload_json,
                observed_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                event_id,
                str(event.get("role") or ""),
                str(event.get("status") or ""),
                str(event.get("terminal_source") or ""),
                str(event.get("text") or ""),
                self._encode(event),
                str(event.get("observed_at") or ""),
            ),
        )

    def _initialize_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._protect_path(self.db_path.parent, directory=True)
        db = self._open_connection()
        try:
            journal_mode = str(db.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            if journal_mode != "wal":
                journal_mode = str(
                    db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                ).lower()
                if journal_mode != "wal":
                    raise RuntimeError(
                        f"failed to enable WAL for local state database: {journal_mode}"
                    )
            current = int(db.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise LocalStoreCompatibilityError(
                    "local state schema is newer than this gptty build: "
                    f"{current} > {SCHEMA_VERSION}"
                )
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS local_runs(
                    run_id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_run_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES local_runs(run_id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_local_run_events_run_seq
                    ON local_run_events(run_id, seq);

                CREATE TABLE IF NOT EXISTS delivery_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_ref TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    observed_at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_delivery_events_conversation_id
                    ON delivery_events(conversation_ref, id);
                CREATE INDEX IF NOT EXISTS idx_delivery_events_type_id
                    ON delivery_events(event_type, id);
                CREATE TABLE IF NOT EXISTS delivery_imports(
                    source_path TEXT PRIMARY KEY,
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS local_sessions(
                    session_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    discovery_hint TEXT,
                    current_conversation TEXT,
                    model TEXT,
                    reasoning_effort TEXT,
                    goal_id TEXT,
                    created_at_ms INTEGER NOT NULL,
                    last_seen_at_ms INTEGER NOT NULL,
                    imported_from TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_local_sessions_kind_seen
                    ON local_sessions(kind, last_seen_at_ms);
                CREATE INDEX IF NOT EXISTS idx_local_sessions_imported_from
                    ON local_sessions(imported_from);
                CREATE TABLE IF NOT EXISTS local_session_imports(
                    source_path TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    imported_at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_local_session_imports_session_id
                    ON local_session_imports(session_id);

                CREATE TABLE IF NOT EXISTS tui_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    terminal_source TEXT NOT NULL,
                    text_value TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    UNIQUE(conversation_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tui_events_conversation_id
                    ON tui_events(conversation_id, id);
                CREATE INDEX IF NOT EXISTS idx_tui_events_terminal
                    ON tui_events(conversation_id, role, status, terminal_source);

                CREATE TABLE IF NOT EXISTS tui_pending(
                    turn_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tui_conversations(
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tui_imports(
                    conversation_id TEXT PRIMARY KEY,
                    imported_at TEXT NOT NULL
                );
                """
            )
            session_columns = {
                str(row[1]) for row in db.execute("PRAGMA table_info(local_sessions)")
            }
            if "reasoning_effort" not in session_columns:
                db.execute("ALTER TABLE local_sessions ADD COLUMN reasoning_effort TEXT")
            if current < SCHEMA_VERSION:
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            db.commit()
        finally:
            db.close()
        self._protect_database_files()

    def _connect(self) -> sqlite3.Connection:
        db = self._open_connection()
        self._protect_database_files()
        return db

    def _open_connection(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=5.0)
        # Install the busy handler before any pragma/transaction that may need a
        # database lock. WAL itself is a persistent database property and is set
        # once by schema bootstrap rather than on every connection.
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA fullfsync=ON")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _protect_database_files(self) -> None:
        self._protect_path(self.db_path, directory=False)
        self._protect_path(Path(f"{self.db_path}-wal"), directory=False)
        self._protect_path(Path(f"{self.db_path}-shm"), directory=False)

    @staticmethod
    def _protect_path(path: Path, *, directory: bool) -> None:
        if os.name == "nt" or not path.exists():
            return
        try:
            path.chmod(0o700 if directory else 0o600)
        except OSError:
            pass

    @staticmethod
    def _session_row(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "session_id": str(row[0]),
            "revision": int(row[1]),
            "kind": str(row[2]),
            "discovery_hint": str(row[3]) if row[3] is not None else None,
            "current_conversation": str(row[4]) if row[4] is not None else None,
            "model": str(row[5]) if row[5] is not None else None,
            "reasoning_effort": str(row[6]) if row[6] is not None else None,
            "goal_id": str(row[7]) if row[7] is not None else None,
            "created_at_ms": int(row[8]),
            "last_seen_at_ms": int(row[9]),
            "imported_from": str(row[10]) if row[10] is not None else None,
        }

    @staticmethod
    def _encode(payload: dict[str, Any]) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_dict(payload: str) -> dict[str, Any] | None:
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None
