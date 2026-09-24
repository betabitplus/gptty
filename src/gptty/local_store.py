from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .profiles import profile_paths

SCHEMA_VERSION = 1
DB_FILENAME = "local-state.sqlite3"


class LocalStoreCompatibilityError(RuntimeError):
    """Raised when local durable state was written by a newer schema."""


def local_store_root(*, profile: str | None, state_path: str | Path) -> Path:
    if profile:
        return profile_paths(profile).profile_dir / "runs"
    return Path(state_path).expanduser().parent / ".gptty_runs"


def local_store_path(*, profile: str | None, state_path: str | Path) -> Path:
    return local_store_root(profile=profile, state_path=state_path) / DB_FILENAME


class LocalEventStore:
    """Transactional authority for local run and TUI observation events."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self._initialize_schema()

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
            existing = db.execute(
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
            if existing is not None:
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
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA fullfsync=ON")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
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
