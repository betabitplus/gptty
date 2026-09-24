from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import GoalState, goal_state_from_dict


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_goal_id(goal: GoalState) -> str:
    if goal.goal_id:
        return goal.goal_id
    goal.goal_id = uuid.uuid4().hex
    return goal.goal_id


class GoalConflictError(RuntimeError):
    """Raised when a stale Goal writer would overwrite a newer revision."""


class GoalStore:
    """Transactional Goal state plus an append-only machine-owned event journal.

    SQLite is authoritative. ``goal.json`` and ``checkpoint.md`` are portable,
    human-readable projections and may be copied for backup or inspection.
    """

    SCHEMA_VERSION = 3

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.root = self.state_path.parent / "goals"
        self.db_path = self.root / "goal-state.sqlite3"
        self.last_projection_error: OSError | None = None

    def goal_dir(self, goal: GoalState | str) -> Path:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        return self.root / goal_id

    def goal_path(self, goal: GoalState | str) -> Path:
        return self.goal_dir(goal) / "goal.json"

    def checkpoint_path(self, goal: GoalState | str) -> Path:
        return self.goal_dir(goal) / "checkpoint.md"

    def current_path(self) -> Path:
        """Legacy singleton pointer retained only for migration compatibility."""
        return self.root / "current"

    def index_path(self) -> Path:
        return self.root / "index.json"

    def clear_current(self) -> None:
        """Remove only the legacy singleton pointer.

        Multi-Goal routing is conversation-scoped and never uses this pointer.
        """
        try:
            self.current_path().unlink()
        except FileNotFoundError:
            pass


    def save(
        self,
        goal: GoalState,
        *,
        event_type: str = "state_saved",
        event_payload: dict[str, Any] | None = None,
    ) -> Path:
        goal_id = ensure_goal_id(goal)
        old_revision = goal.revision
        old_generation = goal.generation
        self.root.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision, generation FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
            if row is None:
                if goal.revision not in {0, 1}:
                    db.rollback()
                    raise GoalConflictError(
                        f"goal {goal_id} has local revision {goal.revision}, but no authoritative row exists"
                    )
                authoritative_revision = 0
                authoritative_generation = max(1, goal.generation)
            else:
                authoritative_revision = int(row[0])
                authoritative_generation = int(row[1])
                if goal.revision != authoritative_revision:
                    db.rollback()
                    raise GoalConflictError(
                        f"goal {goal_id} revision conflict: local={goal.revision}, authoritative={authoritative_revision}"
                    )

            goal.revision = authoritative_revision + 1
            goal.generation = max(authoritative_generation, goal.generation, 1)
            payload = json.dumps(asdict(goal), ensure_ascii=False, sort_keys=True)
            now = _now_iso()
            try:
                db.execute(
                    """
                    INSERT INTO goals(goal_id, revision, generation, state_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(goal_id) DO UPDATE SET
                        revision=excluded.revision,
                        generation=excluded.generation,
                        state_json=excluded.state_json,
                        updated_at=excluded.updated_at
                    """,
                    (goal_id, goal.revision, goal.generation, payload, now),
                )
                refs = list(dict.fromkeys(goal.conversations))
                if goal.conversation_ref and goal.conversation_ref not in refs:
                    refs.append(goal.conversation_ref)
                for index, ref in enumerate(refs, start=1):
                    inferred_generation = (
                        goal.generation
                        if ref == goal.conversation_ref
                        else min(index, goal.generation)
                    )
                    self._bind_conversation_tx(
                        db,
                        goal_id,
                        ref,
                        generation=inferred_generation,
                        allow_replace_terminal=True,
                    )
                self._append_event_tx(
                    db,
                    goal_id,
                    event_type,
                    {
                        "revision": goal.revision,
                        "generation": goal.generation,
                        **(event_payload or {}),
                    },
                )
                db.commit()
            except Exception:
                goal.revision = old_revision
                goal.generation = old_generation
                db.rollback()
                raise

        # Portable projections are deliberately downstream of the authoritative
        # transaction. A projection failure must never turn a committed SQLite
        # transaction into a false "state was not saved" signal.
        self.last_projection_error = None
        try:
            self._write_portable_projection(goal)
            self._write_index_projection()
        except OSError as exc:
            self.last_projection_error = exc
        return self.goal_path(goal)

    def record_event(
        self,
        goal: GoalState | str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            event_id = self._append_event_tx(db, goal_id, event_type, payload or {})
            db.commit()
            return event_id

    def record_observed_event(
        self,
        goal: GoalState | str,
        event_type: str,
        payload: dict[str, Any],
        *,
        event_key: str,
    ) -> bool:
        """Append machine-observed telemetry once without mutating Goal revision."""
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT generation FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
            if row is None:
                db.rollback()
                return False
            if event_type == "conversation_write_committed":
                conversation_ref = str(payload.get("conversation_ref") or "").strip()
                if conversation_ref:
                    self._bind_conversation_tx(
                        db,
                        goal_id,
                        conversation_ref,
                        generation=int(row[0]),
                        allow_replace_terminal=False,
                    )
            seq = int(
                db.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM goal_events WHERE goal_id = ?",
                    (goal_id,),
                ).fetchone()[0]
            )
            cursor = db.execute(
                """
                INSERT OR IGNORE INTO goal_events(
                    goal_id, seq, generation, event_type, payload_json, created_at, event_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal_id,
                    seq,
                    int(row[0]),
                    event_type,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    _now_iso(),
                    event_key,
                ),
            )
            db.commit()
            inserted = cursor.rowcount == 1
        if inserted:
            self.last_projection_error = None
            try:
                self._write_portable_events(goal_id)
                if event_type == "conversation_write_committed":
                    self._write_index_projection()
            except OSError as exc:
                self.last_projection_error = exc
        return inserted

    def operation_committed_conversation(
        self, goal: GoalState | str, operation_id: str | None
    ) -> str | None:
        """Return the uniquely observed committed conversation for an operation.

        The transport write-completed event is journaled independently from Goal
        state so a process crash cannot orphan a newly-created recovery chat. If
        conflicting committed identities are ever observed, fail closed instead of
        guessing which chat owns the Goal.
        """
        if not operation_id:
            return None
        refs: list[str] = []
        for event in self.events(goal):
            if event.get("type") != "conversation_write_committed":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict) or payload.get("operation_id") != operation_id:
                continue
            ref = str(payload.get("conversation_ref") or "").strip()
            if ref and ref not in refs:
                refs.append(ref)
        return refs[0] if len(refs) == 1 else None

    def operation_evidence(
        self, goal: GoalState | str, operation_id: str | None
    ) -> dict[str, int]:
        """Summarize observed tool evidence conservatively.

        Tool results only resolve a preceding unmatched call for the same tool identity.
        We intentionally prefer false "needs reconciliation" over falsely declaring an
        ambiguous side effect resolved.
        """
        if not operation_id:
            return {"tool_calls": 0, "tool_results": 0, "unresolved_tool_calls": 0}
        calls = 0
        results = 0
        unmatched: dict[str, int] = {}
        for event in self.events(goal):
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            if payload.get("operation_id") != operation_id:
                continue
            tool = str(payload.get("tool_name") or "<unknown>").strip() or "<unknown>"
            if event.get("type") == "tool_call_observed":
                calls += 1
                unmatched[tool] = unmatched.get(tool, 0) + 1
            elif event.get("type") == "tool_result_observed":
                results += 1
                if unmatched.get(tool, 0) > 0:
                    unmatched[tool] -= 1
        return {
            "tool_calls": calls,
            "tool_results": results,
            "unresolved_tool_calls": sum(unmatched.values()),
        }

    def operation_reconciliation_evidence(
        self, goal: GoalState | str, operation_id: str | None
    ) -> dict[str, int | bool]:
        """Return machine evidence produced after the latest operation resume.

        An ambiguous original side effect cannot always regain its missing result. In
        that case a recovery turn may inspect external state with a different tool.
        We only call that reconciliation-ready when the resumed turn contains at
        least one matched tool call/result pair and no newly-unresolved calls. The
        semantic claim that the inspection actually proves the old side effect still
        comes from the structured Goal response; the journal keeps both layers
        explicit rather than pretending this is exactly-once proof.
        """
        empty: dict[str, int | bool] = {
            "verification_calls": 0,
            "verification_results": 0,
            "unresolved_verification_calls": 0,
            "ready": False,
        }
        if not operation_id:
            return empty
        relevant = [
            event
            for event in self.events(goal)
            if isinstance(event.get("payload"), dict)
            and event["payload"].get("operation_id") == operation_id
        ]
        resume_positions = [
            index
            for index, event in enumerate(relevant)
            if event.get("type") == "operation_resumed"
        ]
        if not resume_positions:
            return empty
        verification = relevant[resume_positions[-1] + 1 :]
        calls = 0
        results = 0
        matched = 0
        unmatched: dict[str, int] = {}
        for event in verification:
            payload = event["payload"]
            tool = str(payload.get("tool_name") or "<unknown>").strip() or "<unknown>"
            if event.get("type") == "tool_call_observed":
                calls += 1
                unmatched[tool] = unmatched.get(tool, 0) + 1
            elif event.get("type") == "tool_result_observed":
                results += 1
                if unmatched.get(tool, 0) > 0:
                    unmatched[tool] -= 1
                    matched += 1
        unresolved = sum(unmatched.values())
        return {
            "verification_calls": calls,
            "verification_results": results,
            "unresolved_verification_calls": unresolved,
            "ready": matched > 0 and unresolved == 0,
        }

    def events(self, goal: GoalState | str) -> list[dict[str, Any]]:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        if not self.db_path.exists():
            return []
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT seq, generation, event_type, payload_json, created_at
                FROM goal_events WHERE goal_id = ? ORDER BY seq
                """,
                (goal_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for seq, generation, event_type, payload_json, created_at in rows:
            try:
                payload = json.loads(str(payload_json))
            except json.JSONDecodeError:
                payload = {"raw": str(payload_json)}
            result.append(
                {
                    "seq": int(seq),
                    "generation": int(generation),
                    "type": str(event_type),
                    "payload": payload,
                    "created_at": str(created_at),
                }
            )
        return result

    def recovery_context(
        self,
        goal: GoalState | str,
        *,
        max_chars: int = 16000,
        max_events: int = 32,
    ) -> list[str]:
        """Return a bounded projection of the authoritative journal for handoff.

        User steering is treated as durable control input rather than ordinary
        recency data: older steering is projected even after many technical events.
        The complete unbounded history always remains in SQLite/events.jsonl.
        """
        events = self.events(goal)
        selected: list[str] = []
        total = 0

        def add(text: str | None) -> bool:
            nonlocal total
            if not text:
                return False
            normalized = " ".join(text.split()).strip()[:6000]
            if not normalized:
                return False
            if selected and total + len(normalized) > max_chars:
                return False
            selected.append(normalized)
            total += len(normalized)
            return True

        created = next((event for event in events if event.get("type") == "goal_created"), None)
        if created is not None:
            add(self._recovery_event_text(created))

        steering_events = [event for event in events if event.get("type") == "user_steering"]
        if steering_events:
            budget = max(1200, min(8000, max_chars // 2))
            per_item = max(80, min(1200, budget // max(1, len(steering_events))))
            pieces: list[str] = []
            for index, event in enumerate(steering_events, 1):
                payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                raw = " ".join(str(payload.get("text") or "").split()).strip()
                pieces.append(f"{index}. {raw[:per_item]}")
            steering_line = (
                f"user steering history ({len(steering_events)} durable messages): "
                + " | ".join(pieces)
            )
            if len(steering_line) > budget:
                steering_line = steering_line[: budget - 36] + " … [see full Goal journal]"
            add(steering_line)

        recent_candidates = [
            event
            for event in events
            if event.get("type") not in {"goal_created", "user_steering"}
        ]
        recent_texts: list[str] = []
        for event in reversed(recent_candidates):
            text = self._recovery_event_text(event)
            if text:
                recent_texts.append(text)
            if len(recent_texts) >= max_events:
                break
        for text in reversed(recent_texts):
            if not add(text):
                break
        return selected

    @staticmethod
    def _recovery_event_text(event: dict[str, Any]) -> str | None:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        kind = str(event.get("type") or "")
        if kind == "goal_created" and isinstance(payload.get("context_snapshot"), list):
            messages = [str(item) for item in payload["context_snapshot"]]
            recent = " | ".join(" ".join(item.split())[:1200] for item in messages[-6:])
            return f"initial context snapshot: {len(messages)} messages; recent: {recent}"
        if kind == "user_steering":
            return f"user steering: {payload.get('text', '')}"
        if kind in {"turn_terminal", "turn_abnormal", "turn_failed"}:
            body = str(payload.get("body") or payload.get("text") or "").strip()
            status = str(payload.get("signal") or payload.get("status") or kind)
            if body:
                return f"assistant {status}: {body}"
            return f"turn {status}: {payload.get('detail', '')}"
        if kind in {"operation_started", "operation_resumed"}:
            return (
                f"{kind}: {payload.get('operation_id', '')} "
                f"conversation={payload.get('conversation_ref') or 'new-chat'}"
            )
        if kind == "conversation_write_committed":
            return (
                f"committed conversation: {payload.get('conversation_ref', '')} "
                f"for operation {payload.get('operation_id', '')}"
            )
        if kind in {"tool_call_observed", "tool_result_observed"}:
            label = str(payload.get("label") or payload.get("tool_name") or "tool")
            detail = str(payload.get("text") or "").strip()
            return (
                f"{kind}: operation={payload.get('operation_id') or 'unknown'}; "
                f"{label}; {detail[:1800]}"
            )
        if kind == "rollover":
            return (
                f"rollover: generation {payload.get('from_generation')} -> "
                f"{payload.get('to_generation')}; {payload.get('reason', '')}"
            )
        if kind in {"goal_paused", "goal_blocked", "goal_interrupted"}:
            return f"{kind}: {payload.get('reason', '')}"
        return None

    def has_authoritative_goal(self, goal: GoalState | str) -> bool:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        if not self.db_path.exists():
            return False
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
        return row is not None

    def load_for_conversation(self, conversation_ref: str | None) -> GoalState | None:
        ref = str(conversation_ref or "").strip()
        if not ref or not self.db_path.exists():
            return None
        with self._connect() as db:
            row = db.execute(
                """
                SELECT g.state_json
                FROM goal_conversations c
                JOIN goals g ON g.goal_id = c.goal_id
                WHERE c.conversation_ref = ?
                """,
                (ref,),
            ).fetchone()
        return self._decode_goal(str(row[0])) if row is not None else None

    def goal_id_for_conversation(self, conversation_ref: str | None) -> str | None:
        ref = str(conversation_ref or "").strip()
        if not ref or not self.db_path.exists():
            return None
        with self._connect() as db:
            row = db.execute(
                "SELECT goal_id FROM goal_conversations WHERE conversation_ref = ?",
                (ref,),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def list_goals(
        self,
        *,
        statuses: set[str] | None = None,
        limit: int | None = 100,
    ) -> list[GoalState]:
        if not self.db_path.exists():
            return []
        query = "SELECT state_json FROM goals"
        params: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE json_extract(state_json, '$.status') IN ({placeholders})"
            params.extend(sorted(statuses))
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._connect() as db:
            rows = db.execute(query, tuple(params)).fetchall()
        result: list[GoalState] = []
        for row in rows:
            goal = self._decode_goal(str(row[0]))
            if goal is not None:
                result.append(goal)
        return result

    def conversation_goal_map(self) -> dict[str, GoalState]:
        if not self.db_path.exists():
            return {}
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT c.conversation_ref, g.state_json
                FROM goal_conversations c
                JOIN goals g ON g.goal_id = c.goal_id
                """
            ).fetchall()
        result: dict[str, GoalState] = {}
        for conversation_ref, state_json in rows:
            goal = self._decode_goal(str(state_json))
            if goal is not None:
                result[str(conversation_ref)] = goal
        return result

    def bindings_for_goal(self, goal: GoalState | str) -> list[str]:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        if not self.db_path.exists():
            return []
        with self._connect() as db:
            rows = db.execute(
                """SELECT conversation_ref FROM goal_conversations
                   WHERE goal_id = ? ORDER BY bound_at, conversation_ref""",
                (goal_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def bind_conversation(
        self, goal: GoalState | str, conversation_ref: str, *, generation: int | None = None
    ) -> None:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        ref = str(conversation_ref).strip()
        if not ref:
            raise ValueError("conversation_ref is required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT generation FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
            if row is None:
                db.rollback()
                raise KeyError(f"unknown goal: {goal_id}")
            self._bind_conversation_tx(
                db,
                goal_id,
                ref,
                generation=max(1, int(generation or row[0])),
                allow_replace_terminal=True,
            )
            self._append_event_tx(
                db, goal_id, "conversation_bound", {"conversation_ref": ref}
            )
            db.commit()
        try:
            self._write_index_projection()
        except OSError as exc:
            self.last_projection_error = exc

    def unbind_goal(self, goal: GoalState | str) -> list[str]:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        if not self.db_path.exists():
            return []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT conversation_ref FROM goal_conversations WHERE goal_id = ?",
                (goal_id,),
            ).fetchall()
            refs = [str(row[0]) for row in rows]
            db.execute("DELETE FROM goal_conversations WHERE goal_id = ?", (goal_id,))
            if db.execute("SELECT 1 FROM goals WHERE goal_id = ?", (goal_id,)).fetchone():
                self._append_event_tx(
                    db, goal_id, "conversations_unbound", {"conversations": refs}
                )
            db.commit()
        try:
            self._write_index_projection()
        except OSError as exc:
            self.last_projection_error = exc
        return refs

    def load_current(self) -> GoalState | None:
        """Legacy singleton lookup used only to import pre-v3 state."""
        if self.db_path.exists():
            with self._connect() as db:
                row = db.execute(
                    "SELECT goal_id FROM current_goal WHERE slot = 1"
                ).fetchone()
            if row is not None:
                return self.load(str(row[0]))
            return None
        try:
            goal_id = self.current_path().read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return self._load_legacy(goal_id) if goal_id else None


    def load(self, goal_id: str) -> GoalState | None:
        if self.db_path.exists():
            with self._connect() as db:
                row = db.execute(
                    "SELECT state_json FROM goals WHERE goal_id = ?", (goal_id,)
                ).fetchone()
            if row is not None:
                return self._decode_goal(str(row[0]))
        return self._load_legacy(goal_id)

    def _load_legacy(self, goal_id: str) -> GoalState | None:
        path = self.goal_path(goal_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        payload = raw.get("goal") if isinstance(raw, dict) else None
        goal = goal_state_from_dict(payload)
        if goal is not None:
            if goal.goal_id is None:
                goal.goal_id = goal_id
            # Legacy files predate optimistic revisions.
            goal.revision = 0
            goal.generation = max(1, goal.generation)
        return goal

    @staticmethod
    def _decode_goal(payload: str) -> GoalState | None:
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return goal_state_from_dict(raw)

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path, timeout=5.0)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA fullfsync=ON")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=5000")
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS goals(
                goal_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS current_goal(
                slot INTEGER PRIMARY KEY CHECK(slot = 1),
                goal_id TEXT NOT NULL REFERENCES goals(goal_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS goal_conversations(
                conversation_ref TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL REFERENCES goals(goal_id) ON DELETE CASCADE,
                generation INTEGER NOT NULL,
                bound_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_goal_conversations_goal
                ON goal_conversations(goal_id, bound_at);
            CREATE TABLE IF NOT EXISTS goal_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id TEXT NOT NULL REFERENCES goals(goal_id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_key TEXT,
                UNIQUE(goal_id, seq)
            );
            CREATE INDEX IF NOT EXISTS idx_goal_events_goal_seq
                ON goal_events(goal_id, seq);
            """
        )
        columns = {str(row[1]) for row in db.execute("PRAGMA table_info(goal_events)")}
        if "event_key" not in columns:
            db.execute("ALTER TABLE goal_events ADD COLUMN event_key TEXT")
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_goal_events_key "
            "ON goal_events(goal_id, event_key) WHERE event_key IS NOT NULL"
        )
        schema_version = int(db.execute("PRAGMA user_version").fetchone()[0])
        if schema_version < self.SCHEMA_VERSION:
            self._migrate_conversation_bindings_tx(db)
            db.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
        db.commit()
        return db

    def _bind_conversation_tx(
        self,
        db: sqlite3.Connection,
        goal_id: str,
        conversation_ref: str,
        *,
        generation: int,
        allow_replace_terminal: bool,
    ) -> None:
        ref = str(conversation_ref).strip()
        existing = db.execute(
            """SELECT c.goal_id, g.state_json
               FROM goal_conversations c
               JOIN goals g ON g.goal_id = c.goal_id
               WHERE c.conversation_ref = ?""",
            (ref,),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) == goal_id:
                db.execute(
                    "UPDATE goal_conversations SET generation = ? WHERE conversation_ref = ?",
                    (max(1, int(generation)), ref),
                )
                return
            old_goal_id = str(existing[0])
            old_goal = self._decode_goal(str(existing[1]))
            old_status = old_goal.status if old_goal is not None else "unknown"
            if not allow_replace_terminal or old_status not in {"complete", "interrupted"}:
                raise GoalConflictError(
                    f"conversation {ref} is already bound to unfinished goal {old_goal_id}"
                )
            self._append_event_tx(
                db,
                old_goal_id,
                "conversation_released",
                {"conversation_ref": ref, "rebound_to_goal": goal_id},
            )
        db.execute(
            """
            INSERT INTO goal_conversations(conversation_ref, goal_id, generation, bound_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(conversation_ref) DO UPDATE SET
                goal_id=excluded.goal_id,
                generation=excluded.generation,
                bound_at=excluded.bound_at
            """,
            (ref, goal_id, max(1, int(generation)), _now_iso()),
        )

    def _migrate_conversation_bindings_tx(self, db: sqlite3.Connection) -> None:
        # v2 used a singleton current_goal pointer and stored conversation chains
        # only inside Goal JSON. Backfill routing deterministically, newest Goal
        # first, without ever stealing an existing v3 binding.
        rows = db.execute(
            "SELECT goal_id, generation, state_json FROM goals ORDER BY updated_at DESC"
        ).fetchall()
        for goal_id, generation, state_json in rows:
            goal = self._decode_goal(str(state_json))
            if goal is None:
                continue
            refs: list[str] = []
            if goal.conversation_ref:
                refs.append(goal.conversation_ref)
            refs.extend(ref for ref in goal.conversations if ref not in refs)
            for ref in refs:
                db.execute(
                    """INSERT OR IGNORE INTO goal_conversations(
                           conversation_ref, goal_id, generation, bound_at
                       ) VALUES (?, ?, ?, ?)""",
                    (ref, str(goal_id), max(1, int(generation)), _now_iso()),
                )

    def _append_event_tx(
        self,
        db: sqlite3.Connection,
        goal_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        row = db.execute(
            "SELECT generation FROM goals WHERE goal_id = ?", (goal_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown goal: {goal_id}")
        generation = int(row[0])
        seq_row = db.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM goal_events WHERE goal_id = ?",
            (goal_id,),
        ).fetchone()
        seq = int(seq_row[0])
        cursor = db.execute(
            """
            INSERT INTO goal_events(goal_id, seq, generation, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                goal_id,
                seq,
                generation,
                event_type,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                _now_iso(),
            ),
        )
        return int(cursor.lastrowid)

    def _write_portable_projection(self, goal: GoalState) -> None:
        goal_id = ensure_goal_id(goal)
        directory = self.goal_dir(goal_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.SCHEMA_VERSION,
            "goal_id": goal_id,
            "updated_at": _now_iso(),
            "goal": asdict(goal),
            "files": {
                "checkpoint": "checkpoint.md",
                "journal": "events.jsonl",
                "authoritative_store": "../goal-state.sqlite3",
            },
        }
        self._write_json_atomic(directory / "goal.json", payload)
        self._write_text_atomic(
            directory / "checkpoint.md", self._checkpoint_markdown(goal)
        )
        self._write_portable_events(goal_id)

    def _write_index_projection(self) -> None:
        goals = self.list_goals(limit=None)
        payload = {
            "schema": self.SCHEMA_VERSION,
            "updated_at": _now_iso(),
            "goals": [
                {
                    "goal_id": goal.goal_id,
                    "status": goal.status,
                    "generation": goal.generation,
                    "conversation_ref": goal.conversation_ref,
                    "conversations": self.bindings_for_goal(goal),
                    "objective": goal.objective,
                    "turn_count": goal.turn_count,
                    "runner_pid": goal.runner_pid,
                }
                for goal in goals
            ],
        }
        self._write_json_atomic(self.index_path(), payload)

    def _write_portable_events(self, goal: GoalState | str) -> None:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        directory = self.goal_dir(goal_id)
        directory.mkdir(parents=True, exist_ok=True)
        journal = "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            for event in self.events(goal_id)
        )
        self._write_text_atomic(directory / "events.jsonl", journal)

    @staticmethod
    def _checkpoint_markdown(goal: GoalState) -> str:
        checkpoint = goal.checkpoint
        lines = [
            f"# Goal {goal.goal_id or ''}".rstrip(),
            "",
            f"- Status: `{goal.status}`",
            f"- Revision: {goal.revision}",
            f"- Generation: {goal.generation}",
            f"- Turns: {goal.turn_count}",
            f"- Rollovers: {goal.rollover_count}",
            f"- Active conversation: `{goal.conversation_ref or 'none'}`",
            "",
            "## Objective",
            "",
            goal.objective or "_Inherited from the originating conversation._",
            "",
            "## Checkpoint",
            "",
            checkpoint.summary or "_No checkpoint captured yet._",
            "",
        ]
        for heading, values in (
            ("Completed", checkpoint.completed),
            ("Decisions", checkpoint.decisions),
            ("Pending", checkpoint.pending),
        ):
            lines.extend([f"### {heading}", ""])
            if values:
                lines.extend(f"- {value}" for value in values)
            else:
                lines.append("- _none recorded_")
            lines.append("")
        lines.extend(
            [
                "### Next step",
                "",
                checkpoint.next_step or "_not recorded_",
                "",
                "## Recovery context seed",
                "",
            ]
        )
        if goal.context_seed:
            lines.extend(f"- {item}" for item in goal.context_seed)
        else:
            lines.append("- _none captured_")
        lines.extend(["", "## Conversation chain", ""])
        if goal.conversations:
            lines.extend(
                f"- https://chatgpt.com/c/{conversation_id}"
                for conversation_id in goal.conversations
            )
        else:
            lines.append("- _none yet_")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        GoalStore._write_text_atomic(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
