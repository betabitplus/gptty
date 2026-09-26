from __future__ import annotations

import os
import sqlite3
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from gptty.local_store import (
    LocalEventStore,
    LocalStoreCompatibilityError,
    LocalSessionConflictError,
    SCHEMA_VERSION,
)


def _summary(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "running",
        "last_event": "run_started",
        "updated_at": "2026-09-24T00:00:00+00:00",
    }


def _event(kind: str, marker: int | None = None) -> dict[str, object]:
    event: dict[str, object] = {
        "type": kind,
        "timestamp": "2026-09-24T00:00:00+00:00",
    }
    if marker is not None:
        event["marker"] = marker
    return event


def test_run_events_are_transactional_and_recent_query_is_bounded(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    summary = _summary("run-1")
    store.create_run(
        run_id="run-1",
        summary=summary,
        first_event=_event("run_started"),
    )

    for marker in range(30):
        summary = {**summary, "last_event": "token_delta", "marker": marker}
        store.append_run_event(
            run_id="run-1",
            summary=summary,
            event=_event("token_delta", marker),
        )

    all_events = store.run_events("run-1")
    recent = store.run_events("run-1", limit=20)

    assert len(all_events) == 31
    assert len(recent) == 20
    assert recent[0]["marker"] == 10
    assert recent[-1]["marker"] == 29
    assert store.run_summary("run-1")["marker"] == 29


def test_unknown_run_append_rolls_back_without_orphan_event(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    store = LocalEventStore(db_path)

    with pytest.raises(KeyError, match="unknown run_id"):
        store.append_run_event(
            run_id="missing",
            summary=_summary("missing"),
            event=_event("failed"),
        )

    with sqlite3.connect(db_path) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM local_run_events WHERE run_id = 'missing'"
        ).fetchone()[0]
    assert count == 0


def test_two_store_instances_serialize_concurrent_run_event_sequences(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    first = LocalEventStore(db_path)
    second = LocalEventStore(db_path)
    summary = _summary("run-1")
    first.create_run(
        run_id="run-1",
        summary=summary,
        first_event=_event("run_started"),
    )

    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def writer(store: LocalEventStore, base: int) -> None:
        try:
            barrier.wait(timeout=3)
            for offset in range(20):
                marker = base + offset
                store.append_run_event(
                    run_id="run-1",
                    summary={**summary, "marker": marker},
                    event=_event("token_delta", marker),
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(first, 0)),
        threading.Thread(target=writer, args=(second, 100)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert not errors
    events = first.run_events("run-1")
    markers = sorted(
        int(event["marker"])
        for event in events
        if event.get("type") == "token_delta"
    )
    assert markers == list(range(20)) + list(range(100, 120))

    with sqlite3.connect(db_path) as db:
        seqs = [
            int(row[0])
            for row in db.execute(
                "SELECT seq FROM local_run_events WHERE run_id = ? ORDER BY seq",
                ("run-1",),
            )
        ]
    assert seqs == list(range(1, 42))


def test_tui_store_deduplicates_events_and_indexes_terminal_queries(
    tmp_path: Path,
) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    conversation = "conversation-1"
    event = {
        "event_id": "turn-1:terminal",
        "turn_id": "turn-1",
        "observed_at": "2026-09-24T00:00:00+00:00",
        "role": "chat",
        "status": "limit-reached",
        "terminal_source": "conversation_too_large",
        "text": "Conversation limit reached",
    }

    assert store.insert_tui_event(conversation, event) is True
    assert store.insert_tui_event(conversation, event) is False
    assert len(store.tui_events(conversation)) == 1
    assert store.tui_terminal_exists(
        conversation,
        role="chat",
        status="limit-reached",
        text="Conversation limit reached",
        source="conversation_too_large",
    )
    assert store.latest_chat_terminal(conversation) == event


def test_pending_tui_event_is_atomically_popped(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    event = {
        "event_id": "turn-1:user",
        "observed_at": "2026-09-24T00:00:00+00:00",
        "role": "user",
        "text": "hello",
    }

    store.put_pending_tui_event("turn-1", event)

    assert store.pop_pending_tui_event("turn-1") == event
    assert store.pop_pending_tui_event("turn-1") is None


def test_tui_title_preserves_existing_title_when_none_is_supplied(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")

    assert store.set_tui_title("conversation-1", "Useful title") == "Useful title"
    assert store.set_tui_title("conversation-1", None) == "Useful title"
    assert store.tui_title("conversation-1") == "Useful title"


def test_future_schema_is_rejected_without_downgrade(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")

    with pytest.raises(LocalStoreCompatibilityError, match="newer"):
        LocalEventStore(db_path)

    with sqlite3.connect(db_path) as db:
        assert int(db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION + 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits do not apply on Windows")
def test_store_root_and_database_are_owner_only(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    root.chmod(0o777)
    db_path = root / "local.sqlite3"

    store = LocalEventStore(db_path)
    store.create_run(
        run_id="run-1",
        summary=_summary("run-1"),
        first_event=_event("run_started"),
    )

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600

def test_concurrent_terminal_semantic_dedupe_is_transactional(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    stores = [LocalEventStore(db_path), LocalEventStore(db_path)]
    barrier = threading.Barrier(3)
    inserted: list[bool] = []
    errors: list[BaseException] = []

    def writer(store: LocalEventStore, event_id: str) -> None:
        try:
            barrier.wait(timeout=3)
            inserted.append(
                store.insert_tui_terminal_once(
                    "conversation-1",
                    {
                        "event_id": event_id,
                        "observed_at": "2026-09-24T00:00:00+00:00",
                        "role": "chat",
                        "status": "limit-reached",
                        "terminal_source": "stream",
                        "text": "same semantic terminal",
                    },
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(stores[0], "terminal-a")),
        threading.Thread(target=writer, args=(stores[1], "terminal-b")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert not errors
    assert sorted(inserted) == [False, True]
    events = stores[0].tui_events("conversation-1")
    assert len(events) == 1
    assert events[0]["text"] == "same semantic terminal"

def test_schema_v1_migrates_delivery_tables_in_place(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA user_version=1")
        db.commit()

    store = LocalEventStore(db_path)
    event_id = store.append_delivery_event(
        {
            "schema": 1,
            "event": "migrated-delivery",
            "observed_at_ms": 123,
            "conversation_ref": "conversation-1",
        }
    )

    assert event_id > 0
    assert store.delivery_events()[-1][1]["event"] == "migrated-delivery"
    with sqlite3.connect(db_path) as db:
        assert int(db.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION

def test_session_registry_compare_and_swap_rejects_stale_writer(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    created = store.create_session(
        "default",
        kind="default",
        discovery_hint=None,
        current_conversation="conv-a",
        model="model-a",
    )
    assert created["revision"] == 0

    revision = store.save_session(
        "default",
        expected_revision=0,
        current_conversation="conv-b",
        model="model-b",
        goal_id=None,
    )
    assert revision == 1

    with pytest.raises(LocalSessionConflictError, match="changed concurrently"):
        store.save_session(
            "default",
            expected_revision=0,
            current_conversation="conv-stale",
            model="model-stale",
            goal_id=None,
        )

    current = store.get_session("default")
    assert current is not None
    assert current["revision"] == 1
    assert current["current_conversation"] == "conv-b"
    assert current["model"] == "model-b"


def test_session_registry_concurrent_creation_preserves_first_seed(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    stores = [LocalEventStore(db_path), LocalEventStore(db_path)]
    barrier = threading.Barrier(3)
    results: list[dict] = []
    errors: list[BaseException] = []

    def creator(store: LocalEventStore, conversation: str) -> None:
        try:
            barrier.wait(timeout=3)
            results.append(
                store.create_session(
                    "explicit-shared",
                    kind="explicit",
                    discovery_hint="explicit:test",
                    current_conversation=conversation,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=creator, args=(stores[0], "conv-a")),
        threading.Thread(target=creator, args=(stores[1], "conv-b")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert not errors
    current = stores[0].get_session("explicit-shared")
    assert current is not None
    assert current["current_conversation"] in {"conv-a", "conv-b"}
    assert {result["current_conversation"] for result in results} == {
        current["current_conversation"]
    }


def test_existing_schema_v3_without_claim_table_is_upgraded_in_place(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    db_path.parent.mkdir(parents=True)
    source_path = (tmp_path / "legacy-state.json").resolve()
    source_path.write_text("{}\n", encoding="utf-8")

    db = sqlite3.connect(db_path)
    try:
        db.execute(
            """
            CREATE TABLE local_sessions(
                session_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,
                kind TEXT NOT NULL,
                discovery_hint TEXT,
                current_conversation TEXT,
                model TEXT,
                goal_id TEXT,
                created_at_ms INTEGER NOT NULL,
                last_seen_at_ms INTEGER NOT NULL,
                imported_from TEXT
            )
            """
        )
        db.execute(
            """
            INSERT INTO local_sessions(
                session_id, revision, kind, discovery_hint,
                current_conversation, model, goal_id,
                created_at_ms, last_seen_at_ms, imported_from
            )
            VALUES(?, 0, 'runtime', 'cmux:old', 'conv-old', NULL, NULL, 1, 1, ?)
            """,
            ("runtime-old", str(source_path)),
        )
        db.execute("PRAGMA user_version=3")
        db.commit()
    finally:
        db.close()

    store = LocalEventStore(db_path)

    assert store.session_imported_from(source_path) is True
    row, imported = store.create_session_claiming_import(
        "runtime-new",
        source_path=source_path,
        kind="runtime",
        discovery_hint="cmux:new",
        current_conversation="conv-new",
    )
    assert row is None
    assert imported is False
    assert store.get_session("runtime-old") is not None
    with sqlite3.connect(db_path) as check:
        assert (
            check.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='local_session_imports'"
            ).fetchone()
            is not None
        )
        assert int(check.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
        columns = {
            row[1] for row in check.execute("PRAGMA table_info(local_sessions)").fetchall()
        }
        assert "reasoning_effort" in columns
    upgraded = store.get_session("runtime-old")
    assert upgraded is not None
    assert upgraded["reasoning_effort"] is None


def test_legacy_session_import_claim_is_atomic_across_concurrent_sessions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    source_path = tmp_path / "legacy-state.json"
    source_path.write_text("{}\n", encoding="utf-8")
    stores = [LocalEventStore(db_path), LocalEventStore(db_path)]
    barrier = threading.Barrier(3)
    results: list[tuple[dict | None, bool]] = []
    errors: list[BaseException] = []

    def claimant(store: LocalEventStore, session_id: str, conversation: str) -> None:
        try:
            barrier.wait(timeout=3)
            results.append(
                store.create_session_claiming_import(
                    session_id,
                    source_path=source_path,
                    kind="runtime",
                    discovery_hint="cmux:shared",
                    current_conversation=conversation,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=claimant, args=(stores[0], "runtime-a", "conv-a")),
        threading.Thread(target=claimant, args=(stores[1], "runtime-b", "conv-b")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert not errors
    imported = [(row, claimed) for row, claimed in results if claimed]
    rejected = [(row, claimed) for row, claimed in results if row is None]
    assert len(imported) == 1
    assert len(rejected) == 1
    assert imported[0][0] is not None
    assert stores[0].session_imported_from(source_path) is True


def test_legacy_session_import_claim_is_atomic_across_processes(tmp_path: Path) -> None:
    db_path = tmp_path / "runs" / "local.sqlite3"
    source_path = tmp_path / "legacy-state.json"
    source_path.write_text("{}\n", encoding="utf-8")
    start_path = tmp_path / "start"
    result_paths = [tmp_path / "result-a", tmp_path / "result-b"]
    script = """
import sys
import time
from pathlib import Path
from gptty.local_store import LocalEventStore

db_path = Path(sys.argv[1])
source_path = Path(sys.argv[2])
session_id = sys.argv[3]
conversation = sys.argv[4]
start_path = Path(sys.argv[5])
result_path = Path(sys.argv[6])
store = LocalEventStore(db_path)
while not start_path.exists():
    time.sleep(0.01)
row, imported = store.create_session_claiming_import(
    session_id,
    source_path=source_path,
    kind="runtime",
    discovery_hint="cmux:process-race",
    current_conversation=conversation,
)
result_path.write_text(
    f"{imported}:{row is None}",
    encoding="utf-8",
)
"""
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src if not existing_pythonpath else src + os.pathsep + existing_pythonpath
    )
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(db_path),
                str(source_path),
                session_id,
                conversation,
                str(start_path),
                str(result_path),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for session_id, conversation, result_path in (
            ("runtime-process-a", "conv-a", result_paths[0]),
            ("runtime-process-b", "conv-b", result_paths[1]),
        )
    ]
    start_path.write_text("go", encoding="utf-8")

    outputs = [process.communicate(timeout=10) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], outputs
    assert sorted(path.read_text(encoding="utf-8") for path in result_paths) == [
        "False:True",
        "True:False",
    ]
    store = LocalEventStore(db_path)
    assert store.session_imported_from(source_path) is True


def test_retention_run_candidates_skip_running_and_delete_events_transactionally(
    tmp_path: Path,
) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    old_completed = {**_summary("old-completed"), "status": "completed"}
    old_running = {**_summary("old-running"), "status": "running"}
    new_completed = {**_summary("new-completed"), "status": "completed"}
    store.create_run(
        run_id="old-completed",
        summary=old_completed,
        first_event={"type": "run_started", "timestamp": "2026-01-01T00:00:00+00:00"},
    )
    store.create_run(
        run_id="old-running",
        summary=old_running,
        first_event={"type": "run_started", "timestamp": "2026-01-01T00:00:00+00:00"},
    )
    store.create_run(
        run_id="new-completed",
        summary=new_completed,
        first_event={"type": "run_started", "timestamp": "2026-09-25T00:00:00+00:00"},
    )

    candidates = store.run_ids_before("2026-09-01T00:00:00+00:00")

    assert candidates == ["old-completed"]
    assert store.delete_runs(candidates) == 1
    assert store.run_summary("old-completed") is None
    assert store.run_events("old-completed") == []
    assert store.run_summary("old-running") is not None
    assert store.run_summary("new-completed") is not None


def test_retention_prunes_only_old_pending_prompts(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    store.put_pending_tui_event(
        "old-turn",
        {"event_id": "old-turn:user", "observed_at": "2026-01-01T00:00:00+00:00"},
    )
    store.put_pending_tui_event(
        "new-turn",
        {"event_id": "new-turn:user", "observed_at": "2026-09-25T00:00:00+00:00"},
    )

    assert store.prune_pending_tui_before("2026-09-01T00:00:00+00:00") == [
        "old-turn"
    ]
    assert store.pop_pending_tui_event("old-turn") is None
    assert store.pop_pending_tui_event("new-turn") is not None


def test_delete_tui_conversations_removes_events_import_and_metadata(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    event = {
        "event_id": "turn-1:user",
        "role": "user",
        "text": "local copy",
        "observed_at": "2026-01-01T00:00:00+00:00",
    }
    store.import_tui_conversation(
        "conv-12345678",
        events=[event],
        title="Old chat",
        imported_at="2026-01-01T00:00:00+00:00",
    )

    assert store.delete_tui_conversations(["conv-12345678"]) == 1
    assert store.tui_events("conv-12345678") == []
    assert store.tui_imported("conv-12345678") is False
    assert store.tui_title("conv-12345678") is None


def test_privacy_inventory_counts_event_only_archives(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    assert store.insert_tui_event(
        "conv-event1234",
        {
            "event_id": "turn-1:terminal",
            "role": "chat",
            "text": "event-only",
            "status": "unconfirmed",
            "terminal_source": "stream",
            "observed_at": "2026-01-01T00:00:00+00:00",
        },
    ) is True

    inventory = store.privacy_inventory()

    assert inventory["archived_conversations"] == 1


def test_local_session_reasoning_effort_participates_in_create_and_cas(tmp_path: Path) -> None:
    store = LocalEventStore(tmp_path / "runs" / "local.sqlite3")
    created = store.create_session(
        "effort-session",
        kind="explicit",
        discovery_hint="explicit:effort",
        current_conversation="conv-1",
        model=None,
        reasoning_effort="medium",
    )
    assert created["reasoning_effort"] == "medium"

    revision = store.save_session(
        "effort-session",
        expected_revision=created["revision"],
        current_conversation="conv-1",
        model=None,
        reasoning_effort="high",
        goal_id=None,
    )
    reloaded = store.get_session("effort-session")
    assert reloaded is not None
    assert reloaded["revision"] == revision
    assert reloaded["reasoning_effort"] == "high"
