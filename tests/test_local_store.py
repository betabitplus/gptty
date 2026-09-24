from __future__ import annotations

import os
import sqlite3
import stat
import threading
from pathlib import Path

import pytest

from gptty.local_store import (
    LocalEventStore,
    LocalStoreCompatibilityError,
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
