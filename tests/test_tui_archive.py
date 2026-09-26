from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone

from gptty.local_store import LocalEventStore, local_store_path
from gptty.runs import start_run
from gptty.tui_archive import TUIArchive


def _events(archive: TUIArchive, conversation_id: str) -> list[dict]:
    path = archive.conversation_paths(conversation_id)["events"]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_new_chat_prompt_is_pending_until_conversation_identity_is_known(
    tmp_path,
) -> None:
    archive = TUIArchive(tmp_path / "archive")

    turn_id = archive.record_user(
        "hello",
        conversation_ref=None,
        model="gpt-test",
        media_count=1,
    )

    pending = archive.pending_dir / f"{turn_id}.json"
    assert pending.exists()
    assert not archive.conversation_paths("conv-12345678")["events"].exists()

    archive.bind_turn(turn_id, "conv-12345678")

    assert not pending.exists()
    events = _events(archive, "conv-12345678")
    assert [(event["role"], event["text"]) for event in events] == [("user", "hello")]
    assert events[0]["scope"] == "tui-observed"
    assert events[0]["media_count"] == 1


def test_completed_turn_materializes_append_only_events_and_markdown(tmp_path) -> None:
    archive = TUIArchive(tmp_path / "archive")

    turn_id = archive.record_user(
        "question",
        conversation_ref="conv-12345678",
        model=None,
    )
    archive.record_assistant(
        turn_id,
        conversation_ref="conv-12345678",
        text="answer",
        title="Archive Test",
        model="gpt-test",
        status="complete",
    )

    paths = archive.conversation_paths("conv-12345678")
    events = _events(archive, "conv-12345678")
    assert [(event["role"], event["text"]) for event in events] == [
        ("user", "question"),
        ("assistant", "answer"),
    ]
    assert events[1]["status"] == "complete"

    transcript = paths["transcript"].read_text(encoding="utf-8")
    assert "# Archive Test" in transcript
    assert "Scope: `tui-observed`" in transcript
    assert "## USER" in transcript
    assert "question" in transcript
    assert "## ASSISTANT" in transcript
    assert "answer" in transcript

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["conversation_id"] == "conv-12345678"
    assert meta["web_url"] == "https://chatgpt.com/c/conv-12345678"
    assert meta["scope"] == "tui-observed"
    if os.name != "nt":
        assert archive.root.stat().st_mode & 0o777 == 0o700
        assert archive.pending_dir.stat().st_mode & 0o777 == 0o700
        assert archive.conversations_dir.stat().st_mode & 0o777 == 0o700
        assert paths["directory"].stat().st_mode & 0o777 == 0o700
        assert paths["events"].stat().st_mode & 0o777 == 0o600
        assert paths["transcript"].stat().st_mode & 0o777 == 0o600
        assert paths["meta"].stat().st_mode & 0o777 == 0o600


def test_repeated_bind_does_not_duplicate_user_event(tmp_path) -> None:
    archive = TUIArchive(tmp_path / "archive")
    turn_id = archive.record_user(
        "once",
        conversation_ref=None,
        model=None,
    )

    archive.bind_turn(turn_id, "conv-12345678")
    archive.bind_turn(turn_id, "conv-12345678")

    events = _events(archive, "conv-12345678")
    assert [event["event_id"] for event in events] == [f"{turn_id}:user"]


def test_stopped_answer_remains_local_observation(tmp_path) -> None:
    archive = TUIArchive(tmp_path / "archive")
    turn_id = archive.record_user(
        "long answer please",
        conversation_ref="conv-12345678",
        model=None,
    )
    archive.record_assistant(
        turn_id,
        conversation_ref="conv-12345678",
        text="partial",
        title=None,
        model=None,
        status="stopped",
    )

    events = _events(archive, "conv-12345678")
    assert events[-1]["status"] == "stopped"
    transcript = archive.conversation_paths("conv-12345678")["transcript"].read_text(
        encoding="utf-8"
    )
    assert "## ASSISTANT — stopped" in transcript


def test_terminal_marker_is_persisted_as_separate_archive_event(tmp_path) -> None:
    archive = TUIArchive(tmp_path / "archive")
    turn_id = archive.record_user(
        "question",
        conversation_ref="conv-12345678",
        model=None,
    )

    archive.record_terminal(
        turn_id,
        conversation_ref="conv-12345678",
        label="turn",
        status="unconfirmed",
        text="A final ChatGPT completion was not observed; this turn may be incomplete.",
        source="stream",
    )

    events = _events(archive, "conv-12345678")
    assert events[-1]["event_id"] == f"{turn_id}:terminal"
    assert events[-1]["role"] == "turn"
    assert events[-1]["status"] == "unconfirmed"
    assert events[-1]["terminal_source"] == "stream"
    transcript = archive.conversation_paths("conv-12345678")["transcript"].read_text(
        encoding="utf-8"
    )
    assert "## TURN — unconfirmed" in transcript
    assert "A final ChatGPT completion was not observed" in transcript


def test_chat_level_terminal_marker_is_persistent_but_turn_marker_is_not(
    tmp_path,
) -> None:
    archive = TUIArchive(tmp_path / "archive")
    turn_id = archive.record_user(
        "question",
        conversation_ref="conv-12345678",
        model=None,
    )
    archive.record_terminal(
        turn_id,
        conversation_ref="conv-12345678",
        label="turn",
        status="unconfirmed",
        text="turn-only",
        source="stream",
    )
    assert archive.conversation_terminal_marker("conv-12345678") is None

    archive.record_terminal(
        turn_id + "b",
        conversation_ref="conv-12345678",
        label="chat",
        status="limit-reached",
        text="This conversation reached its maximum length; start a new chat to continue.",
        source="stream",
    )

    assert archive.conversation_terminal_marker("conv-12345678") == (
        "chat",
        "limit-reached",
        "This conversation reached its maximum length; start a new chat to continue.",
        "stream",
    )

def test_chat_terminal_resolution_is_append_only_and_restart_safe(tmp_path) -> None:
    root = tmp_path / "archive"
    archive = TUIArchive(root)
    conversation = "conv-resolution"

    archive.record_observed_terminal(
        conversation_ref=conversation,
        label="chat",
        status="unavailable",
        text="ChatGPT web UI cannot load this conversation.",
        source="web-ui",
    )
    assert archive.conversation_terminal_marker(conversation) == (
        "chat",
        "unavailable",
        "ChatGPT web UI cannot load this conversation.",
        "web-ui",
    )

    assert archive.record_chat_terminal_resolution(
        conversation_ref=conversation,
        resolved_status="unavailable",
        source="canonical-read",
    ) is True
    assert archive.conversation_terminal_marker(conversation) is None

    events = archive.store.tui_events(conversation)
    assert events[-1]["terminal_resolution"] is True
    assert events[-1]["resolved_status"] == "unavailable"
    assert events[-1]["terminal_source"] == "canonical-read"
    transcript = archive.conversation_paths(conversation)["transcript"].read_text(
        encoding="utf-8"
    )
    assert "Newer canonical evidence superseded" not in transcript
    assert "## CHAT — resolved" not in transcript
    assert any(
        event.get("status") == "unavailable"
        and event.get("role") == "chat"
        for event in events[:-1]
    )

    restarted = TUIArchive(root, db_path=archive.store.db_path)
    assert restarted.conversation_terminal_marker(conversation) is None
    rebuilt_transcript = restarted.conversation_paths(conversation)[
        "transcript"
    ].read_text(encoding="utf-8")
    assert "Newer canonical evidence superseded" not in rebuilt_transcript
    assert "## CHAT — resolved" not in rebuilt_transcript

    # The same semantic observation is valid again after a resolution. Historical
    # dedupe must not suppress a new recurrence of the current terminal state.
    restarted.record_observed_terminal(
        conversation_ref=conversation,
        label="chat",
        status="unavailable",
        text="ChatGPT web UI cannot load this conversation.",
        source="web-ui",
    )
    assert restarted.conversation_terminal_marker(conversation) == (
        "chat",
        "unavailable",
        "ChatGPT web UI cannot load this conversation.",
        "web-ui",
    )

    restarted.record_observed_terminal(
        conversation_ref=conversation,
        label="chat",
        status="unavailable",
        text="Unavailable again after a later observation.",
        source="web-ui",
    )
    assert restarted.conversation_terminal_marker(conversation) == (
        "chat",
        "unavailable",
        "Unavailable again after a later observation.",
        "web-ui",
    )


def test_chat_terminal_resolution_requires_matching_current_status(tmp_path) -> None:
    archive = TUIArchive(tmp_path / "archive")
    conversation = "conv-limit"

    archive.record_observed_terminal(
        conversation_ref=conversation,
        label="chat",
        status="limit-reached",
        text="This conversation reached its maximum length.",
        source="web-ui",
    )

    assert archive.record_chat_terminal_resolution(
        conversation_ref=conversation,
        resolved_status="unavailable",
        source="canonical-read",
    ) is False
    assert archive.conversation_terminal_marker(conversation) == (
        "chat",
        "limit-reached",
        "This conversation reached its maximum length.",
        "web-ui",
    )


def test_terminal_resolution_cannot_hide_newer_concurrent_chat_marker(tmp_path) -> None:
    root = tmp_path / "archive"
    first = TUIArchive(root)
    second = TUIArchive(root, db_path=first.store.db_path)

    for index in range(20):
        conversation = f"conv-race-{index}"
        first.record_observed_terminal(
            conversation_ref=conversation,
            label="chat",
            status="unavailable",
            text=f"unavailable-{index}",
            source="stream",
        )

        barrier = threading.Barrier(3)
        errors: list[BaseException] = []

        def resolve() -> None:
            try:
                barrier.wait(timeout=2)
                first.record_chat_terminal_resolution(
                    conversation_ref=conversation,
                    resolved_status="unavailable",
                    source="canonical-read",
                )
            except BaseException as error:
                errors.append(error)

        def record_new_terminal() -> None:
            try:
                barrier.wait(timeout=2)
                second.record_observed_terminal(
                    conversation_ref=conversation,
                    label="chat",
                    status="limit-reached",
                    text=f"limit-{index}",
                    source="web-ui",
                )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=resolve),
            threading.Thread(target=record_new_terminal),
        ]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=2)
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

        assert not errors
        assert first.conversation_terminal_marker(conversation) == (
            "chat",
            "limit-reached",
            f"limit-{index}",
            "web-ui",
        )


def test_run_and_tui_archive_share_one_transactional_store(tmp_path) -> None:
    state_path = tmp_path / "gptty_state.json"
    db_path = local_store_path(profile=None, state_path=state_path)
    recorder = start_run(
        profile=None,
        state_path=state_path,
        command="chat",
        conversation_ref="conv-12345678",
    )
    archive = TUIArchive(tmp_path / "archive", db_path=db_path)

    archive.record_user(
        "shared-store",
        conversation_ref="conv-12345678",
        model=None,
    )

    assert recorder.store_file == db_path
    assert archive.store.db_path == db_path
    store = LocalEventStore(db_path)
    assert store.run_summary(recorder.run_id)["conversation_ref"] == "conv-12345678"
    assert store.tui_events("conv-12345678")[0]["text"] == "shared-store"


def test_incremental_archive_append_does_not_reload_full_history(
    monkeypatch,
    tmp_path,
) -> None:
    archive = TUIArchive(tmp_path / "archive")
    turn_id = archive.record_user(
        "first",
        conversation_ref="conv-12345678",
        model=None,
    )

    def fail_full_history(*_args, **_kwargs):
        raise AssertionError("incremental append must not reload full TUI history")

    monkeypatch.setattr(archive.store, "tui_events", fail_full_history)

    archive.record_assistant(
        turn_id,
        conversation_ref="conv-12345678",
        text="second",
        title=None,
        model=None,
        status="complete",
    )

    events = _events(archive, "conv-12345678")
    assert [(event["role"], event["text"]) for event in events] == [
        ("user", "first"),
        ("assistant", "second"),
    ]


def test_projection_failure_does_not_erase_committed_tui_event(
    monkeypatch,
    tmp_path,
) -> None:
    archive = TUIArchive(tmp_path / "archive")
    archive.record_user(
        "first",
        conversation_ref="conv-12345678",
        model=None,
    )

    def fail_projection(*_args, **_kwargs):
        raise OSError("projection unavailable")

    monkeypatch.setattr(archive, "_append_json_event", fail_projection)
    monkeypatch.setattr(archive, "_append_text", fail_projection)

    archive.record_terminal(
        "turn-terminal",
        conversation_ref="conv-12345678",
        label="turn",
        status="unconfirmed",
        text="durable despite projection failure",
        source="stream",
    )

    authoritative = archive.store.tui_events("conv-12345678")
    assert authoritative[-1]["event_id"] == "turn-terminal:terminal"
    assert authoritative[-1]["text"] == "durable despite projection failure"

def test_restart_reconciles_committed_event_missing_from_projections(tmp_path) -> None:
    root = tmp_path / "archive"
    first = TUIArchive(root)
    first.record_user(
        "projected",
        conversation_ref="conv-12345678",
        model=None,
    )
    missing = {
        "schema": 1,
        "event_id": "turn-crash:terminal",
        "turn_id": "turn-crash",
        "observed_at": "2026-09-24T00:00:00+00:00",
        "source": "gptty-tui",
        "scope": "tui-observed",
        "role": "chat",
        "text": "committed before crash",
        "status": "limit-reached",
        "terminal_source": "stream",
        "conversation_id": "conv-12345678",
    }
    assert first.store.insert_tui_event("conv-12345678", missing) is True

    before = _events(first, "conv-12345678")
    assert [event["event_id"] for event in before] == [before[0]["event_id"]]

    restarted = TUIArchive(root, db_path=first.store.db_path)
    assert restarted.conversation_terminal_marker("conv-12345678") == (
        "chat",
        "limit-reached",
        "committed before crash",
        "stream",
    )

    repaired = _events(restarted, "conv-12345678")
    assert [event["event_id"] for event in repaired][-1] == "turn-crash:terminal"
    transcript = restarted.conversation_paths("conv-12345678")["transcript"].read_text(
        encoding="utf-8"
    )
    assert "committed before crash" in transcript


def test_concurrent_archive_writers_keep_projection_order_equal_to_sqlite_order(
    tmp_path,
) -> None:
    root = tmp_path / "archive"
    first = TUIArchive(root)
    second = TUIArchive(root, db_path=first.store.db_path)
    first.record_user(
        "seed",
        conversation_ref="conv-12345678",
        model=None,
    )
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def writer(archive: TUIArchive, turn_id: str, text: str) -> None:
        try:
            barrier.wait(timeout=3)
            archive.record_terminal(
                turn_id,
                conversation_ref="conv-12345678",
                label="turn",
                status="unconfirmed",
                text=text,
                source="stream",
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(first, "turn-a", "A")),
        threading.Thread(target=writer, args=(second, "turn-b", "B")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert not errors
    authoritative = first.store.tui_events("conv-12345678")
    projected = _events(first, "conv-12345678")
    assert [event["event_id"] for event in projected] == [
        event["event_id"] for event in authoritative
    ]


def test_startup_reconciles_old_orphan_pending_prompt_but_keeps_recent(tmp_path) -> None:
    root = tmp_path / "archive"
    first = TUIArchive(root, reconcile_pending=False)
    now = datetime.now(timezone.utc)
    old_at = now - timedelta(days=2)
    recent_at = now - timedelta(hours=1)
    first.store.put_pending_tui_event(
        "old-turn",
        {
            "event_id": "old-turn:user",
            "role": "user",
            "text": "old pending",
            "observed_at": old_at.isoformat(),
        },
    )
    first.store.put_pending_tui_event(
        "recent-turn",
        {
            "event_id": "recent-turn:user",
            "role": "user",
            "text": "recent pending",
            "observed_at": recent_at.isoformat(),
        },
    )
    old_projection = first.pending_dir / "old-turn.json"
    recent_projection = first.pending_dir / "recent-turn.json"
    old_projection.write_text("{}\n", encoding="utf-8")
    recent_projection.write_text("{}\n", encoding="utf-8")
    os.utime(old_projection, (old_at.timestamp(), old_at.timestamp()))
    os.utime(recent_projection, (recent_at.timestamp(), recent_at.timestamp()))

    restarted = TUIArchive(root, db_path=first.store.db_path)

    assert restarted.store.pop_pending_tui_event("old-turn") is None
    assert not old_projection.exists()
    assert restarted.store.pop_pending_tui_event("recent-turn") is not None
    assert recent_projection.exists()


def test_archive_prune_removes_event_only_conversation_from_db_and_projection(
    tmp_path,
) -> None:
    root = tmp_path / "archive"
    archive = TUIArchive(root, reconcile_pending=False)
    conversation_id = "conv-12345678"
    old_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event = {
        "event_id": "turn-1:terminal",
        "role": "chat",
        "text": "event-only durable copy",
        "status": "unconfirmed",
        "terminal_source": "stream",
        "observed_at": old_at.isoformat(),
    }
    assert archive.store.insert_tui_event(conversation_id, event) is True
    directory = archive.conversation_paths(conversation_id)["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "events.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(directory, (old_at.timestamp(), old_at.timestamp()))

    removed = archive.prune_conversations_before(
        datetime(2026, 9, 1, tzinfo=timezone.utc)
    )

    assert removed == 1
    assert not directory.exists()
    assert archive.store.tui_events(conversation_id) == []


def test_source_citation_observations_survive_restart_and_project_markdown(tmp_path) -> None:
    root = tmp_path / "archive"
    archive = TUIArchive(root)
    turn_id = archive.record_user(
        "question",
        conversation_ref="conv-12345678",
        model=None,
    )
    observations = {
        "sources": [
            {
                "kind": "source",
                "source_id": "source-1",
                "url": "https://example.com/article",
                "title": "Example Article",
                "domain": "example.com",
            }
        ],
        "citations": [
            {
                "kind": "citation",
                "citation_id": "citation-1",
                "source_id": "source-1",
                "start_index": 999999,
                "end_index": 1000000,
                "range_coordinate_space": "unknown",
            }
        ],
    }
    archive.record_assistant(
        turn_id,
        conversation_ref="conv-12345678",
        text="answer",
        title="Sources Test",
        model="gpt-test",
        status="complete",
        observations=observations,
    )

    events = archive.store.tui_events("conv-12345678")
    assistant = next(event for event in events if event.get("role") == "assistant")
    assert assistant["observations"]["sources"][0]["source_id"] == "source-1"
    assert (
        assistant["observations"]["citations"][0]["range_coordinate_space"]
        == "unknown"
    )

    transcript = archive.conversation_paths("conv-12345678")["transcript"].read_text(
        encoding="utf-8"
    )
    assert "### Sources" in transcript
    assert "Example Article" in transcript
    assert "https://example.com/article" in transcript
    assert "999999" not in transcript
    assert "1000000" not in transcript

    restarted = TUIArchive(root, db_path=archive.store.db_path)
    restored = restarted.source_citation_observations("conv-12345678")
    assert [item["source_id"] for item in restored["sources"]] == ["source-1"]
    assert [item["citation_id"] for item in restored["citations"]] == ["citation-1"]
    assert restored["citations"][0]["range_coordinate_space"] == "unknown"
