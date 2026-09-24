from __future__ import annotations

import json
import threading

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
