from __future__ import annotations

import json

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
