from __future__ import annotations

import json
import os
import pty

import pytest

from gptty.state import (
    ChatState,
    GoalState,
    StateError,
    load_chat_state,
    save_chat_state,
    session_chat_state_path,
)


def test_load_missing_state_returns_default(tmp_path) -> None:
    state = load_chat_state(tmp_path / "missing.json")

    assert state == ChatState()


def test_save_and_load_state_round_trips(tmp_path) -> None:
    path = tmp_path / "gptty_state.json"

    save_chat_state(path, ChatState(current_conversation="abc", model="gpt-4o"))

    assert load_chat_state(path) == ChatState(current_conversation="abc", model="gpt-4o")
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "current_conversation": "abc",
        "model": "gpt-4o",
    }


def test_goal_state_round_trips_and_old_state_stays_compatible(tmp_path) -> None:
    path = tmp_path / "gptty_state.json"
    state = ChatState(
        current_conversation="conv-goal",
        goal=GoalState(
            conversation_ref="conv-goal",
            status="active",
            objective="Finish it",
            turn_count=2,
            protocol_failures=1,
            reason="still working",
        ),
    )

    save_chat_state(path, state)

    assert load_chat_state(path) == state
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["goal"]["status"] == "active"
    assert payload["goal"]["turn_count"] == 2


def test_load_invalid_json_raises_state_error(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(StateError, match="failed to load state"):
        load_chat_state(path)


def test_load_non_object_json_raises_state_error(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(StateError, match="expected JSON object"):
        load_chat_state(path)


def test_load_state_normalizes_empty_values(tmp_path) -> None:
    path = tmp_path / "gptty_state.json"
    path.write_text(
        json.dumps({"current_conversation": " ", "model": None}),
        encoding="utf-8",
    )

    assert load_chat_state(path) == ChatState(current_conversation=None, model=None)


def test_session_state_path_explicit_ids_are_isolated_siblings(tmp_path) -> None:
    base = tmp_path / "gptty_state.json"
    left = session_chat_state_path(
        base, input_stream=None, environ={"GPTTY_SESSION_ID": "left"}
    )
    right = session_chat_state_path(
        base, input_stream=None, environ={"GPTTY_SESSION_ID": "right"}
    )

    assert left != right
    assert left.parent == base.parent == right.parent
    assert left.name.startswith("gptty_state.session-custom-")
    assert right.name.startswith("gptty_state.session-custom-")
    assert left.suffix == ".json"


def test_session_state_path_prefers_cmux_surface_for_real_tty(tmp_path) -> None:
    base = tmp_path / "gptty_state.json"
    master_fd, slave_fd = pty.openpty()
    try:
        with os.fdopen(slave_fd, "r", encoding="utf-8", closefd=True) as stream:
            first = session_chat_state_path(
                base, input_stream=stream, environ={"CMUX_SURFACE_ID": "surface-A"}
            )
            second = session_chat_state_path(
                base, input_stream=stream, environ={"CMUX_SURFACE_ID": "surface-B"}
            )
    finally:
        os.close(master_fd)

    assert first != second
    assert first.name.startswith("gptty_state.session-cmux-")
    assert second.name.startswith("gptty_state.session-cmux-")


def test_non_tty_does_not_inherit_ambient_terminal_identity(tmp_path) -> None:
    from io import StringIO

    base = tmp_path / "gptty_state.json"
    resolved = session_chat_state_path(
        base,
        input_stream=StringIO(),
        environ={"CMUX_SURFACE_ID": "ambient-surface", "TERM_SESSION_ID": "ambient-term"},
    )

    assert resolved == base
