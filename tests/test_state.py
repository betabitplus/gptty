from __future__ import annotations

import json

import pytest

from gptty.state import ChatState, GoalState, StateError, load_chat_state, save_chat_state


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
