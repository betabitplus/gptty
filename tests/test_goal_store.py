from __future__ import annotations

import json

from gptty.goal_store import GoalStore
from gptty.state import GoalCheckpoint, GoalState


def test_goal_store_writes_portable_json_checkpoint_and_pointer(tmp_path) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(
        goal_id="goal-123",
        conversation_ref="conv-2",
        conversations=["conv-1", "conv-2"],
        context_seed=["user: original request", "assistant: agreed approach"],
        status="active",
        objective="Finish the agreed task",
        turn_count=4,
        rollover_count=1,
        checkpoint=GoalCheckpoint(
            summary="Implementation is half complete.",
            completed=["commit A verified"],
            decisions=["keep API backwards compatible"],
            pending=["finish integration test"],
            next_step="run the live acceptance",
            updated_turn=4,
        ),
    )

    path = store.save(goal)

    assert path == tmp_path / "goals" / "goal-123" / "goal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == 1
    assert payload["goal"]["checkpoint"]["decisions"] == [
        "keep API backwards compatible"
    ]
    assert payload["goal"]["context_seed"] == [
        "user: original request",
        "assistant: agreed approach",
    ]
    checkpoint = store.checkpoint_path(goal).read_text(encoding="utf-8")
    assert "## Recovery context seed" in checkpoint
    assert "commit A verified" in checkpoint
    assert "https://chatgpt.com/c/conv-1" in checkpoint
    assert store.current_path().read_text(encoding="utf-8").strip() == "goal-123"
    assert store.load_current() == goal


def test_goal_store_clear_current_preserves_backupable_goal_directory(tmp_path) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(goal_id="goal-backup", objective="Portable goal")
    store.save(goal)

    store.clear_current()

    assert not store.current_path().exists()
    assert store.goal_path(goal).exists()
    assert store.checkpoint_path(goal).exists()
    assert store.load("goal-backup") == goal
