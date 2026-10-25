from __future__ import annotations

from gptty.queued_turns import (
    QueueBinding,
    QueuedTurnQueue,
)


def _binding(
    *,
    conversation_ref: str | None = "conv-1",
    model: str | None = "model-a",
    goal_id: str | None = None,
    generation: int | None = None,
) -> QueueBinding:
    return QueueBinding(
        conversation_ref=conversation_ref,
        conversation_mode="normal",
        model=model,
        goal_id=goal_id,
        goal_generation=generation,
    )


def test_queue_binds_media_model_conversation_and_generation_immutably() -> None:
    queue = QueuedTurnQueue()
    turn = queue.enqueue(
        "prompt A",
        media=["a.png", "b.png"],
        binding=_binding(goal_id="goal-1", generation=3),
        origin="working",
        turn_id="turn-a",
        queued_at="2026-09-25T20:00:00Z",
    )

    assert turn.turn_id == "turn-a"
    assert turn.text == "prompt A"
    assert turn.media == ("a.png", "b.png")
    assert turn.binding.conversation_ref == "conv-1"
    assert turn.binding.model == "model-a"
    assert turn.binding.goal_id == "goal-1"
    assert turn.binding.goal_generation == 3
    assert turn.origin == "working"
