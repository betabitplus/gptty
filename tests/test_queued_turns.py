from __future__ import annotations

import pytest

from gptty.queued_turns import (
    QueueBinding,
    QueueLimitError,
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


def test_queue_hold_prevents_automatic_pop_until_explicit_release() -> None:
    queue = QueuedTurnQueue()
    queue.enqueue("one", binding=_binding(), origin="working")

    assert queue.hold("stopped by user") is True
    assert queue.held is True
    assert queue.held_reason == "stopped by user"
    with pytest.raises(RuntimeError, match="held"):
        queue.popleft()

    assert queue.release() == 1
    assert queue.held is False
    assert queue.popleft().text == "one"


def test_release_can_explicitly_rebind_all_held_turns() -> None:
    queue = QueuedTurnQueue()
    queue.enqueue("one", binding=_binding(), origin="working")
    queue.enqueue("two", binding=_binding(), origin="working")
    queue.hold("conversation changed")
    new_binding = _binding(conversation_ref="conv-2", model="model-b")

    assert queue.release(binding=new_binding) == 2

    assert [item.binding for item in queue.items()] == [new_binding, new_binding]


def test_binding_match_allows_first_turn_to_acquire_conversation_id() -> None:
    queue = QueuedTurnQueue()
    queue.enqueue(
        "after first turn",
        binding=_binding(conversation_ref=None),
        origin="working",
    )

    assert queue.binding_matches(_binding(conversation_ref="created-conv")) is True
    assert queue.binding_matches(_binding(conversation_ref="created-conv", model="other")) is False


def test_queue_remove_by_position_or_unique_id_prefix() -> None:
    queue = QueuedTurnQueue()
    queue.enqueue("one", binding=_binding(), origin="working", turn_id="alpha111")
    queue.enqueue("two", binding=_binding(), origin="working", turn_id="bravo222")
    queue.enqueue("three", binding=_binding(), origin="working", turn_id="charlie333")

    assert queue.remove("2").text == "two"
    assert queue.remove("char").text == "three"
    assert queue.remove("missing") is None
    assert [item.text for item in queue.items()] == ["one"]


def test_clear_returns_turns_so_media_owners_can_release_files() -> None:
    queue = QueuedTurnQueue()
    queue.enqueue("one", media=["owned.png"], binding=_binding(), origin="working")
    queue.hold("failure")

    removed = queue.clear()

    assert [item.media for item in removed] == [("owned.png",)]
    assert len(queue) == 0
    assert queue.held is False


@pytest.mark.parametrize(
    ("kwargs", "second", "match"),
    [
        ({"max_turns": 1}, {"text": "two"}, "queue limit"),
        ({"max_text_chars": 5}, {"text": "12345"}, "queued text limit"),
        (
            {"max_media_items": 1},
            {"text": "two", "media": ["b.png"]},
            "queued media limit",
        ),
        (
            {"max_media_path_chars": 8},
            {"text": "two", "media": ["12345"]},
            "queued media metadata limit",
        ),
    ],
)
def test_queue_limits_fail_before_accepting_extra_turn(kwargs, second, match) -> None:
    queue = QueuedTurnQueue(**kwargs)
    first_media = ["a.png"] if "max_media_items" in kwargs else (
        ["1234"] if "max_media_path_chars" in kwargs else []
    )
    first_text = "x" if "max_text_chars" in kwargs else "one"
    queue.enqueue(first_text, media=first_media, binding=_binding(), origin="working")

    with pytest.raises(QueueLimitError, match=match):
        queue.enqueue(
            second["text"],
            media=second.get("media", []),
            binding=_binding(),
            origin="working",
        )

    assert len(queue) == 1


class _Renderer:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.events.append(("info", message))

    def warning(self, message: str) -> None:
        self.events.append(("warning", message))


class _Commands:
    def __init__(self, state) -> None:
        self.state = state
        self.pending_media = ["pending.png"]
        self.conversation_ref = state.current_conversation
        self.conversation_mode = "normal"
        self.goal_active = False
        self.released: list[list[str]] = []
        self.take_count = 0

    def take_pending_media(self):
        self.take_count += 1
        items = list(self.pending_media)
        self.pending_media.clear()
        return items

    def release_media(self, media):
        self.released.append(list(media))


def test_enqueue_limit_rejection_keeps_pending_media_unconsumed() -> None:
    from gptty.commands import chat as chat_module
    from gptty.state import ChatState

    state = ChatState(current_conversation="conv-1", model="model-a")
    commands = _Commands(state)
    renderer = _Renderer()
    queue = QueuedTurnQueue(max_turns=1)
    queue.enqueue("existing", binding=_binding(), origin="working")

    accepted = chat_module._enqueue_queued_turn(
        "rejected",
        state=state,
        commands=commands,
        renderer=renderer,
        queued_turns=queue,
        origin="working",
    )

    assert accepted is False
    assert commands.pending_media == ["pending.png"]
    assert commands.take_count == 0
    assert len(queue) == 1
    assert renderer.events[-1][0] == "warning"


def test_queue_command_remove_clear_and_send_release_or_rebind_safely() -> None:
    from gptty.commands import chat as chat_module
    from gptty.state import ChatState

    state = ChatState(current_conversation="conv-current", model="model-current")
    commands = _Commands(state)
    commands.pending_media.clear()
    renderer = _Renderer()
    queue = QueuedTurnQueue()
    queue.enqueue(
        "one",
        media=["owned-a.png"],
        binding=_binding(conversation_ref="conv-old", model="model-old"),
        origin="working",
        turn_id="alpha111",
    )
    queue.enqueue(
        "two",
        media=["owned-b.png"],
        binding=_binding(conversation_ref="conv-old", model="model-old"),
        origin="working",
        turn_id="bravo222",
    )
    queue.hold("context changed")

    assert chat_module._handle_queue_command(
        "/queue remove 1",
        state=state,
        commands=commands,
        renderer=renderer,
        queued_turns=queue,
    )
    assert commands.released == [["owned-a.png"]]
    assert [item.text for item in queue.items()] == ["two"]

    assert chat_module._handle_queue_command(
        "/queue send",
        state=state,
        commands=commands,
        renderer=renderer,
        queued_turns=queue,
    )
    assert queue.held is False
    assert queue.peek().binding.conversation_ref == "conv-current"
    assert queue.peek().binding.model == "model-current"

    assert chat_module._handle_queue_command(
        "/queue clear",
        state=state,
        commands=commands,
        renderer=renderer,
        queued_turns=queue,
    )
    assert commands.released == [["owned-a.png"], ["owned-b.png"]]
    assert len(queue) == 0
