from __future__ import annotations

from types import SimpleNamespace

from gptty.state import (
    ChatState,
    GoalState,
    load_chat_state,
)
from gptty.ui.commands import InteractiveCommands


class FakeUI:
    def __init__(self, *, choices=None, image_paths=None) -> None:
        self.choices = list(choices or [])
        self.image_paths = list(image_paths or [])
        self.seen: list[tuple[str, list[tuple[object, str]]]] = []
        self.history_persistent = True
        self.history_clear_count = 0
        self.settings = SimpleNamespace(history_limit=2_000)

    def set_history_persistent(self, enabled):
        self.history_persistent = bool(enabled)

    def clear_history(self):
        self.history_clear_count += 1

    def choose_searchable(self, message, options, *, default=None):
        self.seen.append((message, list(options)))
        return self.choices.pop(0) if self.choices else default

    async def choose_searchable_async(self, message, options):
        self.seen.append((message, list(options)))
        return self.choices.pop(0) if self.choices else None

    def read_image_path(self):
        return self.image_paths.pop(0) if self.image_paths else None

    async def read_image_path_async(self):
        return self.image_paths.pop(0) if self.image_paths else None


class FakeRenderer:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def clear_context(self):
        self.events.append(("clear_context", None))

    def header(self, **kwargs):
        self.events.append(("header", kwargs))

    def start_elapsed(self, *, initial_elapsed=0.0):
        self.events.append(("start_elapsed", initial_elapsed))

    def finish_elapsed(self):
        self.events.append(("finish_elapsed", None))

    def chat_link(self, ref):
        self.events.append(("chat_link", ref))

    def turn_abort(self):
        self.events.append(("turn_abort", None))

    def info(self, text):
        self.events.append(("info", text))

    def warning(self, text):
        self.events.append(("warning", text))

    def turn_marker(self, label, status, message):
        self.events.append(
            (
                "turn_marker",
                {"label": label, "status": status, "message": message},
            )
        )

    def messages(self, messages):
        self.events.append(("messages", messages))


class FakeClient:
    def __init__(self, *, snapshots=None) -> None:
        self.calls: list[tuple[str, object]] = []
        self.snapshots = list(
            snapshots
            or [
                {
                    "status": SimpleNamespace(status="completed"),
                    "messages": [
                        {"message_id": "u1", "role": "user", "text": "question"},
                        {"message_id": "t1", "role": "tool", "text": "raw tool result"},
                        {
                            "message_id": "call1",
                            "role": "assistant",
                            "recipient": "api_tool.call_tool",
                            "text": "raw tool call",
                        },
                        {"message_id": "a1", "role": "assistant", "text": "answer"},
                    ],
                }
            ]
        )

    def list_conversations(self):
        self.calls.append(("list_conversations", None))
        return [
            {"id": "conv-2", "title": "Second chat", "update_time": 2.0},
            {"id": "conv-1", "title": "First chat", "update_time": 1.0},
        ]

    def conversation_snapshot(self, ref, **options):
        self.calls.append(("snapshot", ref))
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]

    def stop_generation(self, ref, **options):
        self.calls.append(("stop_generation", (ref, options)))
        return {"ok": True, "stopped": True, "conversationId": ref}

    def get_messages(self, ref):
        self.calls.append(("get_messages", ref))
        return [
            {"role": "user", "text": "question"},
            {"role": "assistant", "text": "answer"},
        ]

    def temporary_lifecycle_snapshot(self):
        self.calls.append(("temporary_lifecycle_snapshot", None))
        return {"state": "LIVE", "conversation_id": "temp-1"}

    def end_temporary_chat(self):
        self.calls.append(("end_temporary_chat", None))
        return True

    def list_models(self):
        self.calls.append(("list_models", None))
        return [
            {"slug": "gpt-real-a", "title": "Real A"},
            {"slug": "gpt-real-b", "title": "Real B"},
            {"slug": "disabled", "title": "Disabled", "is_disabled": True},
            {"slug": "work-mode", "title": "Work Mode", "is_work_mode_model": True},
            {"slug": "research", "title": "Deep Research"},
        ]


def make_commands(
    tmp_path, *, state=None, ui=None, client=None, tui_archive=None, runner_id=None
):
    state = state or ChatState()
    ui = ui or FakeUI()
    client = client or FakeClient()
    renderer = FakeRenderer()
    state_path = tmp_path / "gptty_state.json"
    commands = InteractiveCommands(
        state=state,
        state_path=state_path,
        get_client=lambda: client,
        ui=ui,
        renderer=renderer,
        tui_archive=tui_archive,
        runner_id=runner_id,
    )
    return commands, renderer, client, state_path


def finish_pending_resume(commands, client):
    request = commands.take_pending_resume()
    assert request is not None
    snapshot = client.conversation_snapshot(request.conversation_ref)
    commands.complete_resume(request, snapshot)
    return request


def test_resume_lists_real_conversations_and_renders_full_history(tmp_path) -> None:
    ui = FakeUI(choices=["conv-2"])
    commands, renderer, client, state_path = make_commands(tmp_path, ui=ui)

    assert commands.handle("/resume") is None
    assert client.calls == [("list_conversations", None)]
    assert load_chat_state(state_path).current_conversation is None

    finish_pending_resume(commands, client)

    assert client.calls[:2] == [
        ("list_conversations", None),
        ("snapshot", "conv-2"),
    ]
    assert load_chat_state(state_path).current_conversation == "conv-2"
    clear_index = next(
        index
        for index, event in enumerate(renderer.events)
        if event[0] == "clear_context"
    )
    assert clear_index > 0
    rendered = [event for event in renderer.events if event[0] == "messages"][-1][1]
    assert [message.text for message in rendered] == ["question", "answer"]


def test_goal_resume_refuses_to_continue_in_different_conversation(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-2",
        goal=GoalState(conversation_ref="conv-1", status="paused", turn_count=4),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    commands.handle("/goal resume")

    assert state.goal is not None and state.goal.status == "paused"
    assert commands.has_automatic_prompt is False
    assert renderer.events[-1] == (
        "warning",
        "Goal belongs to conv-1. Resume that conversation before /goal resume.",
    )
