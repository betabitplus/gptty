from __future__ import annotations

from types import SimpleNamespace

from gptty.state import ChatState, StateError, load_chat_state
from gptty.ui.commands import InteractiveCommands


class FakeUI:
    def __init__(self, *, choices=None) -> None:
        self.choices = list(choices or [])
        self.seen: list[tuple[str, list[tuple[object, str]]]] = []

    def choose_searchable(self, message, options, *, default=None):
        self.seen.append((message, list(options)))
        return self.choices.pop(0) if self.choices else default


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

    def list_models(self):
        self.calls.append(("list_models", None))
        return [
            {"slug": "gpt-real-a", "title": "Real A"},
            {"slug": "gpt-real-b", "title": "Real B"},
            {"slug": "disabled", "title": "Disabled", "is_disabled": True},
            {"slug": "work-mode", "title": "Work Mode", "is_work_mode_model": True},
            {"slug": "research", "title": "Deep Research"},
        ]


def make_commands(tmp_path, *, state=None, ui=None, client=None):
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
    )
    return commands, renderer, client, state_path


def test_resume_lists_real_conversations_and_renders_full_history(tmp_path) -> None:
    ui = FakeUI(choices=["conv-2"])
    commands, renderer, client, state_path = make_commands(tmp_path, ui=ui)

    assert commands.handle("/resume") is None

    assert client.calls[:2] == [
        ("list_conversations", None),
        ("snapshot", "conv-2"),
    ]
    assert load_chat_state(state_path).current_conversation == "conv-2"
    assert renderer.events[0] == ("clear_context", None)
    rendered = [event for event in renderer.events if event[0] == "messages"][-1][1]
    assert [message.text for message in rendered] == ["question", "answer"]


def test_resume_switches_while_already_attached_without_detach(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, _, client, state_path = make_commands(
        tmp_path,
        state=state,
        ui=FakeUI(choices=["conv-2"]),
    )

    commands.handle("/resume")

    assert client.calls[:2] == [
        ("list_conversations", None),
        ("snapshot", "conv-2"),
    ]
    assert load_chat_state(state_path).current_conversation == "conv-2"


def test_resume_direct_ref_skips_catalog_picker(tmp_path) -> None:
    commands, _, client, state_path = make_commands(tmp_path)

    commands.handle("/resume https://chatgpt.com/c/direct")

    assert ("list_conversations", None) not in client.calls
    assert client.calls[0] == ("snapshot", "direct")
    assert load_chat_state(state_path).current_conversation == "direct"


def test_detach_is_local_only(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, client, state_path = make_commands(tmp_path, state=state)

    commands.handle("/detach")

    assert state.current_conversation is None
    assert load_chat_state(state_path).current_conversation is None
    assert client.calls == []
    assert renderer.events[0] == ("clear_context", None)
    assert "not changed" in renderer.events[-1][1]


def test_model_uses_live_catalog_slug(tmp_path) -> None:
    state = ChatState(model="old")
    commands, renderer, client, state_path = make_commands(
        tmp_path,
        state=state,
        ui=FakeUI(choices=["gpt-real-b"]),
    )

    commands.handle("/model")

    assert client.calls == [("list_models", None)]
    assert state.model == "gpt-real-b"
    assert load_chat_state(state_path).model == "gpt-real-b"
    assert renderer.events[-1] == ("info", "Model: gpt-real-b")


def test_model_picker_excludes_non_chat_modes(tmp_path) -> None:
    ui = FakeUI(choices=[None])
    commands, _, client, _ = make_commands(tmp_path, ui=ui)

    commands.handle("/model")

    assert client.calls == [("list_models", None)]
    _message, options = ui.seen[-1]
    values = [value for value, _label in options]
    assert options[0][1].startswith("Default · latest frontier · High")
    assert "gpt-real-a" in values
    assert "gpt-real-b" in values
    assert "disabled" not in values
    assert "work-mode" not in values
    assert "research" not in values


def test_model_picker_can_reset_to_default(tmp_path) -> None:
    state = ChatState(model="gpt-real-a")
    commands, renderer, client, state_path = make_commands(
        tmp_path,
        state=state,
        ui=FakeUI(choices=[""]),
    )

    commands.handle("/model")

    assert client.calls == [("list_models", None)]
    assert state.model is None
    assert load_chat_state(state_path).model is None
    assert renderer.events[-1] == ("info", "Model: latest frontier · High")


def test_model_default_is_local_only(tmp_path) -> None:
    state = ChatState(model="gpt-real-a")
    commands, renderer, client, state_path = make_commands(tmp_path, state=state)

    commands.handle("/model default")

    assert client.calls == []
    assert state.model is None
    assert load_chat_state(state_path).model is None
    assert renderer.events[-1] == ("info", "Model: latest frontier · High")


def test_model_rejects_slug_not_in_live_catalog(tmp_path) -> None:
    state = ChatState(model="gpt-real-a")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    commands.handle("/model invented")

    assert state.model == "gpt-real-a"
    assert renderer.events[-1][0] == "warning"
    assert "live ChatGPT list" in renderer.events[-1][1]


def test_resume_follows_active_chat_until_completed(tmp_path, monkeypatch) -> None:
    running = {
        "status": SimpleNamespace(status="running"),
        "messages": [{"message_id": "u1", "role": "user", "text": "question"}],
    }
    completed = {
        "status": SimpleNamespace(status="completed"),
        "messages": [
            {"message_id": "u1", "role": "user", "text": "question"},
            {"message_id": "a1", "role": "assistant", "text": "finished"},
        ],
    }
    client = FakeClient(snapshots=[running, completed])
    commands, renderer, _, _ = make_commands(tmp_path, client=client)
    monkeypatch.setattr("gptty.ui.commands.time.sleep", lambda _seconds: None)
    notified: list[bool] = []
    monkeypatch.setattr("gptty.ui.commands.notify_response_complete", lambda: notified.append(True))

    commands.handle("/resume conv-1")

    message_events = [event for event in renderer.events if event[0] == "messages"]
    assert len(message_events) == 2
    assert message_events[-1][1][0].text == "finished"
    assert any(event[0] == "start_elapsed" for event in renderer.events)
    assert ("finish_elapsed", None) in renderer.events
    assert ("chat_link", "conv-1") in renderer.events
    assert notified == [True]
    assert [call[0] for call in client.calls].count("snapshot") == 2


def test_new_clears_current_conversation(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, state_path = make_commands(tmp_path, state=state)

    commands.handle("/new")

    assert state.current_conversation is None
    assert load_chat_state(state_path).current_conversation is None
    assert renderer.events[0] == ("clear_context", None)


def test_state_save_failure_rolls_back_interactive_change(tmp_path, monkeypatch) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    def fail_save(*args, **kwargs) -> None:
        raise StateError("disk failed")

    monkeypatch.setattr("gptty.ui.commands.save_chat_state", fail_save)
    commands.handle("/new")

    assert state.current_conversation == "conv-1"
    assert renderer.events[-1] == ("warning", "disk failed")
