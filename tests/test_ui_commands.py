from __future__ import annotations

from argparse import Namespace
from io import StringIO

from gptty.state import ChatState, StateError, load_chat_state
from gptty.ui.commands import InteractiveCommands
from gptty.ui.state import RecentStore


class FakeUI:
    def __init__(self, *, choices=None, answers=None) -> None:
        self.choices = list(choices or [])
        self.answers = list(answers or [])
        self.settings = object()

    def choose(self, message, options, *, default=None):
        return self.choices.pop(0) if self.choices else default

    def ask(self, message, *, default=""):
        return self.answers.pop(0) if self.answers else default

    def edit_settings(self) -> bool:
        return False


class FakeRenderer:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def info(self, text):
        self.events.append(("info", text))

    def warning(self, text):
        self.events.append(("warning", text))

    def commands(self, commands):
        self.events.append(("commands", commands))

    def messages(self, messages):
        self.events.append(("messages", messages))

    def mapping(self, mapping):
        self.events.append(("mapping", mapping))


class FakeClient:
    def attach_conversation(self, ref):
        return {"conversation_id": f"attached-{ref}"}

    def get_messages(self, ref, **options):
        return {"messages": [{"role": "assistant", "text": "hello"}]}

    def get_status(self, ref):
        return {"status": "completed"}


def make_commands(tmp_path, *, state=None, ui=None, client=None):
    state = state or ChatState()
    ui = ui or FakeUI()
    client = client or FakeClient()
    renderer = FakeRenderer()
    state_path = tmp_path / "gptty_state.json"
    recent = RecentStore(tmp_path / "recent.json")
    commands = InteractiveCommands(
        args=Namespace(auth=str(tmp_path / "auth.json")),
        state=state,
        state_path=state_path,
        get_client=lambda: client,
        reset_client=lambda: None,
        ui=ui,
        renderer=renderer,
        recent=recent,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    return commands, renderer, recent, state_path


def test_attach_updates_state_and_recent_index(tmp_path) -> None:
    commands, renderer, recent, state_path = make_commands(tmp_path)

    assert commands.handle("/attach abc") is None

    assert load_chat_state(state_path).current_conversation == "attached-abc"
    assert recent.list()[0].ref == "attached-abc"
    assert renderer.events[-1] == ("info", "Attached: attached-abc")


def test_switch_uses_local_recent_without_client_call(tmp_path) -> None:
    recent = RecentStore(tmp_path / "recent.json")
    recent.remember("conv-1", label="First")
    recent.remember("conv-2", label="Second")
    state = ChatState(current_conversation="conv-1")
    ui = FakeUI(choices=["conv-2"])
    renderer = FakeRenderer()
    state_path = tmp_path / "gptty_state.json"

    commands = InteractiveCommands(
        args=Namespace(auth=str(tmp_path / "auth.json")),
        state=state,
        state_path=state_path,
        get_client=lambda: (_ for _ in ()).throw(AssertionError("network client must not be used")),
        reset_client=lambda: None,
        ui=ui,
        renderer=renderer,
        recent=recent,
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert commands.handle("/switch") is None
    assert state.current_conversation == "conv-2"
    assert load_chat_state(state_path).current_conversation == "conv-2"


def test_messages_and_status_reuse_current_client(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, _ = make_commands(tmp_path, state=state, ui=FakeUI(answers=["5"]))

    commands.handle("/messages")
    commands.handle("/status")

    assert renderer.events[0][0] == "messages"
    assert renderer.events[0][1][0].text == "hello"
    assert renderer.events[1][0] == "mapping"
    assert renderer.events[1][1]["status"] == "completed"


def test_new_clears_current_conversation(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, _, _, state_path = make_commands(tmp_path, state=state)

    commands.handle("/new")

    assert state.current_conversation is None
    assert load_chat_state(state_path).current_conversation is None


def test_model_cancel_keeps_existing_value(tmp_path) -> None:
    state = ChatState(model="deep")
    commands, _, _, _ = make_commands(tmp_path, state=state, ui=FakeUI(choices=[None]))

    commands.handle("/model")

    assert state.model == "deep"


def test_model_rejects_unknown_manual_value(tmp_path) -> None:
    state = ChatState(model="balanced")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    commands.handle("/model nonsense")

    assert state.model == "balanced"
    assert renderer.events[-1] == ("warning", "Usage: /model [default|fast|balanced|deep]")


def test_state_save_failure_rolls_back_interactive_change(tmp_path, monkeypatch) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    def fail_save(*args, **kwargs) -> None:
        raise StateError("disk failed")

    monkeypatch.setattr("gptty.ui.commands.save_chat_state", fail_save)
    commands.handle("/new")

    assert state.current_conversation == "conv-1"
    assert renderer.events[-1] == ("warning", "disk failed")
