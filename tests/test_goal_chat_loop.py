from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from gptty.commands.chat import run_chat
from gptty.state import ChatState, GoalState, load_chat_state, save_chat_state


class _GoalLoopClient:
    instances: list["_GoalLoopClient"] = []

    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.__class__.instances.append(self)

    def send(self, prompt: str, **options):
        self.calls.append(("send", prompt, None))
        return SimpleNamespace(
            text="GPTTY_GOAL: CONTINUE\nFirst chunk finished; more remains.",
            conversation_id="conv-goal",
            title="Goal loop test",
        )

    def send_to_conversation(self, ref: str, prompt: str, **options):
        self.calls.append(("send_to_conversation", prompt, ref))
        return SimpleNamespace(
            text="GPTTY_GOAL: COMPLETE\nAll agreed work is done and verified.",
            conversation_id=ref,
            title="Goal loop test",
        )


class _FakeSession:
    script = iter(())

    def __init__(self, **kwargs) -> None:
        pass

    def read_prompt(self, *, attachment_count: int = 0) -> str:
        return next(self.script)


class _FakeRenderer:
    instances: list["_FakeRenderer"] = []

    def __init__(self, stdout, settings) -> None:
        self.events: list[tuple[str, object]] = []
        self.__class__.instances.append(self)

    def header(self, **kwargs) -> None:
        self.events.append(("header", kwargs))

    def turn_start(self) -> None:
        self.events.append(("turn_start", None))

    def live_event(self, event) -> None:
        self.events.append(("live_event", event))

    def answer(self, text: str) -> None:
        self.events.append(("answer", text))

    def chat_link(self, ref: str) -> None:
        self.events.append(("chat_link", ref))

    def turn_abort(self) -> None:
        self.events.append(("turn_abort", None))

    def info(self, text: str) -> None:
        self.events.append(("info", text))

    def warning(self, text: str) -> None:
        self.events.append(("warning", text))

    def clear_context(self) -> None:
        self.events.append(("clear_context", None))

    def messages(self, messages) -> None:
        self.events.append(("messages", messages))


def _args(tmp_path):
    return SimpleNamespace(
        state=str(tmp_path / "state.json"),
        auth=str(tmp_path / "auth.json"),
        model=None,
        no_stream=True,
        timeout=90,
        plain=False,
        wait_lock=False,
        lock_timeout=None,
        profile=None,
    )


def test_goal_chat_loop_auto_continues_until_complete_without_intermediate_notification(tmp_path, monkeypatch) -> None:
    _GoalLoopClient.instances.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(['/goal "Finish exactly this test goal"', "/exit"])
    normal_notifications: list[dict[str, object]] = []
    goal_notifications: list[dict[str, object]] = []

    monkeypatch.setattr("gptty.commands.chat.should_use_enhanced_ui", lambda **kwargs: (True, SimpleNamespace()))
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)
    monkeypatch.setattr(
        "gptty.commands.chat.notify_response_complete",
        lambda **kwargs: normal_notifications.append(kwargs),
    )
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: goal_notifications.append(kwargs),
    )

    code = run_chat(
        _args(tmp_path),
        client_factory=_GoalLoopClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = _GoalLoopClient.instances[0]
    assert [call[0] for call in client.calls] == ["send", "send_to_conversation"]
    assert "GPTTY Goal mode is now active" in client.calls[0][1]
    assert "Continue pursuing the active goal" in client.calls[1][1]
    state = load_chat_state(tmp_path / "state.json")
    assert state.current_conversation == "conv-goal"
    assert state.goal is not None
    assert state.goal.status == "complete"
    assert state.goal.turn_count == 2
    assert normal_notifications == []
    assert goal_notifications == [
        {"chat_title": "Goal loop test", "final_response": "All agreed work is done and verified."}
    ]
    renderer = _FakeRenderer.instances[0]
    assert ("info", "Goal · continuing · next turn 2") in renderer.events
    assert ("info", "Goal · complete · 2 turns") in renderer.events


def test_active_goal_is_paused_on_process_restart_and_does_not_auto_resume(tmp_path) -> None:
    state_path = tmp_path / "state.json"
    save_chat_state(
        state_path,
        ChatState(
            current_conversation="conv-1",
            goal=GoalState(conversation_ref="conv-1", status="active", turn_count=7),
        ),
    )
    created: list[object] = []

    class NeverCreateClient:
        def __init__(self, *args, **kwargs) -> None:
            created.append(self)

    code = run_chat(
        _args(tmp_path),
        client_factory=NeverCreateClient,
        input_stream=StringIO("/exit\n"),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    restored = load_chat_state(state_path)
    assert restored.goal is not None
    assert restored.goal.status == "paused"
    assert restored.goal.reason == "gptty restarted while goal was active"
    assert restored.goal.turn_count == 7
    assert created == []


def test_goal_hard_chat_error_interrupts_without_auto_retry(tmp_path, monkeypatch) -> None:
    class HardFailureClient:
        instances: list["HardFailureClient"] = []

        def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
            self.calls: list[str] = []
            self.__class__.instances.append(self)

        def send(self, prompt: str, **options):
            self.calls.append(prompt)
            raise RuntimeError("CHATGPT_CONVERSATION_LIMIT_EXCEEDED")

    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(['/goal "Do the full task"'])
    goal_notifications: list[dict[str, object]] = []
    monkeypatch.setattr("gptty.commands.chat.should_use_enhanced_ui", lambda **kwargs: (True, SimpleNamespace()))
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: goal_notifications.append(kwargs),
    )

    stderr = StringIO()
    code = run_chat(
        _args(tmp_path),
        client_factory=HardFailureClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 1
    client = HardFailureClient.instances[0]
    assert len(client.calls) == 1
    state = load_chat_state(tmp_path / "state.json")
    assert state.goal is not None
    assert state.goal.status == "interrupted"
    assert state.goal.reason == "chat turn failed with exit code 1"
    assert goal_notifications == [
        {
            "chat_title": None,
            "final_response": "Goal interrupted. chat turn failed with exit code 1",
        }
    ]
    assert "CHATGPT_CONVERSATION_LIMIT_EXCEEDED" in stderr.getvalue()
