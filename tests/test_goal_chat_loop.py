from __future__ import annotations

import asyncio
import threading
import time
from io import StringIO
from types import SimpleNamespace

from gptty.commands import chat as chat_module
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

    async def read_prompt_async(self, *, attachment_count: int = 0) -> str:
        try:
            item = next(self.script)
        except StopIteration:
            await asyncio.Future()
            raise AssertionError("unreachable")
        if isinstance(item, tuple):
            predicate, value = item
            while not predicate():
                await asyncio.sleep(0.001)
            return value
        return item

    def set_active_turn(self, controls, *, working_status=None) -> None:
        pass


class _FakeRenderer:
    instances: list["_FakeRenderer"] = []

    def __init__(self, stdout, settings) -> None:
        self.events: list[tuple[str, object]] = []
        self.__class__.instances.append(self)

    def header(self, **kwargs) -> None:
        self.events.append(("header", kwargs))

    def turn_start(self, *, show_elapsed: bool = True) -> None:
        self.events.append(("turn_start", show_elapsed))

    def live_event(self, event) -> None:
        self.events.append(("live_event", event))

    def answer(self, text: str) -> None:
        self.events.append(("answer", text))

    def answer_model(
        self,
        observed_model: str | None,
        *,
        requested_model: str | None = None,
        sent_model: str | None = None,
    ) -> None:
        self.events.append(
            (
                "answer_model",
                {
                    "observed_model": observed_model,
                    "requested_model": requested_model,
                    "sent_model": sent_model,
                },
            )
        )

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


def test_turn_health_status_distinguishes_delivery_and_backend_stall(monkeypatch) -> None:
    now = 500.0
    monkeypatch.setattr(chat_module.time, "monotonic", lambda: now)
    health = chat_module._TurnHealth(last_server_progress_at=470.0)

    assert chat_module._working_status(450.0, 0, health=health) == (
        "working 00:50 · server 00:30 ago"
    )

    health.observe(
        {
            "type": "stream_handoff_ws_reconnecting",
            "attempt": 3,
            "server_idle_seconds": 45.0,
        }
    )
    assert chat_module._working_status(450.0, 0, health=health) == (
        "reconnecting delivery · attempt 3"
    )

    health.observe(
        {
            "type": "stream_handoff_server_quiet",
            "server_idle_seconds": 125.0,
        }
    )
    assert chat_module._working_status(450.0, 0, health=health) == (
        "server quiet 02:05 · no observable progress · waiting safely"
    )

    health.observe(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 305.0,
        }
    )
    assert chat_module._working_status(450.0, 1, health=health) == (
        "PROLONGED SILENCE · no observable server events 05:05"
        " · turn may still be working · do not resend yet · queued 1"
    )

    now = 501.0
    health.observe(
        {
            "type": "stream_handoff_server_resumed",
            "silent_seconds": 306.0,
        }
    )
    assert health.state == "working"
    assert chat_module._working_status(450.0, 0, health=health) == "working 00:51"

    health.observe(
        {
            "type": "assistant_text_delta",
            "message_id": "answer-1",
            "delta": "Final answer",
        }
    )
    now = 507.0
    assert chat_module._working_status(450.0, 0, health=health) == (
        "answer text received · finality unconfirmed 00:06 · do not resend yet"
    )
    health.observe(
        {
            "type": "stream_handoff_server_quiet",
            "server_idle_seconds": 125.0,
        }
    )
    assert chat_module._working_status(450.0, 0, health=health) == (
        "answer text received · finality unconfirmed 02:05"
    )

    health.observe(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 305.0,
        }
    )
    assert chat_module._working_status(450.0, 0, health=health) == (
        "FINALITY UNCONFIRMED · answer text received"
        " · no observable server events 05:05 · do not resend yet"
    )


def test_turn_health_marks_stall_after_failed_tool_result(monkeypatch) -> None:
    now = 700.0
    monkeypatch.setattr(chat_module.time, "monotonic", lambda: now)
    health = chat_module._TurnHealth(last_server_progress_at=690.0)

    health.observe(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_result",
            "text": (
                '{"codexpro_tool":"apply_patch",'
                '"error":"CodexProError: error: corrupt patch at line 13",'
                '"is_error":true}'
            ),
        }
    )
    health.observe(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 605.0,
        }
    )

    assert health.last_tool_error == "CodexProError: error: corrupt patch at line 13"
    status = chat_module._working_status(100.0, 0, health=health)
    assert status == (
        "PROLONGED SILENCE · no observable server events 10:05"
        " · last visible tool failed · turn may still recover"
    )
    assert "Ctrl-C" not in status
    assert "new turn" not in status


def test_goal_chat_loop_auto_continues_until_complete_without_intermediate_notification(
    tmp_path, monkeypatch
) -> None:
    _GoalLoopClient.instances.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            '/goal "Finish exactly this test goal"',
            (
                lambda: bool(_FakeRenderer.instances)
                and ("info", "Goal · complete · 2 turns")
                in _FakeRenderer.instances[0].events,
                "/exit",
            ),
        ]
    )
    normal_notifications: list[dict[str, object]] = []
    goal_notifications: list[dict[str, object]] = []

    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
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
        {
            "chat_title": "Goal loop test",
            "final_response": "All agreed work is done and verified.",
        }
    ]
    renderer = _FakeRenderer.instances[0]
    assert ("info", "Goal · continuing · next turn 2") in renderer.events
    assert ("info", "Goal · complete · 2 turns") in renderer.events


def test_active_goal_is_paused_on_process_restart_and_does_not_auto_resume(
    tmp_path,
) -> None:
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


def test_goal_hard_chat_error_interrupts_without_auto_retry(
    tmp_path, monkeypatch
) -> None:
    class HardFailureClient:
        instances: list["HardFailureClient"] = []

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[str] = []
            self.__class__.instances.append(self)

        def send(self, prompt: str, **options):
            self.calls.append(prompt)
            raise RuntimeError("CHATGPT_CONVERSATION_LIMIT_EXCEEDED")

    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(['/goal "Do the full task"'])
    goal_notifications: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
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


def test_goal_queued_steering_replaces_pending_auto_continuation(
    tmp_path, monkeypatch
) -> None:
    class SteeringClient:
        instances: list["SteeringClient"] = []
        first_started = threading.Event()
        release_first = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str, str | None]] = []
            self.__class__.instances.append(self)

        def send(self, prompt: str, **options):
            self.calls.append(("send", prompt, None))
            self.__class__.first_started.set()
            assert self.__class__.release_first.wait(timeout=2)
            return SimpleNamespace(
                text="GPTTY_GOAL: CONTINUE\nMore work remains.",
                conversation_id="conv-steer",
                title="Steering test",
            )

        def send_to_conversation(self, ref: str, prompt: str, **options):
            self.calls.append(("send_to_conversation", prompt, ref))
            return SimpleNamespace(
                text="GPTTY_GOAL: COMPLETE\nSteering was applied and the goal is complete.",
                conversation_id=ref,
                title="Steering test",
            )

    SteeringClient.instances.clear()
    SteeringClient.first_started.clear()
    SteeringClient.release_first.clear()
    _FakeRenderer.instances.clear()

    def steering_ready() -> bool:
        if not SteeringClient.first_started.is_set():
            return False
        SteeringClient.release_first.set()
        return True

    _FakeSession.script = iter(
        [
            '/goal "Finish the task"',
            (steering_ready, "Prioritize the release notes before finishing"),
            (
                lambda: bool(_FakeRenderer.instances)
                and ("info", "Goal · complete · 2 turns")
                in _FakeRenderer.instances[0].events,
                "/exit",
            ),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=SteeringClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = SteeringClient.instances[0]
    assert [call[0] for call in client.calls] == ["send", "send_to_conversation"]
    assert "Prioritize the release notes before finishing" in client.calls[1][1]
    assert "GPTTY Goal mode remains active" in client.calls[1][1]
    assert "Continue pursuing the active goal" not in client.calls[1][1]
    state = load_chat_state(tmp_path / "state.json")
    assert state.goal is not None
    assert state.goal.status == "complete"
    assert state.goal.turn_count == 2


def test_goal_pause_during_work_finishes_current_turn_and_cancels_auto_continue(
    tmp_path, monkeypatch
) -> None:
    class PauseClient:
        instances: list["PauseClient"] = []
        started = threading.Event()
        release = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[str] = []
            self.__class__.instances.append(self)

        def send(self, prompt: str, **options):
            self.calls.append(prompt)
            self.__class__.started.set()
            assert self.__class__.release.wait(timeout=2)
            return SimpleNamespace(
                text="GPTTY_GOAL: CONTINUE\nCurrent chunk is complete; more remains.",
                conversation_id="conv-pause",
                title="Goal pause test",
            )

        def send_to_conversation(self, ref: str, prompt: str, **options):
            raise AssertionError("paused goal must not auto-continue")

    PauseClient.instances.clear()
    PauseClient.started.clear()
    PauseClient.release.clear()
    _FakeRenderer.instances.clear()

    def pause_ready() -> bool:
        if not PauseClient.started.is_set():
            return False
        PauseClient.started.clear()
        threading.Timer(0.05, PauseClient.release.set).start()
        return True

    _FakeSession.script = iter(
        [
            '/goal "Finish the task"',
            (pause_ready, "/goal pause"),
            (
                lambda: bool(_FakeRenderer.instances)
                and ("info", "Goal · paused") in _FakeRenderer.instances[0].events,
                "/exit",
            ),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    stderr = StringIO()
    code = run_chat(
        _args(tmp_path),
        client_factory=PauseClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 0, stderr.getvalue()
    client = PauseClient.instances[0]
    assert len(client.calls) == 1
    state = load_chat_state(tmp_path / "state.json")
    assert state.goal is not None
    assert state.goal.status == "paused"
    assert state.goal.reason == "paused by user"
    assert state.goal.turn_count == 1
    renderer = _FakeRenderer.instances[0]
    assert (
        "info",
        "Goal · pause pending · current turn will finish",
    ) in renderer.events


def test_stop_command_while_working_preserves_and_sends_queued_prompt(
    tmp_path, monkeypatch
) -> None:
    class StopClient:
        instances: list["StopClient"] = []
        started = threading.Event()
        release = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str | None]] = []
            self.__class__.instances.append(self)

        def send(self, prompt: str, **options):
            self.calls.append(("send", prompt))
            self.__class__.started.set()
            assert self.__class__.release.wait(timeout=2)
            return SimpleNamespace(
                text="Saved partial response",
                conversation_id="conv-stop-command",
                title="Stop command test",
            )

        def send_to_conversation(self, ref: str, prompt: str, **options):
            self.calls.append(("send_to_conversation", prompt))
            return SimpleNamespace(
                text="Queued follow-up sent",
                conversation_id=ref,
                title="Stop command test",
            )

        def stop_generation(self, ref=None, **options):
            self.calls.append(("stop_generation", ref))
            self.__class__.release.set()
            return {"stopped": True, "conversationId": "conv-stop-command"}

    StopClient.instances.clear()
    StopClient.started.clear()
    StopClient.release.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            "Produce a long response",
            (StopClient.started.is_set, "Queued follow-up should run"),
            (
                lambda: bool(_FakeRenderer.instances)
                and ("info", "Queued · 1") in _FakeRenderer.instances[0].events,
                "/stop",
            ),
            (
                lambda: bool(_FakeRenderer.instances)
                and ("answer", "Queued follow-up sent")
                in _FakeRenderer.instances[0].events,
                "/exit",
            ),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=StopClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = StopClient.instances[0]
    assert client.calls[0] == ("send", "Produce a long response")
    assert client.calls[1] == ("stop_generation", None)
    assert client.calls[2] == ("send_to_conversation", "Queued follow-up should run")
    state = load_chat_state(tmp_path / "state.json")
    assert state.current_conversation == "conv-stop-command"
    renderer = _FakeRenderer.instances[0]
    assert ("info", "Stopping ChatGPT…") in renderer.events
    assert ("info", "ChatGPT stopped; finalizing local readback…") in renderer.events
    assert ("info", "Stopped by user.") in renderer.events
    assert ("info", "Queued · 1 · will send next") in renderer.events
    assert ("answer", "Queued follow-up sent") in renderer.events


def test_incomplete_turn_returns_prompt_and_clears_queued_followup(
    tmp_path, monkeypatch
) -> None:
    class IncompleteClient:
        instances: list["IncompleteClient"] = []
        started = threading.Event()
        release = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[str] = []
            self.__class__.instances.append(self)

        def send(self, prompt: str, **options):
            self.calls.append(prompt)
            self.__class__.started.set()
            assert self.__class__.release.wait(timeout=2)
            return SimpleNamespace(
                text="",
                title=None,
                conversation=SimpleNamespace(
                    conversation_id="conv-incomplete",
                    finish_reason="incomplete",
                ),
            )

        def send_to_conversation(self, ref: str, prompt: str, **options):
            raise AssertionError(
                "queued follow-up must be cleared after incomplete turn"
            )

    IncompleteClient.instances.clear()
    IncompleteClient.started.clear()
    IncompleteClient.release.clear()
    _FakeRenderer.instances.clear()

    def queue_ready() -> bool:
        if not IncompleteClient.started.is_set():
            return False
        IncompleteClient.release.set()
        return True

    _FakeSession.script = iter(
        [
            "Start a tool-heavy turn",
            (queue_ready, "Queued follow-up must not run"),
            (
                lambda: bool(_FakeRenderer.instances)
                and (
                    "warning",
                    "ChatGPT stream ended without a final answer; returned control to gptty.",
                )
                in _FakeRenderer.instances[0].events,
                "/exit",
            ),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)
    notifications: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.commands.chat.notify_response_complete",
        lambda **kwargs: notifications.append(kwargs),
    )

    code = run_chat(
        _args(tmp_path),
        client_factory=IncompleteClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = IncompleteClient.instances[0]
    assert client.calls == ["Start a tool-heavy turn"]
    state = load_chat_state(tmp_path / "state.json")
    assert state.current_conversation == "conv-incomplete"
    renderer = _FakeRenderer.instances[0]
    assert (
        "warning",
        "ChatGPT stream ended without a final answer; returned control to gptty.",
    ) in renderer.events
    assert ("info", "Cleared 1 queued prompt after incomplete turn.") in renderer.events
    assert notifications == []


def test_goal_incomplete_turn_interrupts_without_auto_continue(
    tmp_path, monkeypatch
) -> None:
    class IncompleteGoalClient:
        instances: list["IncompleteGoalClient"] = []

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[str] = []
            self.__class__.instances.append(self)

        def send(self, prompt: str, **options):
            self.calls.append(prompt)
            return SimpleNamespace(
                text="",
                title="Incomplete goal",
                conversation=SimpleNamespace(
                    conversation_id="conv-goal-incomplete",
                    finish_reason="incomplete",
                ),
            )

        def send_to_conversation(self, ref: str, prompt: str, **options):
            raise AssertionError("incomplete Goal must not auto-continue")

    IncompleteGoalClient.instances.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            '/goal "Finish the task"',
            (
                lambda: bool(_FakeRenderer.instances)
                and any(
                    event[0] == "warning" and "Goal · interrupted" in str(event[1])
                    for event in _FakeRenderer.instances[0].events
                ),
                "/exit",
            ),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=IncompleteGoalClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    assert len(IncompleteGoalClient.instances[0].calls) == 1
    state = load_chat_state(tmp_path / "state.json")
    assert state.goal is not None
    assert state.goal.status == "interrupted"
    assert state.goal.reason == "ChatGPT stream ended without a final answer"


def test_resume_loading_queues_text_without_concurrent_cwa_request(
    tmp_path, monkeypatch
) -> None:
    class SlowResumeClient:
        instances: list["SlowResumeClient"] = []
        snapshot_started = threading.Event()
        release_snapshot = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.snapshot_active = False
            self.__class__.instances.append(self)

        def conversation_snapshot(self, ref: str):
            self.calls.append(("snapshot", ref))
            self.snapshot_active = True
            self.__class__.snapshot_started.set()
            assert self.__class__.release_snapshot.wait(timeout=2)
            self.snapshot_active = False
            return {
                "status": SimpleNamespace(status="completed"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "old question"},
                    {"message_id": "a1", "role": "assistant", "text": "old answer"},
                ],
            }

        def send_to_conversation(self, ref: str, prompt: str, **options):
            assert not self.snapshot_active, "send must not overlap the resume snapshot"
            self.calls.append(("send_to_conversation", ref))
            return SimpleNamespace(
                text="queued reply",
                conversation_id=ref,
                title="Resumed chat",
            )

    SlowResumeClient.instances.clear()
    SlowResumeClient.snapshot_started.clear()
    SlowResumeClient.release_snapshot.clear()
    _FakeRenderer.instances.clear()

    def queued_send_finished() -> bool:
        if (
            _FakeRenderer.instances
            and ("info", "Queued · 1") in _FakeRenderer.instances[0].events
        ):
            SlowResumeClient.release_snapshot.set()
        if not SlowResumeClient.instances:
            return False
        return any(
            call[0] == "send_to_conversation"
            for call in SlowResumeClient.instances[0].calls
        )

    _FakeSession.script = iter(
        [
            "/resume conv-resume",
            (lambda: SlowResumeClient.snapshot_started.is_set(), "queued after resume"),
            (queued_send_finished, "/exit"),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=SlowResumeClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = SlowResumeClient.instances[0]
    assert client.calls == [
        ("snapshot", "conv-resume"),
        ("send_to_conversation", "conv-resume"),
    ]
    assert (
        load_chat_state(tmp_path / "state.json").current_conversation == "conv-resume"
    )


def test_unfinished_resume_follows_live_events_without_blocking_prompt(
    tmp_path, monkeypatch
) -> None:
    class LiveFollowClient:
        instances: list["LiveFollowClient"] = []

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.follow_count = 0
            self.__class__.instances.append(self)

        def conversation_follow_snapshot(
            self,
            ref: str,
            *,
            emitted_message_ids=(),
            limit=128,
            verify_terminal_status=False,
            terminal_probe_timeout=3.0,
        ):
            self.follow_count += 1
            self.calls.append(("follow", ref))
            if self.follow_count == 1:
                assert limit is None
                assert verify_terminal_status is True
                assert terminal_probe_timeout == 3.0
                return {
                    "status": SimpleNamespace(status="tool_running"),
                    "messages": [
                        {"message_id": "u1", "role": "user", "text": "question"},
                        {
                            "message_id": "a0",
                            "role": "assistant",
                            "text": "first thought",
                        },
                    ],
                    "events": [],
                    "emitted_message_ids": ["old-event"],
                }
            assert verify_terminal_status is False
            if self.follow_count == 2:
                assert "old-event" in emitted_message_ids
                return {
                    "status": SimpleNamespace(status="tool_running"),
                    "messages": [
                        {"message_id": "u1", "role": "user", "text": "question"},
                        {
                            "message_id": "a0",
                            "role": "assistant",
                            "text": "first thought",
                        },
                    ],
                    "events": [
                        {
                            "type": "canonical_intermediate_message",
                            "message_id": "reasoning-2",
                            "message_kind": "reasoning",
                            "text": "second live thought",
                        }
                    ],
                    "emitted_message_ids": ["old-event", "reasoning-2"],
                }
            return {
                "status": SimpleNamespace(status="completed"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "question"},
                    {"message_id": "a0", "role": "assistant", "text": "first thought"},
                    {"message_id": "a1", "role": "assistant", "text": "final answer"},
                ],
                "events": [],
                "emitted_message_ids": ["old-event", "reasoning-2"],
            }

        def send_to_conversation(self, ref: str, prompt: str, **options):
            self.calls.append(("send_to_conversation", ref))
            return SimpleNamespace(
                text="queued reply",
                conversation_id=ref,
                title="Live follow chat",
            )

    LiveFollowClient.instances.clear()
    _FakeRenderer.instances.clear()

    def saw_live_reasoning() -> bool:
        return bool(_FakeRenderer.instances) and any(
            event[0] == "live_event"
            and isinstance(event[1], dict)
            and event[1].get("message_id") == "reasoning-2"
            for event in _FakeRenderer.instances[0].events
        )

    def queued_turn_sent() -> bool:
        return bool(LiveFollowClient.instances) and any(
            call[0] == "send_to_conversation"
            for call in LiveFollowClient.instances[0].calls
        )

    _FakeSession.script = iter(
        [
            "/resume conv-live",
            (saw_live_reasoning, "queued while following"),
            (queued_turn_sent, "/exit"),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)
    monkeypatch.setattr("gptty.commands.chat.FOLLOW_MIN_INTERVAL_SECONDS", 0.001)

    code = run_chat(
        _args(tmp_path),
        client_factory=LiveFollowClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = LiveFollowClient.instances[0]
    assert client.follow_count >= 3
    assert ("send_to_conversation", "conv-live") in client.calls
    renderer = _FakeRenderer.instances[0]
    assert saw_live_reasoning()
    assert any(
        event[0] == "messages"
        and any(getattr(message, "text", "") == "final answer" for message in event[1])
        for event in renderer.events
    )


def test_resume_seed_renders_only_current_turn_intermediate_events() -> None:
    _FakeRenderer.instances.clear()
    renderer = _FakeRenderer(StringIO(), SimpleNamespace())
    snapshot = {
        "status": SimpleNamespace(status="tool_running"),
        "messages": [],
        "events": [
            {
                "type": "canonical_intermediate_message",
                "message_id": "old-reasoning",
                "message_kind": "reasoning",
                "turn_exchange_id": "turn-old",
                "text": "historical reasoning",
            },
            {
                "type": "canonical_intermediate_message",
                "message_id": "current-tool",
                "message_kind": "tool_call",
                "turn_exchange_id": "turn-current",
                "tool_name": "web.run",
                "text": "current tool",
            },
            {
                "type": "canonical_intermediate_message",
                "message_id": "current-reasoning",
                "message_kind": "reasoning",
                "turn_exchange_id": "turn-current",
                "text": "current reasoning",
            },
        ],
        "emitted_message_ids": [
            "old-reasoning",
            "current-tool",
            "current-reasoning",
        ],
        "current_turn_event_ids": ["current-tool", "current-reasoning"],
        "stream_topic_id": "conversation-turn-turn-current",
        "turn_exchange_id": "turn-current",
        "stream_answer_message_id": None,
        "stream_answer_text": "",
    }

    follow = chat_module._seed_enhanced_follow(
        SimpleNamespace(conversation_ref="conv-current"),
        snapshot,
        renderer=renderer,
    )

    assert follow is not None
    assert follow.emitted_message_ids == {
        "old-reasoning",
        "current-tool",
        "current-reasoning",
    }
    rendered_ids = [
        event[1].get("message_id")
        for event in renderer.events
        if event[0] == "live_event" and isinstance(event[1], dict)
    ]
    assert rendered_ids == ["current-tool", "current-reasoning"]


def test_resume_seed_does_not_replay_current_event_already_in_history() -> None:
    _FakeRenderer.instances.clear()
    renderer = _FakeRenderer(StringIO(), SimpleNamespace())
    snapshot = {
        "status": SimpleNamespace(status="tool_running"),
        "messages": [
            {
                "message_id": "current-progress",
                "role": "assistant",
                "text": "progress already shown by resume history",
            }
        ],
        "events": [
            {
                "type": "canonical_intermediate_message",
                "message_id": "current-progress",
                "message_kind": "commentary",
                "turn_exchange_id": "turn-current",
                "text": "progress already shown by resume history",
            }
        ],
        "emitted_message_ids": ["current-progress"],
        "current_turn_event_ids": ["current-progress"],
        "stream_topic_id": "conversation-turn-turn-current",
        "turn_exchange_id": "turn-current",
        "stream_answer_message_id": None,
        "stream_answer_text": "",
    }

    follow = chat_module._seed_enhanced_follow(
        SimpleNamespace(conversation_ref="conv-current"),
        snapshot,
        renderer=renderer,
    )

    assert follow is not None
    assert follow.emitted_message_ids == {"current-progress"}
    assert not any(event[0] == "live_event" for event in renderer.events)


def test_unfinished_resume_prefers_live_stream_without_polling(
    tmp_path, monkeypatch
) -> None:
    class StreamFollowClient:
        instances: list["StreamFollowClient"] = []
        release_stream = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.snapshot_count = 0
            self.stream_count = 0
            self.__class__.instances.append(self)

        def conversation_follow_snapshot(
            self,
            ref: str,
            *,
            emitted_message_ids=(),
            limit=128,
            verify_terminal_status=False,
            terminal_probe_timeout=3.0,
        ):
            self.snapshot_count += 1
            self.calls.append(("snapshot", ref))
            assert limit is None
            return {
                "status": SimpleNamespace(status="tool_running"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "question"},
                    {"message_id": "a1", "role": "assistant", "text": "partial"},
                ],
                "events": [],
                "emitted_message_ids": ["old-event"],
                "stream_topic_id": "conversation-turn-turn-1",
                "turn_exchange_id": "turn-1",
                "stream_answer_message_id": "a1",
                "stream_answer_text": "partial",
            }

        def conversation_follow_stream(
            self,
            ref: str,
            *,
            topic_id,
            emitted_message_ids,
            answer_message_id,
            answer_text,
            timeout,
            limit,
            on_event,
            should_stop,
        ):
            self.stream_count += 1
            self.calls.append(("stream", ref))
            assert topic_id == "conversation-turn-turn-1"
            assert "old-event" in emitted_message_ids
            assert answer_message_id == "a1"
            assert answer_text == "partial"
            assert limit == chat_module.FOLLOW_MESSAGE_LIMIT
            on_event(
                {
                    "type": "canonical_intermediate_message",
                    "message_id": "reasoning-live",
                    "message_kind": "reasoning",
                    "text": "live websocket thought",
                }
            )
            assert self.release_stream.wait(timeout=2.0)
            assert not should_stop()
            on_event(
                {
                    "type": "assistant_text_delta",
                    "message_id": "a1",
                    "sequence": 1,
                    "delta": " final",
                }
            )
            return {
                "stream_completed": True,
                "status": SimpleNamespace(status="completed"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "question"},
                    {"message_id": "a1", "role": "assistant", "text": "partial final"},
                ],
                "events": [],
                "emitted_message_ids": ["old-event", "reasoning-live"],
            }

        def send_to_conversation(self, ref: str, prompt: str, **options):
            self.calls.append(("send_to_conversation", ref))
            return SimpleNamespace(
                text="queued reply",
                conversation_id=ref,
                title="Stream follow chat",
            )

    StreamFollowClient.instances.clear()
    StreamFollowClient.release_stream.clear()
    _FakeRenderer.instances.clear()

    def saw_stream_reasoning() -> bool:
        return bool(_FakeRenderer.instances) and any(
            event[0] == "live_event"
            and isinstance(event[1], dict)
            and event[1].get("message_id") == "reasoning-live"
            for event in _FakeRenderer.instances[0].events
        )

    def queued_prompt_visible() -> bool:
        visible = (
            bool(_FakeRenderer.instances)
            and (
                "info",
                "Queued · 1",
            )
            in _FakeRenderer.instances[0].events
        )
        if visible:
            StreamFollowClient.release_stream.set()
        return visible

    def queued_turn_sent() -> bool:
        return bool(StreamFollowClient.instances) and any(
            call[0] == "send_to_conversation"
            for call in StreamFollowClient.instances[0].calls
        )

    _FakeSession.script = iter(
        [
            "/resume conv-stream",
            (saw_stream_reasoning, "queued while streaming"),
            (queued_prompt_visible, ""),
            (queued_turn_sent, "/exit"),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=StreamFollowClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = StreamFollowClient.instances[0]
    assert client.snapshot_count == 1
    assert client.stream_count == 1
    assert [call[0] for call in client.calls].count("snapshot") == 1
    assert ("send_to_conversation", "conv-stream") in client.calls
    renderer = _FakeRenderer.instances[0]
    assert saw_stream_reasoning()
    assert not any(
        event[0] == "live_event"
        and isinstance(event[1], dict)
        and event[1].get("type") == "assistant_text_delta"
        for event in renderer.events
    )
    assert ("answer", "partial final") in renderer.events


def test_terminal_follow_releases_multiple_queued_prompts_in_order(
    tmp_path, monkeypatch
) -> None:
    class TerminalFollowClient:
        instances: list["TerminalFollowClient"] = []
        release_stream = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.__class__.instances.append(self)

        def conversation_follow_snapshot(
            self,
            ref: str,
            *,
            emitted_message_ids=(),
            limit=128,
            verify_terminal_status=False,
            terminal_probe_timeout=3.0,
        ):
            self.calls.append(("snapshot", ref))
            return {
                "status": SimpleNamespace(status="tool_running"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "question"},
                ],
                "events": [],
                "emitted_message_ids": ["old-event"],
                "stream_topic_id": "conversation-turn-orphaned",
                "turn_exchange_id": "turn-orphaned",
                "stream_answer_message_id": None,
                "stream_answer_text": "",
            }

        def conversation_follow_stream(
            self,
            ref: str,
            *,
            topic_id,
            emitted_message_ids,
            answer_message_id,
            answer_text,
            timeout,
            limit,
            on_event,
            should_stop,
        ):
            self.calls.append(("stream", ref))
            assert self.release_stream.wait(timeout=2.0)
            assert not should_stop()
            on_event(
                {
                    "type": "stream_handoff_terminal_status",
                    "stream_status": "COMPLETE",
                    "last_offset": "1000-0",
                }
            )
            return {
                "stream_completed": True,
                "status": SimpleNamespace(status="completed"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "question"},
                ],
                "events": [],
                "emitted_message_ids": list(emitted_message_ids),
            }

        def send_to_conversation(self, ref: str, prompt: str, **options):
            self.calls.append(("send_to_conversation", prompt))
            return SimpleNamespace(
                text=f"reply:{prompt}",
                conversation_id=ref,
                title="Terminal follow chat",
            )

    class TrackingSession(_FakeSession):
        instances: list["TrackingSession"] = []

        def __init__(self, **kwargs) -> None:
            self.active_updates: list[tuple[object, object]] = []
            self.__class__.instances.append(self)

        def set_active_turn(self, controls, *, working_status=None) -> None:
            self.active_updates.append((controls, working_status))

    TerminalFollowClient.instances.clear()
    TerminalFollowClient.release_stream.clear()
    TrackingSession.instances.clear()
    _FakeRenderer.instances.clear()

    def follow_footer_active() -> bool:
        if not TrackingSession.instances:
            return False
        return any(
            controls is None and callable(status)
            for controls, status in TrackingSession.instances[0].active_updates
        )

    def queued_two() -> bool:
        if not _FakeRenderer.instances:
            return False
        visible = ("info", "Queued · 2") in _FakeRenderer.instances[0].events
        if visible:
            TerminalFollowClient.release_stream.set()
        return visible

    def two_sends_done() -> bool:
        if not TerminalFollowClient.instances:
            return False
        sends = [
            call
            for call in TerminalFollowClient.instances[0].calls
            if call[0] == "send_to_conversation"
        ]
        return len(sends) == 2

    TrackingSession.script = iter(
        [
            "/resume conv-terminal",
            (follow_footer_active, "queued one"),
            (
                lambda: bool(_FakeRenderer.instances)
                and ("info", "Queued · 1") in _FakeRenderer.instances[0].events,
                "queued two",
            ),
            (queued_two, ""),
            (two_sends_done, "/exit"),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", TrackingSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=TerminalFollowClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    sends = [
        call
        for call in TerminalFollowClient.instances[0].calls
        if call[0] == "send_to_conversation"
    ]
    assert sends == [
        ("send_to_conversation", "queued one"),
        ("send_to_conversation", "queued two"),
    ]
    session = TrackingSession.instances[0]
    assert any(
        controls is None and callable(status)
        for controls, status in session.active_updates
    )
    assert any(
        controls is None and status is None
        for controls, status in session.active_updates
    )


def test_stop_command_during_follow_stops_then_sends_queued_prompt(
    tmp_path, monkeypatch
) -> None:
    class FollowStopClient:
        instances: list["FollowStopClient"] = []

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.stopped = False
            self.__class__.instances.append(self)

        def conversation_follow_snapshot(
            self,
            ref: str,
            *,
            emitted_message_ids=(),
            limit=128,
            verify_terminal_status=False,
            terminal_probe_timeout=3.0,
        ):
            self.calls.append(("snapshot", ref))
            return {
                "status": SimpleNamespace(status="tool_running"),
                "messages": [],
                "events": [],
                "emitted_message_ids": [],
                "stream_topic_id": "conversation-turn-stop",
                "stream_answer_message_id": None,
                "stream_answer_text": "",
            }

        def conversation_follow_stream(
            self,
            ref: str,
            *,
            topic_id,
            emitted_message_ids,
            answer_message_id,
            answer_text,
            timeout,
            limit,
            on_event,
            should_stop,
        ):
            self.calls.append(("stream", ref))
            if not self.stopped:
                deadline = time.monotonic() + 2.0
                while not should_stop() and time.monotonic() < deadline:
                    time.sleep(0.002)
                return {
                    "stream_completed": False,
                    "stream_cancelled": should_stop(),
                    "stream_topic_id": topic_id,
                    "emitted_message_ids": list(emitted_message_ids),
                }
            return {
                "stream_completed": True,
                "status": SimpleNamespace(status="completed"),
                "messages": [],
                "events": [],
                "emitted_message_ids": list(emitted_message_ids),
            }

        def stop_generation(self, ref: str, timeout: float = 2.0):
            self.calls.append(("stop_generation", ref))
            self.stopped = True
            return {"stopped": True, "conversationId": ref}

        def send_to_conversation(self, ref: str, prompt: str, **options):
            self.calls.append(("send_to_conversation", prompt))
            return SimpleNamespace(
                text="queued sent after stop",
                conversation_id=ref,
                title="Follow stop chat",
            )

    FollowStopClient.instances.clear()
    _FakeRenderer.instances.clear()

    def stream_started() -> bool:
        return bool(FollowStopClient.instances) and any(
            call[0] == "stream" for call in FollowStopClient.instances[0].calls
        )

    def queued_visible() -> bool:
        return bool(_FakeRenderer.instances) and (
            "info",
            "Queued · 1",
        ) in _FakeRenderer.instances[0].events

    def queued_sent() -> bool:
        return bool(FollowStopClient.instances) and (
            "send_to_conversation",
            "queued after stop",
        ) in FollowStopClient.instances[0].calls

    _FakeSession.script = iter(
        [
            "/resume conv-follow-stop",
            (stream_started, "queued after stop"),
            (queued_visible, "/stop"),
            (queued_sent, "/exit"),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=FollowStopClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = FollowStopClient.instances[0]
    assert ("stop_generation", "conv-follow-stop") in client.calls
    assert client.calls.count(("send_to_conversation", "queued after stop")) == 1
    assert client.calls.index(("stop_generation", "conv-follow-stop")) < client.calls.index(
        ("send_to_conversation", "queued after stop")
    )


def test_nonterminal_follow_keeps_queued_prompt_unsent(
    tmp_path, monkeypatch
) -> None:
    class NonterminalFollowClient:
        instances: list["NonterminalFollowClient"] = []
        release_stream = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.__class__.instances.append(self)

        def conversation_follow_snapshot(
            self,
            ref: str,
            *,
            emitted_message_ids=(),
            limit=128,
            verify_terminal_status=False,
            terminal_probe_timeout=3.0,
        ):
            self.calls.append(("snapshot", ref))
            return {
                "status": SimpleNamespace(status="tool_running"),
                "messages": [],
                "events": [],
                "emitted_message_ids": [],
                "stream_topic_id": "conversation-turn-still-running",
                "stream_answer_message_id": None,
                "stream_answer_text": "",
            }

        def conversation_follow_stream(
            self,
            ref: str,
            *,
            topic_id,
            emitted_message_ids,
            answer_message_id,
            answer_text,
            timeout,
            limit,
            on_event,
            should_stop,
        ):
            self.calls.append(("stream", ref))
            while not self.release_stream.is_set() and not should_stop():
                time.sleep(0.002)
            return {
                "stream_completed": False,
                "stream_cancelled": should_stop(),
                "stream_topic_id": topic_id,
                "emitted_message_ids": list(emitted_message_ids),
            }

        def send_to_conversation(self, ref: str, prompt: str, **options):
            self.calls.append(("send_to_conversation", prompt))
            return SimpleNamespace(
                text="must not send",
                conversation_id=ref,
                title="Nonterminal follow chat",
            )

    NonterminalFollowClient.instances.clear()
    NonterminalFollowClient.release_stream.clear()
    _FakeRenderer.instances.clear()

    def queued_visible_without_send() -> bool:
        if not _FakeRenderer.instances or not NonterminalFollowClient.instances:
            return False
        if ("info", "Queued · 1") not in _FakeRenderer.instances[0].events:
            return False
        assert not any(
            call[0] == "send_to_conversation"
            for call in NonterminalFollowClient.instances[0].calls
        )
        return True

    _FakeSession.script = iter(
        [
            "/resume conv-nonterminal",
            (
                lambda: bool(NonterminalFollowClient.instances)
                and any(
                    call[0] == "stream"
                    for call in NonterminalFollowClient.instances[0].calls
                ),
                "queued but hold",
            ),
            (queued_visible_without_send, "/exit"),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=NonterminalFollowClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )
    NonterminalFollowClient.release_stream.set()

    assert code == 0
    assert not any(
        call[0] == "send_to_conversation"
        for call in NonterminalFollowClient.instances[0].calls
    )


def test_unfinished_resume_does_not_poll_while_live_stream_is_silent(
    tmp_path, monkeypatch
) -> None:
    class SilentStreamClient:
        instances: list["SilentStreamClient"] = []
        release_stream = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.snapshot_count = 0
            self.stream_count = 0
            self.__class__.instances.append(self)

        def conversation_follow_snapshot(
            self,
            ref: str,
            *,
            emitted_message_ids=(),
            limit=128,
            verify_terminal_status=False,
            terminal_probe_timeout=3.0,
        ):
            self.snapshot_count += 1
            self.calls.append(("snapshot", ref))
            assert self.snapshot_count == 1
            assert limit is None
            return {
                "status": SimpleNamespace(status="tool_running"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "question"},
                ],
                "events": [],
                "emitted_message_ids": ["old-event"],
                "stream_topic_id": "conversation-turn-turn-silent",
                "turn_exchange_id": "turn-silent",
                "stream_answer_message_id": None,
                "stream_answer_text": "",
            }

        def conversation_follow_stream(
            self,
            ref: str,
            *,
            topic_id,
            emitted_message_ids,
            answer_message_id,
            answer_text,
            timeout,
            limit,
            on_event,
            should_stop,
        ):
            self.stream_count += 1
            self.calls.append(("stream", ref))
            assert topic_id == "conversation-turn-turn-silent"
            assert "old-event" in emitted_message_ids
            assert answer_message_id is None
            assert answer_text == ""
            assert limit == chat_module.FOLLOW_MESSAGE_LIMIT
            deadline = time.monotonic() + 2.0
            while (
                not self.release_stream.is_set()
                and not should_stop()
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            return {
                "stream_completed": False,
                "stream_cancelled": should_stop(),
                "stream_topic_id": topic_id,
                "emitted_message_ids": list(emitted_message_ids),
            }

    SilentStreamClient.instances.clear()
    SilentStreamClient.release_stream.clear()
    _FakeRenderer.instances.clear()
    silence_started = time.monotonic()

    def stream_remained_poll_free() -> bool:
        if not SilentStreamClient.instances:
            return False
        client = SilentStreamClient.instances[0]
        if client.stream_count != 1 or time.monotonic() - silence_started < 0.15:
            return False
        assert client.snapshot_count == 1
        SilentStreamClient.release_stream.set()
        return True

    _FakeSession.script = iter(
        [
            "/resume conv-silent",
            (stream_remained_poll_free, "/exit"),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=SilentStreamClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = SilentStreamClient.instances[0]
    assert client.stream_count == 1
    assert client.snapshot_count == 1
    assert ("stream", "conv-silent") in client.calls


def test_resume_follow_replaces_corrupt_stream_final_with_canonical_snapshot(
    monkeypatch,
) -> None:
    notifications: list[dict[str, object]] = []
    monkeypatch.setattr(
        chat_module,
        "notify_response_complete",
        lambda **kwargs: notifications.append(kwargs),
    )
    renderer = _FakeRenderer(StringIO(), SimpleNamespace())
    follow = chat_module._EnhancedFollow(
        conversation_ref="conv-live",
        emitted_message_ids=set(),
        seen_messages={"assistant-final": "broken pieces"},
        deadline=10_000.0,
        stream_answer_message_id="assistant-final",
        stream_answer_text="broken pieces",
    )
    snapshot = {
        "status": SimpleNamespace(status="completed"),
        "messages": [
            {
                "message_id": "assistant-final",
                "role": "assistant",
                "text": "complete canonical answer",
            }
        ],
        "events": [],
        "emitted_message_ids": [],
        "stream_terminal_reconciled": True,
    }

    assert not chat_module._apply_enhanced_follow_snapshot(
        follow,
        snapshot,
        renderer=renderer,
    )

    assert ("answer", "complete canonical answer") in renderer.events
    assert not any(event[0] == "messages" for event in renderer.events)
    assert follow.stream_answer_text == "complete canonical answer"
    assert notifications[-1]["final_response"] == "complete canonical answer"


def test_attached_follow_does_not_replay_previously_rendered_intermediate_as_message() -> None:
    renderer = _FakeRenderer(StringIO(), SimpleNamespace())
    follow = chat_module._EnhancedFollow(
        conversation_ref="conv-live",
        emitted_message_ids={"progress-b"},
        seen_messages={},
        deadline=10_000.0,
        stream_answer_message_id="assistant-final",
        stream_answer_text="complete final answer",
        defer_stream_answer_until_terminal=True,
    )

    snapshot = {
        "status": SimpleNamespace(status="completed"),
        "messages": [
            {
                "message_id": "progress-b",
                "role": "assistant",
                "text": "progress already rendered live",
            },
            {
                "message_id": "assistant-final",
                "role": "assistant",
                "text": "complete final answer",
            },
        ],
        "events": [],
        "emitted_message_ids": ["progress-b"],
        "stream_terminal_reconciled": True,
    }

    assert not chat_module._apply_enhanced_follow_snapshot(
        follow,
        snapshot,
        renderer=renderer,
    )

    assert not any(event[0] == "messages" for event in renderer.events)
    assert ("answer", "complete final answer") in renderer.events


def test_attached_follow_renders_late_intermediate_before_deferred_final() -> None:
    renderer = _FakeRenderer(StringIO(), SimpleNamespace())
    follow = chat_module._EnhancedFollow(
        conversation_ref="conv-live",
        emitted_message_ids=set(),
        seen_messages={},
        deadline=10_000.0,
        stream_answer_message_id="assistant-final",
        defer_stream_answer_until_terminal=True,
    )

    chat_module._apply_enhanced_follow_stream_event(
        follow,
        {
            "type": "assistant_text_delta",
            "message_id": "assistant-final",
            "sequence": 1,
            "delta": "complete final answer",
        },
        renderer=renderer,
    )

    assert follow.stream_answer_text == "complete final answer"
    assert not any(
        event[0] == "live_event"
        and isinstance(event[1], dict)
        and event[1].get("type") == "assistant_text_delta"
        for event in renderer.events
    )

    snapshot = {
        "status": SimpleNamespace(status="completed"),
        "messages": [
            {
                "message_id": "assistant-final",
                "role": "assistant",
                "text": "complete final answer",
            }
        ],
        "events": [
            {
                "type": "canonical_intermediate_message",
                "message_id": "late-commentary",
                "message_kind": "commentary",
                "text": "older progress that the live topic missed",
            }
        ],
        "emitted_message_ids": ["late-commentary"],
        "stream_terminal_reconciled": True,
    }

    assert not chat_module._apply_enhanced_follow_snapshot(
        follow,
        snapshot,
        renderer=renderer,
    )

    rendered = [
        event for event in renderer.events if event[0] in {"live_event", "answer"}
    ]
    assert rendered[-2][0] == "live_event"
    assert rendered[-2][1]["message_id"] == "late-commentary"
    assert rendered[-1] == ("answer", "complete final answer")


def test_attached_follow_marks_finality_stall_after_final_text() -> None:
    renderer = _FakeRenderer(StringIO(), SimpleNamespace())
    health = chat_module._TurnHealth(last_server_progress_at=time.monotonic())
    follow = chat_module._EnhancedFollow(
        conversation_ref="conv-live",
        emitted_message_ids=set(),
        seen_messages={},
        deadline=10_000.0,
        health=health,
        stream_answer_message_id="assistant-final",
        defer_stream_answer_until_terminal=True,
    )

    chat_module._apply_enhanced_follow_stream_event(
        follow,
        {
            "type": "assistant_text_delta",
            "message_id": "assistant-final",
            "sequence": 1,
            "delta": "complete final answer",
        },
        renderer=renderer,
    )
    chat_module._apply_enhanced_follow_stream_event(
        follow,
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 305.0,
        },
        renderer=renderer,
    )

    health_event = next(
        event[1]
        for event in renderer.events
        if event[0] == "live_event"
        and isinstance(event[1], dict)
        and event[1].get("type") == "stream_handoff_server_stalled"
    )
    assert health.answer_progress_seen is True
    assert health.state == "stalled"
    assert health_event["final_text_seen"] is True


def test_resume_follow_adapts_poll_budget_and_backs_off_on_rate_limit() -> None:
    renderer = _FakeRenderer(StringIO(), SimpleNamespace())
    follow = chat_module._EnhancedFollow(
        conversation_ref="conv-live",
        emitted_message_ids=set(),
        seen_messages={
            "u1": "question",
            "a0": "first thought",
        },
        deadline=10_000.0,
        next_interval=chat_module.FOLLOW_MIN_INTERVAL_SECONDS,
    )

    idle_snapshot = {
        "status": SimpleNamespace(status="tool_running"),
        "messages": [
            {"message_id": "u1", "role": "user", "text": "question"},
            {"message_id": "a0", "role": "assistant", "text": "first thought"},
        ],
        "events": [],
        "emitted_message_ids": [],
    }
    assert chat_module._apply_enhanced_follow_snapshot(
        follow,
        idle_snapshot,
        renderer=renderer,
    )
    assert follow.next_interval == 30.0

    assert chat_module._apply_enhanced_follow_snapshot(
        follow,
        idle_snapshot,
        renderer=renderer,
    )
    assert follow.next_interval == 60.0

    assert chat_module._apply_enhanced_follow_snapshot(
        follow,
        idle_snapshot,
        renderer=renderer,
    )
    assert follow.next_interval == 60.0

    active_snapshot = {
        **idle_snapshot,
        "events": [
            {
                "type": "canonical_intermediate_message",
                "message_id": "reasoning-2",
                "message_kind": "reasoning",
                "text": "new reasoning",
            }
        ],
        "emitted_message_ids": ["reasoning-2"],
    }
    assert chat_module._apply_enhanced_follow_snapshot(
        follow,
        active_snapshot,
        renderer=renderer,
    )
    assert follow.next_interval == 15.0

    chat_module._backoff_enhanced_follow_after_error(
        follow,
        SimpleNamespace(status_code=429),
        renderer=renderer,
    )
    assert follow.next_interval == 120.0

    chat_module._backoff_enhanced_follow_after_error(
        follow,
        SimpleNamespace(status_code=429),
        renderer=renderer,
    )
    assert follow.next_interval == 240.0

    chat_module._backoff_enhanced_follow_after_error(
        follow,
        SimpleNamespace(status_code=429),
        renderer=renderer,
    )
    assert follow.next_interval == 300.0


def test_exit_during_resume_loading_does_not_wait_for_snapshot(
    tmp_path, monkeypatch
) -> None:
    class BlockingResumeClient:
        instances: list["BlockingResumeClient"] = []
        snapshot_started = threading.Event()
        release_snapshot = threading.Event()

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.__class__.instances.append(self)

        def conversation_snapshot(self, ref: str):
            self.calls.append(("snapshot", ref))
            self.__class__.snapshot_started.set()
            self.__class__.release_snapshot.wait(timeout=5)
            return {"status": SimpleNamespace(status="completed"), "messages": []}

    BlockingResumeClient.instances.clear()
    BlockingResumeClient.snapshot_started.clear()
    BlockingResumeClient.release_snapshot.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            "/resume conv-slow",
            (lambda: BlockingResumeClient.snapshot_started.is_set(), "/exit"),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    try:
        code = run_chat(
            _args(tmp_path),
            client_factory=BlockingResumeClient,
            input_stream=StringIO(),
            stdout=StringIO(),
            stderr=StringIO(),
        )
    finally:
        BlockingResumeClient.release_snapshot.set()

    assert code == 0
    assert BlockingResumeClient.snapshot_started.is_set()
    assert load_chat_state(tmp_path / "state.json").current_conversation is None


def test_unfinished_resume_returns_to_prompt_without_polling(
    tmp_path, monkeypatch
) -> None:
    class UnfinishedResumeClient:
        instances: list["UnfinishedResumeClient"] = []

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str]] = []
            self.__class__.instances.append(self)

        def conversation_snapshot(self, ref: str):
            self.calls.append(("snapshot", ref))
            return {
                "status": SimpleNamespace(status="tool_running"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "old question"},
                    {"message_id": "t1", "role": "tool", "text": "old tool output"},
                ],
            }

    UnfinishedResumeClient.instances.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            "/resume conv-stale",
            (
                lambda: bool(_FakeRenderer.instances)
                and any(
                    event[0] == "warning"
                    and "unfinished turn (status=tool_running)" in str(event[1])
                    for event in _FakeRenderer.instances[0].events
                ),
                "/exit",
            ),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=UnfinishedResumeClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = UnfinishedResumeClient.instances[0]
    assert client.calls == [("snapshot", "conv-stale")]
    assert load_chat_state(tmp_path / "state.json").current_conversation == "conv-stale"


def test_resume_terminal_backend_override_does_not_enter_follow(
    tmp_path, monkeypatch
) -> None:
    class TerminalOverrideResumeClient:
        instances: list["TerminalOverrideResumeClient"] = []

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str, bool, float]] = []
            self.__class__.instances.append(self)

        def conversation_follow_snapshot(
            self,
            ref: str,
            *,
            emitted_message_ids=(),
            limit=128,
            verify_terminal_status=False,
            terminal_probe_timeout=3.0,
        ):
            self.calls.append(
                (
                    "follow",
                    ref,
                    bool(verify_terminal_status),
                    float(terminal_probe_timeout),
                )
            )
            assert emitted_message_ids == ()
            assert limit is None
            return {
                "status": SimpleNamespace(status="completed"),
                "messages": [
                    {"message_id": "u1", "role": "user", "text": "old question"},
                    {"message_id": "t1", "role": "tool", "text": "old tool output"},
                ],
                "events": [],
                "emitted_message_ids": [],
                "backend_stream_status": "COMPLETE",
                "backend_terminal_status_proven": True,
                "canonical_status_overridden": True,
                "canonical_status_before_override": "tool_running",
                "canonical_terminal_text_missing": True,
            }

    TerminalOverrideResumeClient.instances.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            "/resume conv-stale",
            (
                lambda: bool(_FakeRenderer.instances)
                and any(
                    event[0] == "warning"
                    and "Opened chat idle" in str(event[1])
                    for event in _FakeRenderer.instances[0].events
                ),
                "/exit",
            ),
        ]
    )
    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    code = run_chat(
        _args(tmp_path),
        client_factory=TerminalOverrideResumeClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = TerminalOverrideResumeClient.instances[0]
    assert client.calls == [("follow", "conv-stale", True, 3.0)]
    renderer = _FakeRenderer.instances[0]
    infos = [str(event[1]) for event in renderer.events if event[0] == "info"]
    assert "Following active response via live stream…" not in infos
    assert "Following active response in background…" not in infos
    warnings = [str(event[1]) for event in renderer.events if event[0] == "warning"]
    assert not any("unfinished turn" in warning for warning in warnings)
    assert load_chat_state(tmp_path / "state.json").current_conversation == "conv-stale"


def test_working_status_surfaces_exact_codexpro_heartbeat(monkeypatch) -> None:
    class Tracker:
        def snapshot(self, _conversation_ref):
            return chat_module.CodexProActivitySnapshot(
                bound=True,
                last_event_age_seconds=7.0,
                last_tool="bash",
                inflight=True,
                inflight_tool="bash",
                last_heartbeat_age_seconds=7.0,
            )

    now = 900.0
    monkeypatch.setattr(chat_module.time, "monotonic", lambda: now)
    health = chat_module._TurnHealth(
        last_server_progress_at=600.0,
        codexpro_tracker=Tracker(),
        conversation_ref="conversation-1",
    )
    health.observe(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 300.0,
        }
    )

    status = chat_module._working_status(500.0, 0, health=health)
    assert "PROLONGED SILENCE" in status
    assert "CodexPro exact: bash running" in status
    assert "heartbeat 00:07 ago" in status
    assert "do not resend yet" in status


def test_working_status_hides_stale_codexpro_activity(monkeypatch) -> None:
    class Tracker:
        def snapshot(self, _conversation_ref):
            return chat_module.CodexProActivitySnapshot(
                bound=True,
                last_event_age_seconds=90 * 60.0,
                last_tool="bash",
                inflight=False,
            )

    now = 900.0
    monkeypatch.setattr(chat_module.time, "monotonic", lambda: now)
    health = chat_module._TurnHealth(
        last_server_progress_at=600.0,
        codexpro_tracker=Tracker(),
        conversation_ref="conversation-1",
    )
    health.observe(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 300.0,
        }
    )

    status = chat_module._working_status(500.0, 0, health=health)
    assert "PROLONGED SILENCE" in status
    assert "CodexPro" not in status


def test_working_status_labels_idle_reconnect_as_delivery_check(monkeypatch) -> None:
    now = 700.0
    monkeypatch.setattr(chat_module.time, "monotonic", lambda: now)
    health = chat_module._TurnHealth(last_server_progress_at=690.0)
    health.observe(
        {
            "type": "stream_handoff_ws_reconnecting",
            "reason": "topic_idle",
            "attempt": 3,
            "server_idle_seconds": 12.0,
        }
    )

    status = chat_module._working_status(650.0, 0, health=health)
    assert status == "checking delivery · idle lease · attempt 3"

    health.observe(
        {
            "type": "stream_handoff_ws_subscribed",
            "catchup_count": 0,
            "last_offset": "1000-0",
        }
    )
    assert health.state == "working"


def test_working_status_keeps_transport_error_as_reconnecting(monkeypatch) -> None:
    now = 700.0
    monkeypatch.setattr(chat_module.time, "monotonic", lambda: now)
    health = chat_module._TurnHealth(last_server_progress_at=690.0)
    health.observe(
        {
            "type": "stream_handoff_ws_reconnecting",
            "reason": "transport",
            "attempt": 2,
            "server_idle_seconds": 1.0,
        }
    )

    status = chat_module._working_status(650.0, 0, health=health)
    assert status == "reconnecting delivery · attempt 2"
