from __future__ import annotations

import asyncio
import time
from io import StringIO
from types import SimpleNamespace
from typing import ClassVar

from gptty.commands.chat import run_chat
from gptty.goal_store import GoalStore
from gptty.session_state import session_handle
from gptty.state import (
    ChatState,
    GoalState,
    save_chat_state,
)


def _load_session_state(state_path):
    return session_handle(
        state_path=state_path,
        profile=None,
        environ={},
    ).load()


def _load_session_goal(state_path):
    session = _load_session_state(state_path)
    assert session.goal_id is not None
    goal = GoalStore(state_path).load(session.goal_id)
    assert goal is not None
    return session, goal


class _GoalLoopClient:
    instances: ClassVar[list[_GoalLoopClient]] = []

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
            text=(
                "GPTTY_GOAL: COMPLETE\n"
                'GPTTY_CHECKPOINT: {"summary":"done","completed":["verified"],"decisions":[],"pending":[],"next":"none"}\n'
                "All agreed work is done and verified."
            ),
            conversation_id=ref,
            title="Goal loop test",
        )


class _FakeSession:
    script = iter(())
    # A scripted prompt waits at most this long; a regression then fails
    # instead of hanging the whole run.
    deadline_seconds = 10.0

    def __init__(self, **kwargs) -> None:
        pass

    def read_prompt(self, *, attachment_count: int = 0) -> str:
        return next(self.script)

    async def read_prompt_async(self, *, attachment_count: int = 0) -> str:
        try:
            item = next(self.script)
        except StopIteration:
            # The loop may keep a prompt read pending until it exits and cancels it.
            await asyncio.sleep(self.deadline_seconds)
            raise AssertionError(
                f"chat loop still waiting for input {self.deadline_seconds}s after the script ended"
            )
        if isinstance(item, tuple):
            predicate, value = item
            deadline = time.monotonic() + self.deadline_seconds
            while not predicate():
                if time.monotonic() > deadline:
                    raise AssertionError(
                        f"scripted prompt {value!r}: condition not reached in {self.deadline_seconds}s"
                    )
                await asyncio.sleep(0.001)
            return value
        return item

    def set_active_turn(self, controls, *, working_status=None) -> None:
        pass


class _FakeRenderer:
    instances: ClassVar[list[_FakeRenderer]] = []

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

    def turn_marker(self, label: str, status: str, message: str) -> None:
        self.events.append(
            (
                "turn_marker",
                {"label": label, "status": status, "message": message},
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


def test_goal_chat_loop_auto_continues_until_complete_without_intermediate_notification(
    tmp_path, monkeypatch
) -> None:
    _GoalLoopClient.instances.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            '/goal "Finish exactly this test goal"',
            (
                lambda: (
                    bool(_FakeRenderer.instances)
                    and ("info", "Goal · complete · 2 turns")
                    in _FakeRenderer.instances[0].events
                ),
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
    state, goal = _load_session_goal(tmp_path / "state.json")
    assert state.current_conversation == "conv-goal"
    assert goal.status == "complete"
    assert goal.turn_count == 2
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


def test_goal_post_submit_ambiguity_never_auto_resends(tmp_path, monkeypatch) -> None:
    from chatgpt_web_adapter.browser_owned_write_runtime import (
        WRITE_OUTCOME_UNKNOWN,
        BrowserOwnedWriteRuntimeError,
    )

    class AmbiguousWriteClient:
        instances: ClassVar[list[AmbiguousWriteClient]] = []

        def __init__(
            self, auth_file: str = "auth_data.json", timeout: int = 90
        ) -> None:
            self.calls: list[tuple[str, str, str | None]] = []
            self.__class__.instances.append(self)

        def get_messages(self, ref: str):
            return [
                {"role": "user", "text": "Do this exactly once."},
                {"role": "assistant", "text": "Understood."},
            ]

        def send_to_conversation(self, ref: str, prompt: str, **options):
            self.calls.append(("send_to_conversation", prompt, ref))
            raise BrowserOwnedWriteRuntimeError(
                "provider failed after delegation",
                failure_kind=WRITE_OUTCOME_UNKNOWN,
                automatic_retry_allowed=False,
                manual_retry_safe_after_repair=False,
                write_may_have_been_submitted=True,
                reconciliation_required=True,
                request_stage="browser_owned_write",
                status_code=429,
            )

        def send(self, prompt: str, **options):
            self.calls.append(("send", prompt, None))
            raise AssertionError("ambiguous Goal turn must not roll over or resend")

    save_chat_state(
        tmp_path / "state.json",
        ChatState(current_conversation="conv-ambiguous"),
    )
    AmbiguousWriteClient.instances.clear()
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            '/goal "Perform exactly one durable action"',
            (
                lambda: (
                    bool(_FakeRenderer.instances)
                    and any(
                        kind == "warning"
                        and "ambiguous external side effect" in str(message)
                        for kind, message in _FakeRenderer.instances[0].events
                    )
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
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **_kwargs: None,
    )

    code = run_chat(
        _args(tmp_path),
        client_factory=AmbiguousWriteClient,
        input_stream=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    client = AmbiguousWriteClient.instances[0]
    assert [call[0] for call in client.calls] == ["send_to_conversation"]
    assert client.calls[0][2] == "conv-ambiguous"

    _state, goal = _load_session_goal(tmp_path / "state.json")
    assert goal.status == "blocked"
    assert goal.rollover_count == 0
    assert goal.active_operation_id
    events = GoalStore(tmp_path / "state.json").events(goal)
    assert events[-1]["type"] == "goal_blocked_ambiguous_operation"
    classification = events[-1]["payload"]["failure_classification"]
    assert classification["code"] == WRITE_OUTCOME_UNKNOWN
    assert classification["status_code"] == 429
    assert classification["reconciliation_required"] is True
    assert not any(event["type"] == "rollover" for event in events)


def test_second_process_cannot_dispatch_user_message_into_live_owned_goal(
    tmp_path, monkeypatch
) -> None:
    import os

    state_path = tmp_path / "state.json"
    store = GoalStore(state_path)
    store.save(
        GoalState(
            goal_id="goal-live-dispatch-guard",
            conversation_ref="conv-owned",
            conversations=["conv-owned"],
            status="active",
            objective="owned task",
            runner_id="live-owner-token",
            runner_pid=os.getpid(),
        ),
        event_type="goal_created",
    )
    from gptty.goal_lock import try_acquire_goal_lock

    live_lock = try_acquire_goal_lock(
        store.root,
        "goal-live-dispatch-guard",
        runner_id="live-owner-token",
        pid=os.getpid(),
    )
    assert live_lock is not None
    save_chat_state(state_path, ChatState(current_conversation="conv-owned"))
    _FakeRenderer.instances.clear()
    _FakeSession.script = iter(
        [
            "MUST_NOT_BE_SENT",
            (
                lambda: (
                    bool(_FakeRenderer.instances)
                    and any(
                        kind == "warning"
                        and "message dispatch is blocked" in str(message)
                        for kind, message in _FakeRenderer.instances[0].events
                    )
                ),
                "/exit",
            ),
        ]
    )
    created: list[object] = []

    class NeverCreateClient:
        def __init__(self, *args, **kwargs) -> None:
            created.append(self)
            raise AssertionError("remote process must not create a ChatGPT client")

    monkeypatch.setattr(
        "gptty.commands.chat.should_use_enhanced_ui",
        lambda **kwargs: (True, SimpleNamespace()),
    )
    monkeypatch.setattr("gptty.commands.chat.InteractiveSession", _FakeSession)
    monkeypatch.setattr("gptty.commands.chat.PrettyRenderer", _FakeRenderer)

    try:
        code = run_chat(
            _args(tmp_path),
            client_factory=NeverCreateClient,
            input_stream=StringIO(),
            stdout=StringIO(),
            stderr=StringIO(),
        )
    finally:
        live_lock.release()

    assert code == 0
    assert created == []
    authoritative = store.load("goal-live-dispatch-guard")
    assert authoritative is not None
    assert authoritative.status == "active"
    assert authoritative.runner_id == "live-owner-token"


def test_restart_never_binds_stale_chat_to_open_operation_without_commit_evidence(
    tmp_path,
) -> None:
    state_path = tmp_path / "state.json"
    store = GoalStore(state_path)
    operation_id = "goal-no-route:g2:t4"
    goal = GoalState(
        goal_id="goal-no-route",
        generation=2,
        status="active",
        conversations=["conv-old-12345678"],
        rollover_count=1,
        active_operation_id=operation_id,
        active_operation_turn=4,
        runner_id="dead-runner",
        runner_pid=999_999_999,
    )
    store.save(goal, event_type="operation_resumed")
    save_chat_state(
        state_path,
        ChatState(current_conversation="conv-old-12345678", goal=goal),
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
    assert created == []
    recovered = store.load("goal-no-route")
    assert recovered is not None
    assert recovered.status == "paused"
    assert recovered.conversation_ref is None
    assert _load_session_state(state_path).current_conversation is None
