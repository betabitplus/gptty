from __future__ import annotations

from argparse import Namespace
from io import StringIO
from types import SimpleNamespace
from typing import Any, ClassVar

from gptty.commands.chat import (
    LOCAL_QUIT_CODE,
    _send_chat_prompt,
    extract_conversation_ref,
    run_chat,
)
from gptty.state import ChatState, load_chat_state, save_chat_state
from gptty.ui.signals import TurnControlSignals


class Response:
    def __init__(
        self,
        text: str = "reply",
        conversation_id: str | None = "conv-1",
        title: str | None = "Test Chat",
    ) -> None:
        self.text = text
        self.conversation_id = conversation_id
        self.title = title


class FakeGpttyClient:
    instances: ClassVar[list[FakeGpttyClient]] = []

    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        self.auth_file = auth_file
        self.timeout = timeout
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        FakeGpttyClient.instances.append(self)

    def send(self, prompt: str, **options: Any) -> Response:
        self.calls.append(("send", (prompt,), options))
        on_token = options.get("on_token")
        if on_token is not None:
            on_token("reply")
        return Response()

    def send_to_conversation(self, conversation_ref: str, prompt: str, **options: Any) -> Response:
        self.calls.append(("send_to_conversation", (conversation_ref, prompt), options))
        on_token = options.get("on_token")
        if on_token is not None:
            on_token("continued")
        return Response(text="continued", conversation_id=conversation_ref)

    def send_temporary(self, prompt: str, **options: Any) -> Response:
        self.calls.append(("send_temporary", (prompt,), options))
        on_token = options.get("on_token")
        if on_token is not None:
            on_token("temporary reply")
        return Response(text="temporary reply", conversation_id="temp-1", title="Temporary Chat")


def make_args(tmp_path, **overrides: Any) -> Namespace:
    values = {
        "state": str(tmp_path / "gptty_state.json"),
        "auth": "auth_data.json",
        "model": None,
        "no_stream": False,
        "timeout": 90,
    }
    values.update(overrides)
    return Namespace(**values)


def test_first_prompt_calls_send_and_persists_conversation(tmp_path) -> None:
    FakeGpttyClient.instances.clear()
    stdout = StringIO()

    code = run_chat(
        make_args(tmp_path),
        input_stream=StringIO("hello\n/exit\n"),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert stdout.getvalue() == "reply\n"
    assert client.calls == [
        (
            "send",
            ("hello",),
            {
                "stream": True,
                "on_token": client.calls[0][2]["on_token"],
                "on_event": client.calls[0][2]["on_event"],
            },
        ),
    ]
    assert callable(client.calls[0][2]["on_token"])
    assert callable(client.calls[0][2]["on_event"])
    assert load_chat_state(tmp_path / "gptty_state.json").current_conversation == "conv-1"


def test_first_prompt_persists_nested_cwa_conversation_shape(tmp_path) -> None:
    class NestedConversationClient(FakeGpttyClient):
        def send(self, prompt: str, **options: Any):
            self.calls.append(("send", (prompt,), options))
            return SimpleNamespace(
                text="reply",
                conversation=SimpleNamespace(conversation_id="nested-conv"),
            )

    NestedConversationClient.instances.clear()
    code = run_chat(
        make_args(tmp_path, no_stream=True),
        input_stream=StringIO("hello\n/exit\n"),
        client_factory=NestedConversationClient,
        stdout=StringIO(),
    )

    assert code == 0
    assert load_chat_state(tmp_path / "gptty_state.json").current_conversation == "nested-conv"


def test_existing_conversation_uses_send_to_conversation(tmp_path) -> None:
    FakeGpttyClient.instances.clear()
    save_chat_state(tmp_path / "gptty_state.json", ChatState(current_conversation="conv-1"))

    code = run_chat(
        make_args(tmp_path),
        input_stream=StringIO("continue\n/exit\n"),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert client.calls[0][0] == "send_to_conversation"
    assert client.calls[0][1] == ("conv-1", "continue")
    assert client.calls[0][2]["stream"] is True


def test_temporary_turn_uses_session_scoped_send_and_never_persists_temp_id(tmp_path) -> None:
    client = FakeGpttyClient()
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(state_path, ChatState())
    recorded: list[dict[str, object]] = []

    code = _send_chat_prompt(
        client,
        state=ChatState(),
        state_path=state_path,
        profile=None,
        prompt="ephemeral",
        model=None,
        media=None,
        stream=False,
        stdout=StringIO(),
        stderr=StringIO(),
        conversation_mode="temporary",
        attached_ref=None,
        temporary_turn_recorder=lambda **kwargs: recorded.append(kwargs),
    )

    assert code == 0
    assert client.calls == [("send_temporary", ("ephemeral",), {"stream": False})]
    assert load_chat_state(state_path).current_conversation is None
    assert recorded == [
        {
            "prompt": "ephemeral",
            "answer": "temporary reply",
            "conversation_ref": "temp-1",
            "title": "Temporary Chat",
        }
    ]


def test_new_command_clears_conversation_without_sdk_init(tmp_path) -> None:
    FakeGpttyClient.instances.clear()
    save_chat_state(tmp_path / "gptty_state.json", ChatState(current_conversation="conv-1"))
    stdout = StringIO()

    code = run_chat(
        make_args(tmp_path),
        input_stream=StringIO("/new\n/exit\n"),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    assert code == 0
    assert stdout.getvalue() == "Started a new chat.\n"
    assert load_chat_state(tmp_path / "gptty_state.json").current_conversation is None
    assert FakeGpttyClient.instances == []


def test_exit_command_returns_zero_without_sdk_init(tmp_path) -> None:
    FakeGpttyClient.instances.clear()

    code = run_chat(
        make_args(tmp_path),
        input_stream=StringIO("/exit\n"),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    assert code == 0
    assert FakeGpttyClient.instances == []


def test_empty_input_is_ignored(tmp_path) -> None:
    FakeGpttyClient.instances.clear()

    code = run_chat(
        make_args(tmp_path),
        input_stream=StringIO("\nhello\n/exit\n"),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert len(client.calls) == 1
    assert client.calls[0][1] == ("hello",)


def test_no_stream_passes_stream_false_and_prints_response_text(tmp_path) -> None:
    FakeGpttyClient.instances.clear()
    stdout = StringIO()

    code = run_chat(
        make_args(tmp_path, no_stream=True, model="gpt-4o", timeout=12, auth="custom_auth.json"),
        input_stream=StringIO("hello\n/exit\n"),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert client.auth_file == "custom_auth.json"
    assert client.timeout == 12
    assert stdout.getvalue() == "reply\n"
    assert client.calls[0] == (
        "send",
        ("hello",),
        {"stream": False, "model": "gpt-4o"},
    )
    assert load_chat_state(tmp_path / "gptty_state.json").model == "gpt-4o"


def test_completed_enhanced_turn_notifies_with_chat_and_final_response(tmp_path, monkeypatch) -> None:
    class FakeRenderer:
        def answer(self, _text: str) -> None:
            pass

        def chat_link(self, _ref: str) -> None:
            pass

        def turn_abort(self) -> None:
            pass

    client = FakeGpttyClient()
    state = ChatState()
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.commands.chat.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    code = _send_chat_prompt(
        client,
        state=state,
        state_path=tmp_path / "state.json",
        profile=None,
        prompt="Which screenshot is this?",
        model=None,
        media=None,
        stream=False,
        stdout=StringIO(),
        stderr=StringIO(),
        renderer=FakeRenderer(),
    )

    assert code == 0
    assert notified == [
        {
            "chat_title": "Test Chat",
            "final_response": "reply",
        }
    ]


def test_ctrl_c_stops_active_turn_and_keeps_new_chat_attached(tmp_path, monkeypatch) -> None:
    class StopClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def send(self, prompt, **options):
            self.calls.append(("send", prompt))
            return Response(text="partial answer", conversation_id="conv-stopped", title="Stopped Chat")

        def stop_generation(self, ref=None, **options):
            self.calls.append(("stop_generation", ref))
            return {"ok": True, "stopped": True, "conversationId": "conv-stopped"}

    controls = TurnControlSignals()

    class FakeThread:
        def __init__(self, *, target, name=None, daemon=None):
            self.target = target
            self.alive = True
            self.join_calls = 0

        def start(self):
            return None

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            if self.join_calls == 0:
                self.join_calls += 1
                controls.request_stop()
                return None
            self.target()
            self.alive = False

    class FakeRenderer:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def answer(self, text):
            self.events.append(("answer", text))

        def chat_link(self, ref):
            self.events.append(("chat_link", ref))

        def turn_abort(self):
            self.events.append(("turn_abort", None))

        def info(self, text):
            self.events.append(("info", text))

        def warning(self, text):
            self.events.append(("warning", text))

    monkeypatch.setattr("gptty.commands.chat.threading.Thread", FakeThread)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.commands.chat.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )
    client = StopClient()
    renderer = FakeRenderer()
    state = ChatState()

    code = _send_chat_prompt(
        client,
        state=state,
        state_path=tmp_path / "state.json",
        profile=None,
        prompt="keep going for a while",
        model=None,
        media=None,
        stream=False,
        stdout=StringIO(),
        stderr=StringIO(),
        renderer=renderer,
        turn_controls=controls,
    )

    assert code == 0
    assert client.calls == [("stop_generation", None), ("send", "keep going for a while")]
    assert state.current_conversation == "conv-stopped"
    assert load_chat_state(tmp_path / "state.json").current_conversation == "conv-stopped"
    assert ("answer", "partial answer") in renderer.events
    assert ("info", "Stopped by user.") in renderer.events
    assert notified == []


def test_second_ctrl_c_after_confirmed_stop_exits_local_readback_wait(tmp_path, monkeypatch) -> None:
    class StopClient:
        def __init__(self) -> None:
            self.calls = []

        def stop_generation(self, ref=None, **options):
            self.calls.append(("stop_generation", ref))
            return {"ok": True, "stopped": True, "conversationId": "conv-stopped"}

    controls = TurnControlSignals()

    class FakeThread:
        def __init__(self, *, target, name=None, daemon=None):
            self.alive = True

        def start(self):
            return None

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            controls.request_stop()

    class FakeRenderer:
        def __init__(self) -> None:
            self.events = []

        def turn_abort(self):
            self.events.append(("turn_abort", None))

        def info(self, text):
            self.events.append(("info", text))

        def warning(self, text):
            self.events.append(("warning", text))

    monkeypatch.setattr("gptty.commands.chat.threading.Thread", FakeThread)
    client = StopClient()
    renderer = FakeRenderer()
    state = ChatState()
    confirmed = []

    code = _send_chat_prompt(
        client,
        state=state,
        state_path=tmp_path / "state.json",
        profile=None,
        prompt="keep going",
        model=None,
        media=None,
        stream=False,
        stdout=StringIO(),
        stderr=StringIO(),
        renderer=renderer,
        turn_controls=controls,
        on_stop_confirmed=confirmed.append,
    )

    assert code == LOCAL_QUIT_CODE
    assert client.calls == [("stop_generation", None)]
    assert confirmed == ["conv-stopped"]
    assert state.current_conversation == "conv-stopped"
    assert ("info", "ChatGPT stopped; finalizing local readback…") in renderer.events
    assert (
        "info",
        "ChatGPT is already stopped; exiting gptty without waiting for local readback.",
    ) in renderer.events


def test_ctrl_c_reconciles_saved_partial_after_stop_aborts_browser_fetch(tmp_path, monkeypatch) -> None:
    class StopAbortClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def send(self, prompt, **options):
            self.calls.append(("send", prompt))
            raise RuntimeError("CHATGPT_CONVERSATION_REQUEST_FAILED:net::ERR_ABORTED")

        def stop_generation(self, ref=None, **options):
            self.calls.append(("stop_generation", ref))
            return {"ok": True, "stopped": True, "conversationId": "conv-stopped"}

        def conversation_snapshot(self, ref, **options):
            self.calls.append(("conversation_snapshot", ref))
            return {
                "messages": [
                    {"role": "user", "text": "keep going for a while"},
                    {"role": "assistant", "recipient": "all", "text": "saved partial answer"},
                ]
            }

    controls = TurnControlSignals()

    class FakeThread:
        def __init__(self, *, target, name=None, daemon=None):
            self.target = target
            self.alive = True
            self.join_calls = 0

        def start(self):
            return None

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            if self.join_calls == 0:
                self.join_calls += 1
                controls.request_stop()
                return None
            self.target()
            self.alive = False

    class FakeRenderer:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def answer(self, text):
            self.events.append(("answer", text))

        def chat_link(self, ref):
            self.events.append(("chat_link", ref))

        def turn_abort(self):
            self.events.append(("turn_abort", None))

        def info(self, text):
            self.events.append(("info", text))

        def warning(self, text):
            self.events.append(("warning", text))

    monkeypatch.setattr("gptty.commands.chat.threading.Thread", FakeThread)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.commands.chat.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )
    client = StopAbortClient()
    renderer = FakeRenderer()
    state = ChatState()
    stderr = StringIO()

    code = _send_chat_prompt(
        client,
        state=state,
        state_path=tmp_path / "state.json",
        profile=None,
        prompt="keep going for a while",
        model=None,
        media=None,
        stream=False,
        stdout=StringIO(),
        stderr=stderr,
        renderer=renderer,
        turn_controls=controls,
    )

    assert code == 0
    assert client.calls == [
        ("stop_generation", None),
        ("send", "keep going for a while"),
        ("conversation_snapshot", "conv-stopped"),
    ]
    assert state.current_conversation == "conv-stopped"
    assert load_chat_state(tmp_path / "state.json").current_conversation == "conv-stopped"
    assert ("answer", "saved partial answer") in renderer.events
    assert ("info", "Stopped by user.") in renderer.events
    assert "chat request failed" not in stderr.getvalue()
    assert notified == []


def test_ctrl_backslash_exits_locally_without_stopping_chat(tmp_path, monkeypatch) -> None:
    class ContinueClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def send_to_conversation(self, ref, prompt, **options):
            self.calls.append(("send_to_conversation", ref))
            options["on_event"]({"type": "browser_native_write_completed"})
            return Response(text="eventual answer", conversation_id=ref)

        def stop_generation(self, ref=None, **options):
            self.calls.append(("stop_generation", ref))
            return {"ok": True, "stopped": True, "conversationId": ref}

    controls = TurnControlSignals()

    class FakeThread:
        def __init__(self, *, target, name=None, daemon=None):
            self.target = target
            self.alive = True
            self.joins = 0

        def start(self):
            return None

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.joins += 1
            if self.joins == 1:
                controls.request_quit()
                return
            self.target()
            self.alive = False

    class FakeRenderer:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def turn_abort(self):
            self.events.append(("turn_abort", None))

        def info(self, text):
            self.events.append(("info", text))

    monkeypatch.setattr("gptty.commands.chat.threading.Thread", FakeThread)
    client = ContinueClient()
    renderer = FakeRenderer()
    state = ChatState(current_conversation="conv-running")

    code = _send_chat_prompt(
        client,
        state=state,
        state_path=tmp_path / "state.json",
        profile=None,
        prompt="continue in browser",
        model=None,
        media=None,
        stream=False,
        stdout=StringIO(),
        stderr=StringIO(),
        renderer=renderer,
        turn_controls=controls,
    )

    assert code == LOCAL_QUIT_CODE
    assert client.calls == [("send_to_conversation", "conv-running")]
    assert state.current_conversation == "conv-running"
    assert ("info", "Waiting for safe ChatGPT handoff before local exit…") in renderer.events
    assert ("info", "Exited gptty; ChatGPT response continues in browser.") in renderer.events


def test_extract_conversation_ref_reads_dict_attributes_and_nested_conversation() -> None:
    assert extract_conversation_ref({"conversation_url": "https://chatgpt.com/c/abc"}) == (
        "https://chatgpt.com/c/abc"
    )
    assert extract_conversation_ref(Response(conversation_id="abc")) == "abc"
    assert extract_conversation_ref(SimpleNamespace(conversation=SimpleNamespace(conversation_id="nested"))) == "nested"
    assert extract_conversation_ref({"conversation": {"conversation_id": "nested-dict"}}) == "nested-dict"
    assert extract_conversation_ref(object()) is None
