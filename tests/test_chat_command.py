from __future__ import annotations

from argparse import Namespace
from io import StringIO
from types import SimpleNamespace
from typing import Any, ClassVar

from gptty.commands.chat import (
    _send_chat_prompt,
    run_chat,
)
from gptty.session_state import session_handle
from gptty.state import ChatState, load_chat_state, save_chat_state
from gptty.ui.signals import TurnControlSignals


class Response:
    def __init__(
        self,
        text: str = "reply",
        conversation_id: str | None = "conv-1",
        title: str | None = "Test Chat",
        message_id: str | None = None,
    ) -> None:
        self.text = text
        self.conversation_id = conversation_id
        self.title = title
        self.conversation = SimpleNamespace(message_id=message_id)


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

    def send_to_conversation(
        self, conversation_ref: str, prompt: str, **options: Any
    ) -> Response:
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
        return Response(
            text="temporary reply", conversation_id="temp-1", title="Temporary Chat"
        )


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


def _load_command_session(
    tmp_path,
    *,
    explicit_session: str | None = None,
) -> ChatState:
    return session_handle(
        state_path=tmp_path / "gptty_state.json",
        profile=None,
        explicit_session=explicit_session,
        environ={},
    ).load()


def test_terminal_sessions_keep_independent_chat_selection_on_shared_profile(
    tmp_path, monkeypatch
) -> None:
    class SessionAClient(FakeGpttyClient):
        def send(self, prompt: str, **options: Any) -> Response:
            self.calls.append(("send", (prompt,), options))
            return Response(text="A", conversation_id="conv-A")

    class SessionBClient(FakeGpttyClient):
        def send(self, prompt: str, **options: Any) -> Response:
            self.calls.append(("send", (prompt,), options))
            return Response(text="B", conversation_id="conv-B")

    base = tmp_path / "gptty_state.json"
    save_chat_state(base, ChatState())
    args = make_args(tmp_path, no_stream=True)

    monkeypatch.setenv("GPTTY_SESSION_ID", "session-A")
    assert (
        run_chat(
            args,
            input_stream=StringIO("hello A\n/exit\n"),
            client_factory=SessionAClient,
            stdout=StringIO(),
        )
        == 0
    )
    state_a = _load_command_session(tmp_path, explicit_session="session-A")

    monkeypatch.setenv("GPTTY_SESSION_ID", "session-B")
    assert (
        run_chat(
            args,
            input_stream=StringIO("hello B\n/exit\n"),
            client_factory=SessionBClient,
            stdout=StringIO(),
        )
        == 0
    )
    state_b = _load_command_session(tmp_path, explicit_session="session-B")

    assert state_a.current_conversation == "conv-A"
    assert state_b.current_conversation == "conv-B"
    assert _load_command_session(tmp_path).current_conversation is None
    assert load_chat_state(base).current_conversation is None


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
    assert _load_command_session(tmp_path).current_conversation == "conv-1"


def test_existing_conversation_uses_send_to_conversation(tmp_path) -> None:
    FakeGpttyClient.instances.clear()
    save_chat_state(
        tmp_path / "gptty_state.json", ChatState(current_conversation="conv-1")
    )

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


def test_temporary_turn_uses_session_scoped_send_and_never_persists_temp_id(
    tmp_path,
) -> None:
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


def test_ctrl_c_stops_active_turn_and_keeps_new_chat_attached(
    tmp_path, monkeypatch
) -> None:
    class StopClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def send(self, prompt, **options):
            self.calls.append(("send", prompt))
            return Response(
                text="partial answer",
                conversation_id="conv-stopped",
                title="Stopped Chat",
            )

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
                return
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
    assert client.calls == [
        ("stop_generation", None),
        ("send", "keep going for a while"),
    ]
    assert state.current_conversation == "conv-stopped"
    assert (
        load_chat_state(tmp_path / "state.json").current_conversation == "conv-stopped"
    )
    assert ("answer", "partial answer") in renderer.events
    assert ("info", "Stopped by user.") in renderer.events
    assert notified == []
