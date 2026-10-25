from __future__ import annotations

from argparse import Namespace
from io import StringIO
from typing import Any

from gptty.commands.attach import run_attach
from gptty.session_state import session_handle


class Response:
    def __init__(self, conversation_url: str | None = None) -> None:
        self.conversation_url = conversation_url


class FakeGpttyClient:
    instances: list["FakeGpttyClient"] = []

    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        self.auth_file = auth_file
        self.timeout = timeout
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        FakeGpttyClient.instances.append(self)

    def attach_conversation(self, url_or_id: str, **options: Any) -> Response:
        self.calls.append(("attach_conversation", (url_or_id,), options))
        return Response(conversation_url="https://chatgpt.com/c/attached")


def make_args(tmp_path, **overrides: Any) -> Namespace:
    values = {
        "url_or_id": "https://chatgpt.com/c/input",
        "state": str(tmp_path / "gptty_state.json"),
        "auth": "auth_data.json",
        "timeout": 90,
    }
    values.update(overrides)
    return Namespace(**values)


def test_attach_calls_sdk_and_saves_conversation(tmp_path) -> None:
    FakeGpttyClient.instances.clear()
    stdout = StringIO()

    code = run_attach(
        make_args(tmp_path, auth="custom_auth.json", timeout=12),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert client.auth_file == "custom_auth.json"
    assert client.timeout == 12
    assert client.calls == [
        ("attach_conversation", ("https://chatgpt.com/c/input",), {}),
    ]
    state = session_handle(
        state_path=tmp_path / "gptty_state.json",
        profile=None,
        environ={},
    ).load()
    assert state.current_conversation == "https://chatgpt.com/c/attached"
    assert stdout.getvalue() == "Attached conversation: https://chatgpt.com/c/attached\n"
