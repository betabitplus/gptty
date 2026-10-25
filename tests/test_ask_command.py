from __future__ import annotations

from argparse import Namespace
from io import StringIO
from typing import Any, ClassVar

from gptty.commands.ask import run_ask


class Response:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeGpttyClient:
    instances: ClassVar[list[FakeGpttyClient]] = []

    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        self.auth_file = auth_file
        self.timeout = timeout
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        FakeGpttyClient.instances.append(self)

    def send(self, prompt: str, **options: Any) -> Response:
        self.calls.append(("send", prompt, options))
        on_token = options.get("on_token")
        if on_token is not None:
            on_token("hello")
            on_token(" world")
        return Response("hello world")


def make_args(**overrides: Any) -> Namespace:
    values = {
        "prompt": ["explain", "this"],
        "auth": "auth_data.json",
        "model": None,
        "no_stream": True,
        "plain": False,
        "timeout": 90,
        "image": [],
    }
    values.update(overrides)
    return Namespace(**values)


def test_run_ask_streams_tokens() -> None:
    FakeGpttyClient.instances.clear()
    stdout = StringIO()

    code = run_ask(
        make_args(no_stream=False),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert stdout.getvalue() == "hello world\n"
    assert client.calls[0][2]["stream"] is True
    assert callable(client.calls[0][2]["on_token"])
