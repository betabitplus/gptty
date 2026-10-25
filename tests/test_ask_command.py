from __future__ import annotations

from argparse import Namespace
from io import StringIO
from typing import Any

from gptty.commands.ask import run_ask


class Response:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeGpttyClient:
    instances: list["FakeGpttyClient"] = []

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


def test_run_ask_ambiguous_write_requires_reconciliation_and_is_not_retried() -> None:
    from chatgpt_web_adapter.browser_owned_write_runtime import (
        WRITE_OUTCOME_UNKNOWN,
        BrowserOwnedWriteRuntimeError,
    )

    error = BrowserOwnedWriteRuntimeError(
        "provider failed after delegation",
        failure_kind=WRITE_OUTCOME_UNKNOWN,
        automatic_retry_allowed=False,
        manual_retry_safe_after_repair=False,
        write_may_have_been_submitted=True,
        reconciliation_required=True,
        request_stage="browser_owned_write",
        status_code=429,
    )

    class AmbiguousClient(FakeGpttyClient):
        def send(self, prompt: str, **options: Any) -> Response:
            self.calls.append(("send", prompt, options))
            raise error

    FakeGpttyClient.instances.clear()
    stderr = StringIO()

    code = run_ask(
        make_args(),
        client_factory=AmbiguousClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 1
    assert len(FakeGpttyClient.instances[-1].calls) == 1
    assert "may have accepted this turn" in stderr.getvalue()
    assert "reconcile the conversation before retrying" in stderr.getvalue()
