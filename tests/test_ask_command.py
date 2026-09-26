from __future__ import annotations

import json
from argparse import Namespace
from io import StringIO
from pathlib import Path
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


def test_run_ask_non_stream_prints_response_text() -> None:
    FakeGpttyClient.instances.clear()
    stdout = StringIO()

    code = run_ask(
        make_args(no_stream=True),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert stdout.getvalue() == "hello world\n"
    assert client.calls == [
        ("send", "explain this", {"stream": False}),
    ]


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



def test_run_ask_jsonl_stream_is_parseable_and_preserves_provider_observations() -> None:
    class EventClient(FakeGpttyClient):
        def send(self, prompt: str, **options: Any) -> Response:
            self.calls.append(("send", prompt, options))
            on_event = options.get("on_event")
            if on_event is not None:
                on_event(
                    {
                        "type": "canonical_intermediate_message",
                        "message_kind": "tool_call",
                        "message_id": "tool-message-1",
                        "tool_call_id": "tool-call-1",
                        "tool_name": "api_tool.call_tool",
                        "turn_exchange_id": "turn-1",
                    }
                )
            on_token = options.get("on_token")
            if on_token is not None:
                on_token("hello")
                on_token(" world")
            return Response("hello world")

    EventClient.instances.clear()
    stdout = StringIO()

    code = run_ask(
        make_args(format="jsonl", no_stream=False),
        client_factory=EventClient,
        stdout=stdout,
    )

    rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert code == 0
    assert [row["type"] for row in rows] == [
        "run_started",
        "prompt_sent",
        "provider_event",
        "token_delta",
        "token_delta",
        "turn_result",
    ]
    assert all(isinstance(row, dict) for row in rows)
    assert rows[2]["contract"] == "gptty.run.event"
    assert rows[2]["kind"] == "tool"
    assert rows[2]["tool_call_id"] == "tool-call-1"
    assert rows[-1]["contract"] == "gptty.turn.result"
    assert rows[-1]["status"] == "completed"
    assert rows[-1]["text"] == "hello world"
    assert rows[-1]["observations"]["captured"] is True
    assert rows[-1]["observations"]["tools"][0]["tool_call_id"] == "tool-call-1"
    assert "hello world\n" not in stdout.getvalue()


def test_run_ask_jsonl_ambiguous_write_has_stable_machine_failure() -> None:
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

    stdout = StringIO()
    code = run_ask(
        make_args(format="jsonl"),
        client_factory=AmbiguousClient,
        stdout=stdout,
        stderr=StringIO(),
    )

    rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
    failure = rows[-1]
    assert code == 1
    assert failure["contract"] == "gptty.turn.result"
    assert failure["error"]["class"] == "turn_unconfirmed"
    assert failure["error"]["write_may_have_been_submitted"] is True
    assert failure["error"]["reconciliation_required"] is True
    assert failure["error"]["request_stage"] == "browser_owned_write"


def test_run_ask_passes_auth_timeout_and_model() -> None:
    FakeGpttyClient.instances.clear()

    code = run_ask(
        make_args(
            auth="custom_auth.json",
            model="gpt-4o-mini",
            timeout=12,
        ),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert client.auth_file == "custom_auth.json"
    assert client.timeout == 12
    assert client.calls[0][2] == {"stream": False, "model": "gpt-4o-mini"}


def test_run_ask_passes_image_media_to_sdk(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    image_path = tmp_path / "screenshot.png"
    image_path.write_bytes(b"fake image")

    code = run_ask(
        make_args(image=[str(image_path), "https://example.com/chart.webp"]),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert client.calls[0][2] == {
        "stream": False,
        "media": [str(image_path), "https://example.com/chart.webp"],
    }


def test_run_ask_allows_data_uri_media() -> None:
    FakeGpttyClient.instances.clear()

    code = run_ask(
        make_args(image=["data:image/png;base64,AAAA"]),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert client.calls[0][2] == {
        "stream": False,
        "media": ["data:image/png;base64,AAAA"],
    }


def test_run_ask_returns_2_for_missing_local_image(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    stderr = StringIO()

    code = run_ask(
        make_args(image=[str(tmp_path / "missing.png")]),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert FakeGpttyClient.instances == []
    assert "image file does not exist" in stderr.getvalue()


def test_run_ask_uses_stdin_as_prompt() -> None:
    FakeGpttyClient.instances.clear()

    code = run_ask(
        make_args(prompt=[]),
        stdin_text="stdin prompt",
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert client.calls[0][1] == "stdin prompt"


def test_run_ask_rejects_implicit_stdin_and_prompt_combination() -> None:
    FakeGpttyClient.instances.clear()
    stderr = StringIO()

    code = run_ask(
        make_args(prompt=["review", "this"]),
        stdin_text="diff --git",
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert FakeGpttyClient.instances == []
    assert "cannot be combined implicitly" in stderr.getvalue()


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


def test_run_ask_returns_2_for_empty_prompt() -> None:
    stderr = StringIO()

    code = run_ask(
        make_args(prompt=[]),
        stdin_text="  \n",
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert "requires a prompt" in stderr.getvalue()


def test_ask_forwards_reasoning_effort_intent() -> None:
    FakeGpttyClient.instances.clear()

    code = run_ask(
        make_args(effort="medium"),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    assert code == 0
    assert FakeGpttyClient.instances[0].calls[0][2]["reasoning_effort"] == "medium"


def test_ask_rejects_custom_model_plus_effort_before_client_creation() -> None:
    FakeGpttyClient.instances.clear()
    stderr = StringIO()

    code = run_ask(
        make_args(model="gpt-custom", effort="high"),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert FakeGpttyClient.instances == []
    assert "explicit model" in stderr.getvalue()


def test_run_ask_passes_general_file_through_existing_media_contract(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    document = tmp_path / "notes.txt"
    document.write_text("typed file", encoding="utf-8")

    code = run_ask(
        make_args(file=[str(document)]),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    assert code == 0
    assert FakeGpttyClient.instances[0].calls[0][2] == {
        "stream": False,
        "media": [str(document)],
    }


def test_run_ask_returns_2_for_missing_local_general_file(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    stderr = StringIO()

    code = run_ask(
        make_args(file=[str(tmp_path / "missing.txt")]),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert FakeGpttyClient.instances == []
    assert "file does not exist" in stderr.getvalue()
