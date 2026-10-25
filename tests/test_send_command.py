from __future__ import annotations

from argparse import Namespace
from io import StringIO
from pathlib import Path
from typing import Any

from gptty.commands.send import run_send
from gptty.runs import read_run_events, read_run_summary
from gptty.session_state import session_handle
from gptty.state import ChatState, save_chat_state


class Response:
    def __init__(self, text: str = "reply", conversation_id: str | None = "conv-1") -> None:
        self.text = text
        self.conversation_id = conversation_id


class FakeGpttyClient:
    instances: list["FakeGpttyClient"] = []

    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        self.auth_file = auth_file
        self.timeout = timeout
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        FakeGpttyClient.instances.append(self)

    def send(self, prompt: str, **options: Any) -> Response:
        self.calls.append(("send", (prompt,), options))
        on_token = options.get("on_token")
        if on_token is not None:
            on_token("new reply")
        return Response(text="new reply", conversation_id="new-conv")

    def send_to_conversation(self, conversation_ref: str, prompt: str, **options: Any) -> Response:
        self.calls.append(("send_to_conversation", (conversation_ref, prompt), options))
        on_token = options.get("on_token")
        if on_token is not None:
            on_token("reply")
        return Response(text="reply", conversation_id=conversation_ref)


def _load_command_session(tmp_path: Path, *, session: str | None = None) -> ChatState:
    return session_handle(
        state_path=tmp_path / "gptty_state.json",
        profile=None,
        explicit_session=session,
        environ={},
    ).load()


def make_args(tmp_path: Path, **overrides: Any) -> Namespace:
    values = {
        "prompt": ["continue"],
        "stdin_mode": "auto",
        "to": None,
        "new": False,
        "state": str(tmp_path / "gptty_state.json"),
        "auth": "auth_data.json",
        "timeout": 90,
        "model": None,
        "no_stream": True,
        "format": "plain",
        "image": [],
    }
    values.update(overrides)
    return Namespace(**values)


def test_send_uses_attached_conversation_by_default(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    save_chat_state(tmp_path / "gptty_state.json", ChatState(current_conversation="attached-ref"))
    stdout = StringIO()

    code = run_send(
        make_args(tmp_path, auth="custom_auth.json", timeout=12),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert client.auth_file == "custom_auth.json"
    assert client.timeout == 12
    assert client.calls == [
        ("send_to_conversation", ("attached-ref", "continue"), {"stream": False}),
    ]
    assert stdout.getvalue() == "reply\n"
    assert _load_command_session(tmp_path).current_conversation == "attached-ref"


def test_send_to_explicit_conversation_does_not_mutate_attached_state(
    tmp_path: Path,
) -> None:
    FakeGpttyClient.instances.clear()
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(
        state_path,
        ChatState(current_conversation="attached-ref", model="attached-model"),
    )
    stdout = StringIO()

    code = run_send(
        make_args(tmp_path, to="explicit-ref", model="gpt-4o"),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    client = FakeGpttyClient.instances[0]
    state = _load_command_session(tmp_path)
    assert code == 0
    assert client.calls == [
        (
            "send_to_conversation",
            ("explicit-ref", "continue"),
            {"stream": False, "model": "gpt-4o"},
        ),
    ]
    assert state.current_conversation == "attached-ref"
    assert state.model == "attached-model"


def test_send_new_starts_new_conversation_and_saves_ref(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    stdout = StringIO()

    code = run_send(
        make_args(tmp_path, new=True, prompt=["start"], no_stream=False),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    client = FakeGpttyClient.instances[0]
    assert code == 0
    assert stdout.getvalue() == "new reply\n"
    assert client.calls[0][0] == "send"
    assert client.calls[0][1] == ("start",)
    assert client.calls[0][2]["stream"] is True
    assert callable(client.calls[0][2]["on_token"])
    assert _load_command_session(tmp_path).current_conversation == "new-conv"


def test_explicit_remote_send_does_not_touch_concurrently_updated_session(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gptty_state.json"
    seed = session_handle(
        state_path=state_path,
        profile=None,
        environ={},
    )
    seed_state = seed.load()
    seed_state.current_conversation = "initial-ref"
    seed.save(seed_state)

    class ConcurrentClient(FakeGpttyClient):
        def send_to_conversation(
            self,
            conversation_ref: str,
            prompt: str,
            **options: Any,
        ) -> Response:
            self.calls.append(
                ("send_to_conversation", (conversation_ref, prompt), options)
            )
            winner = session_handle(
                state_path=state_path,
                profile=None,
                environ={},
            )
            winner_state = winner.load()
            winner_state.current_conversation = "concurrent-winner"
            winner.save(winner_state)
            return Response(text="remote completed", conversation_id=conversation_ref)

    stderr = StringIO()
    stdout = StringIO()
    code = run_send(
        make_args(tmp_path, to="explicit-ref"),
        client_factory=ConcurrentClient,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert stdout.getvalue() == "remote completed\n"
    assert stderr.getvalue() == ""
    assert _load_command_session(tmp_path).current_conversation == "concurrent-winner"
    assert len(FakeGpttyClient.instances[-1].calls) == 1


def test_send_ambiguous_write_requires_reconciliation_and_is_not_retried(
    tmp_path: Path,
) -> None:
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
        def send_to_conversation(
            self,
            conversation_ref: str,
            prompt: str,
            **options: Any,
        ) -> Response:
            self.calls.append(
                ("send_to_conversation", (conversation_ref, prompt), options)
            )
            raise error

    FakeGpttyClient.instances.clear()
    stderr = StringIO()

    code = run_send(
        make_args(tmp_path, to="explicit-ref"),
        client_factory=AmbiguousClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 1
    assert len(FakeGpttyClient.instances[-1].calls) == 1
    assert "may have accepted this turn" in stderr.getvalue()
    assert "reconcile the conversation before retrying" in stderr.getvalue()

    run_files = sorted((tmp_path / ".gptty_runs").glob("*.json"))
    assert len(run_files) == 1
    summary = read_run_summary(run_files[0])
    events = read_run_events(run_files[0].with_suffix(".jsonl"), from_start=True)
    classification = summary["failure_classification"]
    assert classification["status"] == "unconfirmed"
    assert classification["code"] == WRITE_OUTCOME_UNKNOWN
    assert classification["status_code"] == 429
    assert classification["write_may_have_been_submitted"] is True
    assert classification["reconciliation_required"] is True
    assert events[-1]["failure_classification"] == classification
