from __future__ import annotations

from argparse import Namespace
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

from gptty.commands.send import run_send
from gptty.locks import (
    acquire_conversation_lock,
    conversation_lock_dir,
)
from gptty.state import ChatState, save_chat_state


class Response:
    text = "reply"
    conversation_id = "conv-1"


class FakeGpttyClient:
    instances: ClassVar[list[FakeGpttyClient]] = []

    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        FakeGpttyClient.instances.append(self)

    def send_to_conversation(
        self, conversation_ref: str, prompt: str, **options: Any
    ) -> Response:
        self.calls.append(("send_to_conversation", (conversation_ref, prompt), options))
        return Response()

    def send(self, prompt: str, **options: Any) -> Response:
        self.calls.append(("send", (prompt,), options))
        return Response()


def send_args(tmp_path: Path, **overrides: Any) -> Namespace:
    values = {
        "prompt": ["continue"],
        "stdin_mode": "auto",
        "to": None,
        "new": False,
        "state": str(tmp_path / "gptty_state.json"),
        "auth": "auth_data.json",
        "profile": None,
        "timeout": 90,
        "model": None,
        "no_stream": True,
        "format": "plain",
        "image": [],
        "wait_lock": False,
        "lock_timeout": 0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_send_returns_2_when_conversation_is_locked(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(state_path, ChatState(current_conversation="conv-1"))
    lock = acquire_conversation_lock(
        conversation_ref="conv-1",
        lock_dir=conversation_lock_dir(profile=None, state_path=state_path),
        command="test",
    )

    try:
        stderr = StringIO()
        code = run_send(
            send_args(tmp_path),
            client_factory=FakeGpttyClient,
            stdout=StringIO(),
            stderr=stderr,
        )
    finally:
        lock.release()

    assert code == 2
    assert FakeGpttyClient.instances == []
    assert "conversation still in progress" in stderr.getvalue()
    assert "Conversation: conv-1" in stderr.getvalue()
