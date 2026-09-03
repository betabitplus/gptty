from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gptty.sdk_client import GpttyClient, _ProductRuntimeClient


class FakeSdkClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def send(self, prompt: str, **options: object) -> str:
        self.calls.append(("send", (prompt,), options))
        return "send-result"

    def send_to_conversation(
        self,
        url_or_id: object,
        prompt: str,
        **options: object,
    ) -> str:
        self.calls.append(("send_to_conversation", (url_or_id, prompt), options))
        return "send-to-conversation-result"

    def attach_conversation(self, url_or_id: object, **options: object) -> str:
        self.calls.append(("attach_conversation", (url_or_id,), options))
        return "attach-result"

    def get_messages(self, url_or_id: object, **options: object) -> str:
        self.calls.append(("get_messages", (url_or_id,), options))
        return "messages-result"

    def get_required_action(self, url_or_id: object, **options: object) -> str:
        self.calls.append(("get_required_action", (url_or_id,), options))
        return "required-action-result"

    def get_status(self, url_or_id: object, **options: object) -> str:
        self.calls.append(("get_status", (url_or_id,), options))
        return "status-result"

    def wait_until_completed(self, url_or_id: object, **options: object) -> str:
        self.calls.append(("wait_until_completed", (url_or_id,), options))
        return "wait-result"


class OldFakeSdkClient:
    pass


class FakeProductRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.statuses = [SimpleNamespace(status="running"), SimpleNamespace(status="completed")]
        self.canonical = SimpleNamespace()

    def send(self, prompt: str, **options: object) -> str:
        self.calls.append(("send", (prompt,), options))
        return "runtime-send-result"

    def attach_conversation(self, url_or_id: object, **options: object) -> str:
        self.calls.append(("attach_conversation", (url_or_id,), options))
        return "runtime-attach-result"

    def get_messages(self, url_or_id: object, **options: object) -> str:
        self.calls.append(("get_messages", (url_or_id,), options))
        return "runtime-messages-result"

    def get_status(self, url_or_id: object, **options: object):
        self.calls.append(("get_status", (url_or_id,), options))
        if self.statuses:
            return self.statuses.pop(0)
        return SimpleNamespace(status="completed")


def test_gptty_client_keeps_auth_and_timeout() -> None:
    sdk = FakeSdkClient()

    client = GpttyClient(auth_file="custom_auth.json", timeout=123, sdk_client=sdk)

    assert client.auth_file == Path("custom_auth.json")
    assert client.timeout == 123


def test_send_delegates_to_sdk_client_without_cli_stream_option() -> None:
    sdk = FakeSdkClient()
    client = GpttyClient(sdk_client=sdk)

    result = client.send(
        "hello",
        model="gpt-4o-mini",
        stream=True,
        media=["image.png"],
    )

    assert result == "send-result"
    assert sdk.calls == [
        (
            "send",
            ("hello",),
            {"model": "gpt-4o-mini", "media": ["image.png"]},
        )
    ]


def test_send_to_conversation_delegates_without_cli_stream_option() -> None:
    sdk = FakeSdkClient()
    client = GpttyClient(sdk_client=sdk)

    result = client.send_to_conversation(
        "abc",
        "continue",
        stream=False,
        media=["image.png"],
    )

    assert result == "send-to-conversation-result"
    assert sdk.calls == [
        (
            "send_to_conversation",
            ("abc", "continue"),
            {"media": ["image.png"]},
        )
    ]


def test_conversation_methods_delegate_to_sdk_client() -> None:
    sdk = FakeSdkClient()
    client = GpttyClient(sdk_client=sdk)

    assert client.attach_conversation("abc") == "attach-result"
    assert client.get_messages("abc", limit=5) == "messages-result"
    assert client.get_required_action("abc") == "required-action-result"
    assert client.get_status("abc") == "status-result"
    assert client.wait_until_completed("abc", timeout=30) == "wait-result"

    assert sdk.calls == [
        ("attach_conversation", ("abc",), {}),
        ("get_messages", ("abc",), {"limit": 5}),
        ("get_required_action", ("abc",), {}),
        ("get_status", ("abc",), {}),
        ("wait_until_completed", ("abc",), {"timeout": 30}),
    ]


def test_get_required_action_returns_none_with_old_sdk_client() -> None:
    client = GpttyClient(sdk_client=OldFakeSdkClient())

    assert client.get_required_action("abc") is None


def test_product_runtime_client_maps_cli_send_surface() -> None:
    runtime = FakeProductRuntime()
    client = _ProductRuntimeClient(auth_file="auth.json", timeout=17, runtime=runtime)

    assert client.send("hello", model="high", media=["image.png"], on_token="token-cb") == "runtime-send-result"
    assert client.send_to_conversation("c1", "continue", model="instant") == "runtime-send-result"

    assert runtime.calls[:2] == [
        (
            "send",
            ("hello",),
            {
                "timeout": 17.0,
                "media": ["image.png"],
                "on_token": "token-cb",
                "model_profile": "DEEP",
            },
        ),
        (
            "send",
            ("continue",),
            {
                "conversation": "c1",
                "timeout": 17.0,
                "model_profile": "FAST",
            },
        ),
    ]


def test_product_runtime_client_rejects_unknown_model_name() -> None:
    runtime = FakeProductRuntime()
    client = _ProductRuntimeClient(auth_file="auth.json", timeout=17, runtime=runtime)

    with pytest.raises(ValueError, match="effort profile only"):
        client.send("hello", model="gpt-5.6")

    assert runtime.calls == []


def test_product_runtime_client_delegates_read_surface_and_waits() -> None:
    runtime = FakeProductRuntime()
    client = _ProductRuntimeClient(auth_file="auth.json", timeout=17, runtime=runtime)

    assert client.attach_conversation("c1") == "runtime-attach-result"
    assert client.get_messages("c1", limit=4) == "runtime-messages-result"
    status = client.wait_until_completed("c1", timeout=1, poll_interval=0.001)

    assert status.status == "completed"
    assert runtime.calls[0] == ("attach_conversation", ("c1",), {})
    assert runtime.calls[1] == ("get_messages", ("c1",), {"limit": 4})
    assert runtime.calls[2:] == [
        ("get_status", ("c1",), {}),
        ("get_status", ("c1",), {}),
    ]
