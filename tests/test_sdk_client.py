from __future__ import annotations

from types import SimpleNamespace

from gptty.sdk_client import _ProductRuntimeClient


class FakeProductRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.statuses = [
            SimpleNamespace(status="running"),
            SimpleNamespace(status="completed"),
        ]
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

    def list_conversations(self):
        self.calls.append(("list_conversations", (), {}))
        return "runtime-conversations-result"

    def list_models(self):
        self.calls.append(("list_models", (), {}))
        return "runtime-models-result"

    def conversation_snapshot(self, url_or_id: object, **options: object):
        self.calls.append(("conversation_snapshot", (url_or_id,), options))
        return "runtime-snapshot-result"

    def conversation_follow_snapshot(self, url_or_id: object, **options: object):
        self.calls.append(("conversation_follow_snapshot", (url_or_id,), options))
        return "runtime-follow-snapshot-result"

    def conversation_follow_stream(self, url_or_id: object, **options: object):
        self.calls.append(("conversation_follow_stream", (url_or_id,), options))
        return "runtime-follow-stream-result"

    def stop_generation(self, url_or_id: object = None, **options: object):
        self.calls.append(("stop_generation", (url_or_id,), options))
        return {"ok": True, "stopped": True, "conversationId": url_or_id}

    def end_temporary_chat(self) -> bool:
        self.calls.append(("end_temporary_chat", (), {}))
        return True

    def temporary_lifecycle_snapshot(self) -> dict[str, object]:
        self.calls.append(("temporary_lifecycle_snapshot", (), {}))
        return {"state": "LIVE", "conversation_id": "temp-runtime"}


def test_product_runtime_client_maps_cli_send_surface() -> None:
    runtime = FakeProductRuntime()
    client = _ProductRuntimeClient(auth_file="auth.json", timeout=17, runtime=runtime)

    assert (
        client.send("hello", model="high", media=["image.png"], on_token="token-cb")
        == "runtime-send-result"
    )
    assert (
        client.send_to_conversation("c1", "continue", model="instant")
        == "runtime-send-result"
    )
    assert client.send_temporary("ephemeral") == "runtime-send-result"

    assert runtime.calls[:3] == [
        (
            "send",
            ("hello",),
            {
                "timeout": 17.0,
                "model": "high",
                "media": ["image.png"],
                "on_token": "token-cb",
            },
        ),
        (
            "send",
            ("continue",),
            {
                "conversation": "c1",
                "timeout": 17.0,
                "model": "instant",
            },
        ),
        (
            "send",
            ("ephemeral",),
            {
                "timeout": 17.0,
                "conversation_mode": "temporary",
            },
        ),
    ]


def test_real_adapter_runtime_has_what_gptty_calls() -> None:
    from chatgpt_web_adapter import ChatGPTProductRuntime

    # Everything _ProductRuntimeClient reaches on the runtime. Most of it is looked up with a
    # silent fallback, so only the real classes show a rename on the adapter side.
    surface = [
        "submit",
        "await_final",
        "send",
        "attach_conversation",
        "get_messages",
        "get_status",
        "list_conversations",
        "list_recent_conversations",
        "list_models",
        "conversation_snapshot",
        "conversation_follow_snapshot",
        "conversation_follow_stream",
        "conversation_ui_state",
        "stop_generation",
        "end_temporary_chat",
        "temporary_lifecycle_snapshot",
        "write_transport.governance",
        "canonical.get_required_action",
    ]
    # Built offline: the stub stands in for the signed-in ChatGPT client and is never called;
    # the write transport and the conversation reader on top of it are the adapter's own.
    stub = lambda *args, **kwargs: None
    runtime = ChatGPTProductRuntime(
        SimpleNamespace(get_status=stub, get_messages=stub, attach_conversation=stub),
        transport="browser-owned",
    )
    missing = []
    for path in surface:
        target = runtime
        for name in path.split("."):
            target = getattr(target, name, None)
        if not callable(target):
            missing.append(path)
    assert missing == []
