from __future__ import annotations

import json
from argparse import Namespace
from io import StringIO
from pathlib import Path
from typing import Any

from gptty.commands.send import run_send
from gptty.runs import read_run_events, read_run_summary
from gptty.state import ChatState, save_chat_state


class Response:
    text = "reply"
    conversation_id = "conv-1"


class EmptyResponse:
    text = ""
    conversation_id = "conv-1"


class FakeGpttyClient:
    instances: list["FakeGpttyClient"] = []

    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        FakeGpttyClient.instances.append(self)

    def send_to_conversation(self, conversation_ref: str, prompt: str, **options: Any) -> Response:
        self.calls.append(("send_to_conversation", (conversation_ref, prompt), options))
        on_token = options.get("on_token")
        if on_token is not None:
            on_token("hello")
        return Response()


class FakeProviderEventClient(FakeGpttyClient):
    def send_to_conversation(
        self,
        conversation_ref: str,
        prompt: str,
        **options: Any,
    ) -> Response:
        self.calls.append(("send_to_conversation", (conversation_ref, prompt), options))
        on_event = options.get("on_event")
        if on_event is not None:
            on_event(
                {
                    "type": "canonical_intermediate_message",
                    "message_kind": "tool_call",
                    "message_id": "tool-call-message",
                    "tool_call_id": "tool-call-message",
                    "tool_name": "api_tool.call_tool",
                    "turn_exchange_id": "turn-1",
                    "text": "{\"path\":\"bash\"}",
                }
            )
            on_event(
                {
                    "type": "product_source_observed",
                    "observation_schema": 1,
                    "observation_id": "source-observation:1",
                    "source_id": "source-1",
                    "url": "https://example.com/source",
                    "title": "Source",
                    "domain": "example.com",
                    "source_origin": "canonical_content_references",
                }
            )
            on_event(
                {
                    "type": "product_connector_started",
                    "observation_id": "connector:1:start",
                    "connector_activity_id": "connector-activity:1",
                    "connector_id": "calendar",
                    "connector_name": "Calendar",
                    "operation": "search_events",
                }
            )
            on_event(
                {
                    "type": "product_required_action_started",
                    "observation_id": "action:1:start",
                    "action_id": "action:1",
                    "action_type": "user_authorization",
                    "connector_activity_id": "connector-activity:1",
                    "connector_id": "calendar",
                }
            )
        return Response()


class FakeRequiredActionClient:
    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        pass

    def send_to_conversation(self, conversation_ref: str, prompt: str, **options: Any) -> EmptyResponse:
        return EmptyResponse()

    def get_required_action(self, conversation_ref: str) -> dict[str, object]:
        return {
            "type": "connector_oauth",
            "reason": "Connect Gmail",
            "actions": ["connect", "not_now"],
        }


def make_args(tmp_path: Path, **overrides: Any) -> Namespace:
    values = {
        "prompt": ["continue"],
        "to": None,
        "new": False,
        "state": str(tmp_path / "gptty_state.json"),
        "auth": "auth_data.json",
        "profile": None,
        "timeout": 90,
        "model": None,
        "no_stream": False,
        "format": "plain",
        "image": [],
        "wait_lock": False,
        "lock_timeout": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_send_writes_run_events_for_attached_conversation(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(state_path, ChatState(current_conversation="conv-1"))

    code = run_send(
        make_args(tmp_path),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    run_files = list((tmp_path / ".gptty_runs").glob("*.json"))
    assert len(run_files) == 1
    summary = read_run_summary(run_files[0])
    events = read_run_events(summary["events_file"], from_start=True)
    assert summary["status"] == "completed"
    assert [event["type"] for event in events] == [
        "run_started",
        "prompt_sent",
        "waiting_for_reply",
        "token_delta",
        "completed",
    ]
    assert events[3]["text"] == "hello"


def test_send_records_versioned_typed_provider_events(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(state_path, ChatState(current_conversation="conv-1"))

    code = run_send(
        make_args(tmp_path),
        client_factory=FakeProviderEventClient,
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert code == 0
    run_file = next((tmp_path / ".gptty_runs").glob("*.json"))
    summary = read_run_summary(run_file)
    events = read_run_events(summary["events_file"], from_start=True)
    provider_events = [event for event in events if event["type"] == "provider_event"]

    assert [event["kind"] for event in provider_events] == [
        "tool",
        "source",
        "connector",
        "action",
    ]
    tool_event, source_event, connector_event, action_event = provider_events
    assert tool_event["schema"] == 1
    assert tool_event["contract"] == "gptty.run.event"
    assert tool_event["provenance"] == {
        "producer": "chatgpt-web-adapter",
        "source": "provider-event",
    }
    assert tool_event["tool_call_id"] == "tool-call-message"
    assert tool_event["turn_exchange_id"] == "turn-1"
    assert source_event["source_id"] == "source-1"
    assert source_event["url"] == "https://example.com/source"
    assert connector_event["connector_activity_id"] == "connector-activity:1"
    assert connector_event["connector_id"] == "calendar"
    assert connector_event["operation"] == "search_events"
    assert connector_event["phase"] == "STARTED"
    assert action_event["action_id"] == "action:1"
    assert action_event["action_type"] == "user_authorization"
    assert action_event["connector_activity_id"] == "connector-activity:1"
    assert action_event["phase"] == "STARTED"

    completed = next(event for event in events if event["type"] == "completed")
    observations = completed["turn_result"]["observations"]
    assert observations["connectors"][0]["connector_activity_id"] == "connector-activity:1"
    assert observations["actions"][0]["action_id"] == "action:1"


def test_send_marks_required_action_run_as_failed(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(state_path, ChatState(current_conversation="conv-1"))
    stderr = StringIO()

    code = run_send(
        make_args(tmp_path),
        client_factory=FakeRequiredActionClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 1
    run_files = list((tmp_path / ".gptty_runs").glob("*.json"))
    assert len(run_files) == 1
    summary = read_run_summary(run_files[0])
    events = read_run_events(summary["events_file"], from_start=True)
    assert summary["status"] == "failed"
    assert summary["error"] == "ChatGPT is waiting for a web UI action."
    assert [event["type"] for event in events] == [
        "run_started",
        "prompt_sent",
        "waiting_for_reply",
        "required_action",
        "failed",
    ]
    assert "requires a web UI action" in stderr.getvalue()


def test_send_jsonl_represents_required_action_and_final_failure(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(state_path, ChatState(current_conversation="conv-1"))
    stdout = StringIO()
    stderr = StringIO()

    code = run_send(
        make_args(tmp_path, format="jsonl"),
        client_factory=FakeRequiredActionClient,
        stdout=stdout,
        stderr=stderr,
    )

    rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
    action = next(row for row in rows if row["type"] == "required_action")
    failure = rows[-1]
    assert code == 1
    assert action["kind"] == "action"
    assert action["action_type"] == "connector_oauth"
    assert action["reason"] == "Connect Gmail"
    assert action["actions"] == ["connect", "not_now"]
    assert action["conversation"] == "conv-1"
    assert failure["contract"] == "gptty.turn.result"
    assert failure["error"]["class"] == "required_action"
    assert failure["error"]["code"] == "required_action"
    assert failure["conversation"] == "conv-1"
    assert "requires a web UI action" in stderr.getvalue()

    run_file = next((tmp_path / ".gptty_runs").glob("*.json"))
    summary = read_run_summary(run_file)
    assert summary["turn_result"]["error"]["class"] == "required_action"
