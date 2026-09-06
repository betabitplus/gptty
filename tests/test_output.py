from __future__ import annotations

import json

from gptty.output import (
    OutputMessage,
    normalize_messages,
    normalize_response,
    normalize_status,
    render_live_event,
    render_messages,
    render_response,
    render_status,
)


class MessageObject:
    def __init__(self, role: str, text: str, created_at: str | None = None) -> None:
        self.role = role
        self.text = text
        self.created_at = created_at


class ResponseWithMessages:
    def __init__(self, messages: list[object]) -> None:
        self.messages = messages


class StatusObject:
    def __init__(self, status: str, conversation_id: str) -> None:
        self.status = status
        self.conversation_id = conversation_id


class TextResponse:
    def __init__(self, text: str, conversation_id: str = "conv-1") -> None:
        self.text = text
        self.conversation_id = conversation_id


def test_render_messages_plain() -> None:
    messages = [
        OutputMessage(role="user", text="hello"),
        OutputMessage(role="assistant", text="hi"),
    ]

    assert render_messages(messages, "plain") == "user:\nhello\n\nassistant:\nhi"


def test_render_messages_json() -> None:
    messages = [OutputMessage(role="user", text="hello", created_at="2026-01-01")]

    assert json.loads(render_messages(messages, "json")) == {
        "messages": [
            {"role": "user", "text": "hello", "created_at": "2026-01-01"},
        ]
    }


def test_render_messages_markdown() -> None:
    messages = [OutputMessage(role="assistant", text="hi")]

    assert render_messages(messages, "markdown") == "### assistant\n\nhi"


def test_normalize_messages_from_response_shapes() -> None:
    response = ResponseWithMessages(
        [
            MessageObject("user", "hello", "2026-01-01"),
            {"role": "assistant", "content": [{"text": "hi"}, {"text": " there"}]},
        ]
    )

    assert normalize_messages(response) == [
        OutputMessage(role="user", text="hello", created_at="2026-01-01"),
        OutputMessage(role="assistant", text="hi there"),
    ]


def test_render_status_plain_json_and_markdown() -> None:
    status = {"status": "completed", "conversation": "conv-1"}

    assert render_status(status, "plain") == "status: completed\nconversation: conv-1"
    assert json.loads(render_status(status, "json")) == status
    assert render_status(status, "markdown") == (
        "| Field | Value |\n|---|---|\n| status | completed |\n| conversation | conv-1 |"
    )


def test_normalize_status_from_string_dict_and_object() -> None:
    assert normalize_status("completed") == {"status": "completed"}
    assert normalize_status({"status": "running"}, conversation="conv-1") == {
        "status": "running",
        "conversation": "conv-1",
    }
    assert normalize_status(StatusObject("completed", "conv-1")) == {
        "status": "completed",
        "conversation_id": "conv-1",
    }


def test_render_response_plain_json_and_markdown() -> None:
    response = {"text": "reply", "conversation": "conv-1"}

    assert render_response(response, "plain") == "reply"
    assert json.loads(render_response(response, "json")) == response
    assert render_response(response, "markdown") == "reply"


def test_legacy_activity_placeholders_are_not_rendered() -> None:
    assert render_live_event({"type": "activity_started", "activity_kind": "reasoning", "label": "Thinking…"}) is None
    assert render_live_event({"type": "activity_text_snapshot", "text": "Worked for 12s"}) is None
    assert render_live_event({"type": "activity_started", "tool_name": "api_tool.call_tool"}) is None


def test_render_canonical_intermediate_blocks() -> None:
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "assistant_progress",
            "text": "Reading files…",
        }
    ) == "[thinking]\nReading files…"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "api_tool.call_tool",
            "label": "Reading README…",
            "text": '{"path":"/CodexTool/link/read","args":{"path":"README.md"}}',
        }
    ) == "[tool] read · README.md"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_result",
            "tool_name": "api_tool.call_tool",
            "label": "README read",
            "text": '{"ok":true}',
        }
    ) is None
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "reasoning",
            "label": "Checking context",
            "text": "",
        }
    ) == "[thinking]\nChecking context"


def test_render_tool_calls_use_compact_useful_details() -> None:
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "api_tool.call_tool",
            "label": "Searching workspace...",
            "text": '{"path":"/CodexTool/link/search","args":{"query":"goal mode"}}',
        }
    ) == "[tool] search · goal mode"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "api_tool.call_tool",
            "label": "Running command...",
            "text": '{"path":"/CodexTool/link/bash","args":{"command":"pytest -q"}}',
        }
    ) == "[tool] bash · pytest -q"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "api_tool.call_tool",
            "label": "Reviewing changes...",
            "text": '{"path":"/CodexTool/link/show_changes","args":{}}',
        }
    ) == "[tool] show_changes"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "api_tool.list_resources",
            "label": "Discovering tools...",
            "text": '{"paths":["CodexTool"],"query":"read"}',
        }
    ) == "[tool] list_resources · read"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "api_tool.call_tool",
            "label": "Searching LOCAL_QUIT_CODE...",
            "text": "",
        }
    ) == "[tool] search · LOCAL_QUIT_CODE"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "api_tool.call_tool",
            "label": "Calling list workspaces...",
            "text": "",
        }
    ) == "[tool] list_workspaces"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "api_tool.call_tool",
            "label": "Using tool...",
            "text": "",
        }
    ) is None
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "web.run",
            "label": "Searching the web...",
            "text": "",
        }
    ) == "[tool] web.run · Searching the web"


def test_render_tool_results_only_surfaces_errors() -> None:
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_result",
            "tool_name": "api_tool.call_tool",
            "text": '{"ok":true,"message":"done"}',
        }
    ) is None
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_result",
            "tool_name": "api_tool.call_tool",
            "text": '{"ok":false,"error":"Workspace not found"}',
        }
    ) == "[tool error] api_tool.call_tool · Workspace not found"
    assert render_live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_result",
            "tool_name": "api_tool.call_tool",
            "text": '{"exitCode":2,"stderr":"pytest: bad option\\nmore noise"}',
        }
    ) == "[tool error] api_tool.call_tool · pytest: bad option more noise"


def test_normalize_response_from_shapes() -> None:
    assert normalize_response(TextResponse("reply")) == {
        "text": "reply",
        "conversation": "conv-1",
    }
    assert normalize_response({"content": "reply"}, conversation="fallback") == {
        "text": "reply",
        "conversation": "fallback",
    }
    assert normalize_response("reply") == {"text": "reply"}
