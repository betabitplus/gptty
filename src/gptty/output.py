from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

OutputFormat = Literal["plain", "json", "markdown"]

CONVERSATION_FIELDS = (
    "conversation_url",
    "conversation_id",
    "conversation_ref",
    "current_conversation",
    "url",
    "id",
)


@dataclass(frozen=True)
class OutputMessage:
    role: str
    text: str
    created_at: str | None = None


def normalize_messages(response: Any) -> list[OutputMessage]:
    raw_messages = _extract_raw_messages(response)
    return [_normalize_message(message) for message in raw_messages]


def render_messages(messages: list[OutputMessage], output_format: OutputFormat = "plain") -> str:
    if output_format == "plain":
        return _render_messages_plain(messages)
    if output_format == "json":
        return _json_dump({"messages": [asdict(message) for message in messages]})
    if output_format == "markdown":
        return _render_messages_markdown(messages)
    raise ValueError(f"Unsupported output format: {output_format}")


def normalize_status(response: Any, *, conversation: str | None = None) -> dict[str, Any]:
    if isinstance(response, str):
        data: dict[str, Any] = {"status": response}
    elif isinstance(response, dict):
        data = dict(response)
    else:
        data = _object_fields(response, ("status", "state", "conversation", *CONVERSATION_FIELDS))
        if not data:
            data = {"status": str(response)}

    if conversation and not _has_conversation(data):
        data["conversation"] = conversation
    return data


def render_status(status: dict[str, Any], output_format: OutputFormat = "plain") -> str:
    if output_format == "plain":
        if set(status) == {"status"}:
            return str(status["status"])
        return "\n".join(f"{key}: {_plain_value(value)}" for key, value in status.items())
    if output_format == "json":
        return _json_dump(status)
    if output_format == "markdown":
        return _render_mapping_markdown(status)
    raise ValueError(f"Unsupported output format: {output_format}")


def normalize_response(response: Any, *, conversation: str | None = None) -> dict[str, Any]:
    text = _response_text(response)
    response_conversation = _conversation_ref(response) or conversation
    data: dict[str, Any] = {"text": text}
    if response_conversation:
        data["conversation"] = response_conversation
    return data


def render_response(response: dict[str, Any], output_format: OutputFormat = "plain") -> str:
    if output_format == "plain":
        return str(response.get("text", ""))
    if output_format == "json":
        return _json_dump(response)
    if output_format == "markdown":
        return str(response.get("text", ""))
    raise ValueError(f"Unsupported output format: {output_format}")


def render_live_event(event: Any) -> str | None:
    if not isinstance(event, dict):
        return None
    event_type = event.get("type")
    tool_name = event.get("tool_name")
    if event_type == "canonical_intermediate_message":
        kind = event.get("message_kind")
        text = event.get("text")
        text = text.strip() if isinstance(text, str) else ""
        label = event.get("label")
        label = label.strip() if isinstance(label, str) else ""
        tool = tool_name.strip() if isinstance(tool_name, str) else ""
        if kind == "assistant_progress" and text:
            return f"[thinking]\n{text}"
        if kind == "reasoning":
            rendered = text or label
            return f"[thinking]\n{rendered}" if rendered else None
        if kind == "tool_call":
            return _render_tool_call(tool=tool, text=text, label=label)
        if kind == "tool_result":
            error = _tool_result_error(text)
            return f"[tool error] {tool or 'tool'} · {error}" if error else None
        if kind == "activity":
            return f"[activity] {text or label}" if text or label else None
        return None
    # The early debugger-backed activity stream is intentionally not rendered here.
    # After safe detach, canonical history is the single live source so callers see
    # contextual, completed blocks without duplicate placeholder events such as
    # `[tool] api_tool.call_tool` or `[thinking] Thinking…`.
    return None


def _render_tool_call(*, tool: str, text: str, label: str) -> str:
    payload = _json_mapping(text)

    if tool == "api_tool.call_tool":
        action = _api_tool_action(payload)
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        if action:
            return _tool_line(action, _api_tool_detail(action, args))
        return _tool_line("call_tool", _clean_tool_detail(label))

    if tool == "api_tool.list_resources":
        detail = ""
        query = payload.get("query")
        if isinstance(query, str) and query.strip():
            detail = query
        else:
            paths = payload.get("paths")
            if isinstance(paths, list) and paths and isinstance(paths[0], str):
                detail = paths[0]
        return _tool_line("list_resources", detail or _clean_tool_detail(label))

    return _tool_line(tool or "tool", _clean_tool_detail(label))


def _api_tool_action(payload: dict[str, Any]) -> str:
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        return ""
    return path.rstrip("/").rsplit("/", 1)[-1].strip()


def _api_tool_detail(action: str, args: dict[str, Any]) -> str:
    if action in {"read", "tree"}:
        return _clean_tool_detail(args.get("path"))
    if action == "search":
        return _clean_tool_detail(args.get("query"))
    if action == "bash":
        return _clean_tool_detail(args.get("command"))
    if action == "open_workspace":
        return _clean_tool_detail(args.get("root") or args.get("path"))
    return ""


def _tool_line(name: str, detail: str) -> str:
    return f"[tool] {name} · {detail}" if detail else f"[tool] {name}"


def _clean_tool_detail(value: Any, *, max_chars: int = 160) -> str:
    if not isinstance(value, str):
        return ""
    detail = " ".join(value.split()).strip()
    while detail.endswith("...") or detail.endswith("…"):
        detail = detail[:-3].rstrip() if detail.endswith("...") else detail[:-1].rstrip()
    if len(detail) <= max_chars:
        return detail
    return detail[: max_chars - 1].rstrip() + "…"


def _json_mapping(text: str) -> dict[str, Any]:
    if not text or text[:1] != "{":
        return {}
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tool_result_error(text: str) -> str:
    payload = _json_mapping(text)
    if not payload:
        return ""

    failed = payload.get("is_error") is True or payload.get("ok") is False or payload.get("success") is False
    status = payload.get("status")
    if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
        failed = True
    exit_code = payload.get("exitCode", payload.get("returncode"))
    if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
        failed = True
    if not failed:
        return ""

    for key in ("error", "message", "detail", "stderr"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_tool_detail(value)
        if isinstance(value, dict):
            for nested_key in ("message", "detail", "code"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    return _clean_tool_detail(nested)
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return f"exit {exit_code}"
    if isinstance(status, str) and status.strip():
        return _clean_tool_detail(status)
    return "failed"


def _extract_raw_messages(response: Any) -> list[Any]:
    messages = _field(response, "messages")
    if isinstance(messages, list):
        return messages
    if messages is not None and _is_iterable_messages(messages):
        return list(messages)
    if isinstance(response, list):
        return response
    if _is_iterable_messages(response):
        return list(response)
    return []


def _normalize_message(message: Any) -> OutputMessage:
    role = _message_field(message, "role", "author", default="message")
    text = _message_field(message, "text", "content", "message", default="")
    created_at = _message_field(message, "created_at", "create_time", "timestamp", default="") or None
    return OutputMessage(role=role, text=text, created_at=created_at)


def _message_field(message: Any, *fields: str, default: str) -> str:
    for field in fields:
        value = _field(message, field)
        if value is not None:
            return _stringify_message_value(value)
    return default


def _field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _object_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field in fields:
        field_value = getattr(value, field, None)
        if field_value is not None:
            data[field] = field_value
    return data


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    for field in ("text", "message", "content"):
        value = _field(response, field)
        if value is not None:
            return _stringify_message_value(value)
    if response is None:
        return ""
    return str(response)


def _conversation_ref(response: Any) -> str | None:
    for field in CONVERSATION_FIELDS:
        value = _field(response, field)
        if value:
            return str(value)
    return None


def _has_conversation(data: dict[str, Any]) -> bool:
    return any(key in data and data[key] for key in ("conversation", *CONVERSATION_FIELDS))


def _render_messages_plain(messages: list[OutputMessage]) -> str:
    if not messages:
        return "(no messages)"
    return "\n\n".join(f"{message.role}:\n{message.text}".rstrip() for message in messages)


def _render_messages_markdown(messages: list[OutputMessage]) -> str:
    if not messages:
        return "_No messages._"
    blocks = []
    for message in messages:
        blocks.append(f"### {message.role}\n\n{message.text}".rstrip())
    return "\n\n".join(blocks)


def _render_mapping_markdown(data: dict[str, Any]) -> str:
    lines = ["| Field | Value |", "|---|---|"]
    for key, value in data.items():
        lines.append(f"| {key} | {_markdown_table_value(value)} |")
    return "\n".join(lines)


def _stringify_message_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_stringify_message_part(part) for part in value]
        return "".join(part for part in parts if part)
    return str(value)


def _stringify_message_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        for field in ("text", "content", "value"):
            value = part.get(field)
            if value is not None:
                return _stringify_message_value(value)
    return str(part)


def _plain_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True)
    return str(value)


def _markdown_table_value(value: Any) -> str:
    rendered = _plain_value(value)
    return rendered.replace("|", "\\|").replace("\n", "<br>")


def _json_dump(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _is_iterable_messages(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True
