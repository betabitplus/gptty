from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

AUTOMATION_SCHEMA = 1
RUN_EVENT_CONTRACT = "gptty.run.event"
RUN_SUMMARY_CONTRACT = "gptty.run.summary"
TURN_RESULT_CONTRACT = "gptty.turn.result"

EXIT_SUCCESS = 0
EXIT_REQUEST_FAILED = 1
EXIT_USAGE = 2

LOCAL_RUN_PROVENANCE = {
    "producer": "gptty",
    "source": "local-run",
}


def new_run_id() -> str:
    return uuid.uuid4().hex


def automation_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_required_action(
    action: Any,
    *,
    conversation: Any = None,
) -> dict[str, Any]:
    fields = (
        "action_id",
        "type",
        "reason",
        "connector_id",
        "domain",
        "path",
        "actions",
        "status",
    )
    payload: dict[str, Any] = {}
    for field in fields:
        value = _field(action, field)
        safe = _safe_value(value)
        if safe is not None:
            payload["action_type" if field == "type" else field] = safe
    conversation_ref = _conversation_ref(conversation)
    if conversation_ref:
        payload["conversation"] = conversation_ref
    payload.setdefault("action_type", "required_action")
    payload["provenance"] = {
        "producer": "chatgpt-web-adapter",
        "source": "required-action",
    }
    return payload


def summarize_observations(events: list[dict[str, Any]] | None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "tools": [],
        "actions": [],
        "sources": [],
        "citations": [],
    }
    for event in events or []:
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        target = {
            "tool": "tools",
            "action": "actions",
            "source": "sources",
            "citation": "citations",
        }.get(kind)
        if target is None:
            continue
        record = {
            key: value
            for key, value in event.items()
            if key
            not in {
                "schema",
                "contract",
                "event_id",
                "run_id",
                "type",
                "timestamp",
            }
        }
        grouped[target].append(record)
    return {
        "captured": any(grouped.values()),
        **grouped,
    }


_PROVIDER_SAFE_FIELDS = (
    "conversation_id",
    "conversationId",
    "message_id",
    "parent_message_id",
    "sequence",
    "text",
    "delta",
    "message_kind",
    "turn_exchange_id",
    "tool_name",
    "tool_call_id",
    "observation_schema",
    "observation_id",
    "source_id",
    "url",
    "title",
    "domain",
    "attribution",
    "source_origin",
    "citation_id",
    "citation_index",
    "start_index",
    "end_index",
    "reference_type",
    "display_text",
    "action_id",
    "action_type",
    "activity_id",
    "activity_kind",
    "operation",
    "phase",
    "kind",
    "label",
    "status",
    "reason",
    "attempt",
    "terminal_observed",
    "terminal_source",
    "terminal_error_code",
    "observed_model",
    "observed_reasoning_effort",
)


_ENVELOPE_FIELDS = {
    "schema",
    "contract",
    "event_id",
    "run_id",
    "type",
    "timestamp",
}


def run_event_envelope(
    *,
    run_id: str,
    event_type: str,
    timestamp: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in dict(data or {}).items()
        if key not in _ENVELOPE_FIELDS
    }
    provenance = payload.pop("provenance", None)
    if not isinstance(provenance, dict):
        provenance = dict(LOCAL_RUN_PROVENANCE)
    kind = payload.pop("kind", None)
    if not isinstance(kind, str) or not kind.strip():
        kind = _run_event_kind(event_type)
    return {
        "schema": AUTOMATION_SCHEMA,
        "contract": RUN_EVENT_CONTRACT,
        "event_id": uuid.uuid4().hex,
        "run_id": run_id,
        "type": event_type,
        "kind": kind,
        "timestamp": timestamp,
        "provenance": provenance,
        **payload,
    }


def _run_event_kind(event_type: str) -> str:
    if event_type in {"prompt_sent", "token_delta"}:
        return "message"
    if event_type in {"required_action"}:
        return "action"
    if event_type in {"run_started", "waiting_for_reply", "conversation_bound"}:
        return "turn"
    if event_type in {"completed", "failed"}:
        return "turn"
    return "transport"


def normalize_provider_event(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    event_type = event.get("type")
    if not isinstance(event_type, str) or not event_type.strip():
        return None
    event_type = event_type.strip()

    payload: dict[str, Any] = {
        "kind": _provider_event_kind(event_type, event),
        "provider_event_type": event_type,
        "provenance": {
            "producer": "chatgpt-web-adapter",
            "source": "provider-event",
        },
    }
    for field in _PROVIDER_SAFE_FIELDS:
        if field not in event:
            continue
        value = _safe_value(event[field])
        if value is not None:
            target = "provider_kind" if field == "kind" else field
            payload[target] = value

    conversation = payload.pop("conversationId", None)
    if "conversation_id" not in payload and isinstance(conversation, str):
        payload["conversation_id"] = conversation
    if event_type == "product_citation_observed":
        # CWA deliberately preserves product-provided numeric ranges but their
        # Unicode coordinate space is not release-proven yet. Consumers must
        # treat them as opaque metadata rather than slicing Python strings.
        payload["range_coordinate_space"] = "unknown"
    return payload


def source_citation_bundle(
    events: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    max_sources: int = 64,
    max_citations: int = 128,
) -> dict[str, list[dict[str, Any]]]:
    """Return bounded typed source/citation observations without text inference."""

    source_limit = max(0, int(max_sources))
    citation_limit = max(0, int(max_citations))
    if source_limit == 0:
        return {"sources": [], "citations": []}

    summary = summarize_observations(list(events or ()))
    sources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_citations: set[str] = set()

    for item in summary["sources"]:
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        sources.append(dict(item))
        if len(sources) >= source_limit:
            break

    if citation_limit == 0:
        return {"sources": sources, "citations": []}

    for item in summary["citations"]:
        citation_id = item.get("citation_id")
        source_id = item.get("source_id")
        if (
            not isinstance(citation_id, str)
            or not citation_id
            or citation_id in seen_citations
            or not isinstance(source_id, str)
            or source_id not in seen_sources
        ):
            continue
        seen_citations.add(citation_id)
        record = dict(item)
        record.setdefault("range_coordinate_space", "unknown")
        citations.append(record)
        if len(citations) >= citation_limit:
            break

    return {"sources": sources, "citations": citations}


def _provider_event_kind(event_type: str, event: dict[str, Any]) -> str:
    if event_type in {
        "assistant_text_snapshot",
        "assistant_text_delta",
        "assistant_text_revision",
    }:
        return "message"
    if event_type == "canonical_intermediate_message":
        message_kind = event.get("message_kind")
        if message_kind in {"tool_call", "tool_result"}:
            return "tool"
        return "message"
    if event_type == "product_source_observed":
        return "source"
    if event_type == "product_citation_observed":
        return "citation"
    if event_type in {
        "product_required_action_observed",
        "product_connector_action_observed",
        "product_connector_required_action_observed",
    }:
        return "action"
    if event_type.startswith("activity_"):
        return "activity"
    if (
        event_type.startswith("browser_native_write_")
        or event_type.startswith("stream_handoff_")
        or event_type in {"stream_terminal", "stream_completed"}
    ):
        return "turn"
    return "transport"


def _field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _conversation_ref(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if value is None:
        return None
    for field in (
        "conversation_url",
        "conversation_id",
        "conversation_ref",
        "current_conversation",
        "url",
        "id",
    ):
        candidate = _field(value, field)
        if candidate:
            return str(candidate)
    return None


def _safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_safe_value(item, depth=depth + 1) for item in value[:64]]
        return [item for item in items if item is not None]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:64]:
            if not isinstance(key, str):
                continue
            safe = _safe_value(item, depth=depth + 1)
            if safe is not None:
                result[key] = safe
        return result
    return None
