from __future__ import annotations

from gptty.automation import (
    AUTOMATION_SCHEMA,
    RUN_EVENT_CONTRACT,
    normalize_provider_event,
    run_event_envelope,
    source_citation_bundle,
)


def test_run_event_envelope_is_versioned_and_identified() -> None:
    event = run_event_envelope(
        run_id="run-1",
        event_type="token_delta",
        timestamp="2026-09-25T13:00:00+00:00",
        data={"text": "hello"},
    )

    assert event["schema"] == AUTOMATION_SCHEMA
    assert event["contract"] == RUN_EVENT_CONTRACT
    assert event["run_id"] == "run-1"
    assert event["type"] == "token_delta"
    assert event["text"] == "hello"
    assert isinstance(event["event_id"], str) and event["event_id"]
    assert event["provenance"] == {
        "producer": "gptty",
        "source": "local-run",
    }


def test_normalize_provider_tool_event_preserves_stable_identity() -> None:
    normalized = normalize_provider_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_result",
            "message_id": "message-2",
            "parent_message_id": "message-1",
            "tool_name": "api_tool.call_tool",
            "tool_call_id": "tool-call-1",
            "turn_exchange_id": "turn-1",
            "text": "{\"ok\":true}",
            "ignored_raw_secret": "do-not-copy",
        }
    )

    assert normalized == {
        "kind": "tool",
        "provider_event_type": "canonical_intermediate_message",
        "provenance": {
            "producer": "chatgpt-web-adapter",
            "source": "provider-event",
        },
        "message_id": "message-2",
        "parent_message_id": "message-1",
        "text": "{\"ok\":true}",
        "message_kind": "tool_result",
        "turn_exchange_id": "turn-1",
        "tool_name": "api_tool.call_tool",
        "tool_call_id": "tool-call-1",
    }


def test_normalize_provider_source_and_citation_events_are_typed() -> None:
    source = normalize_provider_event(
        {
            "type": "product_source_observed",
            "observation_schema": 1,
            "observation_id": "source-observation:1",
            "source_id": "source-1",
            "url": "https://example.com/article",
            "title": "Article",
            "domain": "example.com",
            "source_origin": "canonical_content_references",
        }
    )
    citation = normalize_provider_event(
        {
            "type": "product_citation_observed",
            "observation_schema": 1,
            "observation_id": "citation-observation:1",
            "citation_id": "citation-1",
            "source_id": "source-1",
            "citation_index": 0,
            "start_index": 12,
            "end_index": 25,
            "reference_type": "webpage",
            "display_text": "Article",
        }
    )

    assert source is not None and source["kind"] == "source"
    assert source["source_id"] == "source-1"
    assert source["url"] == "https://example.com/article"
    assert citation is not None and citation["kind"] == "citation"
    assert citation["citation_id"] == "citation-1"
    assert citation["source_id"] == "source-1"
    assert citation["start_index"] == 12
    assert citation["end_index"] == 25
    assert citation["range_coordinate_space"] == "unknown"


def test_source_citation_bundle_deduplicates_and_drops_orphans() -> None:
    source = normalize_provider_event(
        {
            "type": "product_source_observed",
            "observation_id": "source-observation:1",
            "source_id": "source-1",
            "url": "https://example.com/article",
            "title": "Article",
        }
    )
    citation = normalize_provider_event(
        {
            "type": "product_citation_observed",
            "observation_id": "citation-observation:1",
            "citation_id": "citation-1",
            "source_id": "source-1",
            "start_index": 5000,
            "end_index": 9000,
        }
    )
    orphan = normalize_provider_event(
        {
            "type": "product_citation_observed",
            "observation_id": "citation-observation:2",
            "citation_id": "citation-2",
            "source_id": "missing-source",
            "start_index": 0,
            "end_index": 1,
        }
    )

    bundle = source_citation_bundle([source, source, citation, citation, orphan])

    assert [item["source_id"] for item in bundle["sources"]] == ["source-1"]
    assert [item["citation_id"] for item in bundle["citations"]] == ["citation-1"]
    assert bundle["citations"][0]["range_coordinate_space"] == "unknown"
    assert source_citation_bundle([source, citation], max_sources=0) == {
        "sources": [],
        "citations": [],
    }
    assert source_citation_bundle([source, citation], max_citations=0) == {
        "sources": [source],
        "citations": [],
    }
