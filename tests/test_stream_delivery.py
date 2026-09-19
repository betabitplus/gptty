from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gptty.stream_delivery import StreamDeliveryJournal


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_stream_delivery_journal_records_safe_delivery_evidence(tmp_path: Path) -> None:
    path = tmp_path / "stream-delivery.jsonl"
    journal = StreamDeliveryJournal(path)
    sample_text = "visible commentary that should only be represented by metadata"

    journal.observe(
        "conversation-1",
        {
            "type": "stream_handoff_ws_reconnecting",
            "topic_id": "conversation-turn-1",
            "reason": "topic_idle",
            "attempt": 2,
            "server_idle_seconds": 24.5,
            "last_offset": "1000-0",
        },
    )
    journal.observe(
        "conversation-1",
        {
            "type": "stream_handoff_ws_subscribed",
            "topic_id": "conversation-turn-1",
            "catchup_count": 0,
            "last_offset": "1000-0",
        },
    )
    journal.observe(
        "conversation-1",
        {
            "type": "canonical_intermediate_message",
            "message_id": "message-1",
            "message_kind": "commentary",
            "turn_exchange_id": "turn-1",
            "source_offset": "1001-0",
            "text": sample_text,
        },
    )
    journal.observe(
        "conversation-1",
        {
            "type": "assistant_text_delta",
            "message_id": "answer-1",
            "sequence": 7,
            "delta": "hello",
        },
    )

    raw = path.read_text(encoding="utf-8")
    assert sample_text not in raw
    assert '"delta": "hello"' not in raw
    assert path.stat().st_mode & 0o777 == 0o600

    rows = _rows(path)
    reconnect = next(
        row for row in rows if row.get("event") == "stream_handoff_ws_reconnecting"
    )
    assert reconnect["conversation_ref"] == "conversation-1"
    assert reconnect["reason"] == "topic_idle"
    assert reconnect["attempt"] == 2
    assert reconnect["last_offset"] == "1000-0"

    subscribed = next(
        row for row in rows if row.get("event") == "stream_handoff_ws_subscribed"
    )
    assert subscribed["catchup_count"] == 0

    canonical = next(
        row for row in rows if row.get("event") == "canonical_intermediate_message"
    )
    assert canonical["message_id"] == "message-1"
    assert canonical["source_offset"] == "1001-0"
    assert canonical["payload_chars"] == len(sample_text)
    assert canonical["payload_sha256"] == (
        "sha256:" + hashlib.sha256(sample_text.encode()).hexdigest()
    )

    delta = next(row for row in rows if row.get("event") == "assistant_text_delta")
    assert delta["message_id"] == "answer-1"
    assert delta["sequence"] == 7
    assert delta["payload_chars"] == 5
    assert delta["payload_sha256"] == (
        "sha256:" + hashlib.sha256(b"hello").hexdigest()
    )


def test_stream_delivery_journal_ignores_unrelated_events(tmp_path: Path) -> None:
    path = tmp_path / "stream-delivery.jsonl"
    journal = StreamDeliveryJournal(path)
    before = len(_rows(path))

    journal.observe(
        "conversation-1",
        {"type": "unrelated_internal_event", "text": "do not persist"},
    )

    assert len(_rows(path)) == before
    assert "do not persist" not in path.read_text(encoding="utf-8")
