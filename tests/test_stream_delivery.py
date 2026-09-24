from __future__ import annotations

import hashlib
import json
import threading
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
            "type": "stream_handoff_terminal_status",
            "topic_id": "conversation-turn-1",
            "stream_status": "COMPLETE",
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

    terminal = next(
        row for row in rows if row.get("event") == "stream_handoff_terminal_status"
    )
    assert terminal["stream_status"] == "COMPLETE"
    assert terminal["last_offset"] == "1000-0"

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

def test_stream_delivery_sqlite_authority_survives_projection_loss(tmp_path: Path) -> None:
    path = tmp_path / "stream-delivery.jsonl"
    journal = StreamDeliveryJournal(path)
    journal.observe(
        "conversation-1",
        {
            "type": "stream_handoff_server_quiet",
            "reason": "quiet",
            "server_idle_seconds": 12.0,
        },
    )

    path.unlink()
    records = journal.records()
    quiet = next(
        row for row in records if row.get("event") == "stream_handoff_server_quiet"
    )
    assert quiet["conversation_ref"] == "conversation-1"
    assert quiet["reason"] == "quiet"


def test_stream_delivery_imports_legacy_backups_once(tmp_path: Path) -> None:
    path = tmp_path / "stream-delivery.jsonl"
    backup = tmp_path / "stream-delivery.jsonl.1"
    backup.write_text(
        json.dumps(
            {
                "schema": 1,
                "event": "legacy-backup",
                "observed_at_ms": 1,
                "conversation_ref": "conversation-old",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "event": "legacy-current",
                "observed_at_ms": 2,
                "conversation_ref": "conversation-old",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = StreamDeliveryJournal(path)
    second = StreamDeliveryJournal(path, db_path=first.store.db_path)

    event_types = [row["event"] for row in second.records()]
    assert event_types.count("legacy-backup") == 1
    assert event_types.count("legacy-current") == 1
    assert event_types.count("journal_start") == 2


def test_stream_delivery_concurrent_writers_share_transactional_store_and_projection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stream-delivery.jsonl"
    first = StreamDeliveryJournal(path)
    second = StreamDeliveryJournal(path, db_path=first.store.db_path)
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def writer(journal: StreamDeliveryJournal, prefix: str) -> None:
        try:
            barrier.wait(timeout=3)
            for index in range(40):
                journal.observe(
                    "conversation-1",
                    {
                        "type": "stream_handoff_ws_reconnecting",
                        "topic_id": f"{prefix}-{index}",
                        "attempt": index,
                    },
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(first, "a")),
        threading.Thread(target=writer, args=(second, "b")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert not errors
    reconnects = [
        row
        for row in first.records()
        if row.get("event") == "stream_handoff_ws_reconnecting"
    ]
    assert len(reconnects) == 80
    projected = _rows(path)
    assert len(
        [
            row
            for row in projected
            if row.get("event") == "stream_handoff_ws_reconnecting"
        ]
    ) == 80


def test_stream_delivery_compacts_projection_without_backup_rotation(tmp_path: Path) -> None:
    path = tmp_path / "stream-delivery.jsonl"
    journal = StreamDeliveryJournal(path, max_bytes=1024 * 1024)

    for index in range(16):
        journal._append(
            {
                "schema": 1,
                "event": "test-padding",
                "observed_at_ms": index,
                "conversation_ref": "conversation-1",
                "padding": "x" * 100_000,
            }
        )

    assert path.stat().st_size <= journal.max_bytes
    assert not (tmp_path / "stream-delivery.jsonl.1").exists()
    padding_records = [
        row for row in journal.records() if row.get("event") == "test-padding"
    ]
    assert len(padding_records) == 16
