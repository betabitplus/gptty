from __future__ import annotations

import json
from pathlib import Path

from gptty.codexpro_activity import (
    CodexProActivityTracker,
    canonical_args_sha256,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _tool_event(*, offset: str, tool: str, args: dict) -> dict:
    return {
        "type": "canonical_intermediate_message",
        "message_kind": "tool_call",
        "source_offset": offset,
        "text": json.dumps(
            {
                "path": f"/CodexPro/link_deadbeef/{tool}",
                "args": args,
            },
            ensure_ascii=False,
        ),
    }


def _start(
    *,
    activity_id: str,
    observed_at_ms: int,
    tool: str,
    args: dict,
    session: str,
) -> dict:
    return {
        "schema": 1,
        "event": "tool_start",
        "activity_id": activity_id,
        "observed_at_ms": observed_at_ms,
        "tool": tool,
        "workspace_id": args.get("workspace_id"),
        "args_sha256": canonical_args_sha256(args),
        "header_fingerprints": {"x-openai-session": session},
        "meta_fingerprints": {"openai/session": session},
    }


def test_tracker_binds_exact_session_and_persists_mapping(tmp_path: Path) -> None:
    journal = tmp_path / "activity.jsonl"
    mapping = tmp_path / "mapping.json"
    args = {
        "workspace_id": "ws_1",
        "path": "README.md",
        "start_line": 1,
        "end_line": 20,
    }
    session = "sha256:" + "a" * 64
    _write_jsonl(
        journal,
        [
            _start(
                activity_id="activity-1",
                observed_at_ms=1_789_764_071_207,
                tool="read",
                args=args,
                session=session,
            ),
            {
                "schema": 1,
                "event": "tool_finish",
                "activity_id": "activity-1",
                "observed_at_ms": 1_789_764_071_210,
                "duration_ms": 3,
                "tool": "read",
                "workspace_id": "ws_1",
                "outcome": "ok",
            },
        ],
    )
    tracker = CodexProActivityTracker(
        journal_path=journal,
        mapping_path=mapping,
        refresh_seconds=0,
    )

    assert tracker.observe_tool_call(
        "conversation-1",
        _tool_event(
            offset="1789764071243-0",
            tool="read",
            args=args,
        ),
    )
    assert tracker.session_for("conversation-1") == session
    assert mapping.stat().st_mode & 0o777 == 0o600

    reloaded = CodexProActivityTracker(
        journal_path=journal,
        mapping_path=mapping,
        refresh_seconds=0,
    )
    assert reloaded.session_for("conversation-1") == session
    snapshot = reloaded.snapshot("conversation-1", now_ms=1_789_764_071_500)
    assert snapshot.bound is True
    assert snapshot.inflight is False
    assert snapshot.last_tool == "read"
    assert snapshot.outcome == "ok"
    assert snapshot.last_event_age_seconds == 0.29


def test_tracker_refuses_ambiguous_session_binding(tmp_path: Path) -> None:
    journal = tmp_path / "activity.jsonl"
    args = {"workspace_id": "ws_1", "path": "same.txt"}
    _write_jsonl(
        journal,
        [
            _start(
                activity_id="a",
                observed_at_ms=1_000_000,
                tool="read",
                args=args,
                session="sha256:" + "a" * 64,
            ),
            _start(
                activity_id="b",
                observed_at_ms=1_000_050,
                tool="read",
                args=args,
                session="sha256:" + "b" * 64,
            ),
        ],
    )
    tracker = CodexProActivityTracker(
        journal_path=journal,
        match_window_ms=1_000,
        refresh_seconds=0,
    )

    assert (
        tracker.observe_tool_call(
            "conversation-1",
            _tool_event(offset="1000025-0", tool="read", args=args),
        )
        is False
    )
    assert tracker.session_for("conversation-1") is None


def test_tracker_reports_inflight_heartbeat_then_finish(tmp_path: Path) -> None:
    journal = tmp_path / "activity.jsonl"
    mapping = tmp_path / "mapping.json"
    args = {"workspace_id": "ws_1", "command": "sleep 30"}
    session = "sha256:" + "c" * 64
    start = _start(
        activity_id="activity-long",
        observed_at_ms=2_000_000,
        tool="bash",
        args=args,
        session=session,
    )
    _write_jsonl(
        journal,
        [
            start,
            {
                "schema": 1,
                "event": "tool_heartbeat",
                "activity_id": "activity-long",
                "observed_at_ms": 2_015_000,
                "elapsed_ms": 15_000,
                "tool": "bash",
                "workspace_id": "ws_1",
            },
        ],
    )
    tracker = CodexProActivityTracker(
        journal_path=journal,
        mapping_path=mapping,
        refresh_seconds=0,
    )
    assert tracker.observe_tool_call(
        "conversation-1",
        _tool_event(offset="2000005-0", tool="bash", args=args),
    )

    snapshot = tracker.snapshot("conversation-1", now_ms=2_020_000)
    assert snapshot.bound is True
    assert snapshot.inflight is True
    assert snapshot.inflight_tool == "bash"
    assert snapshot.last_heartbeat_age_seconds == 5.0
    assert snapshot.last_event_age_seconds == 5.0

    _write_jsonl(
        journal,
        [
            start,
            {
                "schema": 1,
                "event": "tool_heartbeat",
                "activity_id": "activity-long",
                "observed_at_ms": 2_015_000,
                "elapsed_ms": 15_000,
                "tool": "bash",
                "workspace_id": "ws_1",
            },
            {
                "schema": 1,
                "event": "tool_finish",
                "activity_id": "activity-long",
                "observed_at_ms": 2_021_000,
                "duration_ms": 21_000,
                "tool": "bash",
                "workspace_id": "ws_1",
                "outcome": "ok",
            },
        ],
    )
    snapshot = tracker.snapshot("conversation-1", now_ms=2_022_000)
    assert snapshot.inflight is False
    assert snapshot.outcome == "ok"
    assert snapshot.last_event_age_seconds == 1.0


def test_canonical_args_hash_matches_node_key_order_semantics() -> None:
    assert canonical_args_sha256({"b": 2, "a": 1}) == canonical_args_sha256(
        {"a": 1, "b": 2}
    )


def test_tracker_binds_from_snapshot_source_time_ms(tmp_path: Path) -> None:
    journal = tmp_path / "activity.jsonl"
    args = {"workspace_id": "ws_1", "path": "README.md", "start_line": 1, "end_line": 20}
    session = "sha256:" + "d" * 64
    _write_jsonl(
        journal,
        [
            _start(
                activity_id="snapshot-activity",
                observed_at_ms=5_001_100,
                tool="read",
                args=args,
                session=session,
            )
        ],
    )
    tracker = CodexProActivityTracker(
        journal_path=journal,
        match_window_ms=2_000,
        refresh_seconds=0,
    )
    event = _tool_event(offset="5000000-0", tool="read", args=args)
    event.pop("source_offset")
    event["source_time_ms"] = 5_000_000

    assert tracker.observe_tool_call("conversation-snapshot", event)
    assert tracker.session_for("conversation-snapshot") == session
