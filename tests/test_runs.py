from __future__ import annotations

from io import StringIO

import gptty.runs as runs
from gptty.runs import read_run_events, read_run_summary, render_run_status, start_run


def test_start_run_writes_summary_and_events(tmp_path) -> None:
    recorder = start_run(
        profile="work",
        state_path=tmp_path / "gptty_state.json",
        command="send",
        conversation_ref="conv-1",
    )
    recorder.event("prompt_sent")
    recorder.event("token_delta", text="hello")
    recorder.complete()

    summary = read_run_summary(recorder.run_file)
    events = read_run_events(recorder.events_file, from_start=True)

    assert summary["profile"] == "work"
    assert summary["command"] == "send"
    assert summary["conversation_ref"] == "conv-1"
    assert summary["status"] == "completed"
    assert [event["type"] for event in events] == [
        "run_started",
        "prompt_sent",
        "token_delta",
        "completed",
    ]
    assert summary["schema"] == 1
    assert summary["contract"] == "gptty.run.summary"
    assert all(event["schema"] == 1 for event in events)
    assert all(event["contract"] == "gptty.run.event" for event in events)
    assert all(event["run_id"] == recorder.run_id for event in events)
    assert all(isinstance(event["event_id"], str) and event["event_id"] for event in events)


def test_fail_persists_traceback_in_summary_and_event(tmp_path) -> None:
    recorder = start_run(
        profile=None,
        state_path=tmp_path / "gptty_state.json",
        command="chat",
        conversation_ref="conv-1",
    )

    recorder.fail(
        "boom",
        traceback_text="Traceback (most recent call last):\nValueError: boom\n",
    )

    summary = read_run_summary(recorder.run_file)
    events = read_run_events(recorder.events_file, from_start=True)

    assert summary["status"] == "failed"
    assert summary["error"] == "boom"
    assert "ValueError: boom" in summary["traceback"]
    assert events[-1]["type"] == "failed"
    assert events[-1]["message"] == "boom"
    assert "ValueError: boom" in events[-1]["traceback"]




def test_fail_persists_typed_failure_classification(tmp_path) -> None:
    recorder = start_run(
        profile=None,
        state_path=tmp_path / "gptty_state.json",
        command="send",
        conversation_ref="conv-1",
    )
    classification = {
        "label": "turn",
        "status": "unconfirmed",
        "message": "reconcile before retrying",
        "source": "structured",
        "write_may_have_been_submitted": True,
        "reconciliation_required": True,
    }

    recorder.fail("provider detail", failure_classification=classification)

    summary = read_run_summary(recorder.run_file)
    events = read_run_events(recorder.events_file, from_start=True)

    assert summary["error"] == "provider detail"
    assert summary["failure_classification"] == classification
    assert events[-1]["type"] == "failed"
    assert events[-1]["failure_classification"] == classification


def test_sqlite_remains_authority_when_run_projections_are_corrupted(tmp_path) -> None:
    recorder = start_run(
        profile=None,
        state_path=tmp_path / "gptty_state.json",
        command="send",
        conversation_ref="conv-1",
    )
    recorder.event("token_delta", text="authoritative")

    recorder.run_file.write_text("{not-json", encoding="utf-8")
    recorder.events_file.write_text("not-json\n", encoding="utf-8")

    summary = read_run_summary(recorder.run_file)
    events = read_run_events(recorder.events_file, from_start=True)

    assert summary["last_event"] == "token_delta"
    assert [event["type"] for event in events] == ["run_started", "token_delta"]
    assert events[-1]["text"] == "authoritative"


def test_projection_failure_does_not_erase_committed_run_event(
    monkeypatch,
    tmp_path,
) -> None:
    recorder = start_run(
        profile=None,
        state_path=tmp_path / "gptty_state.json",
        command="send",
        conversation_ref="conv-1",
    )

    def fail_projection(*_args, **_kwargs):
        raise OSError("projection unavailable")

    monkeypatch.setattr(runs, "_append_event_projection", fail_projection)

    recorder.event("token_delta", text="survived")

    summary = read_run_summary(recorder.run_file)
    events = read_run_events(recorder.events_file, from_start=True)

    assert summary["status"] == "running"
    assert "projection unavailable" in summary["projection_error"]
    assert events[-1]["type"] == "token_delta"
    assert events[-1]["text"] == "survived"


def test_render_run_status_includes_recent_text(tmp_path) -> None:
    recorder = start_run(
        profile=None,
        state_path=tmp_path / "gptty_state.json",
        command="send",
        conversation_ref="conv-1",
    )
    recorder.event("token_delta", text="hello")
    stdout = StringIO()

    render_run_status(
        summary=read_run_summary(recorder.run_file),
        events=read_run_events(recorder.events_file, from_start=True),
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert "gptty: conversation in progress" in output
    assert "Profile: local files" in output
    assert "Conversation: conv-1" in output
    assert "Assistant:" in output
    assert "hello" in output


def test_render_run_status_only_omits_text(tmp_path) -> None:
    recorder = start_run(
        profile=None,
        state_path=tmp_path / "gptty_state.json",
        command="send",
        conversation_ref="conv-1",
    )
    recorder.event("token_delta", text="hello")
    stdout = StringIO()

    render_run_status(
        summary=read_run_summary(recorder.run_file),
        events=read_run_events(recorder.events_file, from_start=True),
        stdout=stdout,
        status_only=True,
    )

    output = stdout.getvalue()
    assert "Status: running" in output
    assert "Assistant:" not in output
    assert "hello" not in output
