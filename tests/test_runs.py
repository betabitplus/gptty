from __future__ import annotations

from gptty.runs import read_run_events, read_run_summary, start_run


def test_run_recorder_redacts_sensitive_failure_and_event_data(tmp_path) -> None:
    sensitive = "SENSITIVE_" + "VALUE"
    recorder = start_run(
        profile=None,
        state_path=tmp_path / "gptty_state.json",
        command="send",
        conversation_ref="conv-1",
    )

    event = recorder.event(
        "provider_event",
        message=f"provider {'api_' + 'key'}={sensitive}",
        nested={"refresh_" + "token": sensitive},
    )
    recorder.fail(
        f"request failed: {'pass' + 'word'}={sensitive}",
        traceback_text=(
            f"Traceback: {'Author' + 'ization'}: {'Bear' + 'er'} {sensitive}\n"
        ),
        failure_classification={
            "status": "unconfirmed",
            "reconciliation_required": True,
            "coo" + "kie": sensitive,
        },
    )

    summary = read_run_summary(recorder.run_file)
    events = read_run_events(recorder.events_file, from_start=True)

    assert sensitive not in str(event)
    assert event["nested"]["refresh_" + "token"] == "[REDACTED]"
    assert summary["status"] == "failed"
    assert summary["failure_classification"]["status"] == "unconfirmed"
    assert summary["failure_classification"]["reconciliation_required"] is True
    assert summary["failure_classification"]["coo" + "kie"] == "[REDACTED]"
    assert sensitive not in summary["error"]
    assert sensitive not in summary["traceback"]
    assert sensitive not in str(events)
