from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "stream_health_probe.py"
SPEC = importlib.util.spec_from_file_location("stream_health_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_classify_detects_browser_server_divergence_without_local_follower() -> None:
    report = {
        "server": {
            "unfinished": True,
            "turn_exchange_id": "turn-new",
            "current_turn_event_count": 9,
        },
        "browser": {
            "target_open": True,
            "delivery_timeout": True,
        },
        "active_stream": None,
        "process": None,
        "terminal": {"available": False},
    }

    assert probe._classify(report) == [
        "browser_delivery_timed_out_server_continues",
        "server_running_without_local_follower",
        "browser_server_timeline_diverged",
    ]


def test_classify_detects_stale_registry_and_terminal_finality_stall() -> None:
    report = {
        "server": {
            "unfinished": False,
            "turn_exchange_id": "turn-old",
            "current_turn_event_count": 0,
        },
        "browser": {
            "target_open": True,
            "delivery_timeout": False,
        },
        "active_stream": {"pid": 123},
        "process": {"pid": 123, "alive": False},
        "terminal": {
            "available": True,
            "last_status": "FINALITY UNCONFIRMED · answer text received",
        },
    }

    assert probe._classify(report) == [
        "stale_active_stream_registry",
        "terminal_finality_unconfirmed",
    ]



def test_browser_repair_guard_requires_read_only_reload_decision() -> None:
    report = {
        "decision": {
            "action": "reload_browser_read_only",
            "safe_to_send_new_turn": False,
        }
    }
    assert probe._should_repair_browser(report) is True

    report["decision"]["action"] = "wait"
    assert probe._should_repair_browser(report) is False



def test_decision_never_allows_resend_while_server_turn_is_unfinished() -> None:
    report = {
        "server": {"status": "running", "unfinished": True},
        "browser": {"target_open": True, "delivery_timeout": False},
        "active_stream": {"pid": 123},
    }

    decision = probe._decision(report)

    assert decision["action"] == "wait"
    assert decision["safe_to_send_new_turn"] is False
    assert decision["safe_to_stop_generation"] is False


def test_decision_prefers_read_only_recovery_over_new_turn() -> None:
    report = {
        "server": {"status": "tool_running", "unfinished": True},
        "browser": {
            "target_open": True,
            "delivery_timeout": True,
        },
        "active_stream": None,
    }

    decision = probe._decision(report)

    assert decision["action"] == "reload_browser_read_only"
    assert decision["safe_to_send_new_turn"] is False
    assert "do not click Retry" in decision["reason"]


def test_decision_allows_new_turn_only_after_canonical_completion() -> None:
    report = {
        "server": {"status": "completed", "unfinished": False},
        "browser": {},
        "active_stream": None,
    }

    decision = probe._decision(report)

    assert decision == {
        "action": "done",
        "safe_to_send_new_turn": True,
        "safe_to_stop_generation": False,
        "reason": "The observed turn is canonically completed.",
    }
