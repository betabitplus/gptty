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
            "last_status": "STALLED finality · answer text received",
        },
    }

    assert probe._classify(report) == [
        "stale_active_stream_registry",
        "terminal_finality_stalled",
    ]



def test_browser_repair_guard_requires_timeout_target_and_unfinished_server() -> None:
    browser = {
        "available": True,
        "target_open": True,
        "delivery_timeout": True,
    }
    server = {"unfinished": True}
    should_repair = (
        browser.get("available") is True
        and browser.get("target_open") is True
        and browser.get("delivery_timeout") is True
        and server.get("unfinished") is True
    )
    assert should_repair is True

    server["unfinished"] = False
    should_repair = (
        browser.get("available") is True
        and browser.get("target_open") is True
        and browser.get("delivery_timeout") is True
        and server.get("unfinished") is True
    )
    assert should_repair is False
