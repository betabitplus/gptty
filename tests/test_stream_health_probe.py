from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "stream_health_probe.py"
SPEC = importlib.util.spec_from_file_location("stream_health_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


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
