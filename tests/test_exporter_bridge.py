from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gptty.exporter_bridge as bridge


def _completed(payload: dict, *, returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


def test_exporter_failure_redacts_sensitive_diagnostics(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "conversation.md"
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {},
            returncode=1,
            stderr=(
                "Authorization: Bearer secret-token "+ str(Path.home()) + "/auth_data.json"
            ),
        ),
    )

    with pytest.raises(bridge.ExporterBridgeError) as exc_info:
        bridge.export_persistent_conversation(
            "conversation-123",
            output,
            exporter_command=("chatgpt-export-one",),
        )

    message = str(exc_info.value)
    assert "secret-token" not in message
    assert str(Path.home()) not in message
    assert "[REDACTED]" in message
    assert "~/auth_data.json" in message
