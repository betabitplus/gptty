from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import gptty.commands.send as send_command
from gptty import cli


@pytest.fixture
def isolated_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GPTTY_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("GPTTY_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("GPTTY_PROFILE", raising=False)
    return tmp_path


def test_send_uses_environment_profile(
    isolated_profiles: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setenv("GPTTY_PROFILE", "work")

    def fake_read_stdin_text(mode: str) -> None:
        return None

    def fake_run_send(args: Any, *, stdin_text: str | None = None) -> int:
        calls["auth"] = args.auth
        calls["state"] = args.state
        calls["profile"] = args.profile
        return 0

    monkeypatch.setattr(cli, "read_stdin_text", fake_read_stdin_text)
    monkeypatch.setattr(send_command, "run_send", fake_run_send)

    assert cli.main(["send", "--new", "hello"]) == 0

    assert calls == {
        "auth": str(
            isolated_profiles / "data" / "profiles" / "work" / "auth_data.json"
        ),
        "state": str(
            isolated_profiles / "data" / "profiles" / "work" / "gptty_state.json"
        ),
        "profile": "work",
    }
