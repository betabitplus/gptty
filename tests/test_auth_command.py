from __future__ import annotations

import json
from argparse import Namespace
from io import StringIO
from pathlib import Path
from typing import Any

import gptty.commands.auth as auth_command
from gptty.commands.auth import (
    run_auth_logout,
    run_auth_migrate,
    run_auth_refresh,
    run_auth_status,
)


async def fake_auth_runner(**kwargs: Any) -> None:
    fake_auth_runner.calls.append(kwargs)


fake_auth_runner.calls = []


def test_auth_status_returns_1_for_missing_file(tmp_path: Path) -> None:
    stdout = StringIO()

    code = run_auth_status(
        Namespace(auth=str(tmp_path / "missing.json"), format="plain"),
        stdout=stdout,
    )

    assert code == 1
    assert "status: missing" in stdout.getvalue()


def test_auth_status_prints_json(tmp_path: Path) -> None:
    path = tmp_path / "auth_data.json"
    path.write_text(json.dumps({"accessToken": "not-a-jwt"}), encoding="utf-8")
    stdout = StringIO()

    code = run_auth_status(Namespace(auth=str(path), format="json"), stdout=stdout)

    assert code == 0
    assert json.loads(stdout.getvalue())["status"] == "unknown-expiry"


def test_auth_refresh_calls_runner_with_cli_options(tmp_path: Path) -> None:
    fake_auth_runner.calls.clear()
    stdout = StringIO()

    code = run_auth_refresh(
        Namespace(
            auth=str(tmp_path / "auth_data.json"),
            mode="wait",
            timeout=42.0,
            ready_timeout=7.0,
            probe_prompt="Ping",
            credential_store="file",
        ),
        auth_runner=fake_auth_runner,
        stdout=stdout,
    )

    assert code == 0
    assert fake_auth_runner.calls == [
        {
            "output_file": str(tmp_path / "auth_data.json"),
            "auth_timeout": 42.0,
            "mode": "wait",
            "ready_timeout": 7.0,
            "probe_prompt": "Ping",
            "credential_store": "file",
        }
    ]
    assert "auth data refreshed" in stdout.getvalue()


def test_auth_refresh_reports_runner_failure(tmp_path: Path) -> None:
    async def failing_runner(**kwargs: Any) -> None:
        raise RuntimeError("Dependency 'g4f' is not installed")

    stderr = StringIO()

    code = run_auth_refresh(
        Namespace(
            auth=str(tmp_path / "auth_data.json"),
            mode="auto",
            timeout=120.0,
            ready_timeout=0.0,
            probe_prompt="Hello",
        ),
        auth_runner=failing_runner,
        stderr=stderr,
    )

    assert code == 1
    assert "auth refresh failed" in stderr.getvalue()
    assert "gptty-web[auth]" in stderr.getvalue()


def test_auth_migrate_delegates_to_cwa(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "auth_data.json"
    calls: dict[str, object] = {}

    def fake_migrate(auth_file, *, backend):
        calls["auth_file"] = Path(auth_file)
        calls["backend"] = backend

    monkeypatch.setattr(auth_command, "migrate_auth_data", fake_migrate)
    monkeypatch.setattr(
        auth_command,
        "get_auth_status",
        lambda _path: type("Status", (), {"credential_backend": "keyring"})(),
    )
    stdout = StringIO()

    code = run_auth_migrate(
        Namespace(auth=str(path), backend="keyring"),
        stdout=stdout,
    )

    assert code == 0
    assert calls == {"auth_file": path, "backend": "keyring"}
    assert "credential store: keyring" in stdout.getvalue()


def test_auth_logout_delegates_to_cwa(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "auth_data.json"
    calls: list[Path] = []
    monkeypatch.setattr(
        auth_command,
        "clear_auth_data",
        lambda auth_file: calls.append(Path(auth_file)) or True,
    )
    stdout = StringIO()

    code = run_auth_logout(Namespace(auth=str(path)), stdout=stdout)

    assert code == 0
    assert calls == [path]
    assert "reusable auth removed" in stdout.getvalue()


def test_auth_migrate_redacts_sensitive_backend_error(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "auth_data.json"
    sensitive = "SENSITIVE_" + "VALUE"

    def fail_migrate(*_args, **_kwargs):
        raise RuntimeError(f"{'access_' + 'token'}={sensitive}")

    monkeypatch.setattr(auth_command, "migrate_auth_data", fail_migrate)
    stderr = StringIO()

    code = run_auth_migrate(
        Namespace(auth=str(path), backend="keyring"),
        stderr=stderr,
    )

    assert code == 1
    assert sensitive not in stderr.getvalue()
    assert "[REDACTED]" in stderr.getvalue()
