from __future__ import annotations

from typing import Any

import pytest

import gptty.commands.auth as auth_command
from gptty import cli


def test_auth_status_routes_to_auth_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_run_auth_status(args: Any) -> int:
        calls["auth"] = args.auth
        calls["format"] = args.format
        return 0

    monkeypatch.setattr(auth_command, "run_auth_status", fake_run_auth_status)

    assert cli.main(["auth", "status", "--auth", "custom.json", "--format", "json"]) == 0
    assert calls == {"auth": "custom.json", "format": "json"}


def test_auth_refresh_routes_to_auth_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_run_auth_refresh(args: Any) -> int:
        calls["auth"] = args.auth
        calls["mode"] = args.mode
        calls["timeout"] = args.timeout
        calls["ready_timeout"] = args.ready_timeout
        calls["probe_prompt"] = args.probe_prompt
        calls["credential_store"] = args.credential_store
        return 0

    monkeypatch.setattr(auth_command, "run_auth_refresh", fake_run_auth_refresh)

    assert cli.main([
        "auth",
        "refresh",
        "--auth",
        "custom.json",
        "--mode",
        "wait",
        "--timeout",
        "42",
        "--ready-timeout",
        "7",
        "--probe-prompt",
        "Ping",
        "--credential-store",
        "file",
    ]) == 0
    assert calls == {
        "auth": "custom.json",
        "mode": "wait",
        "timeout": 42.0,
        "ready_timeout": 7.0,
        "probe_prompt": "Ping",
        "credential_store": "file",
    }


def test_auth_without_subcommand_returns_2() -> None:
    assert cli.main(["auth"]) == 2


def test_auth_migrate_routes_to_auth_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_run_auth_migrate(args: Any) -> int:
        calls["auth"] = args.auth
        calls["backend"] = args.backend
        return 0

    monkeypatch.setattr(auth_command, "run_auth_migrate", fake_run_auth_migrate)

    assert cli.main([
        "auth",
        "migrate",
        "--auth",
        "custom.json",
        "--backend",
        "file",
    ]) == 0
    assert calls == {"auth": "custom.json", "backend": "file"}


def test_auth_logout_routes_to_auth_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_run_auth_logout(args: Any) -> int:
        calls["auth"] = args.auth
        return 0

    monkeypatch.setattr(auth_command, "run_auth_logout", fake_run_auth_logout)

    assert cli.main(["auth", "logout", "--auth", "custom.json"]) == 0
    assert calls == {"auth": "custom.json"}
