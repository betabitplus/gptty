from __future__ import annotations

from argparse import Namespace
from io import StringIO
from pathlib import Path

import gptty.commands.auth as auth_command
from gptty.commands.auth import run_auth_migrate


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
