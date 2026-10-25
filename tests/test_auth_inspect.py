from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gptty.auth_inspect import inspect_auth_file


def make_token(exp: datetime | None) -> str:
    header = _b64({"alg": "none"})
    payload = {} if exp is None else {"exp": int(exp.timestamp())}
    return f"{header}.{_b64(payload)}.sig"


def _b64(payload: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def test_inspect_auth_file_reports_ok_token(tmp_path: Path) -> None:
    path = tmp_path / "auth_data.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "accessToken": make_token(datetime.now(timezone.utc) + timedelta(hours=2)),
                "cookies": {"session": "yes"},
                "headers": {"user-agent": "test"},
                "proof_token": [1, 2, 3],
                "turnstile_token": "turnstile",
            }
        ),
        encoding="utf-8",
    )

    status = inspect_auth_file(path)

    assert status["status"] == "ok"
    assert status["ok"] is True
    assert status["token_source"] == "file"
    assert status["expired"] is False
    assert status["has_cookies"] is True
    assert status["has_headers"] is True
    assert status["has_proof_token"] is True
    assert status["has_turnstile_token"] is True


def test_inspect_auth_file_reports_expired_token(tmp_path: Path) -> None:
    path = tmp_path / "auth_data.json"
    path.write_text(
        json.dumps({"api_key": make_token(datetime.now(timezone.utc) - timedelta(hours=1))}),
        encoding="utf-8",
    )

    status = inspect_auth_file(path)

    assert status["status"] == "expired"
    assert status["ok"] is False
    assert status["token_source"] == "file"
    assert status["expired"] is True
    assert status["error"] == "access token is expired"
