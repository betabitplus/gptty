from __future__ import annotations

import json
from pathlib import Path

from auth_fetcher import AuthResult


def test_auth_result_delegates_to_cwa_reusable_auth_format(tmp_path: Path) -> None:
    path = tmp_path / "auth_data.json"
    result = AuthResult(
        api_key="token",
        cookies={"session": "yes"},
        headers={"user-agent": "test"},
        expires=None,
        proof_token=[1, 2, 3],
        turnstile_token="turnstile",
    )

    result.to_json(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["accessToken"] == "token"
    assert "api_key" not in data
    assert data["cookies"] == {"session": "yes"}
    assert data["headers"] == {"user-agent": "test"}
    assert "proof_token" not in data
    assert "turnstile_token" not in data


def test_auth_result_reads_access_token_first(tmp_path: Path) -> None:
    path = tmp_path / "auth_data.json"
    path.write_text(
        json.dumps({"accessToken": "new-token", "api_key": "old-token"}),
        encoding="utf-8",
    )

    result = AuthResult.from_json(path)

    assert result.api_key == "new-token"
    assert result.accessToken == "new-token"


def test_auth_result_missing_file_preserves_legacy_bootstrap_signal(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    try:
        AuthResult.from_json(path)
    except FileNotFoundError as error:
        assert str(path) in str(error)
    else:
        raise AssertionError("missing reusable auth must preserve FileNotFoundError")
