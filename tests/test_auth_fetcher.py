from __future__ import annotations

import json
from datetime import datetime, timezone

import auth_fetcher


def test_auth_result_timestamp_is_owned_by_cwa_and_is_utc(tmp_path) -> None:
    target = tmp_path / "auth_data.json"
    auth_fetcher.AuthResult("token", {}, {}, 0, None, None).to_json(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["timestamp"].endswith("Z")
    parsed = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)
