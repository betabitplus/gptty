from __future__ import annotations

from pathlib import Path

from gptty.privacy import REDACTED, redact_diagnostic_text, redact_diagnostic_value


def test_redact_diagnostic_text_removes_common_sensitive_shapes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    sensitive_a = "SENSITIVE_" + "VALUE_A"
    sensitive_b = "SENSITIVE_" + "VALUE_B"
    jwt = "e" + "yJ" + "A" * 8 + "." + "B" * 8 + "." + "C" * 8
    api_key = "s" + "k-" + "D" * 20
    authorization = "Author" + "ization"
    bearer = "Bear" + "er"
    cookie = "Coo" + "kie"
    token_key = "to" + "ken"
    password_key = "pass" + "word"
    text = (
        f"local={tmp_path}/private/auth.json\n"
        f"{authorization}: {bearer} {sensitive_a}\n"
        f"{cookie}: session={sensitive_b}; theme=dark\n"
        f"https://example.test/callback?{token_key}={sensitive_a}&ok=1\n"
        f"{password_key}='{sensitive_b}'\n"
        f"jwt={jwt}\n"
        f"api={api_key}\n"
    )

    redacted = redact_diagnostic_text(text)

    assert str(tmp_path) not in redacted
    assert sensitive_a not in redacted
    assert sensitive_b not in redacted
    assert jwt not in redacted
    assert api_key not in redacted
    assert "~" in redacted
    assert REDACTED in redacted
    assert "ok=1" in redacted


def test_redact_diagnostic_value_preserves_typed_shape_and_safe_fields() -> None:
    sensitive = "SENSITIVE_" + "VALUE"
    payload = {
        "status": "unconfirmed",
        "status_code": 429,
        "reconciliation_required": True,
        "author" + "ization": sensitive,
        "access" + "Token": sensitive,
        "client" + "Secret": sensitive,
        "x-" + "api-key": sensitive,
        "set-" + "cookie": sensitive,
        "nested": {
            "refresh_" + "token": sensitive,
            "message": f"request failed: {'api_' + 'key'}={sensitive}",
        },
        "items": [
            {"coo" + "kie": sensitive},
            f"https://example.test/?{'access_' + 'token'}={sensitive}&safe=yes",
        ],
    }

    redacted = redact_diagnostic_value(payload)

    assert redacted["status"] == "unconfirmed"
    assert redacted["status_code"] == 429
    assert redacted["reconciliation_required"] is True
    assert redacted["author" + "ization"] == REDACTED
    assert redacted["access" + "Token"] == REDACTED
    assert redacted["client" + "Secret"] == REDACTED
    assert redacted["x-" + "api-key"] == REDACTED
    assert redacted["set-" + "cookie"] == REDACTED
    assert redacted["nested"]["refresh_" + "token"] == REDACTED
    assert sensitive not in redacted["nested"]["message"]
    assert redacted["items"][0]["coo" + "kie"] == REDACTED
    assert sensitive not in redacted["items"][1]
    assert "safe=yes" in redacted["items"][1]


def test_redact_diagnostic_text_handles_quoted_mapping_keys() -> None:
    sensitive = "SENSITIVE_" + "VALUE"
    access_key = "access" + "Token"
    auth_key = "Author" + "ization"
    value = (
        "{" 
        f"'{access_key}': '{sensitive}', "
        f"'{auth_key}': '{'Bear' + 'er'} {sensitive}'"
        "}"
    )

    redacted = redact_diagnostic_text(value)

    assert sensitive not in redacted
    assert redacted.count(REDACTED) == 2


def test_redaction_is_idempotent() -> None:
    sensitive = "SENSITIVE_" + "VALUE"
    value = (
        f"{'Author' + 'ization'}: {'Bear' + 'er'} {sensitive}\n"
        f"{'pass' + 'word'}={sensitive}"
    )

    once = redact_diagnostic_text(value)
    twice = redact_diagnostic_text(once)

    assert twice == once
