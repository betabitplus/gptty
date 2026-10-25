from __future__ import annotations

from pathlib import Path

from gptty.privacy import REDACTED, redact_diagnostic_text


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
