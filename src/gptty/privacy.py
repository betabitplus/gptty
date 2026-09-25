from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEY_EXACT = {"authorization", "cookie", "setcookie"}
_SENSITIVE_KEY_SUFFIXES = (
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "apikey",
    "clientsecret",
    "password",
    "passwd",
    "sessiontoken",
    "csrftoken",
)


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    if normalized in _SENSITIVE_KEY_EXACT:
        return True
    return any(normalized.endswith(suffix) for suffix in _SENSITIVE_KEY_SUFFIXES)

_AUTH_HEADER_RE = re.compile(
    r"(?im)\b(authorization\s*:\s*)(?:bearer\s+)?[^\s,;]+"
)
_COOKIE_HEADER_RE = re.compile(r"(?im)\b((?:set-)?cookie\s*:\s*)[^\r\n]+")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:access_token|refresh_token|id_token|api_key|apikey|key|token|auth|"
    r"session_token|csrf_token)=)[^&#\s]+"
)
_ASSIGNMENT_SECRET_RE = re.compile(
    r"""(?ix)
    \b(authorization|cookie|set[_-]?cookie|access[_-]?token|refresh[_-]?token|
       id[_-]?token|api[_-]?key|apikey|client[_-]?secret|password|passwd|
       session[_-]?token|csrf[_-]?token)
    (["']?)
    (\s*[:=]\s*)
    (?:
        "([^"]*)"
        |
        '([^']*)'
        |
        ([^\s,;&]+)
    )
    """
)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")


def _redact_assignment(match: re.Match[str]) -> str:
    key = match.group(1)
    closing_quote = match.group(2)
    separator = match.group(3)
    if match.group(4) is not None:
        return f'{key}{closing_quote}{separator}"{REDACTED}"'
    if match.group(5) is not None:
        return f"{key}{closing_quote}{separator}'{REDACTED}'"
    return f"{key}{closing_quote}{separator}{REDACTED}"


def redact_diagnostic_text(value: str) -> str:
    """Redact credential-like material from diagnostic/support text."""

    text = str(value)
    try:
        home = str(Path.home())
    except RuntimeError:
        home = ""
    if home and home != "/" and home in text:
        text = text.replace(home, "~")
    text = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _COOKIE_HEADER_RE.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _QUERY_SECRET_RE.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _ASSIGNMENT_SECRET_RE.sub(_redact_assignment, text)
    text = _JWT_RE.sub(REDACTED, text)
    text = _API_KEY_RE.sub(REDACTED, text)
    return text


def redact_diagnostic_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact diagnostics while preserving typed structure."""

    if key and _is_sensitive_key(key):
        if value is None:
            return None
        return REDACTED
    if isinstance(value, str):
        return redact_diagnostic_text(value)
    if isinstance(value, dict):
        return {
            item_key: redact_diagnostic_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_diagnostic_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_diagnostic_value(item) for item in value)
    return value
