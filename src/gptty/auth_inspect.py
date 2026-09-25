from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatgpt_web_adapter import get_auth_status

from .privacy import redact_diagnostic_text


def inspect_auth_file(path: str | Path) -> dict[str, Any]:
    auth_path = Path(path)
    status: dict[str, Any] = {
        "auth_file": str(auth_path),
        "exists": auth_path.exists(),
        "readable": False,
        "status": "missing",
        "ok": False,
        "error": None,
        "timestamp": None,
        "token_source": None,
        "has_token": False,
        "expires_at": None,
        "expires_in_seconds": None,
        "expired": None,
        "has_cookies": False,
        "has_headers": False,
        "has_proof_token": False,
        "has_turnstile_token": False,
        "credential_backend": "file",
        "credential_metadata_present": False,
        "keyring_available": False,
        "keyring_backend": None,
    }

    try:
        authority = get_auth_status(auth_path)
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary must stay secret-safe.
        status["exists"] = auth_path.exists()
        status["status"] = "invalid" if status["exists"] else "missing"
        status["error"] = redact_diagnostic_text(str(exc))
        return status

    status["exists"] = authority.file_exists
    status["readable"] = bool(
        authority.file_exists
        or authority.access_token_present
        or authority.session_cookie_present
        or authority.credential_backend == "keyring"
    )
    status["timestamp"] = authority.captured_at
    status["token_source"] = (
        authority.credential_backend if authority.access_token_present else None
    )
    status["has_token"] = authority.access_token_present
    status["has_cookies"] = bool(
        authority.cookies_present
        or authority.session_cookie_present
        or authority.browser_cookie_count
    )
    status["has_headers"] = authority.headers_present
    status["has_proof_token"] = authority.proof_token_present
    status["has_turnstile_token"] = authority.turnstile_token_present
    status["credential_backend"] = authority.credential_backend
    status["credential_metadata_present"] = authority.credential_metadata_present
    status["keyring_available"] = authority.keyring_available
    status["keyring_backend"] = authority.keyring_backend

    if not authority.access_token_present:
        if not authority.file_exists and not authority.session_cookie_present:
            status["status"] = "missing"
            status["error"] = "no reusable authorization material found"
        else:
            status["status"] = "missing-token"
            status["error"] = "no reusable access token found"
        return status

    expiry = authority.access_token_expires_at
    if expiry is None:
        status["status"] = "unknown-expiry"
        status["ok"] = True
        return status

    now = datetime.now(timezone.utc)
    expires_in = int((expiry - now).total_seconds())
    expired = expires_in <= 0
    status["expires_at"] = expiry.isoformat().replace("+00:00", "Z")
    status["expires_in_seconds"] = expires_in
    status["expired"] = expired
    status["ok"] = not expired
    status["status"] = "expired" if expired else "ok"
    if expired:
        status["error"] = "access token is expired"
    return status


def render_auth_status(status: dict[str, Any], output_format: str = "plain") -> str:
    if output_format == "json":
        return json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True)
    if output_format != "plain":
        raise ValueError(f"Unsupported auth status output format: {output_format}")
    return _render_plain_status(status)


def _render_plain_status(status: dict[str, Any]) -> str:
    lines = [
        f"auth file: {status['auth_file']}",
        f"status: {status['status']}",
        f"token: {_present(status['has_token'])}",
        f"credential store: {status.get('credential_backend', 'file')}",
    ]
    if status.get("token_source"):
        lines.append(f"token source: {status['token_source']}")
    if status.get("expires_at"):
        lines.append(f"expires at: {status['expires_at']}")
    if status.get("expires_in_seconds") is not None:
        lines.append(f"expires in: {_format_duration(int(status['expires_in_seconds']))}")
    elif status.get("has_token"):
        lines.append("expires in: unknown")
    lines.extend(
        [
            f"cookies: {_present(status['has_cookies'])}",
            f"headers: {_present(status['has_headers'])}",
            f"proof token: {_present(status['has_proof_token'])}",
            f"turnstile token: {_present(status['has_turnstile_token'])}",
        ]
    )
    if status.get("timestamp"):
        lines.append(f"captured at: {status['timestamp']}")
    if status.get("error"):
        lines.append(f"error: {status['error']}")
    if not status.get("ok"):
        lines.append("next step: run `gptty auth refresh --mode wait`")
    return "\n".join(lines)


def _present(value: Any) -> str:
    return "present" if value else "missing"


def _format_duration(seconds: int) -> str:
    sign = "-" if seconds < 0 else ""
    remaining = abs(seconds)
    days, remaining = divmod(remaining, 86400)
    hours, remaining = divmod(remaining, 3600)
    minutes, _ = divmod(remaining, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return sign + " ".join(parts)
