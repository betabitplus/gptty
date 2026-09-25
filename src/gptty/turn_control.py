from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StopOutcome:
    """Normalized proof-bearing result of one ChatGPT stop request."""

    stopped: bool
    conversation_ref: str | None
    provider: str | None = None
    proof: str | None = None
    stream_status: str | None = None
    identity_verified: bool = False


def request_stop_generation(
    client: Any,
    conversation_ref: str | None,
    *,
    timeout: float,
) -> StopOutcome:
    """Issue one stop request and normalize the provider result without inference."""

    raw = client.stop_generation(conversation_ref, timeout=timeout)
    return normalize_stop_outcome(raw, fallback_conversation_ref=conversation_ref)


def normalize_stop_outcome(
    raw: Any,
    *,
    fallback_conversation_ref: str | None = None,
) -> StopOutcome:
    stopped = _field(raw, "stopped") is True
    conversation_ref = _optional_text(
        _field(raw, "conversationId", "conversation_id")
    )
    if stopped and conversation_ref is None:
        conversation_ref = _optional_text(fallback_conversation_ref)

    return StopOutcome(
        stopped=stopped,
        conversation_ref=conversation_ref,
        provider=_optional_text(_field(raw, "provider")),
        proof=_optional_text(_field(raw, "proof")),
        stream_status=_optional_text(_field(raw, "streamStatus", "stream_status")),
        identity_verified=(
            _field(
                raw,
                "conversationIdentityVerified",
                "conversation_identity_verified",
            )
            is True
        ),
    )


def _field(value: Any, *names: str) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
