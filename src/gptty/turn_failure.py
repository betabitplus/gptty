from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from chatgpt_web_adapter import ConversationTimeoutError, RequestError


@dataclass(frozen=True)
class TurnFailure:
    label: str
    status: str
    message: str
    source: str
    code: str | None = None
    status_code: int | None = None
    request_stage: str | None = None
    write_may_have_been_submitted: bool = False
    reconciliation_required: bool = False

    @property
    def marker(self) -> tuple[str, str, str]:
        return (self.label, self.status, self.message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CONVERSATION_LIMIT_CODES = {
    "conversation_too_large",
    "conversation_limit_exceeded",
}
_CONVERSATION_UNAVAILABLE_CODES = {
    "conversation_unavailable",
    "conversation_not_found",
    "conversation_missing",
}
_VERIFICATION_CODES = {
    "turnstile_required",
    "verification_required",
    "human_verification_required",
}


def classify_turn_failure(error: BaseException) -> TurnFailure:
    """Classify one failed turn, preferring provider-owned structured evidence."""

    status_code = _optional_int(getattr(error, "status_code", None))
    request_stage = _optional_text(getattr(error, "request_stage", None))
    failure_kind = _optional_text(getattr(error, "failure_kind", None))
    reason_code = _optional_text(getattr(error, "reason_code", None))
    terminal_error_code = _optional_text(getattr(error, "terminal_error_code", None))
    write_may_have_been_submitted = (
        getattr(error, "write_may_have_been_submitted", None) is True
    )
    reconciliation_required = (
        getattr(error, "reconciliation_required", None) is True
    )
    structured_code = (
        terminal_error_code
        or reason_code
        or failure_kind
    )
    normalized_code = (structured_code or "").strip().lower()

    # Once the provider says a write may have crossed the commit point, that
    # ambiguity dominates secondary HTTP/status details. Retrying from an HTTP
    # code alone could duplicate the turn.
    if reconciliation_required or write_may_have_been_submitted:
        return TurnFailure(
            label="turn",
            status="unconfirmed",
            message=(
                "ChatGPT may have accepted this turn, but final completion was not "
                "confirmed; reconcile the conversation before retrying."
            ),
            source="structured",
            code=structured_code,
            status_code=status_code,
            request_stage=request_stage,
            write_may_have_been_submitted=write_may_have_been_submitted,
            reconciliation_required=reconciliation_required,
        )

    if normalized_code in _CONVERSATION_LIMIT_CODES:
        return TurnFailure(
            label="chat",
            status="limit-reached",
            message=(
                "This conversation reached its length limit; start a new chat to continue."
            ),
            source="structured",
            code=structured_code,
            status_code=status_code,
            request_stage=request_stage,
        )

    if normalized_code in _CONVERSATION_UNAVAILABLE_CODES:
        return TurnFailure(
            label="chat",
            status="unavailable",
            message="This conversation is no longer available; continue in a new chat.",
            source="structured",
            code=structured_code,
            status_code=status_code,
            request_stage=request_stage,
        )

    if normalized_code in _VERIFICATION_CODES:
        return TurnFailure(
            label="turn",
            status="blocked",
            message="ChatGPT requires browser verification before this turn can continue.",
            source="structured",
            code=structured_code,
            status_code=status_code,
            request_stage=request_stage,
        )

    if status_code == 429:
        return TurnFailure(
            label="turn",
            status="rate-limited",
            message="ChatGPT rate-limited this turn before final completion.",
            source="structured",
            code=structured_code,
            status_code=status_code,
            request_stage=request_stage,
        )

    if isinstance(error, ConversationTimeoutError):
        return TurnFailure(
            label="turn",
            status="failed",
            message=(
                "Response transport timed out before a final assistant completion "
                "was confirmed."
            ),
            source="structured",
            code=structured_code,
            status_code=status_code,
            request_stage=request_stage,
        )

    # RequestError is a typed provider failure even when it lacks a semantic
    # category. Do not invent one from status codes such as 403/404: those can
    # mean transient readback/protection states as well as durable product state.
    if isinstance(error, RequestError):
        compatibility = _classify_compat_text(str(error))
        if compatibility is not None:
            return TurnFailure(
                label=compatibility[0],
                status=compatibility[1],
                message=compatibility[2],
                source="compat-text",
                code=structured_code,
                status_code=status_code,
                request_stage=request_stage,
            )
        return TurnFailure(
            label="turn",
            status="failed",
            message="ChatGPT request ended with an error before final completion.",
            source="structured",
            code=structured_code,
            status_code=status_code,
            request_stage=request_stage,
        )

    compatibility = _classify_compat_text(str(error))
    if compatibility is not None:
        return TurnFailure(
            label=compatibility[0],
            status=compatibility[1],
            message=compatibility[2],
            source="compat-text",
        )

    return TurnFailure(
        label="turn",
        status="failed",
        message="ChatGPT request ended with an error before final completion.",
        source="generic",
    )


def _classify_compat_text(message: str) -> tuple[str, str, str] | None:
    """Compatibility-only classification for legacy/untyped failure paths."""

    normalized = str(message or "").strip().casefold()
    if any(
        token in normalized
        for token in (
            "maximum length",
            "max conversation",
            "conversation too long",
            "conversation length",
            "conversation_limit_exceeded",
            "conversation limit exceeded",
            "start a new chat",
            "new chat to continue",
            "context length",
        )
    ):
        return (
            "chat",
            "limit-reached",
            "This conversation reached its length limit; start a new chat to continue.",
        )
    if any(
        token in normalized
        for token in (
            "conversation not found",
            "conversation unavailable",
            "unable to load conversation",
            "could not load conversation",
            "chat not found",
        )
    ):
        return (
            "chat",
            "unavailable",
            "This conversation is no longer available; continue in a new chat.",
        )
    if "429" in normalized or "rate limit" in normalized:
        return (
            "turn",
            "rate-limited",
            "ChatGPT rate-limited this turn before final completion.",
        )
    if any(
        token in normalized
        for token in ("turnstile", "verify you are human", "verification")
    ):
        return (
            "turn",
            "blocked",
            "ChatGPT requires browser verification before this turn can continue.",
        )
    if "handoff" in normalized and any(
        token in normalized for token in ("final", "completed", "recovery")
    ):
        return (
            "turn",
            "unconfirmed",
            "Stream handoff did not reach a confirmed final assistant completion.",
        )
    if "timeout" in normalized or "timed out" in normalized:
        return (
            "turn",
            "failed",
            "Response transport timed out before a final assistant completion was confirmed.",
        )
    return None


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
