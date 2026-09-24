from __future__ import annotations

from chatgpt_web_adapter import ConversationTimeoutError, RequestError
from chatgpt_web_adapter.browser_owned_write_runtime import (
    WRITE_OUTCOME_UNKNOWN,
    BrowserOwnedWriteRuntimeError,
)

from gptty.turn_failure import TurnFailure, classify_turn_failure


def test_structured_429_does_not_depend_on_message_wording() -> None:
    failure = classify_turn_failure(
        RequestError(
            "localized upstream rejection",
            status_code=429,
            request_stage="conversation_stream",
        )
    )

    assert failure == TurnFailure(
        label="turn",
        status="rate-limited",
        message="ChatGPT rate-limited this turn before final completion.",
        source="structured",
        status_code=429,
        request_stage="conversation_stream",
    )


def test_structured_timeout_uses_exception_type_not_message() -> None:
    failure = classify_turn_failure(
        ConversationTimeoutError(
            "completely unrelated wording",
            timeout=17,
        )
    )

    assert failure.status == "failed"
    assert failure.source == "structured"
    assert "timed out" in failure.message


def test_post_submit_ambiguity_dominates_429_and_requires_reconciliation() -> None:
    error = BrowserOwnedWriteRuntimeError(
        "provider rejected after delegation",
        failure_kind=WRITE_OUTCOME_UNKNOWN,
        automatic_retry_allowed=False,
        manual_retry_safe_after_repair=False,
        write_may_have_been_submitted=True,
        reconciliation_required=True,
        request_stage="browser_owned_write",
        status_code=429,
    )

    failure = classify_turn_failure(error)

    assert failure.status == "unconfirmed"
    assert failure.source == "structured"
    assert failure.code == WRITE_OUTCOME_UNKNOWN
    assert failure.status_code == 429
    assert failure.write_may_have_been_submitted is True
    assert failure.reconciliation_required is True
    assert "reconcile" in failure.message
    assert failure.status != "rate-limited"


def test_structured_403_is_generic_failure_not_verification_guess() -> None:
    failure = classify_turn_failure(
        RequestError(
            "forbidden",
            status_code=403,
            request_stage="conversation_stream",
        )
    )

    assert failure.status == "failed"
    assert failure.source == "structured"
    assert failure.status_code == 403


def test_legacy_rate_limit_text_is_compatibility_only() -> None:
    failure = classify_turn_failure(RuntimeError("backend status=429: rate limited"))

    assert failure.status == "rate-limited"
    assert failure.source == "compat-text"
    assert failure.status_code is None


def test_legacy_verification_text_is_compatibility_only() -> None:
    failure = classify_turn_failure(RuntimeError("Please verify you are human"))

    assert failure.status == "blocked"
    assert failure.source == "compat-text"


def test_plain_unknown_exception_remains_generic() -> None:
    failure = classify_turn_failure(RuntimeError("something novel happened"))

    assert failure.status == "failed"
    assert failure.source == "generic"


def test_structured_reason_code_can_classify_conversation_state() -> None:
    class StructuredError(RuntimeError):
        reason_code = "conversation_unavailable"

    failure = classify_turn_failure(StructuredError("localized text"))

    assert failure.marker == (
        "chat",
        "unavailable",
        "This conversation is no longer available; continue in a new chat.",
    )
    assert failure.source == "structured"
    assert failure.code == "conversation_unavailable"
