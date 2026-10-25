from __future__ import annotations

from chatgpt_web_adapter import RequestError
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
