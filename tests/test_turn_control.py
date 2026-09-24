from __future__ import annotations

from types import SimpleNamespace

from gptty.turn_control import (
    StopOutcome,
    normalize_stop_outcome,
    request_stop_generation,
)


def test_normalize_stop_outcome_preserves_cwa_proof_metadata() -> None:
    outcome = normalize_stop_outcome(
        {
            "stopped": True,
            "conversationId": "conv-1",
            "provider": "wkwebview",
            "proof": "canonical_client_stopped",
            "streamStatus": "IS_STOP_REQUESTED",
        }
    )

    assert outcome == StopOutcome(
        stopped=True,
        conversation_ref="conv-1",
        provider="wkwebview",
        proof="canonical_client_stopped",
        stream_status="IS_STOP_REQUESTED",
    )


def test_normalize_stop_outcome_accepts_snake_case_object_contract() -> None:
    outcome = normalize_stop_outcome(
        SimpleNamespace(
            stopped=True,
            conversation_id="conv-2",
            provider="browser-native",
            proof="browser_stop_control",
            stream_status="COMPLETE",
        )
    )

    assert outcome.conversation_ref == "conv-2"
    assert outcome.provider == "browser-native"
    assert outcome.proof == "browser_stop_control"
    assert outcome.stream_status == "COMPLETE"


def test_normalize_stop_outcome_uses_fallback_only_after_confirmed_stop() -> None:
    confirmed = normalize_stop_outcome(
        {"stopped": True},
        fallback_conversation_ref="conv-fallback",
    )
    not_stopped = normalize_stop_outcome(
        {"stopped": False},
        fallback_conversation_ref="conv-fallback",
    )

    assert confirmed.conversation_ref == "conv-fallback"
    assert not_stopped.conversation_ref is None


def test_normalize_stop_outcome_does_not_promote_truthy_non_boolean_to_proof() -> None:
    outcome = normalize_stop_outcome(
        {"stopped": "true", "conversationId": "conv-untrusted"},
        fallback_conversation_ref="conv-fallback",
    )

    assert outcome.stopped is False
    assert outcome.conversation_ref == "conv-untrusted"


def test_request_stop_generation_forwards_one_request_and_normalizes() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[object, float]] = []

        def stop_generation(self, ref: object, *, timeout: float):
            self.calls.append((ref, timeout))
            return {
                "stopped": True,
                "conversationId": "conv-result",
                "provider": "wkwebview",
                "proof": "stream_status",
            }

    client = Client()
    outcome = request_stop_generation(client, "conv-input", timeout=30.0)

    assert client.calls == [("conv-input", 30.0)]
    assert outcome.stopped is True
    assert outcome.conversation_ref == "conv-result"
    assert outcome.proof == "stream_status"
