from __future__ import annotations

import pytest

from gptty.reconciliation import (
    ChatTerminalEvidence,
    chat_terminal_resolution,
    same_conversation_ref,
    stop_terminal_evidence,
)


UNAVAILABLE = (
    "chat",
    "unavailable",
    "This conversation is unavailable.",
    "stream",
)
LIMIT_REACHED = (
    "chat",
    "limit-reached",
    "This conversation reached its maximum length.",
    "stream",
)


def test_same_conversation_ref_normalizes_supported_chatgpt_url() -> None:
    conversation_id = "conv-12345678"
    assert same_conversation_ref(
        conversation_id,
        f"https://chatgpt.com/c/{conversation_id}",
    )
    assert same_conversation_ref(
        f"https://chat.openai.com/c/{conversation_id}",
        conversation_id,
    )
    assert not same_conversation_ref(conversation_id, "conv-other")
    assert not same_conversation_ref(conversation_id, "https://example.com/c/conv-12345678")


@pytest.mark.parametrize(
    ("marker", "evidence", "expected_status"),
    [
        (
            UNAVAILABLE,
            ChatTerminalEvidence(
                kind="canonical-read",
                source="canonical-read",
                fresh=True,
            ),
            "unavailable",
        ),
        (
            UNAVAILABLE,
            ChatTerminalEvidence(
                kind="canonical-read",
                source="canonical-read",
                fresh=False,
            ),
            None,
        ),
        (
            LIMIT_REACHED,
            ChatTerminalEvidence(
                kind="canonical-read",
                source="canonical-read",
                fresh=True,
            ),
            None,
        ),
        (
            UNAVAILABLE,
            ChatTerminalEvidence(
                kind="canonical-read",
                source="canonical-read",
                fresh=True,
                current_chat_status="unavailable",
            ),
            None,
        ),
        (
            LIMIT_REACHED,
            ChatTerminalEvidence(
                kind="terminal-turn",
                source="browser-stream",
                terminal_observed=True,
                same_conversation=True,
            ),
            "limit-reached",
        ),
        (
            UNAVAILABLE,
            ChatTerminalEvidence(
                kind="terminal-turn",
                source="browser-stream",
                terminal_observed=True,
                same_conversation=True,
            ),
            "unavailable",
        ),
        (
            LIMIT_REACHED,
            ChatTerminalEvidence(
                kind="terminal-turn",
                source="browser-stream",
                terminal_observed=False,
                same_conversation=True,
            ),
            None,
        ),
        (
            LIMIT_REACHED,
            ChatTerminalEvidence(
                kind="terminal-turn",
                source="browser-stream",
                terminal_observed=True,
                same_conversation=False,
            ),
            None,
        ),
        (
            LIMIT_REACHED,
            ChatTerminalEvidence(
                kind="terminal-turn",
                source="browser-stream",
                terminal_observed=True,
                terminal_error_present=True,
                same_conversation=True,
            ),
            None,
        ),
        (
            LIMIT_REACHED,
            ChatTerminalEvidence(
                kind="terminal-turn",
                source="browser-stream",
                terminal_observed=True,
                same_conversation=True,
                current_chat_status="limit-reached",
            ),
            None,
        ),
    ],
)
def test_chat_terminal_resolution_requires_stronger_noncontradictory_evidence(
    marker,
    evidence,
    expected_status,
) -> None:
    resolution = chat_terminal_resolution(marker, evidence)
    assert (resolution.status if resolution is not None else None) == expected_status


def test_chat_terminal_resolution_rejects_non_chat_marker() -> None:
    resolution = chat_terminal_resolution(
        ("turn", "unconfirmed", "No terminal proof.", "stream"),
        ChatTerminalEvidence(
            kind="terminal-turn",
            source="browser-stream",
            terminal_observed=True,
            same_conversation=True,
        ),
    )
    assert resolution is None


def test_stop_terminal_evidence_requires_verified_same_conversation_identity() -> None:
    evidence = stop_terminal_evidence(
        expected_conversation_ref="conv-1",
        stopped=True,
        stopped_conversation_ref="https://chatgpt.com/c/conv-1",
        provider="browser-native",
        proof="browser_stop_control",
        identity_verified=True,
    )

    assert evidence.kind == "stop"
    assert evidence.source == "stop:browser-native:browser_stop_control"
    assert evidence.stop_confirmed is True
    assert evidence.proof_present is True
    assert evidence.identity_verified is True
    assert evidence.same_conversation is True
    assert chat_terminal_resolution(LIMIT_REACHED, evidence) is not None


@pytest.mark.parametrize(
    ("stopped", "proof", "identity_verified", "stopped_ref"),
    [
        (False, "browser_stop_control", True, "conv-1"),
        (True, None, True, "conv-1"),
        (True, "browser_stop_control", False, "conv-1"),
        (True, "browser_stop_control", True, "conv-other"),
    ],
)
def test_stop_terminal_evidence_fails_closed_without_complete_proof(
    stopped,
    proof,
    identity_verified,
    stopped_ref,
) -> None:
    evidence = stop_terminal_evidence(
        expected_conversation_ref="conv-1",
        stopped=stopped,
        stopped_conversation_ref=stopped_ref,
        provider="browser-native",
        proof=proof,
        identity_verified=identity_verified,
    )

    assert chat_terminal_resolution(UNAVAILABLE, evidence) is None
    assert chat_terminal_resolution(LIMIT_REACHED, evidence) is None
