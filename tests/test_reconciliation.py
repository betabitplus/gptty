from __future__ import annotations

import pytest

from gptty.reconciliation import (
    ChatTerminalEvidence,
    chat_terminal_resolution,
    same_conversation_ref,
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
