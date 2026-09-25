from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chatgpt_web_adapter.types import ConversationRef


ChatTerminalMarker = tuple[str, str, str, str | None]
EvidenceKind = Literal["canonical-read", "terminal-turn", "stop"]


@dataclass(frozen=True)
class ChatTerminalEvidence:
    """Typed evidence that may supersede one persisted chat-level terminal marker."""

    kind: EvidenceKind
    source: str
    fresh: bool = False
    terminal_observed: bool | None = None
    terminal_error_present: bool = False
    same_conversation: bool = False
    current_chat_status: str | None = None
    stop_confirmed: bool = False
    proof_present: bool = False
    identity_verified: bool = False


@dataclass(frozen=True)
class ChatTerminalResolution:
    status: str
    source: str


def same_conversation_ref(left: str | None, right: str | None) -> bool:
    """Compare raw ids and supported ChatGPT conversation URLs by canonical id."""

    if not left or not right:
        return False
    try:
        return (
            ConversationRef.from_any(left).conversation_id
            == ConversationRef.from_any(right).conversation_id
        )
    except (TypeError, ValueError):
        return False


def stop_terminal_evidence(
    *,
    expected_conversation_ref: str | None,
    stopped: bool,
    stopped_conversation_ref: str | None,
    provider: str | None,
    proof: str | None,
    identity_verified: bool,
) -> ChatTerminalEvidence:
    provider_name = str(provider or "provider").strip() or "provider"
    proof_name = str(proof or "").strip()
    source = (
        f"stop:{provider_name}:{proof_name}"
        if proof_name
        else f"stop:{provider_name}:unproven"
    )
    return ChatTerminalEvidence(
        kind="stop",
        source=source,
        stop_confirmed=stopped is True,
        proof_present=bool(proof_name),
        identity_verified=identity_verified is True,
        same_conversation=same_conversation_ref(
            expected_conversation_ref,
            stopped_conversation_ref,
        ),
    )


def chat_terminal_resolution(
    marker: ChatTerminalMarker | None,
    evidence: ChatTerminalEvidence,
) -> ChatTerminalResolution | None:
    """Return a resolution only when newer evidence strictly disproves the marker."""

    if marker is None:
        return None

    label = str(marker[0] or "").strip().lower()
    status = str(marker[1] or "").strip().lower()
    source = str(evidence.source or "").strip()
    current_chat_status = str(evidence.current_chat_status or "").strip().lower()

    if label != "chat" or not status or not source:
        return None

    # A current chat-level terminal condition is stronger than any attempt to
    # supersede a historical one. It may be a recurrence or a different issue.
    if current_chat_status:
        return None

    if evidence.kind == "canonical-read":
        # A fresh successful read proves that a previously unavailable
        # conversation is readable again. It does not prove that a chat which
        # reached its write/length limit is writable again.
        if status != "unavailable" or not evidence.fresh:
            return None
        return ChatTerminalResolution(status=status, source=source)

    if evidence.kind == "terminal-turn":
        # A proof-bearing terminal result from a new accepted turn in this exact
        # conversation proves the chat accepted a write. Never derive that from
        # text arrival, a mismatched route, or a response carrying terminal error
        # evidence of its own.
        if status not in {"unavailable", "limit-reached"}:
            return None
        if evidence.terminal_observed is not True:
            return None
        if evidence.terminal_error_present:
            return None
        if not evidence.same_conversation:
            return None
        return ChatTerminalResolution(status=status, source=source)

    if evidence.kind == "stop":
        # A provider-confirmed Stop proves a live turn existed, but only when the
        # provider also supplies explicit stop proof and verified conversation
        # identity. Clicking a control on an unresolved/mismatched route is not
        # authority to mutate persisted truth for a named conversation.
        if status not in {"unavailable", "limit-reached"}:
            return None
        if not evidence.stop_confirmed or not evidence.proof_present:
            return None
        if not evidence.identity_verified or not evidence.same_conversation:
            return None
        return ChatTerminalResolution(status=status, source=source)

    return None
