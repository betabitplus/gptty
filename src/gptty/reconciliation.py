from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chatgpt_web_adapter.types import ConversationRef


ChatTerminalMarker = tuple[str, str, str, str | None]
EvidenceKind = Literal["canonical-read", "terminal-turn"]


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

    return None
