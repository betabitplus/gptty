from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable
import uuid

MAX_QUEUED_TURNS = 24
MAX_QUEUED_TEXT_CHARS = 256_000
MAX_QUEUED_MEDIA_ITEMS = 48
MAX_QUEUED_MEDIA_PATH_CHARS = 32_000


class QueueLimitError(ValueError):
    """Raised when accepting another local queued turn would exceed a bound."""


@dataclass(frozen=True)
class QueueBinding:
    conversation_ref: str | None
    conversation_mode: str
    model: str | None
    reasoning_effort: str | None = None
    goal_id: str | None = None
    goal_generation: int | None = None


@dataclass(frozen=True)
class QueuedTurn:
    turn_id: str
    text: str
    media: tuple[str, ...]
    binding: QueueBinding
    origin: str
    queued_at: str

    @property
    def media_count(self) -> int:
        return len(self.media)


class QueuedTurnQueue:
    """Memory-only queue of immutable user turns with explicit release semantics."""

    def __init__(
        self,
        *,
        max_turns: int = MAX_QUEUED_TURNS,
        max_text_chars: int = MAX_QUEUED_TEXT_CHARS,
        max_media_items: int = MAX_QUEUED_MEDIA_ITEMS,
        max_media_path_chars: int = MAX_QUEUED_MEDIA_PATH_CHARS,
    ) -> None:
        if min(max_turns, max_text_chars, max_media_items, max_media_path_chars) <= 0:
            raise ValueError("queue limits must be positive")
        self._items: deque[QueuedTurn] = deque()
        self.max_turns = int(max_turns)
        self.max_text_chars = int(max_text_chars)
        self.max_media_items = int(max_media_items)
        self.max_media_path_chars = int(max_media_path_chars)
        self._held_reason: str | None = None

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def held(self) -> bool:
        return bool(self._held_reason and self._items)

    @property
    def held_reason(self) -> str | None:
        return self._held_reason if self._items else None

    @property
    def text_chars(self) -> int:
        return sum(len(item.text) for item in self._items)

    @property
    def media_items(self) -> int:
        return sum(len(item.media) for item in self._items)

    @property
    def media_path_chars(self) -> int:
        return sum(len(path) for item in self._items for path in item.media)

    def items(self) -> tuple[QueuedTurn, ...]:
        return tuple(self._items)

    def enqueue(
        self,
        text: str,
        *,
        media: Iterable[str] = (),
        binding: QueueBinding,
        origin: str,
        turn_id: str | None = None,
        queued_at: str | None = None,
    ) -> QueuedTurn:
        normalized = str(text).strip()
        if not normalized:
            raise ValueError("queued turn text cannot be empty")
        media_items = tuple(str(item) for item in media)
        self._check_capacity(normalized, media_items)
        turn = QueuedTurn(
            turn_id=turn_id or uuid.uuid4().hex,
            text=normalized,
            media=media_items,
            binding=binding,
            origin=str(origin or "interactive"),
            queued_at=queued_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self._items.append(turn)
        return turn

    def peek(self) -> QueuedTurn | None:
        return self._items[0] if self._items else None

    def popleft(self) -> QueuedTurn:
        if self.held:
            raise RuntimeError("queued turns are held")
        return self._items.popleft()

    def hold(self, reason: str) -> bool:
        if not self._items:
            self._held_reason = None
            return False
        if self._held_reason:
            return False
        self._held_reason = str(reason).strip() or "explicit hold"
        return True

    def release(self, *, binding: QueueBinding | None = None) -> int:
        if binding is not None:
            self._items = deque(
                replace(item, binding=binding)
                for item in self._items
            )
        count = len(self._items)
        self._held_reason = None
        return count

    def remove(self, selector: str) -> QueuedTurn | None:
        value = str(selector).strip()
        if not value:
            return None
        index: int | None = None
        if value.isdigit():
            candidate = int(value)
            if 1 <= candidate <= len(self._items):
                index = candidate - 1
        if index is None:
            matches = [
                position
                for position, item in enumerate(self._items)
                if item.turn_id.startswith(value)
            ]
            if len(matches) != 1:
                return None
            index = matches[0]
        items = list(self._items)
        removed = items.pop(index)
        self._items = deque(items)
        if not self._items:
            self._held_reason = None
        return removed

    def clear(self) -> tuple[QueuedTurn, ...]:
        removed = tuple(self._items)
        self._items.clear()
        self._held_reason = None
        return removed

    def binding_matches(self, binding: QueueBinding) -> bool:
        return all(_binding_matches(item.binding, binding) for item in self._items)

    def _check_capacity(self, text: str, media: tuple[str, ...]) -> None:
        if len(self._items) >= self.max_turns:
            raise QueueLimitError(f"queue limit reached ({self.max_turns} turns)")
        if self.text_chars + len(text) > self.max_text_chars:
            raise QueueLimitError(
                f"queued text limit reached ({self.max_text_chars} characters)"
            )
        if self.media_items + len(media) > self.max_media_items:
            raise QueueLimitError(
                f"queued media limit reached ({self.max_media_items} attachments)"
            )
        path_chars = sum(len(item) for item in media)
        if self.media_path_chars + path_chars > self.max_media_path_chars:
            raise QueueLimitError(
                "queued media metadata limit reached "
                f"({self.max_media_path_chars} path characters)"
            )


def _binding_matches(accepted: QueueBinding, current: QueueBinding) -> bool:
    if accepted.conversation_mode != current.conversation_mode:
        return False
    if accepted.model != current.model:
        return False
    if accepted.reasoning_effort != current.reasoning_effort:
        return False
    if accepted.goal_id != current.goal_id:
        return False
    if accepted.goal_generation != current.goal_generation:
        return False
    if accepted.conversation_ref is None:
        # A prompt queued while the active first turn is creating a conversation may
        # legitimately observe the resulting conversation id at dispatch. Explicit
        # chat switches are blocked while the queue is non-empty.
        return True
    return accepted.conversation_ref == current.conversation_ref


__all__ = [
    "MAX_QUEUED_MEDIA_ITEMS",
    "MAX_QUEUED_MEDIA_PATH_CHARS",
    "MAX_QUEUED_TEXT_CHARS",
    "MAX_QUEUED_TURNS",
    "QueueBinding",
    "QueueLimitError",
    "QueuedTurn",
    "QueuedTurnQueue",
]
