from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class StateError(RuntimeError):
    """Raised when a gptty state file cannot be loaded or saved."""


@dataclass
class GoalState:
    conversation_ref: str | None = None
    status: str = "paused"
    objective: str | None = None
    turn_count: int = 0
    protocol_failures: int = 0
    reason: str | None = None


@dataclass
class ChatState:
    current_conversation: str | None = None
    model: str | None = None
    goal: GoalState | None = None


def default_chat_state() -> ChatState:
    return ChatState()


def load_chat_state(path: str | Path) -> ChatState:
    state_path = Path(path)
    if not state_path.exists():
        return default_chat_state()

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"failed to load state from {state_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise StateError(f"failed to load state from {state_path}: expected JSON object")

    return ChatState(
        current_conversation=_optional_str(data.get("current_conversation")),
        model=_optional_str(data.get("model")),
        goal=_goal_state(data.get("goal")),
    )


def save_chat_state(path: str | Path, state: ChatState) -> None:
    state_path = Path(path)
    tmp_path = state_path.with_name(f".{state_path.name}.tmp")
    data = asdict(state)
    if data.get("goal") is None:
        data.pop("goal", None)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"

    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(state_path)
    except OSError as exc:
        raise StateError(f"failed to save state to {state_path}: {exc}") from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _goal_state(value: Any) -> GoalState | None:
    if not isinstance(value, dict):
        return None
    status = _optional_str(value.get("status")) or "paused"
    if status not in {"active", "paused", "blocked", "complete", "interrupted"}:
        status = "paused"
    return GoalState(
        conversation_ref=_optional_str(value.get("conversation_ref")),
        status=status,
        objective=_optional_str(value.get("objective")),
        turn_count=_non_negative_int(value.get("turn_count")),
        protocol_failures=_non_negative_int(value.get("protocol_failures")),
        reason=_optional_str(value.get("reason")),
    )


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
