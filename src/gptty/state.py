from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class StateError(RuntimeError):
    """Raised when a gptty state file cannot be loaded or saved."""


@dataclass
class GoalCheckpoint:
    summary: str | None = None
    completed: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    next_step: str | None = None
    updated_turn: int = 0


@dataclass
class GoalState:
    goal_id: str | None = None
    conversation_ref: str | None = None
    conversations: list[str] = field(default_factory=list)
    context_seed: list[str] = field(default_factory=list)
    status: str = "paused"
    objective: str | None = None
    turn_count: int = 0
    protocol_failures: int = 0
    recovery_count: int = 0
    rollover_count: int = 0
    checkpoint: GoalCheckpoint = field(default_factory=GoalCheckpoint)
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
        raise StateError(
            f"failed to load state from {state_path}: expected JSON object"
        )

    return ChatState(
        current_conversation=_optional_str(data.get("current_conversation")),
        model=_optional_str(data.get("model")),
        goal=goal_state_from_dict(data.get("goal")),
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


def goal_state_from_dict(value: Any) -> GoalState | None:
    if not isinstance(value, dict):
        return None
    status = _optional_str(value.get("status")) or "paused"
    if status not in {"active", "paused", "blocked", "complete", "interrupted"}:
        status = "paused"
    return GoalState(
        goal_id=_optional_str(value.get("goal_id")),
        conversation_ref=_optional_str(value.get("conversation_ref")),
        conversations=_string_list(value.get("conversations")),
        context_seed=_string_list(value.get("context_seed")),
        status=status,
        objective=_optional_str(value.get("objective")),
        turn_count=_non_negative_int(value.get("turn_count")),
        protocol_failures=_non_negative_int(value.get("protocol_failures")),
        recovery_count=_non_negative_int(value.get("recovery_count")),
        rollover_count=_non_negative_int(value.get("rollover_count")),
        checkpoint=_goal_checkpoint(value.get("checkpoint")),
        reason=_optional_str(value.get("reason")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _goal_checkpoint(value: Any) -> GoalCheckpoint:
    if not isinstance(value, dict):
        return GoalCheckpoint()
    return GoalCheckpoint(
        summary=_optional_str(value.get("summary")),
        completed=_string_list(value.get("completed")),
        decisions=_string_list(value.get("decisions")),
        pending=_string_list(value.get("pending")),
        next_step=_optional_str(value.get("next_step")),
        updated_turn=_non_negative_int(value.get("updated_turn")),
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        normalized = _optional_str(item)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
