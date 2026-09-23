from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from .state import GoalCheckpoint, GoalState

GOAL_PROTOCOL_PREFIX = "GPTTY_GOAL:"
GOAL_CHECKPOINT_PREFIX = "GPTTY_CHECKPOINT:"
MAX_PROTOCOL_FAILURES = 3
MAX_RECOVERY_ATTEMPTS = 2
MAX_ROLLOVERS = 12


class GoalSignal(str, Enum):
    CONTINUE = "CONTINUE"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ParsedGoalResponse:
    signal: GoalSignal | None
    body: str
    checkpoint: GoalCheckpoint | None = None


_GOAL_LINE_RE = re.compile(
    r"^GPTTY_GOAL:\s*(CONTINUE|COMPLETE|BLOCKED)\s*$",
    re.IGNORECASE,
)


def parse_goal_response(text: str | None) -> ParsedGoalResponse:
    raw = str(text or "")
    lines = raw.splitlines()
    first_content_index = next(
        (index for index, line in enumerate(lines) if line.strip()), None
    )
    if first_content_index is None:
        return ParsedGoalResponse(signal=None, body="")

    match = _GOAL_LINE_RE.fullmatch(lines[first_content_index].strip())
    if match is None:
        return ParsedGoalResponse(signal=None, body=raw.strip())

    signal = GoalSignal(match.group(1).upper())
    checkpoint: GoalCheckpoint | None = None
    checkpoint_index: int | None = None
    for index in range(first_content_index + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if stripped.upper().startswith(GOAL_CHECKPOINT_PREFIX):
            checkpoint_index = index
            checkpoint = _parse_checkpoint(
                stripped[len(GOAL_CHECKPOINT_PREFIX) :].strip()
            )
        break

    body_lines = [
        line
        for index, line in enumerate(lines)
        if index != first_content_index and index != checkpoint_index
    ]
    return ParsedGoalResponse(
        signal=signal,
        body="\n".join(body_lines).strip(),
        checkpoint=checkpoint,
    )


def goal_protocol_instruction() -> str:
    return (
        "At the very start of your FINAL assistant response for this turn, output exactly one of these lines:\n"
        "GPTTY_GOAL: CONTINUE\n"
        "GPTTY_GOAL: COMPLETE\n"
        "GPTTY_GOAL: BLOCKED\n"
        "Immediately after it, output one compact single-line JSON checkpoint in this exact form:\n"
        'GPTTY_CHECKPOINT: {"summary":"current durable state","completed":["verified work"],'
        '"decisions":["important decisions"],"pending":["remaining work"],"next":"next safe action"}\n'
        "Keep the checkpoint factual and compact. Include durable decisions and externally visible/tool-side effects "
        "that must not be repeated after recovery. Use COMPLETE only when the entire agreed goal is finished and "
        "reasonably verified. Use BLOCKED only for a real external blocker that requires user intervention or an "
        "external change. Otherwise use CONTINUE. The end of this model turn is not itself a reason to stop the goal."
    )


def activation_prompt(objective: str | None = None) -> str:
    objective_text = ""
    if objective and objective.strip():
        objective_text = f"\n\nExplicit goal:\n{objective.strip()}"
    return (
        "GPTTY Goal mode is now active. Pursue the task and plan already agreed in this chat autonomously until the "
        "whole agreed scope is complete; do not expand the scope beyond what was agreed. Continue doing useful work "
        "without asking for confirmation unless you are genuinely blocked. Treat the checkpoint as durable recovery "
        "state: it must be sufficient to continue safely in a new chat without repeating completed side effects."
        f"{objective_text}\n\n{goal_protocol_instruction()}\n\nContinue working on the goal now."
    )


def continuation_prompt(*, protocol_recovery: bool = False) -> str:
    recovery = (
        "The previous turn ended without a valid GPTTY_GOAL status line. Treat that turn as unfinished and restore "
        "the protocol. "
        if protocol_recovery
        else ""
    )
    return (
        f"{recovery}Continue pursuing the active goal from this conversation. Do not repeat work that is already "
        "complete. Re-evaluate the current state, continue from the latest progress, and stay within the agreed "
        f"scope.\n\n{goal_protocol_instruction()}"
    )


def abnormal_recovery_prompt(reason: str) -> str:
    return (
        "The previous goal turn ended in a non-standard transport/chat state. Do not blindly repeat the previous "
        "action or tool call. Reconcile the actual current state first, preserve any side effects that already "
        f"happened, and continue from the safest unfinished point. Observed condition: {reason}\n\n"
        f"{goal_protocol_instruction()}"
    )


def rollover_prompt(goal: GoalState, *, reason: str) -> str:
    checkpoint = goal.checkpoint
    conversations = ", ".join(goal.conversations) or "none recorded"
    context_seed = (
        "\n".join(f"- {item}" for item in goal.context_seed) or "- none captured"
    )
    return (
        "GPTTY is continuing an existing Goal in a fresh ChatGPT conversation because the previous conversation "
        "became unusable. This is a recovery handoff, not a new task. Do not repeat completed external actions, "
        "commits, writes, submissions, or tool side effects without first checking whether they already happened.\n\n"
        f"Goal ID: {goal.goal_id or 'unknown'}\n"
        f"Recovery reason: {reason}\n"
        f"Original objective: {goal.objective or 'inherited from the previous conversation'}\n"
        f"Previous conversation chain: {conversations}\n\n"
        "Durable checkpoint:\n"
        f"Summary: {checkpoint.summary or 'not yet captured'}\n"
        f"Completed: {_join_checkpoint(checkpoint.completed)}\n"
        f"Decisions: {_join_checkpoint(checkpoint.decisions)}\n"
        f"Pending: {_join_checkpoint(checkpoint.pending)}\n"
        f"Next safe step: {checkpoint.next_step or 'reconstruct the current state before acting'}\n\n"
        "Recovery context captured when Goal started:\n"
        f"{context_seed}\n\n"
        "Continue the same Goal from this state. If the checkpoint is incomplete, inspect the current external/repo "
        "state before acting rather than guessing or repeating prior work.\n\n"
        f"{goal_protocol_instruction()}"
    )


def steering_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt.rstrip()}\n\n"
        "[GPTTY Goal mode remains active. Treat the user message above as steering/refinement of the existing goal. "
        "Do not stop merely because this turn ends.]\n"
        f"{goal_protocol_instruction()}"
    )


def _parse_checkpoint(payload: str) -> GoalCheckpoint | None:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return GoalCheckpoint(
        summary=_bounded_text(raw.get("summary"), 2400),
        completed=_bounded_list(raw.get("completed")),
        decisions=_bounded_list(raw.get("decisions")),
        pending=_bounded_list(raw.get("pending")),
        next_step=_bounded_text(raw.get("next"), 1200),
    )


def _bounded_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:12]:
        normalized = _bounded_text(item, 700)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def _bounded_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized[:limit] or None


def _join_checkpoint(values: list[str]) -> str:
    return "; ".join(values) if values else "none recorded"
