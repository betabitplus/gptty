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



_INTERNAL_GOAL_USER_PREFIXES = (
    "GPTTY Goal mode is now active.",
    "Continue pursuing the active goal from this conversation.",
    "The previous turn ended without a valid GPTTY_GOAL status line.",
    "The previous goal turn ended in a non-standard transport/chat/process state.",
    "GPTTY is continuing an existing Goal in a fresh ChatGPT conversation",
)
_GOAL_STEERING_MARKER = "\n\n[GPTTY Goal mode remains active."
_GOAL_OPERATION_MARKER = "\n\n[GPTTY durable operation id:"


def sanitize_goal_history_text(role: str, text: str | None) -> str | None:
    """Hide Goal control protocol while preserving user-visible historical content."""
    raw = str(text or "")
    normalized_role = str(role or "").strip().lower()
    if normalized_role == "assistant":
        parsed = parse_goal_response(raw)
        return parsed.body if parsed.signal is not None else raw
    if normalized_role != "user":
        return raw

    stripped = raw.lstrip()
    if (
        GOAL_PROTOCOL_PREFIX in raw
        and any(stripped.startswith(prefix) for prefix in _INTERNAL_GOAL_USER_PREFIXES)
    ):
        return None

    steering_index = raw.find(_GOAL_STEERING_MARKER)
    if steering_index >= 0:
        visible = raw[:steering_index].rstrip()
        return visible or None

    operation_index = raw.find(_GOAL_OPERATION_MARKER)
    if operation_index >= 0:
        visible = raw[:operation_index].rstrip()
        return visible or None
    return raw


def completion_checkpoint_error(parsed: ParsedGoalResponse) -> str | None:
    """Return why COMPLETE is not safe to accept as a durable terminal claim."""
    if parsed.signal is not GoalSignal.COMPLETE:
        return None
    checkpoint = parsed.checkpoint
    if checkpoint is None:
        return "COMPLETE is missing the required structured checkpoint"
    if not checkpoint.summary:
        return "COMPLETE checkpoint has no summary"
    if not checkpoint.completed:
        return "COMPLETE checkpoint has no verified completed work"
    if checkpoint.pending:
        return "COMPLETE checkpoint still lists pending work"
    if not checkpoint.next_step:
        return "COMPLETE checkpoint has no next-state declaration"
    return None

def goal_protocol_instruction() -> str:
    return (
        "At the very start of your FINAL assistant response for this turn, output exactly one of these lines:\n"
        "GPTTY_GOAL: CONTINUE\n"
        "GPTTY_GOAL: COMPLETE\n"
        "GPTTY_GOAL: BLOCKED\n"
        "Immediately after it, output one compact single-line JSON checkpoint in this exact form:\n"
        'GPTTY_CHECKPOINT: {"summary":"current durable state","completed":["verified work"],'
        '"decisions":["important decisions"],"pending":["remaining work"],"next":"next safe action"}\n'
        "Keep the checkpoint factual and compact. `completed` must contain at least one concrete verified completion "
        "claim; include durable decisions and externally visible/tool-side effects "
        "that must not be repeated after recovery. Use COMPLETE only when the entire agreed goal is finished and "
        "reasonably verified; COMPLETE requires a structured checkpoint with pending=[] and a concrete next-state "
        "declaration. Never claim an action, test, commit, or external effect as completed merely because it was "
        "attempted; reconcile machine/external evidence when available. Use BLOCKED only for a real external blocker "
        "that requires user intervention or an "
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


def _goal_anchor(goal: GoalState) -> str:
    checkpoint = goal.checkpoint
    return (
        "Durable active Goal (authoritative for this turn):\n"
        f"Goal ID: {goal.goal_id or 'unknown'}\n"
        f"Generation: {goal.generation}\n"
        f"Objective: {goal.objective or 'inherit the agreed task from the captured Goal context'}\n"
        f"Checkpoint summary: {checkpoint.summary or 'not yet captured'}\n"
        f"Completed: {_join_checkpoint(checkpoint.completed)}\n"
        f"Decisions: {_join_checkpoint(checkpoint.decisions)}\n"
        f"Pending: {_join_checkpoint(checkpoint.pending)}\n"
        f"Next safe step: {checkpoint.next_step or 'reconstruct current state before acting'}\n"
        "Treat this Goal state and newer user steering as the task authority. Do not drift back into unrelated "
        "older work merely because it exists in the conversation history."
    )


def continuation_prompt(
    *, protocol_recovery: bool = False, goal: GoalState | None = None
) -> str:
    recovery = (
        "The previous turn ended without a valid GPTTY_GOAL status line. Treat that turn as unfinished and restore "
        "the protocol. "
        if protocol_recovery
        else ""
    )
    anchor = f"\n\n{_goal_anchor(goal)}" if goal is not None else ""
    return (
        f"{recovery}Continue pursuing the active goal from this conversation. Do not repeat work that is already "
        "complete. Re-evaluate the current state, continue from the latest progress, and stay within the agreed "
        f"scope.{anchor}\n\n{goal_protocol_instruction()}"
    )


def abnormal_recovery_prompt(
    reason: str,
    *,
    goal: GoalState | None = None,
    journal_context: list[str] | None = None,
) -> str:
    anchor = f"\n\n{_goal_anchor(goal)}" if goal is not None else ""
    journal = ""
    if journal_context:
        journal = (
            "\n\nMachine-owned Goal journal (recent/critical recovery evidence):\n"
            + "\n".join(f"- {item}" for item in journal_context)
        )
    open_operation = ""
    if goal is not None and goal.active_operation_id:
        open_operation = (
            f"\n\nOpen durable operation: {goal.active_operation_id}. Its outcome is not fully confirmed. "
            "Reconcile actual machine/external state before repeating any tool call or write from that operation."
        )
    return (
        "The previous goal turn ended in a non-standard transport/chat/process state. Do not blindly repeat the "
        "previous action or tool call. Reconcile the actual current state first, preserve any side effects that already "
        f"happened, and continue only the active Goal. Observed condition: {reason}"
        f"{anchor}{journal}{open_operation}\n\n"
        "Ignore stale unrelated work in the surrounding conversation when it conflicts with the active Goal above. "
        "If the active Goal forbids tools or external changes, that prohibition still applies during recovery.\n\n"
        f"{goal_protocol_instruction()}"
    )


def rollover_prompt(
    goal: GoalState, *, reason: str, journal_context: list[str] | None = None
) -> str:
    checkpoint = goal.checkpoint
    conversations = ", ".join(goal.conversations) or "none recorded"
    context_seed = (
        "\n".join(f"- {item}" for item in goal.context_seed) or "- none captured"
    )
    journal = (
        "\n".join(f"- {item}" for item in (journal_context or []))
        or "- none recorded"
    )
    return (
        "GPTTY is continuing an existing Goal in a fresh ChatGPT conversation because the previous conversation "
        "became unusable. This is a recovery handoff, not a new task. Do not repeat completed external actions, "
        "commits, writes, submissions, or tool side effects without first checking whether they already happened.\n\n"
        f"Goal ID: {goal.goal_id or 'unknown'}\n"
        f"Recovery reason: {reason}\n"
        f"Original objective: {goal.objective or 'inherited from the previous conversation'}\n"
        f"Previous conversation chain: {conversations}\n"
        f"Goal generation: {goal.generation}\n\n"
        "Durable checkpoint:\n"
        f"Summary: {checkpoint.summary or 'not yet captured'}\n"
        f"Completed: {_join_checkpoint(checkpoint.completed)}\n"
        f"Decisions: {_join_checkpoint(checkpoint.decisions)}\n"
        f"Pending: {_join_checkpoint(checkpoint.pending)}\n"
        f"Next safe step: {checkpoint.next_step or 'reconstruct the current state before acting'}\n\n"
        "Recovery context captured when Goal started:\n"
        f"{context_seed}\n\n"
        "Machine-owned Goal journal (recent relevant events):\n"
        f"{journal}\n\n"
        "Continue the same Goal from this state. If the checkpoint is incomplete, inspect the current external/repo "
        "state before acting rather than guessing or repeating prior work.\n\n"
        f"{goal_protocol_instruction()}"
    )



def operation_identity_instruction(operation_id: str) -> str:
    return (
        f"[GPTTY durable operation id: {operation_id}]\n"
        "This turn is durably journaled before dispatch. For any external write, prefer an "
        "idempotency/operation key when the target supports one. If delivery becomes ambiguous, "
        "do not blindly repeat the write: inspect/reconcile the actual external state first."
    )

def steering_prompt(user_prompt: str, *, goal: GoalState | None = None) -> str:
    anchor = f"\n\n{_goal_anchor(goal)}" if goal is not None else ""
    return (
        f"{user_prompt.rstrip()}\n\n"
        "[GPTTY Goal mode remains active. Treat the user message above as steering/refinement of the existing goal. "
        "Do not stop merely because this turn ends.]"
        f"{anchor}\n{goal_protocol_instruction()}"
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
