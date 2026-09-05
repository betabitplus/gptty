from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

GOAL_PROTOCOL_PREFIX = "GPTTY_GOAL:"
MAX_PROTOCOL_FAILURES = 3


class GoalSignal(str, Enum):
    CONTINUE = "CONTINUE"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ParsedGoalResponse:
    signal: GoalSignal | None
    body: str


_GOAL_LINE_RE = re.compile(r"^GPTTY_GOAL:\s*(CONTINUE|COMPLETE|BLOCKED)\s*$", re.IGNORECASE)


def parse_goal_response(text: str | None) -> ParsedGoalResponse:
    raw = str(text or "")
    lines = raw.splitlines()
    first_content_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_content_index is None:
        return ParsedGoalResponse(signal=None, body="")

    match = _GOAL_LINE_RE.fullmatch(lines[first_content_index].strip())
    if match is None:
        return ParsedGoalResponse(signal=None, body=raw.strip())

    signal = GoalSignal(match.group(1).upper())
    body_lines = lines[:first_content_index] + lines[first_content_index + 1 :]
    return ParsedGoalResponse(signal=signal, body="\n".join(body_lines).strip())


def goal_protocol_instruction() -> str:
    return (
        "At the very start of your FINAL assistant response for this turn, output exactly one of these lines:\n"
        "GPTTY_GOAL: CONTINUE\n"
        "GPTTY_GOAL: COMPLETE\n"
        "GPTTY_GOAL: BLOCKED\n"
        "Use COMPLETE only when the entire agreed goal is finished and reasonably verified. "
        "Use BLOCKED only for a real external blocker that requires user intervention or an external change. "
        "Otherwise use CONTINUE. The end of this model turn is not itself a reason to stop the goal."
    )


def activation_prompt(objective: str | None = None) -> str:
    objective_text = ""
    if objective and objective.strip():
        objective_text = f"\n\nExplicit goal:\n{objective.strip()}"
    return (
        "GPTTY Goal mode is now active for this conversation. Pursue the task and plan already agreed in this chat "
        "autonomously until the whole agreed scope is complete; do not expand the scope beyond what was agreed. "
        "Continue doing useful work without asking for confirmation unless you are genuinely blocked."
        f"{objective_text}\n\n{goal_protocol_instruction()}\n\nContinue working on the goal now."
    )


def continuation_prompt(*, protocol_recovery: bool = False) -> str:
    recovery = (
        "The previous turn ended without a valid GPTTY_GOAL status line. Treat that turn as unfinished and restore the protocol. "
        if protocol_recovery
        else ""
    )
    return (
        f"{recovery}Continue pursuing the active goal from this conversation. Do not repeat work that is already complete. "
        "Re-evaluate the current state, continue from the latest progress, and stay within the agreed scope.\n\n"
        f"{goal_protocol_instruction()}"
    )


def steering_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt.rstrip()}\n\n"
        "[GPTTY Goal mode remains active. Treat the user message above as steering/refinement of the existing goal. "
        "Do not stop merely because this turn ends.]\n"
        f"{goal_protocol_instruction()}"
    )
