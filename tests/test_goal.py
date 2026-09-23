from __future__ import annotations

from gptty.goal import (
    GoalSignal,
    activation_prompt,
    continuation_prompt,
    parse_goal_response,
    rollover_prompt,
    steering_prompt,
)
from gptty.state import GoalCheckpoint, GoalState


def test_parse_goal_response_requires_status_at_start() -> None:
    parsed = parse_goal_response("\nGPTTY_GOAL: CONTINUE\n\nStill working.")
    assert parsed.signal is GoalSignal.CONTINUE
    assert parsed.body == "Still working."

    parsed = parse_goal_response("Progress first.\nGPTTY_GOAL: COMPLETE")
    assert parsed.signal is None
    assert parsed.body == "Progress first.\nGPTTY_GOAL: COMPLETE"


def test_parse_goal_response_supports_all_terminal_signals_case_insensitively() -> None:
    assert (
        parse_goal_response("gptty_goal: complete\nDone.").signal is GoalSignal.COMPLETE
    )
    assert (
        parse_goal_response("GPTTY_GOAL: BLOCKED\nNeed login.").signal
        is GoalSignal.BLOCKED
    )


def test_parse_goal_response_handles_empty_or_invalid_status() -> None:
    assert parse_goal_response("").signal is None
    assert parse_goal_response("GPTTY_GOAL: MAYBE\nNot sure.").signal is None


def test_parse_goal_response_extracts_compact_checkpoint_and_hides_protocol_line() -> (
    None
):
    parsed = parse_goal_response(
        "GPTTY_GOAL: CONTINUE\n"
        'GPTTY_CHECKPOINT: {"summary":"half done","completed":["commit A"],'
        '"decisions":["keep API"],"pending":["live test"],"next":"run cmux"}\n'
        "Visible progress."
    )

    assert parsed.signal is GoalSignal.CONTINUE
    assert parsed.body == "Visible progress."
    assert parsed.checkpoint is not None
    assert parsed.checkpoint.summary == "half done"
    assert parsed.checkpoint.completed == ["commit A"]
    assert parsed.checkpoint.decisions == ["keep API"]
    assert parsed.checkpoint.pending == ["live test"]
    assert parsed.checkpoint.next_step == "run cmux"


def test_rollover_prompt_carries_durable_state_and_no_duplicate_side_effect_rule() -> (
    None
):
    goal = GoalState(
        goal_id="goal-1",
        objective="Finish the implementation",
        conversations=["conv-old"],
        context_seed=["user: keep backwards compatibility"],
        checkpoint=GoalCheckpoint(
            summary="Core code is done.",
            completed=["commit A already pushed"],
            decisions=["do not change API"],
            pending=["live cmux acceptance"],
            next_step="run cmux",
        ),
    )

    prompt = rollover_prompt(goal, reason="conversation reached its maximum length")

    assert "Goal ID: goal-1" in prompt
    assert "commit A already pushed" in prompt
    assert "do not change API" in prompt
    assert "user: keep backwards compatibility" in prompt
    assert "Do not repeat completed external actions" in prompt


def test_activation_prompt_carries_explicit_goal_and_completion_contract() -> None:
    prompt = activation_prompt("Finish the current implementation and tests")
    assert "Finish the current implementation and tests" in prompt
    assert "GPTTY_GOAL: CONTINUE" in prompt
    assert "GPTTY_GOAL: COMPLETE" in prompt
    assert "GPTTY_GOAL: BLOCKED" in prompt
    assert "entire agreed goal is finished" in prompt


def test_continuation_prompt_can_recover_missing_protocol() -> None:
    normal = continuation_prompt()
    recovery = continuation_prompt(protocol_recovery=True)
    assert "Do not repeat work that is already complete" in normal
    assert "previous turn ended without a valid GPTTY_GOAL status line" in recovery


def test_steering_prompt_preserves_user_message_and_repeats_protocol() -> None:
    prompt = steering_prompt("Do not touch the other repository.")
    assert prompt.startswith("Do not touch the other repository.")
    assert "steering/refinement" in prompt
    assert "GPTTY_GOAL: COMPLETE" in prompt


def test_goal_aware_recovery_reanchors_objective_checkpoint_and_stale_context_rule() -> None:
    from gptty.goal import abnormal_recovery_prompt

    goal = GoalState(
        goal_id="goal-anchor",
        generation=3,
        objective="Acceptance only; do not call tools or modify anything.",
        active_operation_id="goal-anchor:g3:t7",
        checkpoint=GoalCheckpoint(
            summary="No side effects are allowed.",
            completed=["previous verification retained"],
            pending=["finish acceptance"],
            next_step="verify without tools",
        ),
    )
    prompt = abnormal_recovery_prompt(
        "process restarted",
        goal=goal,
        journal_context=["user steering history: do not call tools"],
    )

    assert "Goal ID: goal-anchor" in prompt
    assert "Generation: 3" in prompt
    assert "Objective: Acceptance only; do not call tools or modify anything." in prompt
    assert "No side effects are allowed." in prompt
    assert "goal-anchor:g3:t7" in prompt
    assert "user steering history: do not call tools" in prompt
    assert "Do not drift back into unrelated older work" in prompt
    assert "If the active Goal forbids tools" in prompt


def test_goal_aware_continuation_and_steering_repeat_objective() -> None:
    goal = GoalState(
        goal_id="goal-repeat",
        objective="Keep API v1 and finish only the agreed verification.",
    )
    continuation = continuation_prompt(goal=goal)
    steering = steering_prompt("Also keep the CLI stable.", goal=goal)
    for prompt in (continuation, steering):
        assert "Goal ID: goal-repeat" in prompt
        assert "Keep API v1 and finish only the agreed verification." in prompt
        assert "Do not drift back into unrelated older work" in prompt


def test_complete_requires_at_least_one_verified_completed_claim() -> None:
    from gptty.goal import completion_checkpoint_error

    parsed = parse_goal_response(
        "GPTTY_GOAL: COMPLETE\n"
        'GPTTY_CHECKPOINT: {"summary":"done","completed":[],"decisions":[],"pending":[],"next":"none"}\n'
        "Done."
    )
    assert completion_checkpoint_error(parsed) == "COMPLETE checkpoint has no verified completed work"
