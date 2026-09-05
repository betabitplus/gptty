from __future__ import annotations

from gptty.goal import (
    GoalSignal,
    activation_prompt,
    continuation_prompt,
    parse_goal_response,
    steering_prompt,
)


def test_parse_goal_response_requires_status_at_start() -> None:
    parsed = parse_goal_response("\nGPTTY_GOAL: CONTINUE\n\nStill working.")
    assert parsed.signal is GoalSignal.CONTINUE
    assert parsed.body == "Still working."

    parsed = parse_goal_response("Progress first.\nGPTTY_GOAL: COMPLETE")
    assert parsed.signal is None
    assert parsed.body == "Progress first.\nGPTTY_GOAL: COMPLETE"


def test_parse_goal_response_supports_all_terminal_signals_case_insensitively() -> None:
    assert parse_goal_response("gptty_goal: complete\nDone.").signal is GoalSignal.COMPLETE
    assert parse_goal_response("GPTTY_GOAL: BLOCKED\nNeed login.").signal is GoalSignal.BLOCKED


def test_parse_goal_response_handles_empty_or_invalid_status() -> None:
    assert parse_goal_response("").signal is None
    assert parse_goal_response("GPTTY_GOAL: MAYBE\nNot sure.").signal is None


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
