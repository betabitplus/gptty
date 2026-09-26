from __future__ import annotations

import pytest

from gptty.reasoning import (
    effort_label,
    model_profile_for_effort,
    normalize_effort,
    validate_model_effort_combination,
)


@pytest.mark.parametrize(
    ("raw", "normalized", "profile"),
    [
        (None, None, None),
        ("default", None, None),
        ("instant", "instant", "FAST"),
        ("medium", "medium", "BALANCED"),
        ("high", "high", "DEEP"),
        (" HIGH ", "high", "DEEP"),
    ],
)
def test_reasoning_effort_normalization_and_profile_mapping(
    raw,
    normalized,
    profile,
) -> None:
    assert normalize_effort(raw) == normalized
    assert model_profile_for_effort(raw) == profile


def test_reasoning_effort_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="default, instant, medium, high"):
        normalize_effort("maximum")


def test_custom_model_plus_explicit_effort_fails_closed() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        validate_model_effort_combination("gpt-custom", "medium")

    validate_model_effort_combination(None, "medium")
    validate_model_effort_combination("gpt-custom", None)


def test_effort_label_keeps_default_policy_explicit() -> None:
    assert effort_label(None) == "Default · High"
    assert effort_label("instant") == "Instant"
