from __future__ import annotations

EFFORT_VALUES: tuple[str, ...] = ("instant", "medium", "high")
EFFORT_TO_MODEL_PROFILE: dict[str, str] = {
    "instant": "FAST",
    "medium": "BALANCED",
    "high": "DEEP",
}


def normalize_effort(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "default", "auto", "none", "off", "-"}:
        return None
    if normalized not in EFFORT_VALUES:
        raise ValueError(
            "reasoning effort must be one of: default, instant, medium, high"
        )
    return normalized


def model_profile_for_effort(value: str | None) -> str | None:
    normalized = normalize_effort(value)
    return EFFORT_TO_MODEL_PROFILE.get(normalized) if normalized is not None else None


def effort_label(value: str | None) -> str:
    normalized = normalize_effort(value)
    return normalized.title() if normalized is not None else "Default · High"


def validate_model_effort_combination(
    model: str | None,
    effort: str | None,
) -> None:
    normalized_effort = normalize_effort(effort)
    if model and normalized_effort is not None:
        raise ValueError(
            "explicit model + explicit reasoning effort is not yet supported by "
            "the product runtime; use the default model or reset effort to default"
        )


__all__ = [
    "EFFORT_TO_MODEL_PROFILE",
    "EFFORT_VALUES",
    "effort_label",
    "model_profile_for_effort",
    "normalize_effort",
    "validate_model_effort_combination",
]
