from __future__ import annotations

from collections.abc import Sequence

EMPTY_PROMPT_ERROR = "prompt argument or piped stdin is required."
PROMPT_STDIN_CONFLICT_ERROR = (
    "stdin and positional prompt cannot be combined implicitly; "
    "use stdin alone, or pass --no-stdin with the positional prompt"
)


def build_prompt(prompt_parts: Sequence[str], stdin_text: str | None = None) -> str:
    prompt = " ".join(part for part in prompt_parts if part).strip()
    context = (stdin_text or "").strip()

    if context and prompt:
        raise ValueError(PROMPT_STDIN_CONFLICT_ERROR)
    if context:
        return context
    if prompt:
        return prompt
    raise ValueError(EMPTY_PROMPT_ERROR)
