from __future__ import annotations

import sys
from typing import Any, TextIO

from ..session_state import SessionStateHandle, session_handle_for_args
from ..state import ChatState


def load_command_session(
    args: Any,
    *,
    stderr: TextIO = sys.stderr,
) -> tuple[SessionStateHandle, ChatState]:
    """Load the transactional local session and surface one-time migration drift."""

    handle = session_handle_for_args(args)
    state = handle.load()
    warning = handle.migration_warning
    if warning is not None:
        print(
            "gptty: legacy chat state could not be imported; "
            f"continuing with transactional local session ({warning})",
            file=stderr,
        )
    return handle, state


def resolve_attached_conversation(
    args: Any,
    *,
    explicit: str | None,
    stderr: TextIO = sys.stderr,
) -> str | None:
    if explicit:
        return str(explicit)
    _, state = load_command_session(args, stderr=stderr)
    return state.current_conversation
