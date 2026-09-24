from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TextIO

from ..sdk_client import GpttyClient
from ..session_state import SessionStateError
from ._client import build_client
from ._session import load_command_session

CONVERSATION_REF_FIELDS = (
    "conversation_url",
    "conversation_id",
    "conversation_ref",
    "url",
    "id",
)


def extract_attached_ref(response: Any, fallback: str) -> str:
    if isinstance(response, dict):
        for field in CONVERSATION_REF_FIELDS:
            value = response.get(field)
            if value:
                return str(value)

    for field in CONVERSATION_REF_FIELDS:
        value = getattr(response, field, None)
        if value:
            return str(value)

    return fallback


def run_attach(
    args: Any,
    *,
    client_factory: Callable[..., Any] = GpttyClient,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    url_or_id = str(getattr(args, "url_or_id"))
    try:
        state_handle, state = load_command_session(args, stderr=stderr)
    except SessionStateError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 1

    client = build_client(client_factory, args)
    try:
        response = client.attach_conversation(url_or_id)
    except Exception as exc:  # noqa: BLE001 - command boundary converts SDK errors to exit codes.
        print(f"gptty: attach failed: {exc}", file=stderr)
        return 1

    conversation_ref = extract_attached_ref(response, fallback=url_or_id)

    try:
        state.current_conversation = conversation_ref
        state_handle.save(state)
    except SessionStateError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 1

    print(f"Attached conversation: {conversation_ref}", file=stdout)
    return 0
