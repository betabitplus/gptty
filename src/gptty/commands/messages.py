from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TextIO

from ..output import OutputMessage as ChatMessage
from ..output import OutputFormat, normalize_messages, render_messages
from ..sdk_client import GpttyClient
from ..session_state import SessionStateError
from ._client import build_client
from ._session import resolve_attached_conversation

NO_CONVERSATION_ERROR = (
    "gptty messages requires a conversation URL/id or an attached conversation. "
    "Run `gptty attach <url-or-id>` first."
)


def run_messages(
    args: Any,
    *,
    client_factory: Callable[..., Any] = GpttyClient,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    try:
        conversation_ref = resolve_conversation_ref(args, stderr=stderr)
    except SessionStateError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 1

    if not conversation_ref:
        print(NO_CONVERSATION_ERROR, file=stderr)
        return 2

    client = build_client(client_factory, args)

    options: dict[str, Any] = {}
    last = getattr(args, "last", None)
    if last is not None:
        options["limit"] = int(last)

    try:
        response = client.get_messages(conversation_ref, **options)
    except Exception as exc:  # noqa: BLE001 - command boundary converts SDK errors to exit codes.
        print(f"gptty: messages request failed: {exc}", file=stderr)
        return 1

    output_format: OutputFormat = getattr(args, "format", "plain")
    print(render_messages(normalize_messages(response), output_format), file=stdout)
    return 0


def resolve_conversation_ref(
    args: Any,
    *,
    stderr: TextIO = sys.stderr,
) -> str | None:
    return resolve_attached_conversation(
        args,
        explicit=getattr(args, "url_or_id", None),
        stderr=stderr,
    )


def format_messages(messages: list[ChatMessage]) -> str:
    return render_messages(messages, "plain")
