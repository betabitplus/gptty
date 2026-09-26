from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TextIO

from ..automation import (
    automation_timestamp,
    new_run_id,
    normalize_provider_event,
    normalize_required_action,
    run_event_envelope,
)
from ..media import MediaInputError, collect_media_inputs
from ..output import (
    OutputFormat,
    normalize_response,
    normalize_turn_failure,
    normalize_turn_result,
    render_jsonl_event,
    render_response,
)
from ..prompt import PROMPT_STDIN_CONFLICT_ERROR, build_prompt
from ..reasoning import validate_model_effort_combination
from ..required_action import render_required_action, required_action_state
from ..sdk_client import GpttyClient
from ..turn_failure import classify_turn_failure
from ._client import build_client


EMPTY_PROMPT_ERROR = "gptty ask requires a prompt argument or piped stdin."


def _build_send_options(
    args: Any,
    *,
    stream: bool,
    media: list[str] | None,
    on_token: Callable[[str], None] | None,
    on_event: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"stream": stream}
    model = getattr(args, "model", None)
    effort = getattr(args, "effort", None)
    if model:
        options["model"] = model
    if effort:
        options["reasoning_effort"] = effort
    if media:
        options["media"] = media
    if on_token is not None:
        options["on_token"] = on_token
    if on_event is not None:
        options["on_event"] = on_event
    return options


def run_ask(
    args: Any,
    *,
    stdin_text: str | None = None,
    client_factory: Callable[..., Any] = GpttyClient,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    output_format: OutputFormat = (
        "plain" if bool(getattr(args, "plain", False)) else getattr(args, "format", "plain")
    )
    jsonl = output_format == "jsonl"

    try:
        prompt = build_prompt(getattr(args, "prompt", []), stdin_text=stdin_text)
    except ValueError as exc:
        raw_message = str(exc)
        conflict = raw_message == PROMPT_STDIN_CONFLICT_ERROR
        message = raw_message if conflict else EMPTY_PROMPT_ERROR
        if jsonl:
            print(
                render_jsonl_event(
                    normalize_turn_failure(
                        {
                            "status": "usage-error",
                            "message": message,
                            "source": "command-boundary",
                        },
                        exit_code=2,
                        error_class="stdin_prompt_conflict" if conflict else "usage",
                    )
                ),
                file=stdout,
            )
        print(message, file=stderr)
        return 2

    try:
        media = collect_media_inputs(args)
    except MediaInputError as exc:
        if jsonl:
            print(
                render_jsonl_event(
                    normalize_turn_failure(
                        {
                            "status": "usage-error",
                            "message": str(exc),
                            "source": "command-boundary",
                        },
                        exit_code=2,
                        error_class="media_input",
                    )
                ),
                file=stdout,
            )
        print(f"gptty: {exc}", file=stderr)
        return 2

    try:
        validate_model_effort_combination(
            getattr(args, "model", None),
            getattr(args, "effort", None),
        )
    except ValueError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 2

    run_id = new_run_id()
    observations: list[dict[str, Any]] = []
    stream = (
        not bool(getattr(args, "no_stream", False))
        and output_format in {"plain", "jsonl"}
    )
    saw_stream_token = False

    def emit(event_type: str, **data: Any) -> dict[str, Any]:
        event = run_event_envelope(
            run_id=run_id,
            event_type=event_type,
            timestamp=automation_timestamp(),
            data=data,
        )
        if jsonl:
            print(render_jsonl_event(event), file=stdout, flush=True)
        return event

    if jsonl:
        emit("run_started", command="ask")
        emit("prompt_sent")

    def on_token(token: str) -> None:
        nonlocal saw_stream_token
        saw_stream_token = True
        if jsonl:
            emit("token_delta", text=token, role="assistant")
        else:
            print(token, end="", file=stdout, flush=True)

    def on_event(event: dict[str, Any]) -> None:
        normalized = normalize_provider_event(event)
        if normalized is None:
            return
        envelope = emit("provider_event", **normalized)
        observations.append(envelope)

    client = build_client(client_factory, args)
    try:
        response = client.send(
            prompt,
            **_build_send_options(
                args,
                stream=stream,
                media=media,
                on_token=on_token if stream else None,
                on_event=on_event if stream and jsonl else None,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - command boundary converts SDK errors to exit codes.
        failure = classify_turn_failure(exc)
        if jsonl:
            print(
                render_jsonl_event(
                    normalize_turn_failure(
                        failure.to_dict(),
                        raw_error=str(exc),
                    )
                ),
                file=stdout,
            )
        print(f"gptty: ask request failed: {failure.message}", file=stderr)
        return 1

    normalized = normalize_response(response)
    response_text = normalized.get("text", "")
    action_state = None
    if not saw_stream_token and not response_text:
        action_state = required_action_state(client, response)

    if action_state is not None:
        action, conversation = action_state
        action_payload = normalize_required_action(action, conversation=conversation)
        if jsonl:
            action_event = emit("required_action", **action_payload)
            observations.append(action_event)
            print(
                render_jsonl_event(
                    normalize_turn_failure(
                        {
                            "status": "blocked",
                            "message": "ChatGPT is waiting for a web UI action.",
                            "source": "required_action",
                            "code": "required_action",
                        },
                        conversation=action_payload.get("conversation"),
                        error_class="required_action",
                    )
                ),
                file=stdout,
            )
        print(render_required_action(action, conversation=conversation), file=stderr)
        return 1

    if jsonl:
        if stream and not saw_stream_token and response_text:
            emit("token_delta", text=response_text, role="assistant")
        print(
            render_jsonl_event(
                normalize_turn_result(
                    response,
                    observations=observations,
                )
            ),
            file=stdout,
        )
    elif output_format == "json":
        print(render_response(normalize_turn_result(response), "json"), file=stdout)
    elif stream:
        if not saw_stream_token:
            print(response_text, file=stdout)
        else:
            print(file=stdout)
    else:
        print(render_response(normalized, output_format), file=stdout)

    return 0
