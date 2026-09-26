from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ..locks import (
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    ConversationLockError,
    acquire_conversation_lock,
    conversation_lock_dir,
    render_lock_error,
    render_lock_timeout,
)
from ..automation import normalize_required_action
from ..media import MediaInputError, collect_media_inputs
from ..output import (
    OutputFormat,
    normalize_response,
    normalize_turn_failure,
    normalize_turn_result,
    render_jsonl_event,
    render_live_event,
    render_response,
)
from ..prompt import PROMPT_STDIN_CONFLICT_ERROR, build_prompt
from ..reasoning import normalize_effort, validate_model_effort_combination
from ..required_action import render_required_action, required_action_state
from ..runs import start_run
from ..sdk_client import GpttyClient
from ..session_state import SessionStateError
from ..turn_failure import classify_turn_failure
from ._client import build_client
from ._session import load_command_session

EMPTY_PROMPT_ERROR = "gptty send requires a prompt argument or piped stdin."
NO_CONVERSATION_ERROR = (
    "gptty send requires an attached conversation, `--to <url-or-id>`, or `--new`. "
    "Run `gptty attach <url-or-id>` first."
)
REQUIRED_ACTION_RUN_ERROR = "ChatGPT is waiting for a web UI action."

CONVERSATION_REF_FIELDS = (
    "conversation_url",
    "conversation_id",
    "conversation_ref",
    "url",
    "id",
)


def run_send(
    args: Any,
    *,
    stdin_text: str | None = None,
    client_factory: Callable[..., Any] = GpttyClient,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    output_format: OutputFormat = getattr(args, "format", "plain")
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

    state_path = Path(getattr(args, "state", "gptty_state.json"))
    try:
        state_handle, state = load_command_session(args, stderr=stderr)
    except SessionStateError as exc:
        if jsonl:
            print(
                render_jsonl_event(
                    normalize_turn_failure(
                        {
                            "status": "local-state-error",
                            "message": str(exc),
                            "source": "local-session",
                        },
                        error_class="local_session_state",
                    )
                ),
                file=stdout,
            )
        print(f"gptty: {exc}", file=stderr)
        return 1

    explicit_ref = getattr(args, "to", None)
    start_new = bool(getattr(args, "new", False))
    conversation_ref = None if start_new else explicit_ref or state.current_conversation
    requested_model = getattr(args, "model", None)
    requested_effort = normalize_effort(getattr(args, "effort", None))
    use_session_policy = explicit_ref is None
    effective_model = (
        requested_model
        if requested_model is not None
        else state.model if use_session_policy else None
    )
    effective_effort = (
        requested_effort
        if getattr(args, "effort", None) is not None
        else state.reasoning_effort if use_session_policy else None
    )
    try:
        validate_model_effort_combination(effective_model, effective_effort)
    except ValueError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 2

    if not start_new and not conversation_ref:
        if jsonl:
            print(
                render_jsonl_event(
                    normalize_turn_failure(
                        {
                            "status": "usage-error",
                            "message": NO_CONVERSATION_ERROR,
                            "source": "command-boundary",
                        },
                        exit_code=2,
                        error_class="usage",
                    )
                ),
                file=stdout,
            )
        print(NO_CONVERSATION_ERROR, file=stderr)
        return 2

    stream = (
        not bool(getattr(args, "no_stream", False))
        and output_format in {"plain", "jsonl"}
    )
    saw_stream_token = False
    observations: list[dict[str, Any]] = []
    recorder = start_run(
        profile=getattr(args, "profile", None),
        state_path=state_path,
        command="send",
        conversation_ref=str(conversation_ref) if conversation_ref else None,
    )

    def emit(event: dict[str, Any] | None) -> None:
        if jsonl and isinstance(event, dict):
            print(render_jsonl_event(event), file=stdout, flush=True)

    emit(recorder.initial_event)
    emit(recorder.event("prompt_sent"))

    def on_token(token: str) -> None:
        nonlocal saw_stream_token
        saw_stream_token = True
        event = recorder.event("token_delta", text=token, role="assistant")
        if jsonl:
            emit(event)
        else:
            print(token, end="", file=stdout, flush=True)

    def on_event(event: dict[str, Any]) -> None:
        recorded = recorder.provider_event(event)
        if recorded is not None:
            observations.append(recorded)
            if jsonl:
                emit(recorded)
        if not jsonl:
            rendered = render_live_event(event)
            if rendered:
                print(rendered, file=stderr, flush=True)

    options: dict[str, Any] = {"stream": stream}
    if effective_model:
        options["model"] = effective_model
    if effective_effort:
        options["reasoning_effort"] = effective_effort
    if media:
        options["media"] = media
    if stream:
        options["on_token"] = on_token
        options["on_event"] = on_event

    lock = None
    if conversation_ref:
        lock_dir = conversation_lock_dir(
            profile=getattr(args, "profile", None),
            state_path=state_path,
        )
        try:
            lock = acquire_conversation_lock(
                conversation_ref=str(conversation_ref),
                lock_dir=lock_dir,
                profile=getattr(args, "profile", None),
                command="send",
                run_id=recorder.run_id,
                run_file=recorder.run_file,
                timeout=_lock_timeout(args),
            )
        except ConversationLockError as exc:
            turn_result = normalize_turn_failure(
                {
                    "status": "locked",
                    "message": "conversation lock could not be acquired",
                    "source": "local-lock",
                },
                conversation=str(conversation_ref),
                exit_code=2,
                raw_error=str(exc),
                error_class="conversation_lock",
            )
            recorder.fail(
                "conversation lock could not be acquired",
                turn_result=turn_result,
            )
            if jsonl:
                print(render_jsonl_event(turn_result), file=stdout)
            _render_lock_failure(exc, args=args, stderr=stderr)
            return 2

    try:
        client = build_client(client_factory, args)
        emit(recorder.event("waiting_for_reply"))

        try:
            if start_new:
                response = client.send(prompt, **options)
            else:
                response = client.send_to_conversation(conversation_ref, prompt, **options)
        except Exception as exc:  # noqa: BLE001 - command boundary converts SDK errors to exit codes.
            failure = classify_turn_failure(exc)
            turn_result = normalize_turn_failure(
                failure.to_dict(),
                conversation=str(conversation_ref) if conversation_ref else None,
                raw_error=str(exc),
            )
            recorder.fail(
                str(exc),
                failure_classification=failure.to_dict(),
                turn_result=turn_result,
            )
            if jsonl:
                print(render_jsonl_event(turn_result), file=stdout)
            print(f"gptty: send request failed: {failure.message}", file=stderr)
            return 1

        updated_ref = extract_conversation_ref(response, fallback=conversation_ref)
        if updated_ref and recorder.summary.get("conversation_ref") != updated_ref:
            emit(recorder.bind_conversation(updated_ref))

        normalized = normalize_response(response, conversation=updated_ref)
        response_text_value = normalized.get("text", "")
        if not saw_stream_token and response_text_value:
            emit(recorder.event("token_delta", text=response_text_value, role="assistant"))

        action_state = None
        if not saw_stream_token and not response_text_value:
            action_state = required_action_state(
                client,
                response,
                fallback_conversation=updated_ref,
            )
        if action_state is not None:
            action, action_conversation = action_state
            action_payload = normalize_required_action(
                action,
                conversation=action_conversation,
            )
            action_event = recorder.event("required_action", **action_payload)
            observations.append(action_event)
            emit(action_event)
            turn_result = normalize_turn_failure(
                {
                    "status": "blocked",
                    "message": REQUIRED_ACTION_RUN_ERROR,
                    "source": "required_action",
                    "code": "required_action",
                },
                conversation=action_payload.get("conversation") or updated_ref,
                error_class="required_action",
            )
            recorder.fail(REQUIRED_ACTION_RUN_ERROR, turn_result=turn_result)
            if jsonl:
                print(render_jsonl_event(turn_result), file=stdout)
            print(
                render_required_action(action, conversation=action_conversation),
                file=stderr,
            )
            return 1

        if stream and not jsonl:
            if saw_stream_token:
                print(file=stdout)
            else:
                print(render_response(normalized, "plain"), file=stdout)
        elif output_format in {"plain", "markdown"}:
            print(render_response(normalized, output_format), file=stdout)

        session_changed = False
        mutates_session = start_new or explicit_ref is None
        if (
            mutates_session
            and updated_ref
            and updated_ref != state.current_conversation
        ):
            state.current_conversation = updated_ref
            session_changed = True
        if (
            mutates_session
            and requested_model is not None
            and requested_model != state.model
        ):
            state.model = requested_model
            session_changed = True
        if (
            mutates_session
            and getattr(args, "effort", None) is not None
            and requested_effort != state.reasoning_effort
        ):
            state.reasoning_effort = requested_effort
            session_changed = True
        if session_changed:
            try:
                state_handle.save(state)
            except SessionStateError as exc:
                emit(
                    recorder.event(
                        "local_session_state_not_updated",
                        message=str(exc),
                    )
                )
                print(
                    "gptty: ChatGPT turn completed, but local session state was not "
                    f"updated: {exc}",
                    file=stderr,
                )

        turn_result = normalize_turn_result(
            response,
            conversation=updated_ref,
            observations=observations,
        )
        recorder.complete(turn_result=turn_result)
        if jsonl:
            print(render_jsonl_event(turn_result), file=stdout)
        elif output_format == "json":
            print(render_response(turn_result, "json"), file=stdout)
        return 0
    finally:
        if lock is not None:
            lock.release()


def _lock_timeout(args: Any) -> float:
    value = getattr(args, "lock_timeout", None)
    if value is not None:
        return max(0.0, float(value))
    if bool(getattr(args, "wait_lock", False)):
        return 120.0
    return DEFAULT_LOCK_TIMEOUT_SECONDS


def _render_lock_failure(exc: ConversationLockError, *, args: Any, stderr: TextIO) -> None:
    if getattr(args, "lock_timeout", None) is not None or bool(getattr(args, "wait_lock", False)):
        render_lock_timeout(exc, stderr=stderr)
    else:
        render_lock_error(exc, stderr=stderr)


def extract_conversation_ref(response: Any, fallback: Any = None) -> str | None:
    def resolve(value: Any, *, depth: int = 0) -> str | None:
        if value is None or depth > 2:
            return None
        if isinstance(value, str):
            return value if value.strip() else None
        if isinstance(value, dict):
            for field in CONVERSATION_REF_FIELDS:
                candidate = value.get(field)
                if candidate:
                    return str(candidate)
            nested = value.get("conversation")
            if nested is not value:
                resolved = resolve(nested, depth=depth + 1)
                if resolved:
                    return resolved
            return None

        for field in CONVERSATION_REF_FIELDS:
            candidate = getattr(value, field, None)
            if candidate:
                return str(candidate)
        nested = getattr(value, "conversation", None)
        if nested is not value:
            return resolve(nested, depth=depth + 1)
        return None

    resolved = resolve(response)
    if resolved:
        return resolved
    if fallback:
        return str(fallback)
    return None


def response_text(response: Any) -> str:
    return normalize_response(response).get("text", "")
