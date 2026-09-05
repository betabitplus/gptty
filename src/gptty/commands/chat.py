from __future__ import annotations

import sys
import threading
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
    render_stale_lock_recovered,
)
from ..output import normalize_messages, render_live_event
from ..runs import RunRecorder, start_run
from ..sdk_client import GpttyClient
from ..state import ChatState, StateError, load_chat_state, save_chat_state
from ..ui.commands import InteractiveCommands
from ..ui.notifications import notify_response_complete
from ..ui.renderer import PrettyRenderer
from ..ui.session import InteractiveSession, should_use_enhanced_ui
from ..ui.signals import TurnControlSignals, turn_control_signals
from ..ui.state import history_path, ui_settings_path

CHAT_HELP = """Commands:
  /help        Show this help
  /new         Start a new chat state
  /exit        Exit chat
  /quit        Exit chat
"""

LOCAL_QUIT_CODE = 97


CONVERSATION_REF_FIELDS = (
    "conversation_id",
    "conversation_url",
    "conversation_ref",
    "url",
    "id",
)


def extract_conversation_ref(response: Any) -> str | None:
    candidates = [response]
    nested = response.get("conversation") if isinstance(response, dict) else getattr(response, "conversation", None)
    if nested is not None:
        candidates.append(nested)

    for candidate in candidates:
        if isinstance(candidate, dict):
            for field in CONVERSATION_REF_FIELDS:
                value = candidate.get(field)
                if value:
                    return str(value)
            continue
        for field in CONVERSATION_REF_FIELDS:
            value = getattr(candidate, field, None)
            if value:
                return str(value)

    return None


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    if isinstance(response, dict):
        for field in ("text", "message", "content"):
            value = response.get(field)
            if value is not None:
                return str(value)
    if response is None:
        return ""
    return str(response)


def response_title(response: Any) -> str | None:
    value = response.get("title") if isinstance(response, dict) else getattr(response, "title", None)
    if not isinstance(value, str):
        return None
    title = " ".join(value.split())
    return title or None


def run_chat(
    args: Any,
    *,
    client_factory: Callable[..., Any] = GpttyClient,
    input_stream: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    state_path = Path(getattr(args, "state", "gptty_state.json"))
    try:
        state = load_chat_state(state_path)
    except StateError as exc:
        print(f"gptty: {exc}", file=stderr)
        return 1

    model = getattr(args, "model", None)
    if model and model != state.model:
        state.model = model
        try:
            save_chat_state(state_path, state)
        except StateError as exc:
            print(f"gptty: {exc}", file=stderr)
            return 1

    client: Any | None = None
    interactive = _is_interactive(input_stream)
    enhanced, ui_settings = should_use_enhanced_ui(
        input_stream=input_stream,
        output_stream=stdout,
        state_path=state_path,
        force_plain=bool(getattr(args, "plain", False)),
    )

    def get_client() -> Any:
        nonlocal client
        if client is None:
            client = client_factory(
                auth_file=getattr(args, "auth", "auth_data.json"),
                timeout=getattr(args, "timeout", 90),
            )
        return client

    ui: InteractiveSession | None = None
    renderer: PrettyRenderer | None = None
    interactive_commands: InteractiveCommands | None = None
    if enhanced:
        ui = InteractiveSession(
            history_file=history_path(state_path),
            settings_file=ui_settings_path(state_path),
            settings=ui_settings,
        )
        renderer = PrettyRenderer(stdout, ui_settings)
        interactive_commands = InteractiveCommands(
            state=state,
            state_path=state_path,
            get_client=get_client,
            ui=ui,
            renderer=renderer,
        )
        renderer.header(
            profile=getattr(args, "profile", None),
            conversation=state.current_conversation,
            model=state.model or "latest frontier · High",
        )

    while True:
        if interactive and not enhanced:
            print("> ", end="", file=stdout, flush=True)

        try:
            if ui is not None:
                attachment_count = interactive_commands.pending_media_count if interactive_commands is not None else 0
                line = ui.read_prompt(attachment_count=attachment_count)
            else:
                line = input_stream.readline()
        except KeyboardInterrupt:
            if interactive_commands is not None:
                interactive_commands.close()
            print(file=stdout)
            return 130
        except EOFError:
            if interactive_commands is not None:
                interactive_commands.close()
            print(file=stdout)
            return 0

        if ui is None and line == "":
            if interactive_commands is not None:
                interactive_commands.close()
            if interactive:
                print(file=stdout)
            return 0

        prompt = line.strip()
        if not prompt:
            continue

        if prompt == "/" and ui is not None:
            selected = ui.choose_command()
            if not selected:
                continue
            prompt = selected

        if prompt.startswith("/"):
            if interactive_commands is not None:
                result = interactive_commands.handle(prompt)
            else:
                result = _handle_chat_command(prompt, state=state, state_path=state_path, stdout=stdout, stderr=stderr)
            if result is not None:
                if interactive_commands is not None:
                    interactive_commands.close()
                return result
            continue

        media = interactive_commands.pending_media if interactive_commands is not None else None
        conversation_mode = interactive_commands.conversation_mode if interactive_commands is not None else "normal"
        attached_ref = interactive_commands.conversation_ref if interactive_commands is not None else state.current_conversation
        turn_client = get_client()
        with turn_control_signals(enabled=renderer is not None) as turn_controls:
            if renderer is not None:
                renderer.turn_start()
            code = _send_chat_prompt(
                turn_client,
                state=state,
                state_path=state_path,
                profile=getattr(args, "profile", None),
                prompt=prompt,
                model=state.model,
                media=media,
                stream=not bool(getattr(args, "no_stream", False)),
                lock_timeout=_lock_timeout(args),
                explicit_lock_wait=bool(getattr(args, "wait_lock", False)) or getattr(args, "lock_timeout", None) is not None,
                stdout=stdout,
                stderr=stderr,
                renderer=renderer,
                turn_controls=turn_controls,
                conversation_mode=conversation_mode,
                attached_ref=attached_ref,
                temporary_turn_recorder=(
                    interactive_commands.record_temporary_turn
                    if interactive_commands is not None and conversation_mode == "temporary"
                    else None
                ),
            )
        if code == LOCAL_QUIT_CODE:
            if interactive_commands is not None:
                interactive_commands.close()
            return 0
        if code != 0:
            if interactive_commands is not None:
                interactive_commands.close()
            return code
        if interactive_commands is not None and media:
            interactive_commands.clear_pending_media()


def _handle_chat_command(
    command: str,
    *,
    state: ChatState,
    state_path: Path,
    stdout: TextIO,
    stderr: TextIO,
) -> int | None:
    if command in {"/exit", "/quit"}:
        return 0
    if command == "/help":
        print(CHAT_HELP.rstrip(), file=stdout)
        return None
    if command == "/new":
        state.current_conversation = None
        try:
            save_chat_state(state_path, state)
        except StateError as exc:
            print(f"gptty: {exc}", file=stderr)
            return 1
        print("Started a new chat.", file=stdout)
        return None

    print(f"Unknown command: {command}. Type /help for available commands.", file=stderr)
    return None


def _send_chat_prompt(
    client: Any,
    *,
    state: ChatState,
    state_path: Path,
    profile: str | None,
    prompt: str,
    model: str | None,
    media: list[str] | None,
    stream: bool,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    explicit_lock_wait: bool = False,
    stdout: TextIO,
    stderr: TextIO,
    renderer: PrettyRenderer | None = None,
    turn_controls: TurnControlSignals | None = None,
    conversation_mode: str = "normal",
    attached_ref: str | None = None,
    temporary_turn_recorder: Callable[..., None] | None = None,
) -> int:
    saw_stream_token = False
    stream_tokens: list[str] = []
    recorder: RunRecorder | None = None
    completed_successfully = False
    stopped_by_user = False
    local_quit_requested = False
    write_committed = threading.Event()
    controls = turn_controls or TurnControlSignals()
    if conversation_mode not in {"normal", "temporary"}:
        raise ValueError(f"unsupported conversation mode: {conversation_mode}")
    is_temporary = conversation_mode == "temporary"
    active_ref = attached_ref if is_temporary else state.current_conversation
    if active_ref and not is_temporary:
        recorder = start_run(
            profile=profile,
            state_path=state_path,
            command="chat",
            conversation_ref=active_ref,
        )
        recorder.event("prompt_sent")

    def on_token(token: str) -> None:
        nonlocal saw_stream_token
        saw_stream_token = True
        stream_tokens.append(token)
        if recorder is not None:
            recorder.event("token_delta", text=token)
        if renderer is None:
            print(token, end="", file=stdout, flush=True)

    def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "browser_native_write_completed":
            write_committed.set()
        if stopped_by_user or local_quit_requested:
            return
        if renderer is not None:
            renderer.live_event(event)
            return
        rendered = render_live_event(event)
        if rendered:
            print(rendered, file=stderr, flush=True)

    options: dict[str, Any] = {"stream": stream}
    if model:
        options["model"] = model
    if media:
        options["media"] = media
    if stream:
        options["on_token"] = on_token
    if stream or renderer is not None:
        options["on_event"] = on_event

    lock = None
    if state.current_conversation:
        lock_dir = conversation_lock_dir(profile=profile, state_path=state_path)
        try:
            lock = acquire_conversation_lock(
                conversation_ref=state.current_conversation,
                lock_dir=lock_dir,
                profile=profile,
                command="chat",
                run_id=recorder.run_id if recorder is not None else None,
                run_file=recorder.run_file if recorder is not None else None,
                timeout=lock_timeout,
            )
        except ConversationLockError as exc:
            if recorder is not None:
                recorder.fail("conversation lock could not be acquired")
            if renderer is not None:
                renderer.turn_abort()
            if explicit_lock_wait:
                render_lock_timeout(exc, stderr=stderr)
            else:
                render_lock_error(exc, stderr=stderr)
            return 2
        render_stale_lock_recovered(lock, stderr=stderr)

    try:
        if recorder is not None:
            recorder.event("waiting_for_reply")
        send_conversation_ref = active_ref

        def perform_send() -> Any:
            if is_temporary:
                return client.send_temporary(prompt, **options)
            if send_conversation_ref:
                return client.send_to_conversation(send_conversation_ref, prompt, **options)
            return client.send(prompt, **options)

        if renderer is None:
            try:
                response = perform_send()
            except Exception as exc:  # noqa: BLE001 - command boundary converts SDK errors to exit codes.
                if recorder is not None:
                    recorder.fail(str(exc))
                print(f"gptty: chat request failed: {exc}", file=stderr)
                return 1
        else:
            outcome: dict[str, Any] = {}

            def send_worker() -> None:
                try:
                    outcome["response"] = perform_send()
                except BaseException as exc:  # noqa: BLE001 - main thread owns Ctrl-C; worker must surface all exits.
                    outcome["error"] = exc

            worker = threading.Thread(target=send_worker, name="gptty-chat-turn", daemon=True)
            worker.start()
            quit_wait_notice_shown = False
            while worker.is_alive():
                worker.join(timeout=0.1)
                if controls.quit_requested.is_set():
                    if write_committed.is_set():
                        local_quit_requested = True
                        renderer.turn_abort()
                        renderer.info("Exited gptty; ChatGPT response continues in browser.")
                        return LOCAL_QUIT_CODE
                    if not quit_wait_notice_shown:
                        quit_wait_notice_shown = True
                        renderer.info("Waiting for safe ChatGPT handoff before local exit…")
                if not controls.consume_stop():
                    continue
                if stopped_by_user:
                    renderer.info("Stop already requested; waiting for ChatGPT to save the partial response…")
                    continue
                renderer.info("Stopping ChatGPT…")
                try:
                    stop_result = client.stop_generation(active_ref, timeout=30.0)
                except Exception as exc:  # noqa: BLE001 - interactive stop is best-effort at this boundary.
                    renderer.warning(f"Stop failed: {exc}")
                    continue

                stopped = (
                    bool(stop_result.get("stopped"))
                    if isinstance(stop_result, dict)
                    else bool(getattr(stop_result, "stopped", False))
                )
                if not stopped:
                    renderer.warning("No active ChatGPT response to stop yet; press Ctrl-C again to retry.")
                    continue

                stopped_by_user = True
                stop_ref = (
                    stop_result.get("conversationId")
                    if isinstance(stop_result, dict)
                    else getattr(stop_result, "conversation_id", None)
                )
                if (
                    isinstance(stop_ref, str)
                    and stop_ref.strip()
                    and not stop_ref.strip().startswith("WEB:")
                ):
                    active_ref = stop_ref.strip()
                    if not is_temporary and not state.current_conversation:
                        state.current_conversation = active_ref
                        try:
                            save_chat_state(state_path, state)
                        except StateError as exc:
                            renderer.warning(str(exc))
                renderer.turn_abort()
                renderer.info("Stop requested; waiting for ChatGPT to save the partial response…")

            if controls.quit_requested.is_set():
                local_quit_requested = True
                renderer.turn_abort()
                renderer.info("Exited gptty; ChatGPT response continues in browser.")
                return LOCAL_QUIT_CODE

            response: Any = None
            error = outcome.get("error")
            if error is not None and stopped_by_user:
                if is_temporary:
                    response = {
                        "text": "".join(stream_tokens),
                        "conversation_id": active_ref,
                    }
                elif state.current_conversation:
                    try:
                        snapshot = client.conversation_snapshot(state.current_conversation)
                        response = _stopped_snapshot_response(
                            snapshot,
                            conversation_ref=state.current_conversation,
                        )
                    except Exception as reconcile_error:  # noqa: BLE001 - confirmed Stop remains a normal user action.
                        renderer.warning(
                            "Stopped ChatGPT, but the saved partial response could not be read yet: "
                            f"{reconcile_error}"
                        )
                        response = {
                            "text": "",
                            "conversation_id": state.current_conversation,
                        }
                else:
                    renderer.warning(
                        "Stopped ChatGPT before the new conversation route was committed; "
                        "use /resume to reopen it if ChatGPT saved the chat."
                    )
                    response = {"text": ""}
                error = None
            if error is not None:
                if isinstance(error, Exception):
                    if recorder is not None:
                        recorder.fail(str(error))
                    renderer.turn_abort()
                    print(f"gptty: chat request failed: {error}", file=stderr)
                    return 1
                raise error
            if "response" in outcome:
                response = outcome["response"]
            elif not stopped_by_user:
                response = None

        text = response_text(response)
        rendered_text = text or "".join(stream_tokens)
        if renderer is not None:
            renderer.answer(rendered_text)
            if stopped_by_user:
                renderer.info("Stopped by user.")
        elif stream:
            if saw_stream_token:
                print(file=stdout)
            else:
                if text and recorder is not None:
                    recorder.event("token_delta", text=text)
                print(text, file=stdout)
        else:
            if text and recorder is not None:
                recorder.event("token_delta", text=text)
            print(text, file=stdout)

        conversation_ref = extract_conversation_ref(response) or active_ref
        if is_temporary:
            if temporary_turn_recorder is not None:
                temporary_turn_recorder(
                    prompt=prompt,
                    answer=rendered_text,
                    conversation_ref=conversation_ref,
                    title=response_title(response),
                )
        elif conversation_ref and conversation_ref != state.current_conversation:
            state.current_conversation = conversation_ref
            try:
                save_chat_state(state_path, state)
            except StateError as exc:
                if recorder is not None:
                    recorder.fail(str(exc))
                print(f"gptty: {exc}", file=stderr)
                return 1

        if renderer is not None and not is_temporary and state.current_conversation:
            renderer.chat_link(state.current_conversation)

        if recorder is not None:
            if stopped_by_user:
                recorder.event("stopped_by_user")
            recorder.complete()
        completed_successfully = True
        return 0
    finally:
        if lock is not None:
            lock.release()
        if renderer is not None:
            renderer.turn_abort()
            if completed_successfully and not stopped_by_user:
                notify_response_complete(
                    chat_title=response_title(response) or ("Temporary Chat" if is_temporary else None),
                    final_response=rendered_text,
                )


def _stopped_snapshot_response(snapshot: Any, *, conversation_ref: str) -> dict[str, str]:
    raw_messages = snapshot.get("messages") if isinstance(snapshot, dict) else getattr(snapshot, "messages", None)
    try:
        messages = list(raw_messages) if raw_messages is not None else []
    except TypeError:
        messages = []

    text = ""
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "assistant":
            continue
        recipient = message.get("recipient") if isinstance(message, dict) else getattr(message, "recipient", None)
        if recipient not in {None, "", "all"}:
            continue
        normalized = normalize_messages([message])
        if normalized:
            text = normalized[0].text
        break

    return {
        "text": text,
        "conversation_id": conversation_ref,
    }


def _lock_timeout(args: Any) -> float:
    value = getattr(args, "lock_timeout", None)
    if value is not None:
        return max(0.0, float(value))
    if bool(getattr(args, "wait_lock", False)):
        return 120.0
    return DEFAULT_LOCK_TIMEOUT_SECONDS


def _is_interactive(input_stream: TextIO) -> bool:
    try:
        return input_stream.isatty()
    except OSError:
        return False
