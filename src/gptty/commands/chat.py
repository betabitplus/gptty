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
    render_stale_lock_recovered,
)
from ..output import render_live_event
from ..runs import RunRecorder, start_run
from ..sdk_client import GpttyClient
from ..state import ChatState, StateError, load_chat_state, save_chat_state
from ..ui.commands import InteractiveCommands
from ..ui.renderer import PrettyRenderer
from ..ui.session import InteractiveSession, should_use_enhanced_ui
from ..ui.state import history_path, ui_settings_path

CHAT_HELP = """Commands:
  /help        Show this help
  /new         Start a new chat state
  /exit        Exit chat
  /quit        Exit chat
"""

CONVERSATION_REF_FIELDS = (
    "conversation_id",
    "conversation_url",
    "conversation_ref",
    "url",
    "id",
)


def extract_conversation_ref(response: Any) -> str | None:
    if isinstance(response, dict):
        for field in CONVERSATION_REF_FIELDS:
            value = response.get(field)
            if value:
                return str(value)

    for field in CONVERSATION_REF_FIELDS:
        value = getattr(response, field, None)
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
        renderer.header(profile=getattr(args, "profile", None), conversation=state.current_conversation)

    while True:
        if interactive and not enhanced:
            print("> ", end="", file=stdout, flush=True)

        try:
            if ui is not None:
                line = ui.read_prompt()
            else:
                line = input_stream.readline()
        except KeyboardInterrupt:
            print(file=stdout)
            return 130
        except EOFError:
            print(file=stdout)
            return 0

        if ui is None and line == "":
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
                return result
            continue

        if renderer is not None:
            renderer.turn_start()
        code = _send_chat_prompt(
            get_client(),
            state=state,
            state_path=state_path,
            profile=getattr(args, "profile", None),
            prompt=prompt,
            model=state.model,
            stream=not bool(getattr(args, "no_stream", False)),
            lock_timeout=_lock_timeout(args),
            explicit_lock_wait=bool(getattr(args, "wait_lock", False)) or getattr(args, "lock_timeout", None) is not None,
            stdout=stdout,
            stderr=stderr,
            renderer=renderer,
        )
        if code != 0:
            return code


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
    stream: bool,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    explicit_lock_wait: bool = False,
    stdout: TextIO,
    stderr: TextIO,
    renderer: PrettyRenderer | None = None,
) -> int:
    saw_stream_token = False
    stream_tokens: list[str] = []
    recorder: RunRecorder | None = None
    if state.current_conversation:
        recorder = start_run(
            profile=profile,
            state_path=state_path,
            command="chat",
            conversation_ref=state.current_conversation,
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
        if renderer is not None:
            renderer.live_event(event)
            return
        rendered = render_live_event(event)
        if rendered:
            print(rendered, file=stderr, flush=True)

    options: dict[str, Any] = {"stream": stream}
    if model:
        options["model"] = model
    if stream:
        options["on_token"] = on_token
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
            if explicit_lock_wait:
                render_lock_timeout(exc, stderr=stderr)
            else:
                render_lock_error(exc, stderr=stderr)
            return 2
        render_stale_lock_recovered(lock, stderr=stderr)

    try:
        if recorder is not None:
            recorder.event("waiting_for_reply")
        try:
            if state.current_conversation:
                response = client.send_to_conversation(state.current_conversation, prompt, **options)
            else:
                response = client.send(prompt, **options)
        except Exception as exc:  # noqa: BLE001 - command boundary converts SDK errors to exit codes.
            if recorder is not None:
                recorder.fail(str(exc))
            print(f"gptty: chat request failed: {exc}", file=stderr)
            return 1

        text = response_text(response)
        if renderer is not None:
            renderer.answer(text or "".join(stream_tokens))
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

        conversation_ref = extract_conversation_ref(response)
        if conversation_ref and conversation_ref != state.current_conversation:
            state.current_conversation = conversation_ref
            try:
                save_chat_state(state_path, state)
            except StateError as exc:
                if recorder is not None:
                    recorder.fail(str(exc))
                print(f"gptty: {exc}", file=stderr)
                return 1

        if recorder is not None:
            recorder.complete()
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


def _is_interactive(input_stream: TextIO) -> bool:
    try:
        return input_stream.isatty()
    except OSError:
        return False
