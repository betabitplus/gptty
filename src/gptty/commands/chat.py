from __future__ import annotations

import asyncio
import shlex
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from prompt_toolkit.patch_stdout import patch_stdout

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


@dataclass
class _EnhancedLoopOutcome:
    exit_code: int | None = None
    command: str | None = None


@dataclass
class _EnhancedTurn:
    task: asyncio.Task[int]
    controls: TurnControlSignals
    result: dict[str, Any]
    goal_turn: bool
    media: list[str]
    started_at: float
    pause_goal_after_turn: bool = False
    exit_after_turn: bool = False


class _PromptAwareStream:
    """Write through prompt_toolkit's patched stdio only while its app is active."""

    def __init__(self, base: TextIO, *, stream_name: str) -> None:
        self._base = base
        self._stream_name = stream_name

    def _target(self) -> TextIO:
        current = getattr(sys, self._stream_name)
        return current if current is not self._base else self._base

    def write(self, text: str) -> int:
        return self._target().write(text)

    def flush(self) -> None:
        self._target().flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


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

    startup_goal_paused = False
    if state.goal is not None and state.goal.status == "active":
        state.goal.status = "paused"
        state.goal.reason = "gptty restarted while goal was active"
        startup_goal_paused = True
        try:
            save_chat_state(state_path, state)
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
        prompt_patch_enabled = stdout is sys.stdout or stderr is sys.stderr
        renderer_stdout: TextIO = (
            _PromptAwareStream(stdout, stream_name="stdout") if stdout is sys.stdout else stdout
        )
        renderer_stderr: TextIO = (
            _PromptAwareStream(stderr, stream_name="stderr") if stderr is sys.stderr else stderr
        )
        renderer = PrettyRenderer(renderer_stdout, ui_settings)
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
        if startup_goal_paused and state.goal is not None:
            renderer.info("Goal · paused after restart · use /goal resume")
        queued_prompts: deque[str] = deque()
        while True:
            outcome = _run_enhanced_loop(
                args=args,
                state=state,
                state_path=state_path,
                get_client=get_client,
                ui=ui,
                renderer=renderer,
                commands=interactive_commands,
                queued_prompts=queued_prompts,
                stdout=renderer_stdout,
                stderr=renderer_stderr,
                patch_stdout_enabled=prompt_patch_enabled,
            )
            if outcome.exit_code is not None:
                interactive_commands.close()
                return outcome.exit_code
            prompt = (outcome.command or "").strip()
            if not prompt:
                continue
            if prompt == "/":
                selected = ui.choose_command()
                if not selected:
                    continue
                prompt = selected
            if not prompt.startswith("/"):
                queued_prompts.append(prompt)
                continue
            result = interactive_commands.handle(prompt)
            if result is not None:
                interactive_commands.close()
                return result

    while True:
        automatic_prompt = interactive_commands.pop_automatic_prompt() if interactive_commands is not None else None
        automatic_turn = automatic_prompt is not None

        if automatic_turn:
            prompt = automatic_prompt or ""
        else:
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

            if interactive_commands is not None:
                prompt = interactive_commands.prepare_goal_user_prompt(prompt)

        media = interactive_commands.pending_media if interactive_commands is not None else None
        conversation_mode = interactive_commands.conversation_mode if interactive_commands is not None else "normal"
        attached_ref = interactive_commands.conversation_ref if interactive_commands is not None else state.current_conversation
        goal_turn = bool(interactive_commands is not None and interactive_commands.goal_active)
        turn_result: dict[str, Any] = {}
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
                notify_completion=not goal_turn,
                result_out=turn_result,
                on_stop_confirmed=(
                    interactive_commands.pause_goal_after_user_stop
                    if goal_turn and interactive_commands is not None
                    else None
                ),
            )
        if code == LOCAL_QUIT_CODE:
            if interactive_commands is not None:
                if goal_turn:
                    interactive_commands.pause_goal_for_local_quit()
                interactive_commands.close()
            return 0
        if code != 0:
            if interactive_commands is not None:
                if goal_turn:
                    interactive_commands.handle_goal_interruption(f"chat turn failed with exit code {code}")
                interactive_commands.close()
            return code
        if interactive_commands is not None and media:
            interactive_commands.clear_pending_media()
        if interactive_commands is not None and goal_turn:
            interactive_commands.handle_goal_turn_result(turn_result)


def _run_enhanced_loop(
    *,
    args: Any,
    state: ChatState,
    state_path: Path,
    get_client: Callable[[], Any],
    ui: InteractiveSession,
    renderer: PrettyRenderer,
    commands: InteractiveCommands,
    queued_prompts: deque[str],
    stdout: TextIO,
    stderr: TextIO,
    patch_stdout_enabled: bool,
) -> _EnhancedLoopOutcome:
    return asyncio.run(
        _run_enhanced_loop_async(
            args=args,
            state=state,
            state_path=state_path,
            get_client=get_client,
            ui=ui,
            renderer=renderer,
            commands=commands,
            queued_prompts=queued_prompts,
            stdout=stdout,
            stderr=stderr,
            patch_stdout_enabled=patch_stdout_enabled,
        )
    )


async def _run_enhanced_loop_async(
    *,
    args: Any,
    state: ChatState,
    state_path: Path,
    get_client: Callable[[], Any],
    ui: InteractiveSession,
    renderer: PrettyRenderer,
    commands: InteractiveCommands,
    queued_prompts: deque[str],
    stdout: TextIO,
    stderr: TextIO,
    patch_stdout_enabled: bool,
) -> _EnhancedLoopOutcome:
    output_context = patch_stdout(raw=True) if patch_stdout_enabled else nullcontext()
    with output_context:
        return await _enhanced_loop_core(
            args=args,
            state=state,
            state_path=state_path,
            get_client=get_client,
            ui=ui,
            renderer=renderer,
            commands=commands,
            queued_prompts=queued_prompts,
            stdout=stdout,
            stderr=stderr,
        )


async def _enhanced_loop_core(
    *,
    args: Any,
    state: ChatState,
    state_path: Path,
    get_client: Callable[[], Any],
    ui: InteractiveSession,
    renderer: PrettyRenderer,
    commands: InteractiveCommands,
    queued_prompts: deque[str],
    stdout: TextIO,
    stderr: TextIO,
) -> _EnhancedLoopOutcome:
    active: _EnhancedTurn | None = None
    prompt_task: asyncio.Task[str] | None = None
    accepting_input = True

    while True:
        if active is None:
            next_prompt: str | None = None
            automatic_turn = False
            if queued_prompts:
                commands.clear_automatic_prompts()
                next_prompt = queued_prompts.popleft()
            else:
                next_prompt = commands.pop_automatic_prompt()
                automatic_turn = next_prompt is not None
            if next_prompt is not None:
                active = _start_enhanced_turn(
                    args=args,
                    state=state,
                    state_path=state_path,
                    get_client=get_client,
                    ui=ui,
                    renderer=renderer,
                    commands=commands,
                    queued_prompts=queued_prompts,
                    stdout=stdout,
                    stderr=stderr,
                    prompt=next_prompt,
                    automatic_turn=automatic_turn,
                )

        if prompt_task is None and accepting_input:
            prompt_task = asyncio.create_task(
                ui.read_prompt_async(attachment_count=commands.pending_media_count)
            )

        wait_for: set[asyncio.Task[Any]] = set()
        if prompt_task is not None:
            wait_for.add(prompt_task)
        if active is not None:
            wait_for.add(active.task)
        if not wait_for:
            return _EnhancedLoopOutcome(exit_code=0)

        done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

        if prompt_task is not None and prompt_task in done:
            finished_prompt = prompt_task
            prompt_task = None
            try:
                raw = finished_prompt.result()
            except KeyboardInterrupt:
                if active is None:
                    return _EnhancedLoopOutcome(exit_code=130)
                active.controls.request_stop()
            except EOFError:
                if active is None:
                    return _EnhancedLoopOutcome(exit_code=0)
                active.controls.request_quit()
                accepting_input = False
            else:
                prompt = raw.strip()
                if prompt:
                    if active is None:
                        if prompt.startswith("/"):
                            return _EnhancedLoopOutcome(command=prompt)
                        queued_prompts.append(prompt)
                    else:
                        accepting_input = _handle_working_input(
                            prompt,
                            active=active,
                            commands=commands,
                            renderer=renderer,
                            queued_prompts=queued_prompts,
                        )
                        _refresh_active_turn_ui(ui, active, queued_prompts)

        if active is not None and active.task in done:
            finished_turn = active
            active = None
            await asyncio.sleep(0)
            ui.set_active_turn(None)
            commands.release_media(finished_turn.media)
            outcome = await _finish_enhanced_turn(
                finished_turn,
                commands=commands,
                renderer=renderer,
                stderr=stderr,
                prompt_task=prompt_task,
                queued_prompts=queued_prompts,
            )
            if outcome is not None:
                return outcome
            accepting_input = True


async def _finish_enhanced_turn(
    turn: _EnhancedTurn,
    *,
    commands: InteractiveCommands,
    renderer: PrettyRenderer,
    stderr: TextIO,
    prompt_task: asyncio.Task[str] | None,
    queued_prompts: deque[str],
) -> _EnhancedLoopOutcome | None:
    try:
        code = turn.task.result()
    except BaseException as exc:  # noqa: BLE001 - async orchestration boundary.
        renderer.turn_abort()
        print(f"gptty: chat request failed: {exc}", file=stderr)
        await _cancel_prompt_task(prompt_task)
        return _EnhancedLoopOutcome(exit_code=1)

    if code == LOCAL_QUIT_CODE:
        if turn.goal_turn:
            commands.pause_goal_for_local_quit()
        await _cancel_prompt_task(prompt_task)
        return _EnhancedLoopOutcome(exit_code=0)
    if code != 0:
        if turn.goal_turn:
            commands.handle_goal_interruption(f"chat turn failed with exit code {code}")
        await _cancel_prompt_task(prompt_task)
        return _EnhancedLoopOutcome(exit_code=code)

    if turn.result.get("stopped_by_user"):
        queued_count = len(queued_prompts)
        queued_prompts.clear()
        commands.clear_automatic_prompts()
        if queued_count:
            renderer.info(f"Cleared {queued_count} queued prompt{'s' if queued_count != 1 else ''} after Stop.")

    if turn.goal_turn:
        commands.handle_goal_turn_result(turn.result)
        if turn.pause_goal_after_turn and commands.goal_active:
            commands.handle("/goal pause")

    if turn.exit_after_turn or turn.controls.quit_requested.is_set():
        if turn.goal_turn and commands.goal_active:
            commands.pause_goal_for_local_quit()
        await _cancel_prompt_task(prompt_task)
        return _EnhancedLoopOutcome(exit_code=0)
    return None


async def _cancel_prompt_task(task: asyncio.Task[str] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, KeyboardInterrupt, EOFError):
        pass


def _start_enhanced_turn(
    *,
    args: Any,
    state: ChatState,
    state_path: Path,
    get_client: Callable[[], Any],
    ui: InteractiveSession,
    renderer: PrettyRenderer,
    commands: InteractiveCommands,
    queued_prompts: deque[str],
    stdout: TextIO,
    stderr: TextIO,
    prompt: str,
    automatic_turn: bool,
) -> _EnhancedTurn:
    if automatic_turn:
        media: list[str] = []
    else:
        prompt = commands.prepare_goal_user_prompt(prompt)
        media = commands.take_pending_media()

    conversation_mode = commands.conversation_mode
    attached_ref = commands.conversation_ref
    goal_turn = commands.goal_active
    turn_result: dict[str, Any] = {}
    controls = TurnControlSignals()
    started_at = time.monotonic()
    renderer.turn_start(show_elapsed=False)

    loop = asyncio.get_running_loop()

    def stop_confirmed(ref: str | None) -> None:
        loop.call_soon_threadsafe(commands.pause_goal_after_user_stop, ref)

    task = asyncio.create_task(
        asyncio.to_thread(
            _send_chat_prompt,
            get_client(),
            state=state,
            state_path=state_path,
            profile=getattr(args, "profile", None),
            prompt=prompt,
            model=state.model,
            media=media or None,
            stream=not bool(getattr(args, "no_stream", False)),
            lock_timeout=_lock_timeout(args),
            explicit_lock_wait=bool(getattr(args, "wait_lock", False))
            or getattr(args, "lock_timeout", None) is not None,
            stdout=stdout,
            stderr=stderr,
            renderer=renderer,
            turn_controls=controls,
            conversation_mode=conversation_mode,
            attached_ref=attached_ref,
            temporary_turn_recorder=(
                commands.record_temporary_turn if conversation_mode == "temporary" else None
            ),
            notify_completion=not goal_turn,
            result_out=turn_result,
            on_stop_confirmed=stop_confirmed if goal_turn else None,
        )
    )
    active = _EnhancedTurn(
        task=task,
        controls=controls,
        result=turn_result,
        goal_turn=goal_turn,
        media=media,
        started_at=started_at,
    )
    _refresh_active_turn_ui(ui, active, queued_prompts)
    return active


def _refresh_active_turn_ui(
    ui: InteractiveSession,
    active: _EnhancedTurn,
    queued_prompts: deque[str],
) -> None:
    ui.set_active_turn(
        active.controls,
        working_status=lambda: _working_status(active.started_at, len(queued_prompts)),
    )


def _working_status(started_at: float, queued_count: int) -> str:
    elapsed = max(0, int(time.monotonic() - started_at))
    minutes, seconds = divmod(elapsed, 60)
    status = f"working {minutes:02d}:{seconds:02d}"
    if queued_count:
        status += f" · queued {queued_count}"
    return status


def _handle_working_input(
    prompt: str,
    *,
    active: _EnhancedTurn,
    commands: InteractiveCommands,
    renderer: PrettyRenderer,
    queued_prompts: deque[str],
) -> bool:
    if not prompt.startswith("/"):
        queued_prompts.append(prompt)
        renderer.info(f"Queued · {len(queued_prompts)}")
        return True

    if prompt == "/":
        renderer.info(
            "While working: /stop · /exit · /goal pause · /goal status · /image PATH · /paste"
        )
        return True

    try:
        parts = shlex.split(prompt)
    except ValueError as exc:
        renderer.warning(f"Invalid command: {exc}")
        return True
    if not parts:
        return True
    name = parts[0].lstrip("/").lower()
    argv = parts[1:]

    if name == "stop":
        if argv:
            renderer.warning("/stop takes no arguments.")
            return True
        active.controls.request_stop()
        return True

    if name in {"exit", "quit"}:
        if argv:
            renderer.warning(f"/{name} takes no arguments.")
            return True
        active.exit_after_turn = True
        active.controls.request_quit()
        return False

    if name == "goal":
        action = argv[0].lower() if len(argv) == 1 else ""
        if action == "pause" and active.goal_turn and commands.goal_active:
            active.pause_goal_after_turn = True
            renderer.info("Goal · pause pending · current turn will finish")
            return True
        if action == "status" or (not argv and commands.goal_active):
            if active.pause_goal_after_turn:
                renderer.info("Goal · active · pause pending")
            else:
                commands.handle(prompt)
            return True
        renderer.warning("While working, only /goal pause and /goal status are available.")
        return True

    if name == "image":
        if not argv:
            renderer.warning("While working, use /image PATH or /image clear.")
            return True
        commands.handle(prompt)
        return True

    if name == "paste":
        commands.handle(prompt)
        return True

    renderer.warning(f"/{name} is unavailable while working; stop the current response first.")
    return True


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
    notify_completion: bool = True,
    result_out: dict[str, Any] | None = None,
    on_stop_confirmed: Callable[[str | None], None] | None = None,
) -> int:
    if result_out is not None:
        result_out.clear()
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
                    local_quit_requested = True
                    renderer.turn_abort()
                    renderer.info("ChatGPT is already stopped; exiting gptty without waiting for local readback.")
                    return LOCAL_QUIT_CODE
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
                if on_stop_confirmed is not None:
                    on_stop_confirmed(active_ref)
                renderer.turn_abort()
                renderer.info("ChatGPT stopped; finalizing local readback…")

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
        if result_out is not None:
            result_out.update(
                text=rendered_text,
                title=response_title(response),
                conversation_ref=conversation_ref,
                stopped_by_user=stopped_by_user,
            )
        completed_successfully = True
        return 0
    finally:
        if lock is not None:
            lock.release()
        if renderer is not None:
            renderer.turn_abort()
            if completed_successfully and not stopped_by_user and notify_completion:
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
