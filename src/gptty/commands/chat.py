from __future__ import annotations

import asyncio
import shlex
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, TextIO

from prompt_toolkit.patch_stdout import patch_stdout

from ..codexpro_activity import CodexProActivitySnapshot, CodexProActivityTracker
from ..stream_delivery import StreamDeliveryJournal
from ..locks import (
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    ConversationLockError,
    acquire_conversation_lock,
    conversation_lock_dir,
    render_lock_error,
    render_lock_timeout,
    render_stale_lock_recovered,
)
from ..output import _tool_result_error, normalize_messages, render_live_event
from ..runs import RunRecorder, start_run
from ..sdk_client import GpttyClient
from ..state import ChatState, StateError, load_chat_state, save_chat_state
from ..ui.commands import (
    UNFINISHED_STATUSES,
    InteractiveCommands,
    ResumeRequest,
    _last_assistant_message_text,
    _message_identity,
    _message_text,
    _snapshot_messages,
    _snapshot_status,
)
from ._client import build_client
from ..ui.notifications import notify_response_complete
from ..ui.renderer import PrettyRenderer
from ..ui.session import InteractiveSession, should_use_enhanced_ui
from ..ui.signals import (
    TurnControlSignals,
    routed_turn_control_signals,
    turn_control_signals,
)
from ..ui.state import history_path, ui_settings_path

CHAT_HELP = """Commands:
  /help        Show this help
  /new         Start a new chat state
  /exit        Exit chat
  /quit        Exit chat
"""

LOCAL_QUIT_CODE = 97
FOLLOW_MIN_INTERVAL_SECONDS = 15.0
FOLLOW_MAX_IDLE_INTERVAL_SECONDS = 60.0
FOLLOW_RATE_LIMIT_BACKOFF_SECONDS = 120.0
FOLLOW_RATE_LIMIT_MAX_BACKOFF_SECONDS = 300.0
FOLLOW_TIMEOUT_SECONDS = 2 * 60 * 60
FOLLOW_MESSAGE_LIMIT = 128
ANSWER_FINALITY_PENDING_SECONDS = 5.0
CODEXPRO_RECENT_ACTIVITY_MAX_AGE_SECONDS = 5 * 60.0


@dataclass
class _EnhancedLoopOutcome:
    exit_code: int | None = None
    command: str | None = None


@dataclass
class _TurnHealth:
    last_server_progress_at: float
    state: str = "working"
    server_idle_seconds: float = 0.0
    reconnect_attempt: int = 0
    delivery_recoveries: int = 0
    answer_progress_seen: bool = False
    last_tool_error: str = ""
    codexpro_tracker: CodexProActivityTracker | None = None
    delivery_journal: StreamDeliveryJournal | None = None
    conversation_ref: str | None = None
    reconnect_reason: str = ""

    def observe(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        now = time.monotonic()
        if event_type in {
            "browser_native_write_identity_resolved",
            "browser_native_write_completed",
        }:
            candidate = event.get("conversation_id") or event.get("conversationId")
            if isinstance(candidate, str) and candidate.strip():
                self.conversation_ref = candidate.strip()
        if self.delivery_journal is not None:
            try:
                self.delivery_journal.observe(self.conversation_ref, event)
            except Exception:
                # Delivery diagnostics must never interfere with a live turn.
                pass
        if event_type == "canonical_intermediate_message":
            if (
                self.codexpro_tracker is not None
                and self.conversation_ref
                and event.get("message_kind") == "tool_call"
            ):
                try:
                    self.codexpro_tracker.observe_tool_call(self.conversation_ref, event)
                except Exception:
                    # Observability must never break a live ChatGPT turn.
                    pass
            self.last_server_progress_at = now
            self.server_idle_seconds = 0.0
            self.answer_progress_seen = False
            message_kind = event.get("message_kind")
            text = event.get("text")
            tool_error = (
                _tool_result_error(text)
                if message_kind == "tool_result" and isinstance(text, str)
                else ""
            )
            self.last_tool_error = tool_error
            self.state = "working"
            return
        if event_type in {
            "assistant_text_snapshot",
            "assistant_text_delta",
            "assistant_text_revision",
        }:
            self.last_server_progress_at = now
            self.server_idle_seconds = 0.0
            self.answer_progress_seen = True
            self.last_tool_error = ""
            self.state = "working"
            return
        if event_type == "stream_handoff_ws_reconnecting":
            attempt = event.get("attempt")
            if isinstance(attempt, int) and not isinstance(attempt, bool):
                self.reconnect_attempt = attempt
            reason = event.get("reason")
            self.reconnect_reason = reason.strip() if isinstance(reason, str) else ""
            idle = event.get("server_idle_seconds")
            if isinstance(idle, (int, float)) and not isinstance(idle, bool):
                self.server_idle_seconds = max(self.server_idle_seconds, float(idle))
            if self.state not in {"quiet", "stalled"}:
                self.state = "reconnecting"
            return
        if event_type == "stream_handoff_ws_subscribed":
            if self.state == "reconnecting":
                self.state = "working"
            return
        if event_type == "stream_handoff_delivery_recovered":
            self.delivery_recoveries += 1
            self.last_server_progress_at = now
            self.server_idle_seconds = 0.0
            self.reconnect_reason = ""
            self.state = "working"
            return
        if event_type == "stream_handoff_server_quiet":
            idle = event.get("server_idle_seconds")
            if isinstance(idle, (int, float)) and not isinstance(idle, bool):
                self.server_idle_seconds = float(idle)
            self.state = "quiet"
            return
        if event_type == "stream_handoff_server_stalled":
            idle = event.get("server_idle_seconds")
            if isinstance(idle, (int, float)) and not isinstance(idle, bool):
                self.server_idle_seconds = float(idle)
            self.state = "stalled"
            return
        if event_type == "stream_handoff_server_resumed":
            self.last_server_progress_at = now
            self.server_idle_seconds = 0.0
            self.reconnect_reason = ""
            self.state = "working"

    def codexpro_snapshot(self) -> CodexProActivitySnapshot:
        if self.codexpro_tracker is None:
            return CodexProActivitySnapshot()
        try:
            return self.codexpro_tracker.snapshot(self.conversation_ref)
        except Exception:
            return CodexProActivitySnapshot()


@dataclass
class _EnhancedTurn:
    task: asyncio.Task[int]
    controls: TurnControlSignals
    result: dict[str, Any]
    goal_turn: bool
    media: list[str]
    started_at: float
    health: _TurnHealth
    pause_goal_after_turn: bool = False
    exit_after_turn: bool = False


@dataclass
class _EnhancedResume:
    request: ResumeRequest
    future: asyncio.Future[tuple[bool, Any]]


@dataclass
class _EnhancedFollow:
    conversation_ref: str
    emitted_message_ids: set[str]
    seen_messages: dict[str, str]
    deadline: float
    health: _TurnHealth | None = None
    started_at: float = field(default_factory=time.monotonic)
    next_interval: float = FOLLOW_MIN_INTERVAL_SECONDS
    stream_topic_id: str | None = None
    stream_answer_message_id: str | None = None
    stream_answer_text: str = ""
    defer_stream_answer_until_terminal: bool = False
    stream_disabled: bool = False
    mode: str | None = None
    timer: asyncio.Task[None] | None = None
    future: asyncio.Future[tuple[bool, Any]] | None = None
    event_queue: asyncio.Queue[dict[str, Any]] | None = None
    event_task: asyncio.Task[dict[str, Any]] | None = None
    stop_requested: bool = False
    stopped_by_user: bool = False


class _ThreadsafeRendererProxy:
    """Marshal PrettyRenderer method calls onto the asyncio/UI thread."""

    def __init__(
        self,
        renderer: PrettyRenderer,
        loop: asyncio.AbstractEventLoop,
        *,
        application: Any | None,
    ) -> None:
        self._renderer = renderer
        self._loop = loop
        self._application = application

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._renderer, name)
        if not callable(value):
            return value

        def publish(*args: Any, **kwargs: Any) -> None:
            callback = partial(value, *args, **kwargs)

            def render_now() -> None:
                try:
                    app_context = getattr(self._application, "context", None)
                    if app_context is not None and self._application.is_running:
                        app_context.copy().run(callback)
                    else:
                        callback()
                except (RuntimeError, EOFError):
                    # The prompt application may already be shutting down. Fall
                    # back to the renderer directly while the loop still exists.
                    callback()

            try:
                self._loop.call_soon_threadsafe(render_now)
            except RuntimeError:
                # The loop may already be closed during local quit/shutdown.
                pass

        return publish


CONVERSATION_REF_FIELDS = (
    "conversation_id",
    "conversation_url",
    "conversation_ref",
    "url",
    "id",
)


def extract_conversation_ref(response: Any) -> str | None:
    candidates = [response]
    nested = (
        response.get("conversation")
        if isinstance(response, dict)
        else getattr(response, "conversation", None)
    )
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
    value = (
        response.get("title")
        if isinstance(response, dict)
        else getattr(response, "title", None)
    )
    if not isinstance(value, str):
        return None
    title = " ".join(value.split())
    return title or None


def response_finish_reason(response: Any) -> str | None:
    conversation = (
        response.get("conversation")
        if isinstance(response, dict)
        else getattr(response, "conversation", None)
    )
    if isinstance(conversation, dict):
        value = conversation.get("finish_reason")
    else:
        value = getattr(conversation, "finish_reason", None)
    if not isinstance(value, str):
        value = (
            response.get("finish_reason")
            if isinstance(response, dict)
            else getattr(response, "finish_reason", None)
        )
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


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
            client = build_client(client_factory, args)
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
        prompt_patch_enabled = False
        transcript_stream = getattr(ui, "transcript_stream", None)
        if callable(transcript_stream):
            renderer_stdout: TextIO = transcript_stream(
                stdout,
                stream_name="stdout",
            )
            renderer_stderr: TextIO = transcript_stream(
                stderr,
                stream_name="stderr",
            )
        else:
            # Keep lightweight test/dummy sessions usable without requiring the
            # full terminal transcript implementation.
            renderer_stdout = stdout
            renderer_stderr = stderr
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
        automatic_prompt = (
            interactive_commands.pop_automatic_prompt()
            if interactive_commands is not None
            else None
        )
        automatic_turn = automatic_prompt is not None

        if automatic_turn:
            prompt = automatic_prompt or ""
        else:
            if interactive and not enhanced:
                print("> ", end="", file=stdout, flush=True)

            try:
                if ui is not None:
                    attachment_count = (
                        interactive_commands.pending_media_count
                        if interactive_commands is not None
                        else 0
                    )
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
                    result = _handle_chat_command(
                        prompt,
                        state=state,
                        state_path=state_path,
                        stdout=stdout,
                        stderr=stderr,
                    )
                if result is not None:
                    if interactive_commands is not None:
                        interactive_commands.close()
                    return result
                continue

            if interactive_commands is not None:
                prompt = interactive_commands.prepare_goal_user_prompt(prompt)

        media = (
            interactive_commands.pending_media
            if interactive_commands is not None
            else None
        )
        conversation_mode = (
            interactive_commands.conversation_mode
            if interactive_commands is not None
            else "normal"
        )
        attached_ref = (
            interactive_commands.conversation_ref
            if interactive_commands is not None
            else state.current_conversation
        )
        goal_turn = bool(
            interactive_commands is not None and interactive_commands.goal_active
        )
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
                explicit_lock_wait=bool(getattr(args, "wait_lock", False))
                or getattr(args, "lock_timeout", None) is not None,
                stdout=stdout,
                stderr=stderr,
                renderer=renderer,
                turn_controls=turn_controls,
                conversation_mode=conversation_mode,
                attached_ref=attached_ref,
                temporary_turn_recorder=(
                    interactive_commands.record_temporary_turn
                    if interactive_commands is not None
                    and conversation_mode == "temporary"
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
                    interactive_commands.handle_goal_interruption(
                        f"chat turn failed with exit code {code}"
                    )
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
    with routed_turn_control_signals(
        enabled=True,
        controls=lambda: ui.active_turn_controls,
    ):
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
    start_ui = getattr(ui, "start_async", None)
    stop_ui = getattr(ui, "stop_async", None)
    if callable(start_ui):
        await start_ui()
    try:
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
    finally:
        if callable(stop_ui):
            await stop_ui()


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
    activity_tracker = CodexProActivityTracker(
        mapping_path=state_path.parent / "codexpro-session-map.json"
    )
    delivery_journal = StreamDeliveryJournal(
        state_path.parent / "stream-delivery.jsonl"
    )
    active: _EnhancedTurn | None = None
    active_resume: _EnhancedResume | None = None
    active_follow: _EnhancedFollow | None = None
    pending_follow_command: str | None = None
    prompt_task: asyncio.Task[str] | None = None
    accepting_input = True

    while True:
        if active_follow is not None and time.monotonic() >= active_follow.deadline:
            active_follow.stop_requested = True
            _cancel_enhanced_follow_timer(active_follow)
            _cancel_enhanced_follow_event_task(active_follow)
            renderer.info(
                "Stopped following after 2 hours; conversation remains attached."
            )
            ui.set_active_turn(None)
            active_follow = None

        if (
            active_follow is not None
            and pending_follow_command is not None
            and active_follow.future is None
        ):
            _cancel_enhanced_follow_timer(active_follow)
            command = pending_follow_command
            pending_follow_command = None
            result = await commands.handle_async(command)
            if result is not None:
                return _EnhancedLoopOutcome(exit_code=result)
            if (
                commands.has_pending_resume
                or commands.conversation_ref != active_follow.conversation_ref
            ):
                ui.set_active_turn(None)
                active_follow = None
            else:
                if command.split(maxsplit=1)[0].lower() == "/stop":
                    active_follow.stopped_by_user = True
                active_follow.stop_requested = False

        if active is None and active_resume is None and commands.has_pending_resume:
            if active_follow is not None:
                active_follow.stop_requested = True
                _cancel_enhanced_follow_timer(active_follow)
                _cancel_enhanced_follow_event_task(active_follow)
                ui.set_active_turn(None)
                active_follow = None
            active_resume = _start_enhanced_resume(
                get_client=get_client,
                commands=commands,
            )

        if active_follow is not None and active is None and active_resume is None:
            if active_follow.future is None and active_follow.timer is None:
                if not _start_enhanced_follow_stream(
                    active_follow,
                    get_client=get_client,
                ):
                    _schedule_enhanced_follow_timer(active_follow)

        if active is None and active_resume is None and active_follow is None:
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
                    activity_tracker=activity_tracker,
                    delivery_journal=delivery_journal,
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

        wait_for: set[asyncio.Future[Any]] = set()
        if prompt_task is not None:
            wait_for.add(prompt_task)
        if active is not None:
            wait_for.add(active.task)
        if active_resume is not None:
            wait_for.add(active_resume.future)
        if active_follow is not None:
            if active_follow.timer is not None:
                wait_for.add(active_follow.timer)
            if active_follow.future is not None:
                wait_for.add(active_follow.future)
            if active_follow.event_task is not None:
                wait_for.add(active_follow.event_task)
        if not wait_for:
            return _EnhancedLoopOutcome(exit_code=0)

        done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

        if prompt_task is not None and prompt_task in done:
            finished_prompt = prompt_task
            prompt_task = None
            try:
                raw = finished_prompt.result()
            except KeyboardInterrupt:
                if active is not None:
                    active.controls.request_stop()
                elif active_resume is not None:
                    renderer.info(
                        "Conversation is still loading; use Ctrl-\\ or /exit to exit gptty."
                    )
                elif active_follow is not None:
                    pending_follow_command = "/stop"
                    active_follow.stop_requested = True
                    _cancel_enhanced_follow_timer(active_follow)
                else:
                    return _EnhancedLoopOutcome(exit_code=130)
            except EOFError:
                if active is None:
                    if active_resume is not None:
                        commands.pause_goal_for_local_quit()
                    return _EnhancedLoopOutcome(exit_code=0)
                active.controls.request_quit()
                accepting_input = False
            else:
                recorder = getattr(stdout, "record_prompt", None)
                if callable(recorder) and raw:
                    recorder(raw)
                prompt = raw.strip()
                if prompt:
                    if (
                        active_follow is not None
                        and active is None
                        and active_resume is None
                    ):
                        if not prompt.startswith("/"):
                            queued_prompts.append(prompt)
                            renderer.info(f"Queued · {len(queued_prompts)}")
                            _refresh_active_follow_ui(
                                ui,
                                active_follow,
                                queued_prompts,
                            )
                        elif prompt.split(maxsplit=1)[0].lower() in {"/exit", "/quit"}:
                            result = commands.handle(prompt)
                            return _EnhancedLoopOutcome(
                                exit_code=0 if result is None else result
                            )
                        elif prompt == "/":
                            renderer.info(
                                "While following: text queues · /stop · /exit · other commands run between follow reads"
                            )
                        else:
                            pending_follow_command = prompt
                            active_follow.stop_requested = True
                            _cancel_enhanced_follow_timer(active_follow)
                    elif active is None and active_resume is None:
                        if prompt == "/":
                            ui.reopen_command_completion()
                        elif prompt.startswith("/"):
                            result = await commands.handle_async(prompt)
                            if result is not None:
                                return _EnhancedLoopOutcome(exit_code=result)
                        else:
                            queued_prompts.append(prompt)
                    elif active_resume is not None and active is None:
                        outcome = _handle_resume_loading_input(
                            prompt,
                            commands=commands,
                            renderer=renderer,
                            queued_prompts=queued_prompts,
                        )
                        if outcome is not None:
                            return outcome
                    else:
                        accepting_input = _handle_working_input(
                            prompt,
                            active=active,
                            commands=commands,
                            renderer=renderer,
                            queued_prompts=queued_prompts,
                        )
                        _refresh_active_turn_ui(ui, active, queued_prompts)

        if active_resume is not None and active_resume.future in done:
            finished_resume = active_resume
            active_resume = None
            ok, payload = finished_resume.future.result()
            resumed = False
            if ok:
                resumed = commands.complete_resume(finished_resume.request, payload)
                if resumed:
                    active_follow = _seed_enhanced_follow(
                        finished_resume.request,
                        payload,
                        renderer=renderer,
                        activity_tracker=activity_tracker,
                        delivery_journal=delivery_journal,
                    )
                    if active_follow is not None:
                        _refresh_active_follow_ui(ui, active_follow, queued_prompts)
            else:
                commands.fail_resume(finished_resume.request, payload)
            if not resumed:
                queued_count = len(queued_prompts)
                queued_prompts.clear()
                commands.clear_automatic_prompts()
                if queued_count:
                    renderer.info(
                        f"Cleared {queued_count} queued prompt{'s' if queued_count != 1 else ''} after failed resume."
                    )
            accepting_input = True

        if (
            active_follow is not None
            and active_follow.event_task is not None
            and active_follow.event_task in done
        ):
            finished_event_task = active_follow.event_task
            active_follow.event_task = None
            try:
                stream_event = finished_event_task.result()
            except asyncio.CancelledError:
                stream_event = None
            if stream_event is not None:
                _apply_enhanced_follow_stream_event(
                    active_follow,
                    stream_event,
                    renderer=renderer,
                )
            _restart_enhanced_follow_event_task(active_follow)

        if (
            active_follow is not None
            and active_follow.timer is not None
            and active_follow.timer in done
        ):
            active_follow.timer = None
            if not active_follow.stop_requested:
                _start_enhanced_follow_poll(active_follow, get_client=get_client)

        if (
            active_follow is not None
            and active_follow.future is not None
            and active_follow.future in done
        ):
            finished_follow = active_follow.future
            follow_mode = active_follow.mode
            active_follow.future = None
            active_follow.mode = None
            ok, payload = finished_follow.result()
            keep_following = True

            if follow_mode == "stream":
                _drain_enhanced_follow_stream_events(
                    active_follow,
                    renderer=renderer,
                )
                _cancel_enhanced_follow_event_task(active_follow)
                active_follow.event_queue = None
                if (
                    ok
                    and isinstance(payload, dict)
                    and payload.get("stream_completed") is True
                ):
                    keep_following = _apply_enhanced_follow_snapshot(
                        active_follow,
                        payload,
                        renderer=renderer,
                    )
                elif ok:
                    if not active_follow.stop_requested:
                        active_follow.stream_disabled = True
                        renderer.warning(
                            "Live stream ended before completion; falling back to canonical polling."
                        )
                else:
                    active_follow.stream_disabled = True
                    active_follow.next_interval = FOLLOW_MIN_INTERVAL_SECONDS
                    stream_error = str(payload)
                    if "maximum length for this conversation" in stream_error.lower():
                        renderer.warning(
                            "Live stream rejected for this long conversation; continuing via canonical polling."
                        )
                    else:
                        renderer.warning(
                            f"Live stream unavailable; falling back to canonical polling: {payload}"
                        )
            elif ok:
                keep_following = _apply_enhanced_follow_snapshot(
                    active_follow,
                    payload,
                    renderer=renderer,
                )
            else:
                _backoff_enhanced_follow_after_error(
                    active_follow,
                    payload,
                    renderer=renderer,
                )

            if not keep_following:
                _cancel_enhanced_follow_timer(active_follow)
                _cancel_enhanced_follow_event_task(active_follow)
                ui.set_active_turn(None)
                active_follow = None
            elif active_follow.stop_requested and pending_follow_command is None:
                active_follow.stop_requested = False

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


def _start_enhanced_resume(
    *,
    get_client: Callable[[], Any],
    commands: InteractiveCommands,
) -> _EnhancedResume | None:
    request = commands.take_pending_resume()
    if request is None:
        return None

    loop = asyncio.get_running_loop()
    future: asyncio.Future[tuple[bool, Any]] = loop.create_future()
    client = get_client()

    def publish(result: tuple[bool, Any]) -> None:
        if not future.done():
            future.set_result(result)

    def worker() -> None:
        try:
            follow_snapshot = getattr(client, "conversation_follow_snapshot", None)
            if callable(follow_snapshot):
                payload = follow_snapshot(
                    request.conversation_ref,
                    emitted_message_ids=(),
                    limit=None,
                    verify_terminal_status=True,
                    terminal_probe_timeout=3.0,
                )
            else:
                payload = client.conversation_snapshot(request.conversation_ref)
            result: tuple[bool, Any] = (True, payload)
        except BaseException as exc:  # noqa: BLE001 - background resume boundary.
            result = (False, exc)
        try:
            loop.call_soon_threadsafe(publish, result)
        except RuntimeError:
            # The user may exit while a slow snapshot is still finishing. The
            # daemon worker must never keep gptty alive or touch a closed loop.
            pass

    threading.Thread(
        target=worker,
        name="gptty-resume-snapshot",
        daemon=True,
    ).start()
    return _EnhancedResume(request=request, future=future)


def _seed_enhanced_follow(
    request: ResumeRequest,
    snapshot: Any,
    *,
    renderer: PrettyRenderer,
    activity_tracker: CodexProActivityTracker | None = None,
    delivery_journal: StreamDeliveryJournal | None = None,
) -> _EnhancedFollow | None:
    if not isinstance(snapshot, dict) or "emitted_message_ids" not in snapshot:
        return None
    status = _snapshot_status(snapshot)
    if status not in UNFINISHED_STATUSES:
        return None
    emitted = {
        str(message_id).strip()
        for message_id in snapshot.get("emitted_message_ids", [])
        if str(message_id).strip()
    }
    messages = _snapshot_messages(snapshot)
    seen = {_message_identity(message): _message_text(message) for message in messages}
    stream_topic_id = snapshot.get("stream_topic_id")
    if not isinstance(stream_topic_id, str) or not stream_topic_id.strip():
        stream_topic_id = None
    else:
        stream_topic_id = stream_topic_id.strip()
    stream_answer_message_id = snapshot.get("stream_answer_message_id")
    if (
        not isinstance(stream_answer_message_id, str)
        or not stream_answer_message_id.strip()
    ):
        stream_answer_message_id = None
    else:
        stream_answer_message_id = stream_answer_message_id.strip()
    stream_answer_text = snapshot.get("stream_answer_text")
    if not isinstance(stream_answer_text, str):
        stream_answer_text = ""
    health = _TurnHealth(
        last_server_progress_at=time.monotonic(),
        answer_progress_seen=bool(stream_answer_text),
        codexpro_tracker=activity_tracker,
        delivery_journal=delivery_journal,
        conversation_ref=request.conversation_ref,
    )

    renderer.info(
        "Following active response via live stream…"
        if stream_topic_id
        else "Following active response in background…"
    )
    current_turn_event_ids = {
        str(message_id).strip()
        for message_id in snapshot.get("current_turn_event_ids", [])
        if str(message_id).strip()
    }
    raw_events = snapshot.get("events")
    if current_turn_event_ids and isinstance(raw_events, list):
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            message_id = event.get("message_id")
            normalized_message_id = (
                message_id.strip()
                if isinstance(message_id, str) and message_id.strip()
                else None
            )
            if normalized_message_id in current_turn_event_ids:
                health.observe(event)
                # complete_resume() has already rendered snapshot messages. If
                # canonical history contains this same intermediate message,
                # replaying the event here produces a visible duplicate on
                # attach. Only render seed events that history did not show.
                if normalized_message_id not in seen:
                    renderer.live_event(event)
    now = time.monotonic()
    return _EnhancedFollow(
        conversation_ref=request.conversation_ref,
        emitted_message_ids=emitted,
        seen_messages=seen,
        deadline=now + FOLLOW_TIMEOUT_SECONDS,
        health=health,
        started_at=now,
        next_interval=FOLLOW_MIN_INTERVAL_SECONDS,
        stream_topic_id=stream_topic_id,
        stream_answer_message_id=stream_answer_message_id,
        stream_answer_text=stream_answer_text,
        # Attached follow can join after earlier commentary/tool events. Because
        # terminal reconciliation may recover those missed events, rendering the
        # final answer eagerly can put older progress after the final text.
        defer_stream_answer_until_terminal=True,
    )


def _cancel_enhanced_follow_timer(follow: _EnhancedFollow) -> None:
    timer = follow.timer
    follow.timer = None
    if timer is not None and not timer.done():
        timer.cancel()


def _schedule_enhanced_follow_timer(follow: _EnhancedFollow) -> bool:
    if follow.stop_requested or follow.future is not None or follow.timer is not None:
        return False
    if time.monotonic() >= follow.deadline:
        return False
    follow.timer = asyncio.create_task(asyncio.sleep(follow.next_interval))
    return True


def _cancel_enhanced_follow_event_task(follow: _EnhancedFollow) -> None:
    task = follow.event_task
    follow.event_task = None
    if task is not None and not task.done():
        task.cancel()


def _start_enhanced_follow_stream(
    follow: _EnhancedFollow,
    *,
    get_client: Callable[[], Any],
) -> bool:
    topic_id = follow.stream_topic_id
    if (
        follow.stream_disabled
        or not isinstance(topic_id, str)
        or not topic_id
        or follow.future is not None
    ):
        return False

    loop = asyncio.get_running_loop()
    future: asyncio.Future[tuple[bool, Any]] = loop.create_future()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    client = get_client()
    emitted = tuple(sorted(follow.emitted_message_ids))
    remaining = max(1.0, follow.deadline - time.monotonic())

    def publish(result: tuple[bool, Any]) -> None:
        if not future.done():
            future.set_result(result)

    def publish_event(event: Any) -> None:
        if not isinstance(event, dict):
            return
        queue.put_nowait(event)

    def relay_event(event: Any) -> None:
        try:
            loop.call_soon_threadsafe(publish_event, event)
        except RuntimeError:
            pass

    def should_stop() -> bool:
        return follow.stop_requested or time.monotonic() >= follow.deadline

    def worker() -> None:
        try:
            stream_follow = getattr(client, "conversation_follow_stream", None)
            if not callable(stream_follow):
                raise RuntimeError("live topic follow is unavailable")
            result: tuple[bool, Any] = (
                True,
                stream_follow(
                    follow.conversation_ref,
                    topic_id=topic_id,
                    emitted_message_ids=emitted,
                    answer_message_id=follow.stream_answer_message_id,
                    answer_text=follow.stream_answer_text,
                    timeout=remaining,
                    limit=FOLLOW_MESSAGE_LIMIT,
                    on_event=relay_event,
                    should_stop=should_stop,
                ),
            )
        except BaseException as exc:  # noqa: BLE001 - background follow boundary.
            result = (False, exc)
        try:
            loop.call_soon_threadsafe(publish, result)
        except RuntimeError:
            pass

    follow.mode = "stream"
    follow.future = future
    follow.event_queue = queue
    follow.event_task = asyncio.create_task(queue.get())
    threading.Thread(
        target=worker,
        name="gptty-resume-stream-follow",
        daemon=True,
    ).start()
    return True


def _apply_enhanced_follow_stream_event(
    follow: _EnhancedFollow,
    event: Any,
    *,
    renderer: PrettyRenderer,
) -> None:
    if not isinstance(event, dict):
        return
    event_type = event.get("type")
    if follow.health is not None:
        follow.health.observe(event)
        if event_type in {
            "stream_handoff_server_quiet",
            "stream_handoff_server_stalled",
        }:
            event = {
                **event,
                "final_text_seen": follow.health.answer_progress_seen,
                "last_tool_error": follow.health.last_tool_error or None,
            }
    message_id = event.get("message_id")
    normalized_message_id = (
        message_id.strip()
        if isinstance(message_id, str) and message_id.strip()
        else None
    )

    if event_type == "canonical_intermediate_message":
        if normalized_message_id is not None:
            follow.emitted_message_ids.add(normalized_message_id)
        renderer.live_event(event)
        return

    if event_type not in {
        "assistant_text_snapshot",
        "assistant_text_delta",
        "assistant_text_revision",
    }:
        renderer.live_event(event)
        return

    if (
        normalized_message_id is not None
        and normalized_message_id != follow.stream_answer_message_id
    ):
        follow.stream_answer_message_id = normalized_message_id
        follow.stream_answer_text = ""

    if event_type == "assistant_text_delta":
        delta = event.get("delta")
        if isinstance(delta, str) and delta:
            follow.stream_answer_text += delta
    else:
        text = event.get("text")
        if isinstance(text, str):
            follow.stream_answer_text = text

    if follow.stream_answer_message_id:
        follow.seen_messages[follow.stream_answer_message_id] = (
            follow.stream_answer_text
        )
    if not follow.defer_stream_answer_until_terminal:
        renderer.live_event(event)


def _drain_enhanced_follow_stream_events(
    follow: _EnhancedFollow,
    *,
    renderer: PrettyRenderer,
) -> None:
    task = follow.event_task
    if task is not None and task.done() and not task.cancelled():
        follow.event_task = None
        try:
            event = task.result()
        except asyncio.CancelledError:
            event = None
        if event is not None:
            _apply_enhanced_follow_stream_event(follow, event, renderer=renderer)

    queue = follow.event_queue
    if queue is None:
        return
    while True:
        try:
            event = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        _apply_enhanced_follow_stream_event(follow, event, renderer=renderer)


def _restart_enhanced_follow_event_task(follow: _EnhancedFollow) -> None:
    if (
        follow.mode == "stream"
        and follow.future is not None
        and follow.event_queue is not None
        and follow.event_task is None
    ):
        follow.event_task = asyncio.create_task(follow.event_queue.get())


def _start_enhanced_follow_poll(
    follow: _EnhancedFollow,
    *,
    get_client: Callable[[], Any],
) -> None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[tuple[bool, Any]] = loop.create_future()
    client = get_client()
    emitted = tuple(sorted(follow.emitted_message_ids))

    def publish(result: tuple[bool, Any]) -> None:
        if not future.done():
            future.set_result(result)

    def worker() -> None:
        try:
            result: tuple[bool, Any] = (
                True,
                client.conversation_follow_snapshot(
                    follow.conversation_ref,
                    emitted_message_ids=emitted,
                    limit=FOLLOW_MESSAGE_LIMIT,
                ),
            )
        except BaseException as exc:  # noqa: BLE001 - background follow boundary.
            result = (False, exc)
        try:
            loop.call_soon_threadsafe(publish, result)
        except RuntimeError:
            pass

    follow.mode = "poll"
    follow.future = future
    threading.Thread(
        target=worker,
        name="gptty-resume-follow",
        daemon=True,
    ).start()


def _apply_enhanced_follow_snapshot(
    follow: _EnhancedFollow,
    snapshot: Any,
    *,
    renderer: PrettyRenderer,
) -> bool:
    if not isinstance(snapshot, dict):
        renderer.warning("Live follow stopped: invalid canonical snapshot.")
        return False

    events = snapshot.get("events")
    raw_event_items = (
        [event for event in events if isinstance(event, dict)]
        if isinstance(events, list)
        else []
    )
    # Preserve the IDs that were rendered before this snapshot. The snapshot's
    # emitted_message_ids are merged below and may include newly discovered
    # events that still need rendering in this reconciliation pass.
    previously_emitted_message_ids = set(follow.emitted_message_ids)
    event_items: list[dict[str, Any]] = []
    for event in raw_event_items:
        message_id = event.get("message_id")
        normalized_message_id = (
            message_id.strip()
            if isinstance(message_id, str) and message_id.strip()
            else None
        )
        if (
            normalized_message_id is not None
            and normalized_message_id in previously_emitted_message_ids
        ):
            continue
        event_items.append(event)
    event_ids = {
        str(event.get("message_id")).strip()
        for event in event_items
        if event.get("message_id")
    }
    for event in event_items:
        renderer.live_event(event)

    emitted = snapshot.get("emitted_message_ids")
    if isinstance(emitted, (list, tuple, set, frozenset)):
        follow.emitted_message_ids.update(
            str(message_id).strip() for message_id in emitted if str(message_id).strip()
        )

    current = _snapshot_messages(snapshot)
    status = _snapshot_status(snapshot)
    corrected_final_identity: str | None = None
    corrected_final_text = ""
    deferred_final_identity: str | None = None
    deferred_final_text = ""
    if status == "completed" and follow.stream_answer_message_id:
        for message in reversed(current):
            identity = _message_identity(message)
            if identity != follow.stream_answer_message_id:
                continue
            text = _message_text(message)
            if follow.defer_stream_answer_until_terminal and text:
                deferred_final_identity = identity
                deferred_final_text = text
            elif text and text != follow.stream_answer_text:
                corrected_final_identity = identity
                corrected_final_text = text
            break

    changed: list[Any] = []
    for message in current:
        identity = _message_identity(message)
        text = _message_text(message)
        previous = follow.seen_messages.get(identity)
        follow.seen_messages[identity] = text
        if (
            previous == text
            or identity in event_ids
            or identity in previously_emitted_message_ids
        ):
            continue
        if identity in {deferred_final_identity, corrected_final_identity}:
            continue
        changed.append(message)
    if changed:
        renderer.messages(normalize_messages(changed))
    if deferred_final_identity is not None:
        renderer.answer(deferred_final_text)
        follow.stream_answer_text = deferred_final_text
    elif corrected_final_identity is not None:
        renderer.answer(corrected_final_text)
        follow.stream_answer_text = corrected_final_text

    if (
        event_items
        or changed
        or deferred_final_identity is not None
        or corrected_final_identity is not None
    ):
        follow.next_interval = FOLLOW_MIN_INTERVAL_SECONDS
    else:
        follow.next_interval = min(
            FOLLOW_MAX_IDLE_INTERVAL_SECONDS,
            max(FOLLOW_MIN_INTERVAL_SECONDS, follow.next_interval * 2.0),
        )

    if status == "completed":
        renderer.chat_link(follow.conversation_ref)
        if not follow.stopped_by_user:
            notify_response_complete(
                chat_title=None,
                final_response=_last_assistant_message_text(current),
            )
        return False
    if status == "awaiting_tool_approval":
        renderer.warning("Conversation is waiting for tool approval.")
        return False
    if status not in UNFINISHED_STATUSES:
        renderer.info(f"Follow stopped: status={status or 'unknown'}")
        return False
    return True


def _backoff_enhanced_follow_after_error(
    follow: _EnhancedFollow,
    error: Any,
    *,
    renderer: PrettyRenderer,
) -> None:
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        follow.next_interval = min(
            FOLLOW_RATE_LIMIT_MAX_BACKOFF_SECONDS,
            max(FOLLOW_RATE_LIMIT_BACKOFF_SECONDS, follow.next_interval * 2.0),
        )
        renderer.warning(
            f"Live follow rate limited; backing off {int(follow.next_interval)}s."
        )
        return

    follow.next_interval = min(
        FOLLOW_MAX_IDLE_INTERVAL_SECONDS,
        max(FOLLOW_MIN_INTERVAL_SECONDS, follow.next_interval * 2.0),
    )
    renderer.warning(
        f"Live follow read failed; retrying after {int(follow.next_interval)}s: {error}"
    )


def _handle_resume_loading_input(
    prompt: str,
    *,
    commands: InteractiveCommands,
    renderer: PrettyRenderer,
    queued_prompts: deque[str],
) -> _EnhancedLoopOutcome | None:
    if not prompt.startswith("/"):
        queued_prompts.append(prompt)
        renderer.info(f"Queued · {len(queued_prompts)}")
        return None

    try:
        parts = shlex.split(prompt)
    except ValueError as exc:
        renderer.warning(f"Invalid command: {exc}")
        return None
    if not parts:
        return None

    name = parts[0].lstrip("/").lower()
    argv = parts[1:]
    if name in {"exit", "quit"}:
        if argv:
            renderer.warning(f"/{name} takes no arguments.")
            return None
        result = commands.handle("/exit")
        return _EnhancedLoopOutcome(exit_code=0 if result is None else result)
    if name == "":
        renderer.info("While loading a conversation: /exit · Ctrl-\\ quit")
        return None

    renderer.warning(
        f"/{name} is unavailable while the conversation is loading; queued text will send after resume."
    )
    return None


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
        renderer.turn_abort()
        if turn.goal_turn:
            commands.pause_goal_for_local_quit()
        await _cancel_prompt_task(prompt_task)
        return _EnhancedLoopOutcome(exit_code=0)
    if code != 0:
        renderer.turn_abort()
        if turn.goal_turn:
            commands.handle_goal_interruption(f"chat turn failed with exit code {code}")
        await _cancel_prompt_task(prompt_task)
        return _EnhancedLoopOutcome(exit_code=code)

    incomplete_turn = bool(turn.result.get("incomplete_without_terminal"))
    stopped_by_user = bool(turn.result.get("stopped_by_user"))
    final_text = str(turn.result.get("text") or "")
    if incomplete_turn:
        renderer.turn_abort()
        renderer.warning(
            "ChatGPT stream ended without a final answer; returned control to gptty."
        )
    else:
        renderer.answer(final_text)
    if stopped_by_user:
        renderer.info("Stopped by user.")
        if queued_prompts:
            renderer.info(
                f"Queued · {len(queued_prompts)} · will send next"
            )
    conversation_ref = turn.result.get("conversation_ref")
    if (
        not turn.result.get("is_temporary")
        and isinstance(conversation_ref, str)
        and conversation_ref.strip()
    ):
        renderer.chat_link(conversation_ref.strip())

    if turn.result.get("stopped_by_user"):
        commands.clear_automatic_prompts()

    incomplete_turn = bool(turn.result.get("incomplete_without_terminal"))
    if incomplete_turn:
        queued_count = len(queued_prompts)
        queued_prompts.clear()
        commands.clear_automatic_prompts()
        if queued_count:
            renderer.info(
                f"Cleared {queued_count} queued prompt{'s' if queued_count != 1 else ''} after incomplete turn."
            )
        if turn.goal_turn:
            commands.handle_goal_interruption(
                "ChatGPT stream ended without a final answer"
            )
    elif turn.goal_turn:
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
    activity_tracker: CodexProActivityTracker | None,
    delivery_journal: StreamDeliveryJournal | None,
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
    health = _TurnHealth(
        last_server_progress_at=started_at,
        codexpro_tracker=activity_tracker,
        delivery_journal=delivery_journal,
        conversation_ref=commands.conversation_ref or state.current_conversation,
    )
    renderer.turn_start(show_elapsed=False)

    loop = asyncio.get_running_loop()
    threaded_renderer = _ThreadsafeRendererProxy(
        renderer,
        loop,
        application=getattr(ui, "application", None),
    )

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
            renderer=threaded_renderer,
            turn_controls=controls,
            conversation_mode=conversation_mode,
            attached_ref=attached_ref,
            temporary_turn_recorder=(
                commands.record_temporary_turn
                if conversation_mode == "temporary"
                else None
            ),
            notify_completion=not goal_turn,
            result_out=turn_result,
            on_stop_confirmed=stop_confirmed if goal_turn else None,
            defer_final_rendering=True,
            turn_health=health,
        )
    )
    active = _EnhancedTurn(
        task=task,
        controls=controls,
        result=turn_result,
        goal_turn=goal_turn,
        media=media,
        started_at=started_at,
        health=health,
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
        working_status=lambda: _working_status(
            active.started_at,
            len(queued_prompts),
            health=active.health,
        ),
    )


def _refresh_active_follow_ui(
    ui: InteractiveSession,
    follow: _EnhancedFollow,
    queued_prompts: deque[str],
) -> None:
    ui.set_active_turn(
        None,
        working_status=lambda: _working_status(
            follow.started_at,
            len(queued_prompts),
            health=follow.health,
        ),
    )


def _format_status_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes:02d}:{seconds:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _codexpro_status_suffix(snapshot: CodexProActivitySnapshot) -> str:
    if not snapshot.bound:
        return ""
    if snapshot.inflight:
        tool = snapshot.inflight_tool or snapshot.last_tool or "tool"
        heartbeat_age = snapshot.last_heartbeat_age_seconds
        if heartbeat_age is not None and heartbeat_age <= 45.0:
            return (
                f" · CodexPro exact: {tool} running"
                f" · heartbeat {_format_status_duration(heartbeat_age)} ago"
            )
        if (
            heartbeat_age is None
            and snapshot.last_event_age_seconds is not None
            and snapshot.last_event_age_seconds <= 20.0
        ):
            return (
                f" · CodexPro exact: {tool} in flight"
                f" · started {_format_status_duration(snapshot.last_event_age_seconds)} ago"
            )
    if snapshot.last_event_age_seconds is not None:
        if snapshot.last_event_age_seconds <= CODEXPRO_RECENT_ACTIVITY_MAX_AGE_SECONDS:
            return (
                " · CodexPro exact activity "
                f"{_format_status_duration(snapshot.last_event_age_seconds)} ago"
            )
        return ""
    return " · CodexPro session mapped"


def _working_status(
    started_at: float,
    queued_count: int,
    *,
    health: _TurnHealth | None = None,
) -> str:
    elapsed = max(0.0, time.monotonic() - started_at)
    elapsed_label = _format_status_duration(elapsed)
    codexpro = health.codexpro_snapshot() if health is not None else CodexProActivitySnapshot()
    if health is not None and health.state == "stalled":
        if health.answer_progress_seen:
            status = (
                "FINALITY UNCONFIRMED"
                " · answer text received"
                f" · no observable server events {_format_status_duration(health.server_idle_seconds)}"
                " · do not resend yet"
            )
        elif health.last_tool_error:
            status = (
                "PROLONGED SILENCE"
                f" · no observable server events {_format_status_duration(health.server_idle_seconds)}"
                " · last visible tool failed"
                " · turn may still recover"
            )
        else:
            status = (
                "PROLONGED SILENCE"
                f" · no observable server events {_format_status_duration(health.server_idle_seconds)}"
                " · turn may still be working"
                " · do not resend yet"
            )
    elif health is not None and health.state == "quiet":
        if health.answer_progress_seen:
            status = (
                "answer text received"
                f" · finality unconfirmed {_format_status_duration(health.server_idle_seconds)}"
            )
        else:
            status = (
                f"server quiet {_format_status_duration(health.server_idle_seconds)}"
                " · no observable progress"
                " · waiting safely"
            )
    elif health is not None and health.state == "reconnecting":
        if health.reconnect_reason == "topic_idle":
            status = f"checking delivery · idle lease · attempt {health.reconnect_attempt}"
        else:
            status = f"reconnecting delivery · attempt {health.reconnect_attempt}"
    else:
        status = f"working {elapsed_label}"
        if health is not None:
            progress_age = max(0.0, time.monotonic() - health.last_server_progress_at)
            if (
                health.answer_progress_seen
                and progress_age >= ANSWER_FINALITY_PENDING_SECONDS
            ):
                status = (
                    "answer text received"
                    f" · finality unconfirmed {_format_status_duration(progress_age)}"
                    " · do not resend yet"
                )
            elif progress_age >= 30.0:
                status += f" · server { _format_status_duration(progress_age) } ago"
    if health is not None:
        progress_age = max(0.0, time.monotonic() - health.last_server_progress_at)
        if health.state in {"quiet", "stalled"} or progress_age >= 30.0:
            status += _codexpro_status_suffix(codexpro)
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
        renderer.warning(
            "While working, only /goal pause and /goal status are available."
        )
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

    renderer.warning(
        f"/{name} is unavailable while working; stop the current response first."
    )
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

    print(
        f"Unknown command: {command}. Type /help for available commands.", file=stderr
    )
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
    defer_final_rendering: bool = False,
    turn_health: _TurnHealth | None = None,
) -> int:
    if result_out is not None:
        result_out.clear()
    saw_stream_token = False
    stream_tokens: list[str] = []
    recorder: RunRecorder | None = None
    completed_successfully = False
    stopped_by_user = False
    local_quit_requested = False
    incomplete_turn = False
    write_committed = threading.Event()
    write_conversation_ref: str | None = None
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
        nonlocal active_ref, write_conversation_ref
        event_type = event.get("type")
        if turn_health is not None:
            turn_health.observe(event)
            if event_type in {
                "stream_handoff_server_quiet",
                "stream_handoff_server_stalled",
            }:
                event = {
                    **event,
                    "final_text_seen": turn_health.answer_progress_seen,
                    "last_tool_error": turn_health.last_tool_error or None,
                }
        if (
            recorder is not None
            and isinstance(event_type, str)
            and event_type.startswith("stream_handoff_")
        ):
            recorder.event(
                "stream_health",
                event_type=event_type,
                reason=event.get("reason"),
                attempt=event.get("attempt"),
                server_idle_seconds=event.get("server_idle_seconds"),
                silent_seconds=event.get("silent_seconds"),
                catchup_count=event.get("catchup_count"),
                last_offset=event.get("last_offset"),
                last_tool_error=event.get("last_tool_error"),
            )
        if event_type in {
            "browser_native_write_identity_resolved",
            "browser_native_write_completed",
        }:
            candidate = event.get("conversation_id") or event.get("conversationId")
            if (
                isinstance(candidate, str)
                and candidate.strip()
                and not candidate.strip().startswith("WEB:")
            ):
                write_conversation_ref = candidate.strip()
                if not is_temporary and not active_ref:
                    active_ref = write_conversation_ref
            if event_type == "browser_native_write_completed":
                write_committed.set()
        if stopped_by_user or local_quit_requested:
            return
        if renderer is not None:
            renderer.live_event(event)
            return
        rendered = render_live_event(event)
        if rendered:
            print(rendered, file=stderr, flush=True)

    def persist_committed_conversation_for_local_quit() -> None:
        nonlocal active_ref
        if is_temporary or state.current_conversation or not write_conversation_ref:
            return
        state.current_conversation = write_conversation_ref
        active_ref = write_conversation_ref
        try:
            save_chat_state(state_path, state)
        except StateError as exc:
            if renderer is not None:
                renderer.warning(str(exc))
            else:
                print(str(exc), file=stderr)

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
                return client.send_to_conversation(
                    send_conversation_ref, prompt, **options
                )
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

            worker = threading.Thread(
                target=send_worker, name="gptty-chat-turn", daemon=True
            )
            worker.start()
            quit_wait_notice_shown = False
            stop_pending = False
            stop_notice_shown = False
            stop_requires_conversation_ref = (
                getattr(client, "browser_authority_backend", None) == "wkwebview"
            )
            while worker.is_alive():
                worker.join(timeout=0.1)
                if controls.quit_requested.is_set():
                    if write_committed.is_set():
                        persist_committed_conversation_for_local_quit()
                        local_quit_requested = True
                        renderer.turn_abort()
                        renderer.info(
                            "Exited gptty; ChatGPT response continues in browser."
                        )
                        return LOCAL_QUIT_CODE
                    if not quit_wait_notice_shown:
                        quit_wait_notice_shown = True
                        renderer.info(
                            "Waiting for safe ChatGPT handoff before local exit…"
                        )
                stop_requested_now = controls.consume_stop()
                if stop_requested_now:
                    if stopped_by_user:
                        local_quit_requested = True
                        renderer.turn_abort()
                        renderer.info(
                            "ChatGPT is already stopped; exiting gptty without waiting for local readback."
                        )
                        return LOCAL_QUIT_CODE
                    stop_pending = True
                    if not stop_notice_shown:
                        stop_notice_shown = True
                        renderer.info("Stopping ChatGPT…")

                if not stop_pending:
                    continue

                stop_target = active_ref or write_conversation_ref
                if stop_requires_conversation_ref and not stop_target:
                    continue

                try:
                    stop_result = client.stop_generation(stop_target, timeout=30.0)
                except Exception as exc:  # noqa: BLE001 - interactive stop is best-effort at this boundary.
                    renderer.warning(f"Stop failed: {exc}")
                    stop_pending = False
                    stop_notice_shown = False
                    continue

                stopped = (
                    bool(stop_result.get("stopped"))
                    if isinstance(stop_result, dict)
                    else bool(getattr(stop_result, "stopped", False))
                )
                if not stopped:
                    renderer.warning(
                        "No active ChatGPT response to stop yet; press Ctrl-C again to retry."
                    )
                    stop_pending = False
                    stop_notice_shown = False
                    continue

                stop_pending = False
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
                persist_committed_conversation_for_local_quit()
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
                        snapshot = client.conversation_snapshot(
                            state.current_conversation
                        )
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
                        recorder.fail(
                            str(error),
                            traceback_text="".join(
                                traceback.format_exception(
                                    type(error),
                                    error,
                                    error.__traceback__,
                                )
                            ),
                        )
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
        finish_reason = response_finish_reason(response)
        incomplete_turn = finish_reason == "incomplete"
        if renderer is not None and not defer_final_rendering:
            if incomplete_turn:
                renderer.turn_abort()
                renderer.warning(
                    "ChatGPT stream ended without a final answer; returned control to gptty."
                )
            else:
                renderer.answer(rendered_text)
            if stopped_by_user:
                renderer.info("Stopped by user.")
        elif renderer is None and incomplete_turn:
            print("gptty: ChatGPT stream ended without a final answer.", file=stderr)
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

        if (
            renderer is not None
            and not defer_final_rendering
            and not is_temporary
            and state.current_conversation
        ):
            renderer.chat_link(state.current_conversation)

        if recorder is not None:
            if stopped_by_user:
                recorder.event("stopped_by_user")
            if incomplete_turn:
                recorder.event("incomplete_without_terminal")
            recorder.complete()
        if result_out is not None:
            result_out.update(
                text=rendered_text,
                title=response_title(response),
                conversation_ref=conversation_ref,
                finish_reason=finish_reason,
                stopped_by_user=stopped_by_user,
                incomplete_without_terminal=incomplete_turn,
                is_temporary=is_temporary,
            )
        completed_successfully = True
        return 0
    finally:
        if lock is not None:
            lock.release()
        if renderer is not None and not defer_final_rendering:
            renderer.turn_abort()
        if (
            renderer is not None
            and completed_successfully
            and not stopped_by_user
            and not incomplete_turn
            and notify_completion
        ):
            notify_response_complete(
                chat_title=response_title(response)
                or ("Temporary Chat" if is_temporary else None),
                final_response=rendered_text,
            )


def _stopped_snapshot_response(
    snapshot: Any, *, conversation_ref: str
) -> dict[str, str]:
    raw_messages = (
        snapshot.get("messages")
        if isinstance(snapshot, dict)
        else getattr(snapshot, "messages", None)
    )
    try:
        messages = list(raw_messages) if raw_messages is not None else []
    except TypeError:
        messages = []

    text = ""
    for message in reversed(messages):
        role = (
            message.get("role")
            if isinstance(message, dict)
            else getattr(message, "role", None)
        )
        if role != "assistant":
            continue
        recipient = (
            message.get("recipient")
            if isinstance(message, dict)
            else getattr(message, "recipient", None)
        )
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
