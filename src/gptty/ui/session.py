from __future__ import annotations

import asyncio
import os
import signal
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import FuzzyCompleter, PathCompleter, WordCompleter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition, has_focus, to_filter
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import CompletionsMenu, Float, FloatContainer, Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.shortcuts import CompleteStyle, choice
from prompt_toolkit.utils import get_cwidth

from .signals import TurnControlSignals
from .state import UISettings, UIStateError, load_ui_settings, ui_settings_path


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("new", "Start a new ChatGPT conversation"),
    CommandSpec("temporary", "Start a new Temporary ChatGPT conversation"),
    CommandSpec("resume", "Resume a real ChatGPT conversation"),
    CommandSpec("detach", "Detach locally from the current conversation"),
    CommandSpec("stop", "Stop the active ChatGPT response"),
    CommandSpec("goal", "Run the current task until complete or blocked"),
    CommandSpec("export", "Export the attached conversation to Markdown"),
    CommandSpec("image", "Attach an image to the next prompt"),
    CommandSpec("paste", "Attach the clipboard image to the next prompt"),
    CommandSpec("model", "Choose a real ChatGPT model"),
    CommandSpec("exit", "Exit gptty chat"),
)


def _text_width(value: str) -> int:
    return sum(get_cwidth(char) for char in value)


def _clip_toolbar(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if _text_width(value) <= width:
        return value
    if width == 1:
        return "…"
    remaining = width - 1
    chars: list[str] = []
    used = 0
    for char in value:
        char_width = get_cwidth(char)
        if used + char_width > remaining:
            break
        chars.append(char)
        used += char_width
    return "".join(chars) + "…"


def _fit_toolbar(candidates: tuple[str, ...], width: int | None) -> str:
    if not candidates:
        return ""
    if width is None:
        return candidates[0]
    for candidate in candidates:
        if _text_width(candidate) <= width:
            return candidate
    return _clip_toolbar(candidates[-1], width)


def _compact_active_status(status: str) -> str:
    compact = status
    replacements = (
        ("no observable server events ", "server silent "),
        (" · turn may still be working", ""),
        (" · turn may still recover", ""),
        (" · no observable progress", ""),
        (" · waiting safely", ""),
        (" · answer text received", " · answer received"),
        (" · do not resend yet", " · don't resend"),
        (" · finality unconfirmed ", " · finality "),
        ("CodexPro exact activity ", "CodexPro active "),
        ("CodexPro exact: ", "CodexPro "),
        (" · heartbeat ", " · hb "),
    )
    for old, new in replacements:
        compact = compact.replace(old, new)
    return compact


@dataclass
class _TranscriptLine:
    fragments: list[tuple[str, str]] = field(default_factory=list)
    chars: int = 0
    _wrapped_width: int | None = None
    _wrapped_rows: tuple[tuple[tuple[str, str], ...], ...] = ()

    def _invalidate_wrap(self) -> None:
        self._wrapped_width = None
        self._wrapped_rows = ()

    def append(self, style: str, text: str) -> None:
        if not text:
            return
        if self.fragments and self.fragments[-1][0] == style:
            previous_style, previous_text = self.fragments[-1]
            self.fragments[-1] = (previous_style, previous_text + text)
        else:
            self.fragments.append((style, text))
        self.chars += len(text)
        self._invalidate_wrap()

    def trim_prefix(self, count: int) -> int:
        remaining = max(0, int(count))
        if remaining <= 0 or not self.fragments:
            return 0
        removed_chars = 0
        kept: list[tuple[str, str]] = []
        for style, text in self.fragments:
            if remaining <= 0:
                kept.append((style, text))
                continue
            if len(text) <= remaining:
                removed_chars += len(text)
                remaining -= len(text)
                continue
            removed = text[:remaining]
            tail = text[remaining:]
            removed_chars += len(removed)
            remaining = 0
            if tail:
                kept.append((style, tail))
        self.fragments = kept
        self.chars = max(0, self.chars - removed_chars)
        self._invalidate_wrap()
        return removed_chars

    @staticmethod
    def _append_row_fragment(
        row: list[tuple[str, str]], style: str, text: str
    ) -> None:
        if not text:
            return
        if row and row[-1][0] == style:
            previous_style, previous_text = row[-1]
            row[-1] = (previous_style, previous_text + text)
        else:
            row.append((style, text))

    def wrapped(self, width: int) -> tuple[tuple[tuple[str, str], ...], ...]:
        width = max(1, int(width))
        if self._wrapped_width == width:
            return self._wrapped_rows
        if not self.fragments:
            rows: list[list[tuple[str, str]]] = [[]]
        else:
            rows = [[]]
            used = 0
            for style, text in self.fragments:
                segment_start = 0
                for index, char in enumerate(text):
                    char_width = max(0, get_cwidth(char))
                    if used > 0 and char_width > 0 and used + char_width > width:
                        self._append_row_fragment(
                            rows[-1], style, text[segment_start:index]
                        )
                        rows.append([])
                        used = 0
                        segment_start = index
                    used += char_width
                self._append_row_fragment(rows[-1], style, text[segment_start:])
        self._wrapped_width = width
        self._wrapped_rows = tuple(tuple(row) for row in rows)
        return self._wrapped_rows


class _TranscriptControl(UIControl):
    def __init__(
        self,
        content_provider: Callable[[int, int], list[list[tuple[str, str]]]],
        *,
        scroll_handler: Callable[[MouseEvent], object],
    ) -> None:
        self._content_provider = content_provider
        self._scroll_handler = scroll_handler

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self._content_provider(max(1, width), max(1, height))
        return UIContent(
            get_line=lambda index: lines[index],
            line_count=len(lines),
            show_cursor=False,
        )

    def mouse_handler(self, mouse_event: MouseEvent) -> object:
        if mouse_event.event_type in {
            MouseEventType.SCROLL_UP,
            MouseEventType.SCROLL_DOWN,
        }:
            return self._scroll_handler(mouse_event)
        return NotImplemented


class TranscriptStream:
    """Text stream that feeds the prompt_toolkit-owned transcript viewport."""

    supports_rich_ansi = True
    supports_live = False

    def __init__(
        self,
        session: "InteractiveSession",
        base: TextIO,
        *,
        stream_name: str,
    ) -> None:
        self._session = session
        self._base = base
        self._stream_name = stream_name

    @property
    def encoding(self) -> str:
        return getattr(self._base, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._base, "errors", None) or "strict"

    def fileno(self) -> int:
        return self._base.fileno()

    def isatty(self) -> bool:
        # Rich ANSI is forced explicitly by PrettyRenderer. Returning False here
        # keeps Rich Live/cursor-control output out of the transcript model.
        return False

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._session.append_transcript(text)
        return len(text)

    def write_stream_fragment(self, text: str) -> int:
        self._session.append_transcript(text)
        return len(text)

    def record_prompt(self, text: str) -> None:
        if not text:
            return
        lines = text.rstrip().splitlines() or [""]
        rendered = [f"❯ {lines[0]}"]
        rendered.extend(f"  {line}" for line in lines[1:])
        self._session.append_transcript("\n".join(rendered) + "\n")

    def flush(self) -> None:
        return None

    def clear(self) -> None:
        self._session.clear_transcript()


class InteractiveSession:
    def __init__(
        self,
        *,
        history_file: str | Path,
        settings_file: str | Path,
        settings: UISettings | None = None,
        prompt_input: Any | None = None,
        prompt_output: Any | None = None,
    ) -> None:
        self.history_file = Path(history_file)
        self.settings_file = Path(settings_file)
        self.settings = settings or load_ui_settings(self.settings_file)
        self._prompt_input = prompt_input
        self._prompt_output = prompt_output
        self._turn_controls: TurnControlSignals | None = None
        self._working_status: Callable[[], str] | None = None
        self._transcript_lines: deque[_TranscriptLine] = deque([_TranscriptLine()])
        self._transcript_chars = 0
        self._transcript_char_limit = 1_000_000
        self._transcript_line_limit = 20_000
        self._transcript_follow_tail = True
        self._transcript_has_new_output = False
        self._transcript_scroll_row = 0
        self._transcript_max_scroll = 0
        self._transcript_view_width = 1
        self._transcript_view_height = 1
        self._transcript_window: Window | None = None
        self._input_window: Window | None = None
        self._transcript_control: _TranscriptControl | None = None
        self._footer_control: FormattedTextControl | None = None
        self._attachment_count = 0
        self._prompt_override: str | None = None
        self._persistent_input_queue: asyncio.Queue[object] | None = None
        self._application_task: asyncio.Task[Any] | None = None
        self._default_accept_handler: Callable[[Any], bool] | None = None
        self._persistent_cancel = object()
        self._picker_active = False
        self._picker_previous_prompt: str | None = None
        self._picker_previous_completer: Any | None = None
        self._command_completer: Any | None = None
        self._session: PromptSession[str]
        self._build_session()

    def _build_session(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        command_words = [f"/{spec.name}" for spec in COMMANDS]
        meta = {f"/{spec.name}": spec.description for spec in COMMANDS}
        completer = FuzzyCompleter(
            WordCompleter(command_words, meta_dict=meta, sentence=True),
            enable_fuzzy=True,
        )
        self._command_completer = completer
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add(Keys.ControlC, eager=True)
        def _control_c(event: Any) -> None:
            if self._picker_active:
                self._cancel_picker()
                return
            if self._turn_controls is not None:
                self._turn_controls.request_stop()
                return
            event.app.exit(exception=KeyboardInterrupt())

        @bindings.add("escape", filter=Condition(lambda: self._picker_active), eager=True)
        def _cancel_picker_key(event: Any) -> None:
            self._cancel_picker()

        @bindings.add(Keys.ControlBackslash, eager=True)
        def _control_backslash(event: Any) -> None:
            if self._turn_controls is not None:
                self._turn_controls.request_quit()
                return
            event.app.exit(exception=EOFError())

        @bindings.add(Keys.PageUp, eager=True)
        def _page_up(event: Any) -> None:
            self.scroll_transcript(-self._transcript_page_size())

        @bindings.add(Keys.PageDown, eager=True)
        def _page_down(event: Any) -> None:
            self.scroll_transcript(self._transcript_page_size())

        @bindings.add(Keys.ScrollUp, eager=True)
        def _scroll_up(event: Any) -> None:
            self.scroll_transcript(-3)

        @bindings.add(Keys.ScrollDown, eager=True)
        def _scroll_down(event: Any) -> None:
            self.scroll_transcript(3)

        empty_input = Condition(
            lambda: not self._picker_active
            and not self._session.default_buffer.text
            and self._session.default_buffer.complete_state is None
        )

        @bindings.add(Keys.Up, filter=empty_input, eager=True)
        def _alternate_scroll_up(event: Any) -> None:
            self.scroll_transcript(-3)

        @bindings.add(Keys.Down, filter=empty_input, eager=True)
        def _alternate_scroll_down(event: Any) -> None:
            self.scroll_transcript(3)

        @bindings.add(Keys.ControlEnd, eager=True)
        def _scroll_bottom(event: Any) -> None:
            self.scroll_transcript_to_bottom()

        editing_mode = EditingMode.VI if self.settings.editor == "vi" else EditingMode.EMACS
        kwargs: dict[str, Any] = {
            "message": self._prompt_text,
            "history": FileHistory(str(self.history_file)),
            "auto_suggest": AutoSuggestFromHistory(),
            "completer": completer,
            "complete_while_typing": True,
            "reserve_space_for_menu": 0,
            "multiline": False,
            "key_bindings": bindings,
            "editing_mode": editing_mode,
            "bottom_toolbar": None,
            "mouse_support": False,
            "refresh_interval": 1.0,
        }
        if self._prompt_input is not None:
            kwargs["input"] = self._prompt_input
        if self._prompt_output is not None:
            kwargs["output"] = self._prompt_output
        self._session = PromptSession(**kwargs)
        self._session.app.ttimeoutlen = 0.05
        self._bound_input_window_height()
        self._install_transcript_layout()
        if os.name != "nt" and hasattr(signal, "SIGWINCH"):
            # prompt_toolkit also polls terminal size every 0.5s by default.
            # On POSIX main-thread TTYs SIGWINCH is authoritative; keeping both
            # produces a second resize callback after the first redraw.
            self._session.app.terminal_size_polling_interval = None

    def _prompt_text(self) -> str:
        if self._prompt_override is not None:
            return self._prompt_override
        if self._attachment_count:
            suffix = "s" if self._attachment_count != 1 else ""
            return f"[{self._attachment_count} image{suffix}] ❯ "
        return "❯ "

    def _input_window_height(self) -> Dimension:
        return Dimension(min=1, max=8)

    def _bound_input_window_height(self) -> None:
        """Keep the prompt at the bottom and grow it only for visible content/menu."""
        input_window = self._session.app.layout.current_window
        input_window.height = self._input_window_height
        input_window.dont_extend_height = to_filter(True)
        self._input_window = input_window

    def _strip_inner_completion_floats(self, container: Any) -> None:
        seen: set[int] = set()

        def visit(node: Any) -> None:
            node_id = id(node)
            if node_id in seen:
                return
            seen.add(node_id)
            if isinstance(node, FloatContainer):
                node.floats = [
                    item
                    for item in node.floats
                    if "Completion" not in type(item.content).__name__
                ]
            for child in getattr(node, "children", ()) or ():
                visit(child)
            for attr in ("content", "alternative_content"):
                child = getattr(node, attr, None)
                if child is not None and child is not node:
                    visit(child)

        visit(container)

    def _install_transcript_layout(self) -> None:
        app = self._session.app
        original = app.layout.container
        self._strip_inner_completion_floats(original)
        children = list(getattr(original, "children", ()))
        self._transcript_control = _TranscriptControl(
            self._visible_transcript,
            scroll_handler=self._transcript_mouse_handler,
        )
        self._transcript_window = Window(
            content=self._transcript_control,
            wrap_lines=False,
            always_hide_cursor=True,
            height=Dimension(weight=1),
        )
        self._footer_control = FormattedTextControl(self._bottom_toolbar)
        footer = Window(
            content=self._footer_control,
            height=1,
            style="reverse",
            always_hide_cursor=True,
        )
        body = HSplit([self._transcript_window, *children, footer])
        completion_popup = CompletionsMenu(
            max_height=8,
            scroll_offset=1,
            extra_filter=has_focus(self._session.default_buffer),
            display_arrows=True,
        )
        app.layout = Layout(
            FloatContainer(
                content=body,
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=completion_popup,
                        allow_cover_cursor=False,
                        z_index=100,
                    )
                ],
            ),
            focused_element=self._session.default_buffer,
        )
        # PromptSession builds its Application/Renderer for line mode by default.
        # Both flags must agree or resize redraws leak old full-screen frames into
        # the normal terminal screen.
        app.full_screen = True
        app.renderer.full_screen = True
        self.scroll_transcript_to_bottom()

    def transcript_stream(self, base: TextIO, *, stream_name: str) -> TranscriptStream:
        return TranscriptStream(self, base, stream_name=stream_name)

    def append_transcript(self, text: str) -> None:
        if not text:
            return
        normalized = text.replace("\r\n", "\n").replace("\r", "")
        if "\x1b" in normalized:
            try:
                parsed = to_formatted_text(ANSI(normalized))
                fragments = [(style, value) for style, value, *_rest in parsed]
            except Exception:
                fragments = [("", normalized)]
        else:
            fragments = [("", normalized)]
        for style, value in fragments:
            if not value:
                continue
            parts = value.split("\n")
            for index, part in enumerate(parts):
                if part:
                    self._transcript_lines[-1].append(style, part)
                    self._transcript_chars += len(part)
                if index < len(parts) - 1:
                    self._transcript_chars += 1
                    self._transcript_lines.append(_TranscriptLine())
        self._trim_transcript()
        if not self._transcript_follow_tail:
            self._transcript_has_new_output = True
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def clear_transcript(self) -> None:
        self._transcript_lines = deque([_TranscriptLine()])
        self._transcript_chars = 0
        self._transcript_has_new_output = False
        self._transcript_follow_tail = True
        self._transcript_scroll_row = 0
        self._transcript_max_scroll = 0
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def _trim_transcript(self) -> None:
        width = max(1, self._transcript_view_width)
        while len(self._transcript_lines) > 1 and (
            self._transcript_chars > self._transcript_char_limit
            or len(self._transcript_lines) > self._transcript_line_limit
        ):
            first = self._transcript_lines.popleft()
            removed_rows = len(first.wrapped(width))
            self._transcript_chars = max(0, self._transcript_chars - first.chars - 1)
            if not self._transcript_follow_tail:
                self._transcript_scroll_row = max(
                    0, self._transcript_scroll_row - removed_rows
                )
        if (
            self._transcript_chars > self._transcript_char_limit
            and self._transcript_lines
        ):
            first = self._transcript_lines[0]
            before_rows = len(first.wrapped(width))
            excess = self._transcript_chars - self._transcript_char_limit
            removed = first.trim_prefix(excess)
            self._transcript_chars = max(0, self._transcript_chars - removed)
            if not self._transcript_follow_tail and removed:
                after_rows = len(first.wrapped(width))
                self._transcript_scroll_row = max(
                    0,
                    self._transcript_scroll_row - max(0, before_rows - after_rows),
                )

    def _formatted_transcript(self) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        lines = list(self._transcript_lines)
        for index, line in enumerate(lines):
            fragments.extend(line.fragments)
            if index < len(lines) - 1:
                fragments.append(("", "\n"))
        return fragments

    def _visible_transcript(
        self, width: int, height: int
    ) -> list[list[tuple[str, str]]]:
        width = max(1, int(width))
        height = max(1, int(height))
        self._transcript_view_width = width
        self._transcript_view_height = height
        total_rows = sum(
            len(line.wrapped(width)) for line in self._transcript_lines
        )
        self._transcript_max_scroll = max(0, total_rows - height)
        if self._transcript_follow_tail:
            self._transcript_scroll_row = self._transcript_max_scroll
        else:
            self._transcript_scroll_row = min(
                self._transcript_max_scroll,
                max(0, self._transcript_scroll_row),
            )

        start = self._transcript_scroll_row
        visible: list[list[tuple[str, str]]] = []
        row_cursor = 0
        for line in self._transcript_lines:
            wrapped = line.wrapped(width)
            next_cursor = row_cursor + len(wrapped)
            if next_cursor <= start:
                row_cursor = next_cursor
                continue
            offset = max(0, start - row_cursor)
            for row in wrapped[offset:]:
                visible.append(list(row))
                if len(visible) >= height:
                    return visible
            row_cursor = next_cursor
        return visible or [[]]

    def _transcript_mouse_handler(self, event: MouseEvent) -> object:
        if event.event_type == MouseEventType.SCROLL_UP:
            self.scroll_transcript(-3)
            return None
        if event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll_transcript(3)
            return None
        return NotImplemented

    def _transcript_page_size(self) -> int:
        if self._transcript_view_height > 1:
            return max(3, self._transcript_view_height - 1)
        try:
            rows = int(self._session.app.output.get_size().rows)
        except Exception:
            return 10
        return max(3, rows - 5)

    def scroll_transcript(self, delta: int) -> None:
        if self._transcript_control is None or delta == 0:
            return
        if self._transcript_follow_tail:
            self._transcript_scroll_row = self._transcript_max_scroll
        self._transcript_follow_tail = False
        self._transcript_scroll_row = min(
            self._transcript_max_scroll,
            max(0, self._transcript_scroll_row + int(delta)),
        )
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def scroll_transcript_to_bottom(self) -> None:
        self._transcript_follow_tail = True
        self._transcript_has_new_output = False
        self._transcript_scroll_row = self._transcript_max_scroll
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def _persistent_accept(self, buffer: Any) -> bool:
        queue = self._persistent_input_queue
        if queue is None:
            return True
        state = buffer.complete_state
        completion = state.current_completion if state is not None else None
        if completion is None and state is not None and state.completions:
            completion = state.completions[0]
        if completion is not None:
            buffer.apply_completion(completion)
        queue.put_nowait(buffer.text)
        return False

    def _restore_picker_state(self) -> None:
        buffer = self._session.default_buffer
        self._picker_active = False
        self._prompt_override = self._picker_previous_prompt
        buffer.completer = self._picker_previous_completer or self._command_completer
        self._picker_previous_prompt = None
        self._picker_previous_completer = None
        buffer.reset()
        self._session.app.invalidate()

    def _cancel_picker(self) -> None:
        if not self._picker_active:
            return
        queue = self._persistent_input_queue
        self._restore_picker_state()
        if queue is not None:
            queue.put_nowait(self._persistent_cancel)

    async def start_async(self) -> None:
        task = self._application_task
        if task is not None and not task.done():
            return
        self._persistent_input_queue = asyncio.Queue()
        buffer = self._session.default_buffer
        self._default_accept_handler = buffer.accept_handler
        buffer.accept_handler = self._persistent_accept
        self._application_task = asyncio.create_task(self._session.app.run_async())
        await asyncio.sleep(0)
        output = self._session.app.output
        try:
            # Keep native mouse selection. In alternate screen mode Ghostty/xterm
            # translate wheel scrolling into cursor up/down when DECSET 1007 is on.
            output.write_raw("\x1b[?1007h")
            output.flush()
        except Exception:
            pass

    async def stop_async(self) -> None:
        task = self._application_task
        if task is None:
            return
        output = self._session.app.output
        try:
            output.write_raw("\x1b[?1007l")
            output.flush()
        except Exception:
            pass
        if not task.done():
            try:
                self._session.app.exit()
            except Exception:
                pass
        with suppress(BaseException):
            await task
        self._application_task = None
        self._persistent_input_queue = None
        self._picker_active = False
        buffer = self._session.default_buffer
        if self._default_accept_handler is not None:
            buffer.accept_handler = self._default_accept_handler
        self._default_accept_handler = None

    async def _next_persistent_input(self) -> object:
        queue = self._persistent_input_queue
        app_task = self._application_task
        if queue is None or app_task is None:
            raise RuntimeError("interactive application is not running")
        queue_task = asyncio.create_task(queue.get())
        done, _ = await asyncio.wait(
            {queue_task, app_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if queue_task in done:
            return queue_task.result()
        queue_task.cancel()
        with suppress(asyncio.CancelledError):
            await queue_task
        try:
            await app_task
        except (KeyboardInterrupt, EOFError):
            raise
        raise EOFError()

    def reopen_command_completion(self) -> None:
        buffer = self._session.default_buffer
        buffer.reset()
        buffer.text = "/"
        buffer.cursor_position = 1
        buffer.start_completion(select_first=False)
        self._session.app.invalidate()

    async def choose_searchable_async(
        self,
        message: str,
        options: list[tuple[Any, str]],
    ) -> Any | None:
        if not options:
            return None
        await self.start_async()
        labels: list[str] = []
        values_by_label: dict[str, Any] = {}
        for value, raw_label in options:
            label = str(raw_label).strip() or str(value)
            if label in values_by_label:
                label = f"{label}  [{value}]"
            labels.append(label)
            values_by_label[label] = value

        buffer = self._session.default_buffer
        self._picker_previous_completer = buffer.completer
        self._picker_previous_prompt = self._prompt_override
        self._picker_active = True
        self._prompt_override = f"{message}: "
        buffer.completer = FuzzyCompleter(
            WordCompleter(labels, sentence=True),
            enable_fuzzy=True,
        )
        buffer.reset()
        buffer.start_completion(select_first=False)
        self._session.app.invalidate()
        try:
            raw = await self._next_persistent_input()
            if raw is self._persistent_cancel:
                return None
            selected = str(raw).strip()
            if not selected:
                return None
            if selected in values_by_label:
                return values_by_label[selected]
            for value, _label in options:
                if str(value) == selected:
                    return value
            matches = [
                value
                for label, value in values_by_label.items()
                if selected.casefold() in label.casefold()
            ]
            return matches[0] if len(matches) == 1 else None
        finally:
            if self._picker_active or self._picker_previous_completer is not None:
                self._restore_picker_state()

    def read_prompt(self, *, attachment_count: int = 0) -> str:
        marker = f"[{attachment_count} image{'s' if attachment_count != 1 else ''}] " if attachment_count else ""
        return self._session.prompt(f"{marker}❯ ")

    async def read_prompt_async(self, *, attachment_count: int = 0) -> str:
        self._attachment_count = attachment_count
        await self.start_async()
        self._session.app.invalidate()
        raw = await self._next_persistent_input()
        if raw is self._persistent_cancel:
            return ""
        return str(raw)

    @property
    def application(self) -> Any:
        return self._session.app

    @property
    def active_turn_controls(self) -> TurnControlSignals | None:
        return self._turn_controls

    def set_active_turn(
        self,
        controls: TurnControlSignals | None,
        *,
        working_status: Callable[[], str] | None = None,
    ) -> None:
        self._turn_controls = controls
        self._working_status = working_status
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def _bottom_toolbar(self) -> str:
        width = self._toolbar_width()
        scroll_suffix = ""
        compact_scroll_suffix = ""
        if not self._transcript_follow_tail:
            if self._transcript_has_new_output:
                scroll_suffix = " · ↓ new · Ctrl-End bottom"
                compact_scroll_suffix = " · ↓ new"
            else:
                scroll_suffix = " · Ctrl-End bottom"
                compact_scroll_suffix = " · ↑ scroll"
        if self._turn_controls is not None or self._working_status is not None:
            status = self._working_status() if self._working_status is not None else "working"
            compact = _compact_active_status(status)
            return _fit_toolbar(
                (
                    f" {status} · / commands · Ctrl-C stop · Ctrl-\\ quit{scroll_suffix}",
                    f" {status} · Ctrl-C stop{scroll_suffix}",
                    f" {compact} · Ctrl-C stop{compact_scroll_suffix}",
                    f" {compact}{compact_scroll_suffix}",
                ),
                width,
            )
        return _fit_toolbar(
            (
                f" / actions · Ctrl-R history · Alt-Enter newline{scroll_suffix}",
                f" / actions · Ctrl-R history{scroll_suffix}",
                f" / actions{compact_scroll_suffix}",
            ),
            width,
        )

    def _toolbar_width(self) -> int | None:
        try:
            size = self._session.app.output.get_size()
            columns = int(size.columns)
        except Exception:
            return None
        if columns <= 0:
            return None
        # Leave the terminal's last column unused. Some terminals auto-wrap
        # when the final cell is painted, which makes a one-line toolbar jump
        # during SIGWINCH redraws.
        return max(1, columns - 1)

    def read_image_path(self) -> str | None:
        kwargs: dict[str, Any] = {
            "completer": PathCompleter(expanduser=True),
            "complete_while_typing": True,
            "bottom_toolbar": "Enter attach · Esc/Ctrl-C cancel · drag a file here also works",
        }
        if self._prompt_input is not None:
            kwargs["input"] = self._prompt_input
        if self._prompt_output is not None:
            kwargs["output"] = self._prompt_output
        session = PromptSession(**kwargs)
        try:
            value = session.prompt("Image path: ").strip()
        except (KeyboardInterrupt, EOFError):
            return None
        return value or None

    def choose_command(self) -> str | None:
        selected = self.choose(
            "Actions",
            [(f"/{spec.name}", f"/{spec.name:<10} {spec.description}") for spec in COMMANDS],
        )
        return str(selected) if selected else None

    def choose(self, message: str, options: list[tuple[Any, str]], *, default: Any | None = None) -> Any | None:
        if not options:
            return None
        cancel_bindings = KeyBindings()

        @cancel_bindings.add("escape")
        def _cancel(event: Any) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        try:
            return choice(
                message,
                options=options,
                default=default,
                bottom_toolbar="↑↓ select · Enter confirm · Esc cancel",
                show_frame=False,
                key_bindings=cancel_bindings,
            )
        except (KeyboardInterrupt, EOFError):
            return None

    def choose_searchable(
        self,
        message: str,
        options: list[tuple[Any, str]],
    ) -> Any | None:
        if not options:
            return None
        labels: list[str] = []
        values_by_label: dict[str, Any] = {}
        for value, raw_label in options:
            label = str(raw_label).strip() or str(value)
            if label in values_by_label:
                label = f"{label}  [{value}]"
            labels.append(label)
            values_by_label[label] = value

        picker_bindings = KeyBindings()

        @picker_bindings.add("escape")
        def _cancel_picker(event: Any) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        @picker_bindings.add("enter")
        def _accept_picker(event: Any) -> None:
            buffer = event.current_buffer
            state = buffer.complete_state
            completion = state.current_completion if state is not None else None
            if completion is not None:
                buffer.apply_completion(completion)
                event.app.exit(result=buffer.text)
                return
            buffer.validate_and_handle()

        kwargs: dict[str, Any] = {
            "completer": FuzzyCompleter(WordCompleter(labels, sentence=True), enable_fuzzy=True),
            "complete_while_typing": True,
            "complete_style": CompleteStyle.COLUMN,
            "key_bindings": picker_bindings,
            "bottom_toolbar": "↑↓ browse · type to filter · Enter resume · Esc/Ctrl-C cancel",
        }
        if self._prompt_input is not None:
            kwargs["input"] = self._prompt_input
        if self._prompt_output is not None:
            kwargs["output"] = self._prompt_output
        picker = PromptSession(**kwargs)
        try:
            selected = picker.prompt(
                f"{message}: ",
                pre_run=lambda: get_app().current_buffer.start_completion(select_first=False),
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not selected:
            return None
        if selected in values_by_label:
            return values_by_label[selected]
        for value, _label in options:
            if str(value) == selected:
                return value
        matches = [
            value
            for label, value in values_by_label.items()
            if selected.casefold() in label.casefold()
        ]
        return matches[0] if len(matches) == 1 else None

def should_use_enhanced_ui(
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    state_path: str | Path,
    force_plain: bool = False,
) -> tuple[bool, UISettings]:
    settings_file = ui_settings_path(state_path)
    try:
        settings = load_ui_settings(settings_file)
    except UIStateError:
        settings = UISettings()
    if force_plain or os.environ.get("TERM", "").lower() == "dumb" or "NO_COLOR" in os.environ:
        return False, settings
    if not (_isatty(input_stream) and _isatty(output_stream)):
        return False, settings
    if settings.pretty == "off":
        return False, settings
    return True, settings


def _isatty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except OSError:
        return False
