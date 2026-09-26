from __future__ import annotations

import asyncio
from bisect import bisect_right
import os
import signal
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Callable, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import (
    Completer,
    Completion,
    ConditionalCompleter,
    FuzzyCompleter,
    PathCompleter,
    WordCompleter,
)
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER, EditingMode
from prompt_toolkit.filters import Condition, has_focus, is_done
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import CompletionsMenu, Float, FloatContainer, Layout
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, SearchBufferControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import (
    AppendAutoSuggestion,
    ConditionalProcessor,
    ReverseSearchProcessor,
    TabsProcessor,
)
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.shortcuts import CompleteStyle, choice
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.markdown import Markdown

from .command_registry import COMMANDS, CommandSpec
from .history import PrivatePromptHistory
from .signals import TurnControlSignals
from .state import UISettings, UIStateError, load_ui_settings, ui_settings_path
from .terminal_capabilities import TerminalCapabilities
from .terminal_safety import sanitize_terminal_text


ACTIVE_STATUS_REFRESH_SECONDS = 1.0
TABSTOP = 4


@dataclass
class _PersistentPromptRuntime:
    app: Application[str]
    default_buffer: Buffer
    search_buffer: Buffer
    mouse_support: bool = False


def _command_meta(spec: CommandSpec) -> str:
    if spec.usage:
        return f"{spec.description} · {spec.usage}"
    return spec.description


class _ContextualCommandCompleter(Completer):
    def __init__(self, commands: tuple[CommandSpec, ...]) -> None:
        self._commands = commands
        self._by_name = {spec.name: spec for spec in commands}

    def get_completions(self, document: Any, complete_event: Any) -> Any:
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return

        if " " not in text:
            typed = text[1:].strip().lower()
            for spec in self._commands:
                if typed and not spec.name.startswith(typed):
                    continue
                yield Completion(
                    f"/{spec.name}",
                    start_position=-len(text),
                    display=f"/{spec.name}",
                    display_meta=_command_meta(spec),
                )
            return

        command_token, remainder = text.split(" ", 1)
        command_name = command_token[1:].strip().lower()
        spec = self._by_name.get(command_name)
        if spec is None or not spec.options:
            return

        # Only complete the first subcommand token. Once a second argument begins,
        # command completion is no longer useful.
        if any(char.isspace() for char in remainder.strip()):
            return
        typed = remainder.strip().lower()
        for option in spec.options:
            if typed and not option.value.lower().startswith(typed):
                continue
            yield Completion(
                option.value,
                start_position=-len(typed),
                display=option.value,
                display_meta=option.description,
            )


def _text_width(value: str) -> int:
    return sum(get_cwidth(char) for char in value)


def _tab_span(display_position: int, *, tabstop: int = TABSTOP) -> int:
    stop = max(1, int(tabstop))
    return stop - (max(0, int(display_position)) % stop)


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


def _ansi_fragments(value: str) -> list[tuple[str, str]]:
    safe = sanitize_terminal_text(value, allow_sgr=True)
    if not safe:
        return []
    if "\x1b" not in safe:
        return [("", safe)]
    try:
        parsed = to_formatted_text(ANSI(safe))
        return [(style, text) for style, text, *_rest in parsed]
    except Exception:
        # A parser failure must never reintroduce raw escape bytes.
        return [("", sanitize_terminal_text(safe))]


def _ansi_rows(value: str) -> tuple[tuple[tuple[str, str], ...], ...]:
    if not value:
        return ((),)
    fragments = _ansi_fragments(value)

    rows: list[list[tuple[str, str]]] = [[]]
    for style, text in fragments:
        if not text:
            continue
        parts = text.split("\n")
        for index, part in enumerate(parts):
            if part:
                _TranscriptLine._append_row_fragment(rows[-1], style, part)
            if index < len(parts) - 1:
                rows.append([])
    return tuple(tuple(row) for row in rows)


def _render_markdown_rows(
    value: str, width: int
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Render raw Markdown for one viewport width without freezing that width in state."""
    width = max(1, int(width))
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
        soft_wrap=False,
        width=width,
        height=1_000_000,
    )
    console.print(Markdown(value))
    rendered = output.getvalue()
    rows = list(_ansi_rows(rendered))
    # Console.print adds one terminal newline after the renderable. The transcript
    # block itself owns content rows; the caller keeps the semantic line boundary.
    if rendered.endswith("\n") and rows and not rows[-1]:
        rows.pop()
    return tuple(rows) or ((),)


@dataclass
class _TranscriptLine:
    fragments: list[tuple[str, str]] = field(default_factory=list)
    chars: int = 0
    rule_title: str | None = None
    rule_style: str = ""
    markdown_text: str | None = None
    _wrapped_width: int | None = None
    _wrapped_rows: tuple[tuple[tuple[str, str], ...], ...] = ()
    _markdown_wrapped_cache: dict[
        int, tuple[tuple[tuple[str, str], ...], ...]
    ] = field(default_factory=dict)

    def _invalidate_wrap(self) -> None:
        self._wrapped_width = None
        self._wrapped_rows = ()
        self._markdown_wrapped_cache.clear()

    def append(self, style: str, text: str) -> None:
        if not text:
            return
        if self.markdown_text is not None:
            raise RuntimeError("cannot append terminal fragments to a Markdown transcript block")
        if self.fragments and self.fragments[-1][0] == style:
            previous_style, previous_text = self.fragments[-1]
            self.fragments[-1] = (previous_style, previous_text + text)
        else:
            self.fragments.append((style, text))
        self.chars += len(text)
        self._invalidate_wrap()

    def trim_prefix(self, count: int) -> int:
        remaining = max(0, int(count))
        if remaining <= 0:
            return 0
        if self.markdown_text is not None:
            removed_chars = min(remaining, len(self.markdown_text))
            self.markdown_text = self.markdown_text[removed_chars:]
            self.chars = max(0, self.chars - removed_chars)
            self._invalidate_wrap()
            return removed_chars
        if not self.fragments:
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
        if self.markdown_text is not None:
            cached = self._markdown_wrapped_cache.get(width)
            if cached is None:
                cached = _render_markdown_rows(self.markdown_text, width)
                self._markdown_wrapped_cache[width] = cached
                while len(self._markdown_wrapped_cache) > 4:
                    oldest_width = next(iter(self._markdown_wrapped_cache))
                    self._markdown_wrapped_cache.pop(oldest_width, None)
            self._wrapped_width = width
            self._wrapped_rows = cached
            return cached
        if self.rule_title is not None:
            title = f" {self.rule_title.strip()} "
            if _text_width(title) >= width:
                rendered = _clip_toolbar(self.rule_title.strip(), width)
            else:
                remaining = width - _text_width(title)
                left = remaining // 2
                right = remaining - left
                rendered = ("─" * left) + title + ("─" * right)
            self._wrapped_width = width
            self._wrapped_rows = (((self.rule_style, rendered),),)
            return self._wrapped_rows
        if not self.fragments:
            rows: list[list[tuple[str, str]]] = [[]]
        else:
            rows = [[]]
            used = 0
            display_position = 0
            for style, text in self.fragments:
                for char in text:
                    if char == "\t":
                        span = _tab_span(display_position)
                        display_position += span
                        for _ in range(span):
                            if used > 0 and used + 1 > width:
                                rows.append([])
                                used = 0
                            self._append_row_fragment(rows[-1], style, " ")
                            used += 1
                        continue

                    char_width = max(0, get_cwidth(char))
                    if used > 0 and char_width > 0 and used + char_width > width:
                        rows.append([])
                        used = 0
                    self._append_row_fragment(rows[-1], style, char)
                    used += char_width
                    display_position += 1
        self._wrapped_width = width
        self._wrapped_rows = tuple(tuple(row) for row in rows)
        return self._wrapped_rows


class _TranscriptRowIndex:
    """Incremental wrapped-row index for the bounded transcript."""

    def __init__(self, first_line: _TranscriptLine) -> None:
        self._lines: list[_TranscriptLine] = [first_line]
        self._row_starts: list[int] = [0]
        self._row_counts: list[int] = [0]
        self._head = 0
        self._width = 0
        self._dirty_from: int | None = 0

    def reset(self, first_line: _TranscriptLine) -> None:
        self._lines = [first_line]
        self._row_starts = [0]
        self._row_counts = [0]
        self._head = 0
        self._width = 0
        self._dirty_from = 0

    def append(self, line: _TranscriptLine) -> None:
        self._lines.append(line)
        self._row_starts.append(0)
        self._row_counts.append(0)
        physical = len(self._lines) - 1
        self._dirty_from = (
            physical
            if self._dirty_from is None
            else min(self._dirty_from, physical)
        )

    def replace_last(self, line: _TranscriptLine) -> None:
        if self._head >= len(self._lines):
            self.reset(line)
            return
        self._lines[-1] = line
        self.mark_changed(-1)

    def mark_changed(self, active_index: int) -> None:
        active_count = len(self._lines) - self._head
        if active_count <= 0:
            return
        logical = active_index if active_index >= 0 else active_count + active_index
        logical = min(max(0, logical), active_count - 1)
        physical = self._head + logical
        self._dirty_from = (
            physical
            if self._dirty_from is None
            else min(self._dirty_from, physical)
        )

    def drop_first(self) -> None:
        if self._head >= len(self._lines):
            return
        self._head += 1
        if self._dirty_from is not None and self._dirty_from < self._head:
            self._dirty_from = self._head if self._head < len(self._lines) else None
        if self._head >= 4096 and self._head * 2 >= len(self._lines):
            self._compact()

    def _compact(self) -> None:
        if self._head <= 0:
            return
        active_lines = self._lines[self._head :]
        active_counts = self._row_counts[self._head :]
        base = self._row_starts[self._head] if active_lines else 0
        active_starts = [value - base for value in self._row_starts[self._head :]]
        dirty = self._dirty_from
        self._lines = active_lines
        self._row_counts = active_counts
        self._row_starts = active_starts
        self._dirty_from = None if dirty is None else max(0, dirty - self._head)
        self._head = 0

    def _sync(self, width: int) -> None:
        width = max(1, int(width))
        if self._head >= len(self._lines):
            return
        if width != self._width:
            self._compact()
            self._width = width
            self._row_starts = [0] * len(self._lines)
            self._row_counts = [0] * len(self._lines)
            self._dirty_from = 0

        dirty = self._dirty_from
        if dirty is None:
            return
        start = max(self._head, dirty)
        if start >= len(self._lines):
            self._dirty_from = None
            return
        next_start = (
            self._row_starts[start]
            if start == self._head
            else self._row_starts[start - 1] + self._row_counts[start - 1]
        )
        for physical in range(start, len(self._lines)):
            self._row_starts[physical] = next_start
            count = len(self._lines[physical].wrapped(width))
            self._row_counts[physical] = count
            next_start += count
        self._dirty_from = None

    def total_rows(self, width: int) -> int:
        self._sync(width)
        if self._head >= len(self._lines):
            return 0
        base = self._row_starts[self._head]
        last = len(self._lines) - 1
        return self._row_starts[last] + self._row_counts[last] - base

    def visible_rows(
        self, width: int, start: int, height: int
    ) -> list[list[tuple[str, str]]]:
        self._sync(width)
        height = max(0, int(height))
        if height <= 0 or self._head >= len(self._lines):
            return []
        total = self.total_rows(width)
        if total <= 0:
            return []
        start = min(max(0, int(start)), max(0, total - 1))
        base = self._row_starts[self._head]
        target = base + start
        physical = max(
            self._head,
            bisect_right(
                self._row_starts,
                target,
                lo=self._head,
                hi=len(self._lines),
            )
            - 1,
        )
        offset = max(0, target - self._row_starts[physical])
        visible: list[list[tuple[str, str]]] = []
        while physical < len(self._lines) and len(visible) < height:
            wrapped = self._lines[physical].wrapped(width)
            for row in wrapped[offset:]:
                visible.append(list(row))
                if len(visible) >= height:
                    break
            physical += 1
            offset = 0
        return visible


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

    def write_markdown(self, text: str) -> None:
        self._session.append_markdown(text)

    def record_prompt(self, text: str) -> None:
        if not text:
            return
        self._session.ensure_transcript_line_boundary()
        lines = text.rstrip().splitlines() or [""]
        rendered = [f"\x1b[1;7m YOU \x1b[0m❯ {lines[0]}"]
        rendered.extend(f"       {line}" for line in lines[1:])
        self._session.append_transcript("\n".join(rendered) + "\n")

    def write_rule(self, title: str, *, style: str | None = None) -> None:
        self._session.append_rule(title, style=style)

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
        initial_transcript_line = _TranscriptLine()
        self._transcript_lines: deque[_TranscriptLine] = deque([initial_transcript_line])
        self._transcript_row_index = _TranscriptRowIndex(initial_transcript_line)
        self._transcript_trimmed = False
        self._transcript_trim_marker = _TranscriptLine(
            rule_title="older local transcript trimmed",
            rule_style="fg:#888888",
        )
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
        self._search_window: Window | None = None
        self._input_visual_cache: tuple[str, int, str, list[tuple[int, int]]] | None = None
        self._transcript_control: _TranscriptControl | None = None
        self._footer_control: FormattedTextControl | None = None
        self._attachment_count = 0
        self._prompt_override: str | None = None
        self._persistent_input_queue: asyncio.Queue[object] | None = None
        self._application_task: asyncio.Task[Any] | None = None
        self._status_tick_task: asyncio.Task[None] | None = None
        self._persistent_cancel = object()
        self._picker_active = False
        self._picker_previous_prompt: str | None = None
        self._picker_previous_completer: Any | None = None
        self._command_completer: Any | None = None
        self._terminal_capabilities: TerminalCapabilities | None = None
        self._history = PrivatePromptHistory(
            self.history_file,
            limit=self.settings.history_limit,
        )
        self._session: _PersistentPromptRuntime
        self._build_session()
        self._terminal_capabilities = TerminalCapabilities(self._session.app.output)

    def _build_session(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        completer = ConditionalCompleter(
            _ContextualCommandCompleter(COMMANDS),
            Condition(lambda: get_app().current_buffer.text.startswith("/")),
        )
        self._command_completer = completer

        default_buffer = Buffer(
            name=DEFAULT_BUFFER,
            history=self._history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            complete_while_typing=True,
            multiline=False,
            accept_handler=self._persistent_accept,
        )
        search_buffer = Buffer(name=SEARCH_BUFFER)
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add(
            Keys.Enter,
            filter=Condition(lambda: get_app().current_buffer is default_buffer),
            eager=True,
        )
        def _accept_input(event: Any) -> None:
            event.current_buffer.validate_and_handle()

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

        @bindings.add(
            Keys.ControlD,
            filter=Condition(
                lambda: get_app().current_buffer is default_buffer
                and not default_buffer.text
            ),
            eager=True,
        )
        def _eof(event: Any) -> None:
            event.app.exit(exception=EOFError())

        main_input = Condition(
            lambda: not self._picker_active
            and get_app().current_buffer is default_buffer
            and default_buffer.complete_state is None
        )
        main_draft = main_input & Condition(lambda: bool(default_buffer.text))
        visual_draft = main_draft & Condition(self._input_has_multiple_visual_rows)
        # DECSET 1007 delivers wheel motion as Up/Down keys. A single visual
        # draft row has no vertical cursor navigation to preserve, so keep wheel
        # scrolling attached to the transcript until the draft actually wraps.
        transcript_arrow_scroll = main_input & ~visual_draft
        empty_input = main_input & Condition(lambda: not default_buffer.text)

        @bindings.add(Keys.PageUp, filter=visual_draft, eager=True)
        def _draft_page_up(event: Any) -> None:
            self._move_input_visual_rows(-self._input_page_size())

        @bindings.add(Keys.PageDown, filter=visual_draft, eager=True)
        def _draft_page_down(event: Any) -> None:
            self._move_input_visual_rows(self._input_page_size())

        @bindings.add(Keys.PageUp, filter=empty_input, eager=True)
        def _page_up(event: Any) -> None:
            self.scroll_transcript(-self._transcript_page_size())

        @bindings.add(Keys.PageDown, filter=empty_input, eager=True)
        def _page_down(event: Any) -> None:
            self.scroll_transcript(self._transcript_page_size())

        @bindings.add(Keys.ScrollUp, eager=True)
        def _scroll_up(event: Any) -> None:
            self.scroll_transcript(-3)

        @bindings.add(Keys.ScrollDown, eager=True)
        def _scroll_down(event: Any) -> None:
            self.scroll_transcript(3)

        @bindings.add(Keys.Up, filter=visual_draft, eager=True)
        def _draft_up(event: Any) -> None:
            self._move_input_visual_rows(-1)

        @bindings.add(Keys.Down, filter=visual_draft, eager=True)
        def _draft_down(event: Any) -> None:
            self._move_input_visual_rows(1)

        @bindings.add(Keys.Up, filter=transcript_arrow_scroll, eager=True)
        def _alternate_scroll_up(event: Any) -> None:
            self.scroll_transcript(-3)

        @bindings.add(Keys.Down, filter=transcript_arrow_scroll, eager=True)
        def _alternate_scroll_down(event: Any) -> None:
            self.scroll_transcript(3)

        @bindings.add("c-p", filter=main_input, eager=True)
        def _history_previous(event: Any) -> None:
            event.current_buffer.history_backward()

        @bindings.add("c-n", filter=main_input, eager=True)
        def _history_next(event: Any) -> None:
            event.current_buffer.history_forward()

        @bindings.add(Keys.ControlUp, filter=visual_draft, eager=True)
        @bindings.add(Keys.ControlHome, filter=main_draft, eager=True)
        def _draft_start(event: Any) -> None:
            self._move_input_to_edge(end=False)

        @bindings.add(Keys.ControlDown, filter=visual_draft, eager=True)
        @bindings.add(Keys.ControlEnd, filter=main_draft, eager=True)
        def _draft_end(event: Any) -> None:
            self._move_input_to_edge(end=True)

        @bindings.add(Keys.ControlEnd, filter=empty_input, eager=True)
        def _scroll_bottom(event: Any) -> None:
            self.scroll_transcript_to_bottom()

        search_control = SearchBufferControl(
            buffer=search_buffer,
            input_processors=[ReverseSearchProcessor()],
            ignore_case=False,
        )
        default_control = BufferControl(
            buffer=default_buffer,
            search_buffer_control=search_control,
            input_processors=[
                TabsProcessor(tabstop=TABSTOP, char1=" ", char2=" "),
                ConditionalProcessor(
                    AppendAutoSuggestion(),
                    has_focus(default_buffer) & ~is_done,
                ),
            ],
            include_default_input_processors=True,
            preview_search=True,
        )
        input_window = Window(
            content=default_control,
            height=self._input_window_height,
            dont_extend_height=True,
            get_line_prefix=self._input_line_prefix,
            wrap_lines=True,
        )
        self._input_window = input_window
        search_window = Window(
            content=search_control,
            height=Dimension(min=1, max=1),
            dont_extend_height=True,
        )
        self._search_window = search_window

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
        input_stack = HSplit(
            [
                ConditionalContainer(
                    input_window,
                    Condition(lambda: get_app().layout.current_control != search_control),
                ),
                ConditionalContainer(
                    search_window,
                    Condition(lambda: get_app().layout.current_control == search_control),
                ),
            ]
        )
        body = HSplit([self._transcript_window, input_stack])
        completion_popup = CompletionsMenu(
            max_height=12,
            scroll_offset=1,
            extra_filter=has_focus(default_buffer),
            display_arrows=True,
        )
        completion_layer = FloatContainer(
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
        )
        layout = Layout(
            HSplit([completion_layer, footer]),
            focused_element=default_buffer,
        )
        editing_mode = (
            EditingMode.VI if self.settings.editor == "vi" else EditingMode.EMACS
        )
        app_kwargs: dict[str, Any] = {
            "layout": layout,
            "key_bindings": bindings,
            "full_screen": True,
            "mouse_support": False,
            "editing_mode": editing_mode,
            "refresh_interval": None,
        }
        if self._prompt_input is not None:
            app_kwargs["input"] = self._prompt_input
        if self._prompt_output is not None:
            app_kwargs["output"] = self._prompt_output
        app: Application[str] = Application(**app_kwargs)
        app.ttimeoutlen = 0.05
        if os.name != "nt" and hasattr(signal, "SIGWINCH"):
            app.terminal_size_polling_interval = None

        self._session = _PersistentPromptRuntime(
            app=app,
            default_buffer=default_buffer,
            search_buffer=search_buffer,
            mouse_support=False,
        )
        self.scroll_transcript_to_bottom()

    def _prompt_text(self) -> str:
        if self._prompt_override is not None:
            return self._prompt_override
        if self._attachment_count:
            suffix = "s" if self._attachment_count != 1 else ""
            return f"[{self._attachment_count} image{suffix}] ❯ "
        return "❯ "

    def _input_window_height(self) -> Dimension:
        return Dimension(min=1, max=8)

    def _input_line_prefix(self, line_number: int, wrap_count: int) -> str:
        if line_number == 0 and wrap_count == 0:
            return self._prompt_text()
        return ""

    def _input_page_size(self) -> int:
        render_info = self._input_window.render_info if self._input_window is not None else None
        if render_info is not None:
            return max(1, int(render_info.window_height) - 1)
        return max(1, self._input_window_height().max - 1)

    def _move_input_to_edge(self, *, end: bool) -> None:
        buffer = self._session.default_buffer
        buffer.cursor_position = len(buffer.text) if end else 0
        buffer.preferred_column = None
        self._session.app.invalidate()

    def _input_visual_positions(self) -> list[tuple[int, int]]:
        """Return the wrapped draft row/column for every cursor position."""
        buffer = self._session.default_buffer
        render_info = self._input_window.render_info if self._input_window is not None else None
        if render_info is not None:
            width = max(1, int(render_info.window_width))
        else:
            try:
                width = max(1, int(self._session.app.output.get_size().columns))
            except Exception:
                width = 80

        text = buffer.text
        prompt = self._prompt_text()
        cache = self._input_visual_cache
        if cache is not None:
            cached_text, cached_width, cached_prompt, cached_positions = cache
            if cached_text is text and cached_width == width and cached_prompt == prompt:
                return cached_positions

        prompt_width = min(width - 1, _text_width(prompt))
        row = 0
        col = 0
        capacity = max(1, width - prompt_width)
        positions: list[tuple[int, int]] = [(row, col)]
        display_position = 0
        for char in text:
            if char == "\n":
                row += 1
                col = 0
                capacity = width
                display_position = 0
                positions.append((row, col))
                continue

            if char == "\t":
                span = _tab_span(display_position)
                display_position += span
                for _ in range(span):
                    col += 1
                    while col >= capacity:
                        col -= capacity
                        row += 1
                        capacity = width
                positions.append((row, col))
                continue

            char_width = max(0, get_cwidth(char))
            display_position += 1
            if char_width > 0 and col + char_width > capacity:
                row += 1
                col = 0
                capacity = width
            col += char_width
            while col >= capacity:
                col -= capacity
                row += 1
                capacity = width
            positions.append((row, col))
        self._input_visual_cache = (text, width, prompt, positions)
        return positions

    def _input_has_multiple_visual_rows(self) -> bool:
        positions = self._input_visual_positions()
        return bool(positions and positions[-1][0] > 0)

    def _move_input_visual_rows(self, delta: int) -> None:
        """Move through wrapped screen rows instead of logical newline rows."""
        if delta == 0:
            return

        buffer = self._session.default_buffer
        if not buffer.text or not self._input_has_multiple_visual_rows():
            return

        before = buffer.cursor_position
        positions = self._input_visual_positions()
        current_y, current_x = positions[before]
        target_y = current_y + int(delta)
        candidates: list[tuple[int, int]] = []
        for index, (row, col) in enumerate(positions):
            if row != target_y:
                continue
            if delta > 0 and index <= before:
                continue
            if delta < 0 and index >= before:
                continue
            candidates.append((abs(col - current_x), index))

        if candidates:
            _, target = min(candidates)
            buffer.cursor_position = target
        elif delta > 0 and before < len(buffer.text):
            buffer.cursor_position = len(buffer.text)
        elif delta < 0 and before > 0:
            buffer.cursor_position = 0

        buffer.preferred_column = None
        self._session.app.invalidate()

    def transcript_stream(self, base: TextIO, *, stream_name: str) -> TranscriptStream:
        return TranscriptStream(self, base, stream_name=stream_name)

    def _append_transcript_line(self, line: _TranscriptLine | None = None) -> _TranscriptLine:
        value = line or _TranscriptLine()
        self._transcript_lines.append(value)
        self._transcript_row_index.append(value)
        return value

    def _replace_transcript_last(self, line: _TranscriptLine) -> None:
        self._transcript_lines[-1] = line
        self._transcript_row_index.replace_last(line)

    def _mark_transcript_trimmed(self, width: int) -> int:
        if self._transcript_trimmed:
            return 0
        self._transcript_trimmed = True
        return len(self._transcript_trim_marker.wrapped(width))

    def append_transcript(self, text: str) -> None:
        if not text:
            return
        fragments = _ansi_fragments(text)
        for style, value in fragments:
            if not value:
                continue
            parts = value.split("\n")
            for index, part in enumerate(parts):
                if part:
                    self._transcript_lines[-1].append(style, part)
                    self._transcript_row_index.mark_changed(-1)
                    self._transcript_chars += len(part)
                if index < len(parts) - 1:
                    self._transcript_chars += 1
                    self._append_transcript_line()
        self._trim_transcript()
        if not self._transcript_follow_tail:
            self._transcript_has_new_output = True
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def append_markdown(self, text: str) -> None:
        value = sanitize_terminal_text(str(text))
        if not value:
            return
        current = self._transcript_lines[-1]
        if (
            current.fragments
            or current.rule_title is not None
            or current.markdown_text is not None
        ):
            self._append_transcript_line()
        self._replace_transcript_last(
            _TranscriptLine(chars=len(value), markdown_text=value)
        )
        self._append_transcript_line()
        self._transcript_chars += len(value) + 1
        self._trim_transcript()
        if not self._transcript_follow_tail:
            self._transcript_has_new_output = True
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def ensure_transcript_line_boundary(self) -> None:
        current = self._transcript_lines[-1]
        if (
            not current.fragments
            and current.rule_title is None
            and current.markdown_text is None
        ):
            return
        self._transcript_chars += 1
        self._append_transcript_line()
        self._trim_transcript()

    def append_rule(self, title: str, *, style: str | None = None) -> None:
        title = sanitize_terminal_text(str(title)).strip()
        if not title:
            return
        current = self._transcript_lines[-1]
        if (
            current.fragments
            or current.rule_title is not None
            or current.markdown_text is not None
        ):
            self._append_transcript_line()
        self._replace_transcript_last(
            _TranscriptLine(
                chars=len(title),
                rule_title=title,
                rule_style="fg:#888888" if style == "dim" else "",
            )
        )
        self._append_transcript_line()
        self._transcript_chars += len(title) + 1
        self._trim_transcript()
        if not self._transcript_follow_tail:
            self._transcript_has_new_output = True
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def clear_transcript(self) -> None:
        initial_line = _TranscriptLine()
        self._transcript_lines = deque([initial_line])
        self._transcript_row_index.reset(initial_line)
        self._transcript_trimmed = False
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
            self._transcript_row_index.drop_first()
            marker_rows = self._mark_transcript_trimmed(width)
            self._transcript_chars = max(0, self._transcript_chars - first.chars - 1)
            if not self._transcript_follow_tail:
                self._transcript_scroll_row = max(
                    0,
                    self._transcript_scroll_row - removed_rows + marker_rows,
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
            if removed:
                self._transcript_row_index.mark_changed(0)
                marker_rows = self._mark_transcript_trimmed(width)
                if not self._transcript_follow_tail:
                    after_rows = len(first.wrapped(width))
                    self._transcript_scroll_row = max(
                        0,
                        self._transcript_scroll_row
                        - max(0, before_rows - after_rows)
                        + marker_rows,
                    )

    def _formatted_transcript(self) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        lines = list(self._transcript_lines)
        if self._transcript_trimmed:
            lines.insert(0, self._transcript_trim_marker)
        width = max(1, self._transcript_view_width)
        for index, line in enumerate(lines):
            if line.markdown_text is not None:
                rows = line.wrapped(width)
                for row_index, row in enumerate(rows):
                    fragments.extend(row)
                    if row_index < len(rows) - 1:
                        fragments.append(("", "\n"))
            elif line.rule_title is not None:
                fragments.extend(line.wrapped(width)[0])
            else:
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

        marker_rows = (
            self._transcript_trim_marker.wrapped(width)
            if self._transcript_trimmed
            else ()
        )
        content_rows = self._transcript_row_index.total_rows(width)
        total_rows = len(marker_rows) + content_rows
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
        marker_count = len(marker_rows)
        content_start = max(0, start - marker_count)
        if marker_count and start < marker_count:
            for row in marker_rows[start:]:
                visible.append(list(row))
                if len(visible) >= height:
                    return visible
            content_start = 0

        remaining = height - len(visible)
        if remaining > 0:
            visible.extend(
                self._transcript_row_index.visible_rows(
                    width,
                    content_start,
                    remaining,
                )
            )
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
        if delta > 0 and self._transcript_scroll_row >= self._transcript_max_scroll:
            self._transcript_follow_tail = True
            self._transcript_has_new_output = False
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

    def _cancel_status_tick(self) -> asyncio.Task[None] | None:
        task = self._status_tick_task
        self._status_tick_task = None
        if task is not None and not task.done():
            task.cancel()
        return task

    def _sync_status_tick(self) -> None:
        app_task = self._application_task
        if (
            self._working_status is None
            or app_task is None
            or app_task.done()
        ):
            self._cancel_status_tick()
            return
        task = self._status_tick_task
        if task is not None and not task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._status_tick_task = asyncio.create_task(self._status_tick_loop())

    async def _status_tick_loop(self) -> None:
        current = asyncio.current_task()
        try:
            while self._working_status is not None:
                await asyncio.sleep(ACTIVE_STATUS_REFRESH_SECONDS)
                app_task = self._application_task
                if app_task is None or app_task.done() or self._working_status is None:
                    return
                try:
                    self._session.app.invalidate()
                except Exception:
                    return
        except asyncio.CancelledError:
            return
        finally:
            if self._status_tick_task is current:
                self._status_tick_task = None

    def _restore_terminal_capabilities(self) -> None:
        capabilities = self._terminal_capabilities
        if capabilities is not None:
            capabilities.restore()

    async def start_async(self) -> None:
        task = self._application_task
        if task is not None and not task.done():
            return
        self._persistent_input_queue = asyncio.Queue()
        buffer = self._session.default_buffer
        buffer.accept_handler = self._persistent_accept
        self._application_task = asyncio.create_task(self._session.app.run_async())
        self._application_task.add_done_callback(
            lambda _task: self._restore_terminal_capabilities()
        )
        await asyncio.sleep(0)
        self._sync_status_tick()
        if (
            self._application_task is not None
            and not self._application_task.done()
            and self._terminal_capabilities is not None
        ):
            self._terminal_capabilities.enter_persistent_tui()

    async def stop_async(self) -> None:
        status_tick = self._cancel_status_tick()
        if status_tick is not None:
            with suppress(asyncio.CancelledError):
                await status_tick
        self._restore_terminal_capabilities()
        task = self._application_task
        if task is None:
            return
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
        buffer.accept_handler = self._persistent_accept

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
        marker = (
            f"[{attachment_count} image{'s' if attachment_count != 1 else ''}] "
            if attachment_count
            else ""
        )
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add(Keys.ControlBackslash)
        def _quit_local(event: Any) -> None:
            event.app.exit(exception=EOFError())

        kwargs: dict[str, Any] = {
            "history": self._history,
            "auto_suggest": AutoSuggestFromHistory(),
            "completer": self._command_completer,
            "complete_while_typing": True,
            "multiline": False,
            "editing_mode": (
                EditingMode.VI if self.settings.editor == "vi" else EditingMode.EMACS
            ),
            "mouse_support": False,
            "key_bindings": bindings,
        }
        if self._prompt_input is not None:
            kwargs["input"] = self._prompt_input
        if self._prompt_output is not None:
            kwargs["output"] = self._prompt_output
        return PromptSession(**kwargs).prompt(f"{marker}❯ ")

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

    def set_history_persistent(self, enabled: bool) -> None:
        self._history.set_persistent(enabled)

    def clear_history(self) -> None:
        self._history.clear()

    @property
    def history_persistent(self) -> bool:
        return self._history.persistent

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
        self._sync_status_tick()
        try:
            self._session.app.invalidate()
        except Exception:
            pass

    def _command_toolbar_hint(self) -> str | None:
        if self._picker_active:
            return None
        buffer = self._session.default_buffer
        text = buffer.text
        if not text.startswith("/") or "\n" in text:
            return None

        command_token, separator, remainder = text.partition(" ")
        command_name = command_token[1:].strip().lower()
        spec = next((item for item in COMMANDS if item.name == command_name), None)
        if spec is None:
            return None

        if separator and remainder.strip():
            first_arg = remainder.strip().split()[0].lower()
            option = next((item for item in spec.options if item.value == first_arg), None)
            if option is not None:
                return f" /{spec.name} {option.value} · {option.description}"

        detail = spec.usage or spec.description
        return f" /{spec.name} · {detail}"

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
        command_hint = self._command_toolbar_hint()
        if command_hint is not None:
            return _fit_toolbar((command_hint,), width)
        return _fit_toolbar(
            (
                f" / actions · Ctrl-P/N history · Ctrl-R search · Alt-Enter newline{scroll_suffix}",
                f" / actions · Ctrl-P/N history · Ctrl-R search{scroll_suffix}",
                f" / actions · Ctrl-R history{compact_scroll_suffix}",
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

    async def read_image_path_async(self) -> str | None:
        await self.start_async()
        buffer = self._session.default_buffer
        self._picker_previous_completer = buffer.completer
        self._picker_previous_prompt = self._prompt_override
        self._picker_active = True
        self._prompt_override = "Image path: "
        buffer.completer = PathCompleter(expanduser=True)
        buffer.reset()
        self._session.app.invalidate()
        try:
            raw = await self._next_persistent_input()
            if raw is self._persistent_cancel:
                return None
            value = str(raw).strip()
            return value or None
        finally:
            if self._picker_active or self._picker_previous_completer is not None:
                self._restore_picker_state()

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
            [(f"/{spec.name}", f"/{spec.name:<10} {_command_meta(spec)}") for spec in COMMANDS],
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
