from __future__ import annotations

import asyncio
from io import StringIO
import os
import signal

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output.base import Size

from gptty.ui.session import (
    COMMANDS,
    InteractiveSession,
    _text_width,
    should_use_enhanced_ui,
)
from gptty.ui.signals import TurnControlSignals
from gptty.ui.state import UISettings, save_ui_settings, ui_settings_path


class TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class ResizableDummyOutput(DummyOutput):
    def __init__(self, columns: int) -> None:
        super().__init__()
        self.columns = columns

    def get_size(self) -> Size:
        return Size(rows=24, columns=self.columns)


def test_enhanced_ui_auto_requires_tty(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    state_path = tmp_path / "gptty_state.json"

    enabled, _ = should_use_enhanced_ui(
        input_stream=TTYStringIO(),
        output_stream=TTYStringIO(),
        state_path=state_path,
    )
    disabled, _ = should_use_enhanced_ui(
        input_stream=StringIO(),
        output_stream=TTYStringIO(),
        state_path=state_path,
    )

    assert enabled is True
    assert disabled is False


def test_pretty_off_disables_enhanced_ui(tmp_path) -> None:
    state_path = tmp_path / "gptty_state.json"
    save_ui_settings(ui_settings_path(state_path), UISettings(pretty="off"))

    enabled, _ = should_use_enhanced_ui(
        input_stream=TTYStringIO(),
        output_stream=TTYStringIO(),
        state_path=state_path,
    )

    assert enabled is False


def test_pretty_on_never_forces_prompt_toolkit_into_non_tty(tmp_path) -> None:
    state_path = tmp_path / "gptty_state.json"
    save_ui_settings(ui_settings_path(state_path), UISettings(pretty="on"))

    enabled, _ = should_use_enhanced_ui(
        input_stream=StringIO(),
        output_stream=TTYStringIO(),
        state_path=state_path,
    )

    assert enabled is False


def test_posix_session_uses_sigwinch_without_redundant_size_polling(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
    )

    if os.name != "nt" and hasattr(signal, "SIGWINCH"):
        assert session.application.terminal_size_polling_interval is None


def test_transcript_layout_is_fullscreen_with_pinned_footer(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )

    root = session.application.layout.container
    assert session.application.full_screen is True
    assert session.application.renderer.full_screen is True
    assert session._session.mouse_support is True
    assert session._transcript_window is not None
    assert root.children[0] is session._transcript_window
    assert root.children[-1].content is session._footer_control
    assert root.children[-1].height == 1


def test_transcript_auto_follow_uses_real_bottom_not_sentinel(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    session.append_transcript("\n".join(f"line {index}" for index in range(75)))

    session._visible_transcript(80, 20)
    assert session._transcript_max_scroll == 55
    assert session._transcript_scroll_row == 55

    session.scroll_transcript(-10)
    assert session._transcript_follow_tail is False
    session._visible_transcript(80, 20)
    assert session._transcript_scroll_row == 45


def test_transcript_control_materializes_only_viewport_rows(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    session.append_transcript("\n".join(f"line {index}" for index in range(5000)))
    control = session._transcript_control
    assert control is not None

    content = control.create_content(width=100, height=24)

    assert content.line_count == 24
    assert session._transcript_max_scroll == 4976
    assert session._transcript_scroll_row == 4976


def test_transcript_stream_keeps_ansi_as_formatted_text_and_can_clear(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    stream = session.transcript_stream(StringIO(), stream_name="stdout")

    stream.write("\x1b[31mhello\x1b[0m\n")

    fragments = session._formatted_transcript()
    assert "".join(fragment[1] for fragment in fragments) == "hello\n"
    assert any("ansired" in fragment[0] for fragment in fragments)

    stream.clear()
    assert session._formatted_transcript() == []


def test_scroll_up_freezes_transcript_and_marks_new_output(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    session.append_transcript("\n".join(f"line {index}" for index in range(80)))
    session._visible_transcript(80, 20)
    assert session._transcript_scroll_row == 60

    session.scroll_transcript(-5)
    assert session._transcript_scroll_row == 55
    assert session._transcript_follow_tail is False

    session.append_transcript("\nnew output")
    session._visible_transcript(80, 20)

    assert session._transcript_scroll_row == 55
    assert session._transcript_has_new_output is True
    assert "new" in session._bottom_toolbar()

    session.scroll_transcript_to_bottom()
    session._visible_transcript(80, 20)
    assert session._transcript_follow_tail is True
    assert session._transcript_has_new_output is False
    assert session._transcript_scroll_row == session._transcript_max_scroll
    assert "↓ new" not in session._bottom_toolbar()


def test_transcript_mouse_wheel_scrolls_without_changing_input_focus(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    session.append_transcript("\n".join(f"line {index}" for index in range(80)))
    session._visible_transcript(80, 20)
    control = session._transcript_control
    assert control is not None
    focused = session.application.layout.current_buffer

    result = control.mouse_handler(
        MouseEvent(
            position=Point(x=60, y=10),
            event_type=MouseEventType.SCROLL_UP,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
    )

    assert result is None
    assert session._transcript_scroll_row == 57
    assert session._transcript_follow_tail is False
    assert session.application.layout.current_buffer is focused


def test_raw_sgr_mouse_wheel_scrolls_transcript(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            session.append_transcript("".join(f"line {i}\n" for i in range(80)))
            task = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)
            before = session._transcript_scroll_row

            pipe.send_bytes(b"\x1b[<64;10;5M")
            await asyncio.sleep(0.05)

            assert session._transcript_follow_tail is False
            assert session._transcript_scroll_row == before - 3

            pipe.send_text("done\r")
            assert await task == "done"

    asyncio.run(scenario())


def test_raw_pageup_then_ctrl_end_toggles_transcript_follow(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            session.append_transcript("".join(f"line {i}\n" for i in range(80)))
            task = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)

            pipe.send_bytes(b"\x1b[5~")
            await asyncio.sleep(0.05)
            assert session._transcript_follow_tail is False

            session.append_transcript("late output\n")
            assert session._transcript_has_new_output is True
            assert "↓ new" in session._bottom_toolbar()

            pipe.send_bytes(b"\x1b[1;5F")
            await asyncio.sleep(0.05)
            assert session._transcript_follow_tail is True
            assert session._transcript_has_new_output is False

            pipe.send_text("done\r")
            assert await task == "done"

    asyncio.run(scenario())


def test_transcript_stream_records_submitted_multiline_prompt(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    stream = session.transcript_stream(StringIO(), stream_name="stdout")

    stream.record_prompt("hello\nworld")

    rendered = "".join(fragment[1] for fragment in session._formatted_transcript())
    assert rendered == "❯ hello\n  world\n"


def test_transcript_buffer_is_bounded(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    session._transcript_char_limit = 20

    session.append_transcript("old line\n")
    session.append_transcript("newer line\n")
    session.append_transcript("latest line\n")

    rendered = "".join(fragment[1] for fragment in session._formatted_transcript())
    assert len(rendered) <= 20
    assert "latest line" in rendered


def test_prompt_session_reads_input_and_persists_history(tmp_path) -> None:
    with create_pipe_input() as pipe:
        session = InteractiveSession(
            history_file=tmp_path / "history",
            settings_file=tmp_path / "ui.json",
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        pipe.send_text("hello\r")

        assert session.read_prompt() == "hello"

    assert "hello" in (tmp_path / "history").read_text(encoding="utf-8")


def test_alt_enter_inserts_newline_before_submit(tmp_path) -> None:
    with create_pipe_input() as pipe:
        session = InteractiveSession(
            history_file=tmp_path / "history",
            settings_file=tmp_path / "ui.json",
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        pipe.send_text("first\x1b\rsecond\r")

        assert session.read_prompt() == "first\nsecond"


def test_searchable_picker_accepts_unique_fuzzy_text(tmp_path) -> None:
    with create_pipe_input() as pipe:
        session = InteractiveSession(
            history_file=tmp_path / "history",
            settings_file=tmp_path / "ui.json",
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        pipe.send_text("Second\r")

        assert session.choose_searchable(
            "Resume",
            [("conv-1", "First chat"), ("conv-2", "Second chat")],
        ) == "conv-2"


def test_searchable_picker_browses_visible_list_with_arrows(tmp_path) -> None:
    with create_pipe_input() as pipe:
        session = InteractiveSession(
            history_file=tmp_path / "history",
            settings_file=tmp_path / "ui.json",
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        pipe.send_text("\x1b[B\r")

        assert session.choose_searchable(
            "Resume",
            [("conv-1", "First chat"), ("conv-2", "Second chat")],
        ) == "conv-1"


def test_searchable_picker_escape_cancels(tmp_path) -> None:
    with create_pipe_input() as pipe:
        session = InteractiveSession(
            history_file=tmp_path / "history",
            settings_file=tmp_path / "ui.json",
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        pipe.send_text("\x1b")

        assert session.choose_searchable(
            "Resume",
            [("conv-1", "First chat"), ("conv-2", "Second chat")],
        ) is None


def test_active_toolbar_reflows_to_current_terminal_width(tmp_path) -> None:
    output = ResizableDummyOutput(160)
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=output,
    )
    status = (
        "PROLONGED SILENCE · no observable server events 06:46"
        " · turn may still be working · do not resend yet"
        " · CodexPro exact activity 00:04"
    )
    session.set_active_turn(
        TurnControlSignals(),
        working_status=lambda: status,
    )

    expected_prefixes = {
        160: " PROLONGED SILENCE · no observable server events 06:46",
        120: " PROLONGED SILENCE · server silent 06:46",
        80: " PROLONGED SILENCE · server silent 06:46",
        50: " PROLONGED SILENCE · server silent 06:46",
    }
    rendered: dict[int, str] = {}
    for columns, prefix in expected_prefixes.items():
        output.columns = columns
        toolbar = session._bottom_toolbar()
        rendered[columns] = toolbar
        assert _text_width(toolbar) <= columns - 1
        assert toolbar.startswith(prefix)

    assert rendered[160] != rendered[120]
    assert rendered[120] != rendered[80]
    assert "Ctrl-C stop" in rendered[120]
    assert "Ctrl-C stop" not in rendered[80]
    assert "CodexPro active 00:04" in rendered[80]
    assert rendered[50].endswith("…")


def test_status_only_follow_toolbar_is_active_without_turn_controls(tmp_path) -> None:
    output = ResizableDummyOutput(120)
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=output,
    )

    session.set_active_turn(
        None,
        working_status=lambda: "server quiet 05:00 · queued 1",
    )

    toolbar = session._bottom_toolbar()
    assert toolbar.startswith(" server quiet 05:00 · queued 1")
    assert "Ctrl-C stop" in toolbar
    assert session.active_turn_controls is None


def test_status_only_follow_ctrl_c_keeps_follow_stop_semantics(tmp_path) -> None:
    with create_pipe_input() as pipe:
        session = InteractiveSession(
            history_file=tmp_path / "history",
            settings_file=tmp_path / "ui.json",
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        session.set_active_turn(
            None,
            working_status=lambda: "server quiet 05:00 · queued 1",
        )
        pipe.send_text("\x03")

        with pytest.raises(KeyboardInterrupt):
            session.read_prompt()

    assert session.active_turn_controls is None


def test_status_only_follow_ctrl_backslash_keeps_local_quit_semantics(tmp_path) -> None:
    with create_pipe_input() as pipe:
        session = InteractiveSession(
            history_file=tmp_path / "history",
            settings_file=tmp_path / "ui.json",
            prompt_input=pipe,
            prompt_output=DummyOutput(),
        )
        session.set_active_turn(
            None,
            working_status=lambda: "server quiet 05:00 · queued 1",
        )
        pipe.send_text("\x1c")

        with pytest.raises(EOFError):
            session.read_prompt()

    assert session.active_turn_controls is None


def test_idle_toolbar_reflows_without_using_last_terminal_column(tmp_path) -> None:
    output = ResizableDummyOutput(30)
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=output,
    )

    toolbar = session._bottom_toolbar()
    assert _text_width(toolbar) <= 29
    assert toolbar.startswith(" / actions")

    output.columns = 10
    toolbar = session._bottom_toolbar()
    assert _text_width(toolbar) <= 9
    assert toolbar.startswith(" / act")
    assert toolbar.endswith("…")


def test_command_registry_exposes_session_actions() -> None:
    names = {spec.name for spec in COMMANDS}
    assert names == {
        "new",
        "temporary",
        "resume",
        "detach",
        "stop",
        "goal",
        "export",
        "image",
        "paste",
        "model",
        "exit",
    }
