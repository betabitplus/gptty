from __future__ import annotations

import asyncio
from io import StringIO
import os
import signal

import pytest
from prompt_toolkit.history import FileHistory
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
    def __init__(self, columns: int, rows: int = 24) -> None:
        super().__init__()
        self.columns = columns
        self.rows = rows

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)


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
    completion_layer = root.children[0]
    body = completion_layer.content
    footer = root.children[-1]
    assert session.application.full_screen is True
    assert session.application.renderer.full_screen is True
    assert session._session.mouse_support is False
    assert session._transcript_window is not None
    assert body.children[0] is session._transcript_window
    assert footer.content is session._footer_control
    assert footer.height == 1
    assert session._footer_control not in [getattr(child, "content", None) for child in body.children]


def test_reverse_history_search_keeps_transcript_geometry(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80, rows=20),
            )
            session.append_transcript("alpha\nbeta\ngamma\n")
            task = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)

            transcript_window = session._transcript_window
            assert transcript_window is not None
            assert transcript_window.render_info is not None
            before_height = transcript_window.render_info.window_height
            assert before_height == 18

            pipe.send_bytes(b"\x12")
            await asyncio.sleep(0.05)

            assert session.application.layout.current_buffer.name == "SEARCH_BUFFER"
            assert transcript_window.render_info is not None
            assert transcript_window.render_info.window_height == before_height

            pipe.send_bytes(b"\x1b")
            await asyncio.sleep(0.12)
            assert session.application.layout.current_buffer.name == "DEFAULT_BUFFER"

            pipe.send_text("done\r")
            assert await task == "done"
            await session.stop_async()

    asyncio.run(scenario())


def test_prompt_stays_at_bottom_and_grows_only_with_content(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(100, rows=40),
            )
            task = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)

            input_window = session._input_window
            transcript_window = session._transcript_window
            assert input_window is not None
            assert transcript_window is not None
            assert input_window.render_info is not None
            assert transcript_window.render_info is not None
            assert input_window.render_info.window_height == 1
            assert transcript_window.render_info.window_height == 38

            session._session.default_buffer.text = "one\ntwo\nthree\nfour"
            session.application.invalidate()
            await asyncio.sleep(0.05)
            assert input_window.render_info is not None
            assert transcript_window.render_info is not None
            assert input_window.render_info.window_height == 4
            assert transcript_window.render_info.window_height == 35

            session._session.default_buffer.text = "\n".join(f"line {i}" for i in range(20))
            session.application.invalidate()
            await asyncio.sleep(0.05)
            assert input_window.render_info is not None
            assert transcript_window.render_info is not None
            assert input_window.render_info.window_height == 8
            assert transcript_window.render_info.window_height == 31

            session._session.default_buffer.reset()
            session._session.default_buffer.insert_text("/")
            session.application.invalidate()
            await asyncio.sleep(0.05)
            assert session._session.default_buffer.complete_state is not None
            assert input_window.render_info is not None
            assert input_window.render_info.window_height == 1

            session._session.default_buffer.cancel_completion()
            session.application.invalidate()
            await asyncio.sleep(0.05)
            assert input_window.render_info is not None
            assert input_window.render_info.window_height == 1

            session._session.default_buffer.text = "done"
            pipe.send_text("\r")
            assert await task == "done"
            await session.stop_async()

    asyncio.run(scenario())


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


def test_transcript_strips_osc8_hyperlink_controls(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    stream = session.transcript_stream(StringIO(), stream_name="stdout")
    url = "https://chatgpt.com/c/6aafcb0b-5edc-83eb-9463-ba69ebd90547"

    stream.write(f"chat: \x1b]8;id=7924233;{url}\x1b\\{url}\x1b]8;;\x1b\\\n")

    text = "".join(fragment[1] for fragment in session._formatted_transcript())
    assert text == f"chat: {url}\n"
    assert "8;id=" not in text


def test_transcript_markdown_reflows_from_raw_source_on_resize(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(177),
    )
    stream = session.transcript_stream(StringIO(), stream_name="stdout")
    markdown = (
        "Short answer.\n\n"
        "- alpha\n- beta\n\n"
        "```python\ndef hello(name):\n    return f\"hello {name}\"\n```\n\n"
        "| Name | Value |\n|---|---|\n| Alpha | 123 |\n| Beta | 456 |\n\n"
        "> quoted line\n> second line"
    )

    stream.write_markdown(markdown)

    blocks = [line for line in session._transcript_lines if line.markdown_text is not None]
    assert len(blocks) == 1
    block = blocks[0]
    assert block.markdown_text == markdown
    assert block.fragments == []

    wide = block.wrapped(177)
    narrow = block.wrapped(88)
    fresh_narrow = block.wrapped(88)

    assert narrow is fresh_narrow
    assert narrow != wide
    narrow_text = "\n".join(
        "".join(fragment[1] for fragment in row).rstrip() for row in narrow
    )
    assert "Short answer." in narrow_text
    assert "• alpha" in narrow_text
    assert "def hello(name):" in narrow_text
    assert 'return f"hello {name}"' in narrow_text
    assert "Alpha" in narrow_text and "123" in narrow_text
    assert "quoted line second line" in narrow_text


def test_transcript_markdown_short_messages_do_not_gain_blank_rows_after_resize(
    tmp_path,
) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(177),
    )
    stream = session.transcript_stream(StringIO(), stream_name="stdout")
    stream.write("assistant\n")
    stream.write_markdown("Listed available actions")
    stream.write("\nassistant\n")
    stream.write_markdown("SECOND_CLEAN_DONE")

    def logical_rows(width: int) -> list[str]:
        rows = session._visible_transcript(width, 40)
        return ["".join(fragment[1] for fragment in row).rstrip() for row in rows]

    wide = logical_rows(177)
    narrow = logical_rows(88)

    assert wide == narrow
    assert narrow == [
        "assistant",
        "Listed available actions",
        "",
        "assistant",
        "SECOND_CLEAN_DONE",
        "",
    ]


def test_transcript_rules_reflow_semantically_on_resize(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(100),
    )
    stream = session.transcript_stream(StringIO(), stream_name="stdout")
    stream.write_rule("ChatGPT", style="dim")

    wide = session._visible_transcript(100, 10)
    narrow = session._visible_transcript(44, 10)
    wide_rule = "".join(fragment[1] for fragment in wide[0])
    narrow_rule = "".join(fragment[1] for fragment in narrow[0])

    assert "ChatGPT" in wide_rule
    assert "ChatGPT" in narrow_rule
    assert _text_width(wide_rule) == 100
    assert _text_width(narrow_rule) == 44
    assert "\n" not in narrow_rule


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


def test_scrolling_down_to_bottom_restores_follow_tail(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    session.append_transcript("\n".join(f"line {index}" for index in range(80)))
    session._visible_transcript(80, 20)
    session.scroll_transcript(-8)
    session.append_transcript("\nlate output")
    session._visible_transcript(80, 20)

    assert session._transcript_follow_tail is False
    assert session._transcript_has_new_output is True

    session.scroll_transcript(1000)
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


def test_alternate_scroll_cursor_key_scrolls_transcript(tmp_path) -> None:
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

            # Ghostty/xterm DECSET 1007 translates wheel events to cursor keys
            # when the application does not capture the mouse.
            pipe.send_bytes(b"\x1b[A")
            await asyncio.sleep(0.05)

            assert session._transcript_follow_tail is False
            assert session._transcript_scroll_row == before - 3

            pipe.send_text("done\r")
            assert await task == "done"
            await session.stop_async()

    asyncio.run(scenario())


def test_alternate_scroll_still_scrolls_with_single_line_draft(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            session.append_transcript("".join(f"line {i}\n" for i in range(80)))
            await session.start_async()
            buffer = session._session.default_buffer
            buffer.text = "draft message"
            buffer.cursor_position = len(buffer.text)
            session.application.invalidate()
            await asyncio.sleep(0.05)
            before = session._transcript_scroll_row
            cursor_before = buffer.cursor_position

            pipe.send_bytes(b"\x1b[A")
            await asyncio.sleep(0.05)

            assert session._transcript_follow_tail is False
            assert session._transcript_scroll_row == before - 3
            assert buffer.text == "draft message"
            assert buffer.cursor_position == cursor_before

            await session.stop_async()

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
            await session.stop_async()

    asyncio.run(scenario())


def test_picker_escape_cancels_quickly_without_restarting_app(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            await session.start_async()
            app_task = session._application_task
            assert app_task is not None
            picker = asyncio.create_task(
                session.choose_searchable_async(
                    "Pick one",
                    [("one", "One"), ("two", "Two")],
                )
            )
            await asyncio.sleep(0.05)
            pipe.send_bytes(b"\x1b")

            assert await asyncio.wait_for(picker, timeout=0.2) is None
            assert session._application_task is app_task
            assert not app_task.done()
            await session.stop_async()

    asyncio.run(scenario())


def test_image_path_prompt_reuses_persistent_application(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            await session.start_async()
            app_task = session._application_task
            assert app_task is not None and not app_task.done()

            picker = asyncio.create_task(session.read_image_path_async())
            await asyncio.sleep(0.05)
            assert session._picker_active is True
            assert session._prompt_override == "Image path: "

            pipe.send_text("/tmp/gptty-nonexistent-image.png\r")
            assert await picker == "/tmp/gptty-nonexistent-image.png"
            assert session._application_task is app_task
            assert not app_task.done()
            assert session._picker_active is False
            assert session._prompt_override is None

            await session.stop_async()

    asyncio.run(scenario())


def test_image_path_prompt_escape_restores_main_prompt(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            picker = asyncio.create_task(session.read_image_path_async())
            await asyncio.sleep(0.05)
            app_task = session._application_task

            pipe.send_bytes(b"\x1b")
            assert await asyncio.wait_for(picker, timeout=0.2) is None
            assert app_task is not None and not app_task.done()
            assert session._picker_active is False
            assert session._prompt_override is None

            await session.stop_async()

    asyncio.run(scenario())


def test_enter_applies_selected_command_completion_without_restarting_app(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            task = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)
            app_task = session._application_task
            assert app_task is not None

            pipe.send_text("/")
            await asyncio.sleep(0.05)
            assert session._session.default_buffer.complete_state is not None
            pipe.send_text("\r")

            assert await task == "/new"
            await asyncio.sleep(0.05)
            assert session._session.default_buffer.text == ""
            assert session._session.default_buffer.complete_state is None
            assert session._application_task is app_task
            assert not app_task.done()
            await session.stop_async()

    asyncio.run(scenario())


def test_down_arrow_still_browses_command_completion_menu(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            task = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)

            pipe.send_text("/")
            await asyncio.sleep(0.05)
            buffer = session._session.default_buffer
            assert buffer.complete_state is not None
            assert buffer.complete_state.current_completion is None

            pipe.send_bytes(b"\x1b[B")
            await asyncio.sleep(0.03)
            assert buffer.complete_state is not None
            first = buffer.complete_state.current_completion
            assert first is not None
            assert first.text == "/new"

            pipe.send_bytes(b"\x1b[B")
            await asyncio.sleep(0.03)
            assert buffer.complete_state is not None
            selected = buffer.complete_state.current_completion
            assert selected is not None
            assert selected.text == "/temporary"

            pipe.send_text("\r")
            assert await task == "/temporary"
            await session.stop_async()

    asyncio.run(scenario())


def test_contextual_command_completion_exposes_usage_and_subcommands(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(120),
            )
            task = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)
            buffer = session._session.default_buffer

            pipe.send_text("/")
            await asyncio.sleep(0.05)
            assert buffer.complete_state is not None
            top = {item.text: item for item in buffer.complete_state.completions}
            assert "list [all] | open <id> | pause | resume | status | doctor | trace [N] | criteria | clear" in top["/goal"].display_meta_text
            assert "Enter: choose chat" in top["/resume"].display_meta_text

            for raw, expected in (
                ("/goal ", ["list", "open", "pause", "resume", "status", "doctor", "trace", "criteria", "clear"]),
                ("/goal re", ["resume"]),
                ("/image ", ["clear"]),
                ("/model ", ["default"]),
            ):
                buffer.reset()
                buffer.insert_text(raw)
                await asyncio.sleep(0.05)
                assert buffer.complete_state is not None
                assert [item.text for item in buffer.complete_state.completions] == expected

            for raw in ("/resume ", "/stop ", "/export "):
                buffer.reset()
                buffer.insert_text(raw)
                await asyncio.sleep(0.05)
                assert buffer.complete_state is None

            await session.stop_async()
            task.cancel()

    asyncio.run(scenario())


def test_command_toolbar_is_contextual_and_never_hides_active_turn_status(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(120),
            )
            await session.start_async()
            buffer = session._session.default_buffer

            buffer.text = "/resume "
            session.application.invalidate()
            await asyncio.sleep(0.02)
            assert "Enter: choose chat" in session._bottom_toolbar()

            buffer.text = "/goal resume"
            session.application.invalidate()
            await asyncio.sleep(0.02)
            assert "Resume the attached paused or blocked goal" in session._bottom_toolbar()

            controls = TurnControlSignals()
            session.set_active_turn(controls, working_status=lambda: "working · queued 1")
            assert "working · queued 1" in session._bottom_toolbar()
            assert "paused or blocked" not in session._bottom_toolbar()

            session.set_active_turn(None)
            await session.stop_async()

    asyncio.run(scenario())


def test_persistent_application_survives_multiple_submits(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80),
            )
            first = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.05)
            app_task = session._application_task
            assert app_task is not None and not app_task.done()

            pipe.send_text("one\r")
            assert await first == "one"
            assert session._application_task is app_task
            assert not app_task.done()

            second = asyncio.create_task(session.read_prompt_async())
            await asyncio.sleep(0.02)
            pipe.send_text("two\r")
            assert await second == "two"
            assert session._application_task is app_task
            assert not app_task.done()

            await session.stop_async()
            assert app_task.done()

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
    assert rendered == " YOU ❯ hello\n       world\n"
    assert session._transcript_lines[0].fragments[0] == ("bold reverse", " YOU ")


def test_record_prompt_starts_new_line_after_unterminated_stream_fragment(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=ResizableDummyOutput(80),
    )
    stream = session.transcript_stream(StringIO(), stream_name="stdout")
    stream.write_stream_fragment("FINAL_WITHOUT_NEWLINE")

    stream.record_prompt("/new")

    rendered = "".join(fragment[1] for fragment in session._formatted_transcript())
    assert rendered == "FINAL_WITHOUT_NEWLINE\n YOU ❯ /new\n"


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


def test_single_line_draft_arrows_do_not_replace_text_from_history(tmp_path) -> None:
    history = tmp_path / "history"
    FileHistory(str(history)).append_string("previous prompt")

    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=history,
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80, rows=20),
            )
            await session.start_async()
            await asyncio.sleep(0.05)
            buffer = session._session.default_buffer
            buffer.text = "current draft"
            buffer.cursor_position = len(buffer.text)
            session.application.invalidate()
            await asyncio.sleep(0.02)

            pipe.send_bytes(b"\x1b[A")
            await asyncio.sleep(0.02)
            assert buffer.text == "current draft"
            assert buffer.cursor_position == len(buffer.text)

            pipe.send_bytes(b"\x1b[B")
            await asyncio.sleep(0.02)
            assert buffer.text == "current draft"
            assert buffer.cursor_position == len(buffer.text)

            await session.stop_async()

    asyncio.run(scenario())


def test_ctrl_p_n_browse_history_and_restore_current_draft(tmp_path) -> None:
    history = tmp_path / "history"
    file_history = FileHistory(str(history))
    file_history.append_string("older prompt")
    file_history.append_string("newer\nmultiline")

    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=history,
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80, rows=20),
            )
            await session.start_async()
            await asyncio.sleep(0.05)
            buffer = session._session.default_buffer
            buffer.text = "current draft"
            buffer.cursor_position = len(buffer.text)
            session.application.invalidate()
            await asyncio.sleep(0.02)

            pipe.send_bytes(b"\x10")
            await asyncio.sleep(0.02)
            assert buffer.text == "newer\nmultiline"

            pipe.send_bytes(b"\x10")
            await asyncio.sleep(0.02)
            assert buffer.text == "older prompt"

            pipe.send_bytes(b"\x0e")
            await asyncio.sleep(0.02)
            assert buffer.text == "newer\nmultiline"

            pipe.send_bytes(b"\x0e")
            await asyncio.sleep(0.02)
            assert buffer.text == "current draft"

            await session.stop_async()

    asyncio.run(scenario())


def test_long_draft_down_arrow_traverses_wrapped_rows_to_end(tmp_path) -> None:
    text = (
        "@CodexTool внимательно почитай что бы дальше продолжить работу над Sphinx\n"
        "Needs\n"
        "https://chatgpt.com/c/6aa71f0e-2284-83ed-a373-0050985707e1\n"
        "https://chatgpt.com/c/6aa7f385-c678-83eb-9785-95965bb2b130\n"
        "https://chatgpt.com/c/6aa85c3d-09d4-83eb-9747-15810cd97a65\n"
        "https://chatgpt.com/c/6aa98224-a978-83eb-9252-dbcedae0768e\n"
        "https://chatgpt.com/c/6aabf65a-9260-83ed-83c4-a75dfbf01582\n"
        "https://chatgpt.com/c/6aad111b-d2f4-83ed-aec6-a6e5f3fce721\n"
        "https://chatgpt.com/c/6aaec65c-1c5c-83ed-97b9-eaa218e1c02d\n"
        "поясни что мы делали, нюансы с которыми столкнулись и решения которые приняли. "
        "Далее поясни на чем мы остановили и что дальше делаем. Войди в контекст работы короче, "
        "и будь готов продолжать. Иди последовательно и изучи всю историю целиком. "
        "Обязательно все читай полностью через codexpro инструменты включая ссылки на чаты."
    )

    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(80, rows=20),
            )
            await session.start_async()
            buffer = session._session.default_buffer
            buffer.text = text
            buffer.cursor_position = 0
            session.application.invalidate()
            await asyncio.sleep(0.05)

            positions = [0]
            rows = [0]
            for _ in range(24):
                pipe.send_bytes(b"\x1b[B")
                await asyncio.sleep(0.01)
                positions.append(buffer.cursor_position)
                rows.append(buffer.document.cursor_position_row)
                if buffer.cursor_position == len(text):
                    break

            assert positions == sorted(set(positions))
            assert buffer.cursor_position == len(text)
            assert rows.count(9) >= 4
            assert buffer.text == text

            reverse_positions = [buffer.cursor_position]
            for _ in range(24):
                pipe.send_bytes(b"\x1b[A")
                await asyncio.sleep(0.01)
                reverse_positions.append(buffer.cursor_position)
                if buffer.cursor_position == 0:
                    break

            assert reverse_positions == sorted(set(reverse_positions), reverse=True)
            assert buffer.cursor_position == 0
            assert buffer.text == text

            await session.stop_async()

    asyncio.run(scenario())


def test_long_draft_ctrl_arrows_and_ctrl_home_end_jump_to_edges(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(60, rows=16),
            )
            await session.start_async()
            buffer = session._session.default_buffer
            buffer.text = "alpha\n" + ("wrapped " * 40)
            buffer.cursor_position = len(buffer.text) // 2
            session.application.invalidate()
            await asyncio.sleep(0.05)

            pipe.send_bytes(b"\x1b[1;5A")
            await asyncio.sleep(0.02)
            assert buffer.cursor_position == 0

            pipe.send_bytes(b"\x1b[1;5B")
            await asyncio.sleep(0.02)
            assert buffer.cursor_position == len(buffer.text)

            pipe.send_bytes(b"\x1b[1;5H")
            await asyncio.sleep(0.02)
            assert buffer.cursor_position == 0

            pipe.send_bytes(b"\x1b[1;5F")
            await asyncio.sleep(0.02)
            assert buffer.cursor_position == len(buffer.text)

            await session.stop_async()

    asyncio.run(scenario())


def test_long_draft_page_keys_move_inside_input_not_transcript(tmp_path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe:
            session = InteractiveSession(
                history_file=tmp_path / "history",
                settings_file=tmp_path / "ui.json",
                prompt_input=pipe,
                prompt_output=ResizableDummyOutput(50, rows=18),
            )
            session.append_transcript("\n".join(f"transcript {index}" for index in range(100)))
            await session.start_async()
            buffer = session._session.default_buffer
            buffer.text = "draft " * 120
            buffer.cursor_position = 0
            session.application.invalidate()
            await asyncio.sleep(0.05)
            scroll_before = session._transcript_scroll_row

            pipe.send_bytes(b"\x1b[6~")
            await asyncio.sleep(0.02)
            assert buffer.cursor_position > 0
            assert session._transcript_scroll_row == scroll_before

            pipe.send_bytes(b"\x1b[5~")
            await asyncio.sleep(0.02)
            assert buffer.cursor_position == 0
            assert session._transcript_scroll_row == scroll_before

            await session.stop_async()

    asyncio.run(scenario())


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
        "reload",
        "stop",
        "goal",
        "export",
        "image",
        "paste",
        "model",
        "exit",
    }
