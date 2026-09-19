from __future__ import annotations

from io import StringIO
import os
import signal

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.base import Size

from gptty.ui.session import (
    COMMANDS,
    InteractiveSession,
    _text_width,
    _wrap_replay_lines,
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


def test_wrap_replay_lines_uses_terminal_display_width() -> None:
    assert _wrap_replay_lines(["abcdefgh", "界界界", ""], 4) == [
        "abcd",
        "efgh",
        "界界",
        "界",
        "",
    ]


def test_wrap_replay_lines_redraws_rich_rule_for_new_width() -> None:
    line = "──────────── working ────────────"

    wrapped = _wrap_replay_lines([line], 20)

    assert wrapped == ["───── working ──────"]
    assert _text_width(wrapped[0]) == 20


def test_posix_session_uses_sigwinch_without_redundant_size_polling(tmp_path) -> None:
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
    )

    if os.name != "nt" and hasattr(signal, "SIGWINCH"):
        assert session.application.terminal_size_polling_interval is None


def test_resize_replays_recent_transcript_before_redraw(tmp_path, monkeypatch) -> None:
    output = ResizableDummyOutput(80)
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=output,
    )
    requested_limits: list[int] = []
    session.set_resize_replay(
        lambda max_lines: requested_limits.append(max_lines) or ["one", "two"]
    )
    app = session.application
    events: list[object] = []

    monkeypatch.setattr(
        app.renderer,
        "erase",
        lambda **kwargs: events.append(("erase", kwargs)),
    )
    monkeypatch.setattr(app.output, "erase_screen", lambda: events.append("erase_screen"))
    monkeypatch.setattr(
        app.output,
        "cursor_goto",
        lambda row, column: events.append(("cursor_goto", row, column)),
    )
    monkeypatch.setattr(app.output, "write", lambda text: events.append(("write", text)))
    monkeypatch.setattr(app.output, "flush", lambda: events.append("flush"))
    monkeypatch.setattr(
        app.renderer,
        "reset",
        lambda **kwargs: events.append(("reset", kwargs)),
    )
    monkeypatch.setattr(
        app.renderer,
        "report_absolute_cursor_row",
        lambda row: events.append(("cursor_row", row)),
    )
    monkeypatch.setattr(app, "_redraw", lambda: events.append("redraw"))

    session._on_resize()

    assert requested_limits == [21]
    assert events == [
        ("erase", {"leave_alternate_screen": False}),
        "erase_screen",
        ("cursor_goto", 0, 0),
        "flush",
        ("reset", {"leave_alternate_screen": False}),
        ("write", "one\ntwo"),
        ("write", "\n"),
        "flush",
        ("cursor_row", 3),
        "redraw",
    ]


def test_resize_during_run_in_terminal_defers_and_coalesces_replay(
    tmp_path, monkeypatch
) -> None:
    output = ResizableDummyOutput(80)
    session = InteractiveSession(
        history_file=tmp_path / "history",
        settings_file=tmp_path / "ui.json",
        prompt_output=output,
    )
    session.set_resize_replay(lambda _max_lines: ["one"])
    app = session.application
    events: list[object] = []
    callbacks: list[object] = []

    class Future:
        def add_done_callback(self, callback) -> None:
            callbacks.append(callback)

    app._running_in_terminal = True
    app._running_in_terminal_f = Future()
    monkeypatch.setattr(
        app.renderer,
        "erase",
        lambda **kwargs: events.append(("erase", kwargs)),
    )
    monkeypatch.setattr(app.output, "erase_screen", lambda: events.append("erase_screen"))
    monkeypatch.setattr(
        app.output,
        "cursor_goto",
        lambda row, column: events.append(("cursor_goto", row, column)),
    )
    monkeypatch.setattr(app.output, "write", lambda text: events.append(("write", text)))
    monkeypatch.setattr(app.output, "flush", lambda: events.append("flush"))
    monkeypatch.setattr(
        app.renderer,
        "reset",
        lambda **kwargs: events.append(("reset", kwargs)),
    )
    monkeypatch.setattr(
        app.renderer,
        "report_absolute_cursor_row",
        lambda row: events.append(("cursor_row", row)),
    )
    monkeypatch.setattr(app, "_redraw", lambda: events.append("redraw"))

    session._on_resize()
    session._on_resize()

    assert events == []
    assert len(callbacks) == 1
    assert session._resize_replay_pending is True

    app._running_in_terminal = False
    callbacks[0](None)

    assert session._resize_replay_pending is False
    assert events == [
        ("erase", {"leave_alternate_screen": False}),
        "erase_screen",
        ("cursor_goto", 0, 0),
        "flush",
        ("reset", {"leave_alternate_screen": False}),
        ("write", "one"),
        ("write", "\n"),
        "flush",
        ("cursor_row", 2),
        "redraw",
    ]


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
