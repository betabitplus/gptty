from __future__ import annotations

from io import StringIO

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
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
