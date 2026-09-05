from __future__ import annotations

from io import StringIO

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from gptty.ui.session import COMMANDS, InteractiveSession, should_use_enhanced_ui
from gptty.ui.state import UISettings, save_ui_settings, ui_settings_path


class TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


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


def test_command_registry_exposes_session_actions() -> None:
    names = {spec.name for spec in COMMANDS}
    assert names == {"new", "resume", "detach", "image", "paste", "model", "exit"}
