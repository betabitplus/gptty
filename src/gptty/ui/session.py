from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle, choice

from .state import UISettings, UIStateError, load_ui_settings, ui_settings_path


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("new", "Start a new ChatGPT conversation"),
    CommandSpec("resume", "Resume a real ChatGPT conversation"),
    CommandSpec("detach", "Detach locally from the current conversation"),
    CommandSpec("model", "Choose a real ChatGPT model"),
    CommandSpec("exit", "Exit gptty chat"),
)


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
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        editing_mode = EditingMode.VI if self.settings.editor == "vi" else EditingMode.EMACS
        kwargs: dict[str, Any] = {
            "history": FileHistory(str(self.history_file)),
            "auto_suggest": AutoSuggestFromHistory(),
            "completer": completer,
            "complete_while_typing": True,
            "multiline": False,
            "key_bindings": bindings,
            "editing_mode": editing_mode,
            "bottom_toolbar": " / actions · Ctrl-R history · Alt-Enter newline",
        }
        if self._prompt_input is not None:
            kwargs["input"] = self._prompt_input
        if self._prompt_output is not None:
            kwargs["output"] = self._prompt_output
        self._session = PromptSession(**kwargs)

    def read_prompt(self) -> str:
        return self._session.prompt("❯ ")

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
