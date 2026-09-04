from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import choice, prompt

from .state import (
    UISettings,
    UIStateError,
    load_ui_settings,
    save_ui_settings,
    ui_settings_path,
)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("new", "Start a new ChatGPT conversation"),
    CommandSpec("switch", "Switch to a recent local conversation"),
    CommandSpec("attach", "Attach by ChatGPT URL or conversation ID"),
    CommandSpec("model", "Change the effort/model profile"),
    CommandSpec("messages", "Show recent messages"),
    CommandSpec("status", "Show conversation status"),
    CommandSpec("export", "Export the current conversation"),
    CommandSpec("profile", "Choose the active gptty profile"),
    CommandSpec("auth", "Inspect or refresh web-session auth"),
    CommandSpec("settings", "Change interactive UI settings"),
    CommandSpec("help", "Show all interactive commands"),
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

    def ask(self, message: str, *, default: str = "") -> str | None:
        try:
            value = prompt(f"{message}: ", default=default)
        except (KeyboardInterrupt, EOFError):
            return None
        return value.strip()

    def edit_settings(self) -> bool:
        changed = False
        while True:
            action = self.choose(
                "Settings",
                [
                    ("pretty", f"Pretty UI              {self.settings.pretty}"),
                    ("markdown", f"Markdown rendering     {'on' if self.settings.markdown else 'off'}"),
                    ("thinking", f"Thinking blocks        {'show' if self.settings.thinking else 'hide'}"),
                    ("tools", f"Tool activity          {self.settings.tools}"),
                    ("editor", f"Input mode             {self.settings.editor}"),
                    ("done", "Done"),
                ],
                default="done",
            )
            if action in {None, "done"}:
                break
            if action == "pretty":
                selected = self.choose("Pretty UI", [(v, v) for v in ("auto", "on", "off")], default=self.settings.pretty)
                if selected:
                    self.settings.pretty = str(selected)
                    changed = True
            elif action == "markdown":
                selected = self.choose("Markdown rendering", [(True, "On"), (False, "Off")], default=self.settings.markdown)
                if selected is not None:
                    self.settings.markdown = bool(selected)
                    changed = True
            elif action == "thinking":
                selected = self.choose("Thinking blocks", [(True, "Show"), (False, "Hide")], default=self.settings.thinking)
                if selected is not None:
                    self.settings.thinking = bool(selected)
                    changed = True
            elif action == "tools":
                selected = self.choose("Tool activity", [("compact", "Compact"), ("hidden", "Hidden")], default=self.settings.tools)
                if selected:
                    self.settings.tools = str(selected)
                    changed = True
            elif action == "editor":
                selected = self.choose("Input mode", [("emacs", "Emacs"), ("vi", "Vi")], default=self.settings.editor)
                if selected:
                    self.settings.editor = str(selected)
                    changed = True
                    self._build_session()
        if changed:
            save_ui_settings(self.settings_file, self.settings)
        return changed

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


def command_help() -> list[tuple[str, str]]:
    return [(spec.name, spec.description) for spec in COMMANDS]


def _isatty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except OSError:
        return False
