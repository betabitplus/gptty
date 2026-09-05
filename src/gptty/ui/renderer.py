from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TextIO

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

from ..output import OutputMessage
from .state import UISettings


@dataclass
class RenderState:
    last_block: str | None = None
    turn_active: bool = False


class PrettyRenderer:
    """Line-oriented Rich renderer that keeps normal terminal scrollback intact."""

    def __init__(self, stdout: TextIO, settings: UISettings) -> None:
        self.stdout = stdout
        self.settings = settings
        self.console = Console(file=stdout, highlight=False, soft_wrap=False)
        self.state = RenderState()

    def header(
        self,
        *,
        profile: str | None = None,
        conversation: str | None = None,
        model: str | None = None,
    ) -> None:
        self.console.print(Rule("ChatGPT", style="dim"))
        details: list[str] = []
        if profile:
            details.append(f"profile: {profile}")
        if conversation:
            details.append(f"conversation: {_short_ref(conversation)}")
        if model:
            details.append(f"model: {model}")
        if details:
            self.console.print(" · ".join(details), style="dim")
            self.console.print()

    def clear_context(self) -> None:
        self.console.clear()
        self.state = RenderState()

    def turn_start(self) -> None:
        if self.state.turn_active:
            self.console.print()
        self.console.print(Rule("working", style="dim"))
        self.console.print()
        self.state = RenderState(last_block="boundary", turn_active=True)

    def live_event(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        if event.get("type") != "canonical_intermediate_message":
            return
        kind = event.get("message_kind")
        text = _clean(event.get("text"))
        label = _clean(event.get("label"))
        tool_name = _clean(event.get("tool_name")) or "tool"

        if kind in {"assistant_progress", "reasoning"}:
            if not self.settings.thinking:
                return
            rendered = text or label
            if rendered:
                self.thinking(rendered)
            return

        if kind == "tool_call" and self.settings.tools != "hidden":
            self.tool(tool_name, label)
            return

        if kind == "activity":
            rendered = text or label
            if rendered:
                self.activity(rendered)

    def thinking(self, text: str) -> None:
        self._gap_before("thinking")
        self.console.print(Text("Thinking", style="dim italic"))
        self.console.print(Text(text, style="dim"))
        self.state.last_block = "thinking"

    def tool(self, tool_name: str, label: str = "") -> None:
        self._gap_before("tool")
        line = Text()
        line.append("◇ ", style="dim")
        line.append(tool_name, style="bold")
        if label:
            line.append(f"  {label}")
        self.console.print(line)
        self.state.last_block = "tool"

    def activity(self, text: str) -> None:
        self._gap_before("activity")
        self.console.print(Text(text, style="dim"))
        self.state.last_block = "activity"

    def answer(self, text: str) -> None:
        self.console.print()
        self.console.print(Rule("answer"))
        self.console.print()
        if self.settings.markdown and text:
            self.console.print(Markdown(text))
        else:
            self.console.print(text)
        self.state.last_block = "answer"
        self.state.turn_active = False

    def info(self, text: str) -> None:
        self._gap_before("info")
        self.console.print(text)
        self.state.last_block = "info"

    def warning(self, text: str) -> None:
        self._gap_before("warning")
        self.console.print(Text(text, style="bold"))
        self.state.last_block = "warning"

    def messages(self, messages: list[OutputMessage]) -> None:
        self._gap_before("messages")
        if not messages:
            self.console.print("(no messages)", style="dim")
            self.state.last_block = "messages"
            return
        for index, message in enumerate(messages):
            if index:
                self.console.print()
            self.console.print(Text(message.role, style="bold"))
            if self.settings.markdown and message.text:
                self.console.print(Markdown(message.text))
            else:
                self.console.print(message.text)
        self.state.last_block = "messages"

    def _gap_before(self, block: str) -> None:
        previous = self.state.last_block
        if previous is None or previous == "boundary":
            return
        if block == "tool" and previous == "tool":
            return
        self.console.print()


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _short_ref(ref: str, *, max_len: int = 44) -> str:
    if len(ref) <= max_len:
        return ref
    return f"…{ref[-(max_len - 1):]}"
