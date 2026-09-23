from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TextIO

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

from ..output import OutputMessage, RevisionTextState, _tool_result_error, render_tool_call_parts
from .state import UISettings


@dataclass
class RenderState:
    last_block: str | None = None
    turn_active: bool = False


@dataclass
class _ElapsedStatus:
    started_at: float
    initial_elapsed: float = 0.0
    finished_at: float | None = None
    label: str = "elapsed"
    hint: str | None = "Ctrl-C stop · Ctrl-\\ quit"

    def __rich__(self) -> Text:
        now = self.finished_at if self.finished_at is not None else time.monotonic()
        seconds = self.initial_elapsed + max(0.0, now - self.started_at)
        suffix = f" · {self.hint}" if self.hint else ""
        return Text(f"{self.label} {_format_elapsed(seconds)}{suffix}", style="dim")


class PrettyRenderer:
    """Line-oriented Rich renderer for terminal or transcript-backed output."""

    def __init__(self, stdout: TextIO, settings: UISettings) -> None:
        self.stdout = stdout
        self.settings = settings
        force_terminal = True if getattr(stdout, "supports_rich_ansi", False) else None
        self.console = Console(
            file=stdout,
            highlight=False,
            soft_wrap=False,
            force_terminal=force_terminal,
        )
        self._supports_live = bool(
            getattr(stdout, "supports_live", self.console.is_terminal)
        )
        self.state = RenderState()
        self._elapsed_status: _ElapsedStatus | None = None
        self._elapsed_live: Live | None = None
        self._answer_state = RevisionTextState()
        self._answer_stream_started = False
        self._answer_stream_suppressed = False

    def _rule(self, title: str, *, style: str | None = None) -> None:
        write_rule = getattr(self.stdout, "write_rule", None)
        if callable(write_rule):
            write_rule(title, style=style)
            return
        self.console.print(Rule(title, style=style))

    def _markdown(self, text: str) -> None:
        write_markdown = getattr(self.stdout, "write_markdown", None)
        if callable(write_markdown):
            write_markdown(text)
            return
        self.console.print(Markdown(text))

    def header(
        self,
        *,
        profile: str | None = None,
        conversation: str | None = None,
        model: str | None = None,
        temporary: bool = False,
    ) -> None:
        self._rule("ChatGPT", style="dim")
        details: list[str] = []
        if profile:
            details.append(f"profile: {profile}")
        if model:
            details.append(f"model: {model}")
        if temporary:
            details.append("temporary chat")
        if details:
            self.console.print(" · ".join(details), style="dim")
        if conversation:
            self.chat_link(conversation)
        if details or conversation:
            self.console.print()

    def chat_link(self, conversation: str) -> None:
        url = _conversation_url(conversation)
        line = Text("chat: ", style="dim")
        line.append(url, style=f"underline link {url}")
        self.console.print(line)

    def answer_model(
        self,
        observed_model: str | None,
        *,
        requested_model: str | None = None,
        sent_model: str | None = None,
    ) -> None:
        observed = str(observed_model or "").strip()
        requested = str(requested_model or "").strip()
        sent = str(sent_model or "").strip()

        line = Text("model: ", style="dim")
        line.append(observed or "unknown")
        if requested and requested != observed:
            line.append(" · requested: ", style="dim")
            line.append(requested)
        if sent and sent not in {observed, requested}:
            line.append(" · sent: ", style="dim")
            line.append(sent)
        self.console.print(line)

    def turn_marker(self, label: str, status: str, message: str) -> None:
        prefix = str(label or "turn").strip() or "turn"
        state = str(status or "abnormal").strip() or "abnormal"
        detail = str(message or "").strip()
        caution_states = {
            "filtered",
            "incomplete",
            "truncated",
            "unconfirmed",
            "unresolved",
        }
        accent = "yellow" if state.lower() in caution_states else "bright_red"

        header = Text()
        header.append("▌", style=f"bold {accent}")
        header.append(" !  ", style=f"bold {accent}")
        header.append(prefix.upper(), style="dim")
        header.append(" · ", style="dim")
        header.append(state.upper(), style=f"bold {accent}")

        body = Text()
        body.append("▌", style=f"bold {accent}")
        body.append("    ")
        body.append(detail or "ChatGPT ended this turn in a non-standard state.")

        self.console.print()
        self.console.print(header, soft_wrap=True)
        self.console.print(body, soft_wrap=True)
        self.console.print()

    def clear_context(self) -> None:
        self.turn_abort()
        clear = getattr(self.stdout, "clear", None)
        if callable(clear):
            clear()
        else:
            self.console.clear()
        self.state = RenderState()

    def turn_start(self, *, show_elapsed: bool = True) -> None:
        if self.state.turn_active:
            self.console.print()
        self._answer_state = RevisionTextState()
        self._answer_stream_started = False
        self._answer_stream_suppressed = False
        self._rule("working", style="dim")
        self.console.print()
        self.state = RenderState(last_block="boundary", turn_active=True)
        if show_elapsed:
            self.start_elapsed()

    def start_elapsed(self, *, initial_elapsed: float = 0.0) -> None:
        self._stop_elapsed(label="stopped")
        self._elapsed_status = _ElapsedStatus(
            started_at=time.monotonic(),
            initial_elapsed=max(0.0, float(initial_elapsed)),
        )
        self.state.turn_active = True
        if self._supports_live:
            self._elapsed_live = Live(
                self._elapsed_status,
                console=self.console,
                refresh_per_second=1,
                transient=False,
            )
            self._elapsed_live.start(refresh=True)

    def finish_elapsed(self) -> None:
        self._stop_elapsed(label="elapsed")
        self.state.turn_active = False

    def turn_abort(self) -> None:
        if self.state.turn_active or self._elapsed_status is not None or self._answer_stream_started:
            self._stop_elapsed(label="stopped")
            if self._answer_stream_started and not self._answer_stream_suppressed:
                self.console.print()
            self._answer_stream_started = False
            self._answer_stream_suppressed = False
            self.state.turn_active = False

    def turn_stop_pending(self) -> None:
        """Stop elapsed UI while preserving streamed answer state for canonical readback."""
        self._stop_elapsed(label="stopped")
        self.state.turn_active = False

    def _stop_elapsed(self, *, label: str) -> None:
        status = self._elapsed_status
        if status is None:
            return
        status.label = label
        status.hint = None
        status.finished_at = time.monotonic()
        if self._elapsed_live is not None:
            self._elapsed_live.update(status, refresh=True)
            self._elapsed_live.stop()
        self._elapsed_live = None
        self._elapsed_status = None

    def live_event(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        if event_type == "stream_handoff_delivery_recovered":
            count = event.get("catchup_count")
            recovered = (
                f" · replayed {count} events"
                if isinstance(count, int) and not isinstance(count, bool)
                else ""
            )
            self.info(f"Delivery recovered{recovered}")
            return
        if event_type == "stream_handoff_server_quiet":
            idle = event.get("server_idle_seconds")
            seconds = float(idle) if isinstance(idle, (int, float)) else 0.0
            if event.get("final_text_seen") is True:
                self.warning(
                    "Answer text received"
                    f" · finality still unconfirmed after {_format_elapsed(seconds)}"
                    " · waiting read-only"
                )
            else:
                self.warning(
                    "Server quiet"
                    f" · no observable ChatGPT events for {_format_elapsed(seconds)}"
                    " · this does not prove the turn stopped"
                )
            return
        if event_type == "stream_handoff_server_stalled":
            idle = event.get("server_idle_seconds")
            seconds = float(idle) if isinstance(idle, (int, float)) else 0.0
            tool_error = event.get("last_tool_error")
            if event.get("final_text_seen") is True:
                self.warning(
                    "Finality unconfirmed"
                    " · answer text received"
                    f" · no observable server events for {_format_elapsed(seconds)}"
                    " · do not resend yet"
                )
            elif isinstance(tool_error, str) and tool_error.strip():
                self.warning(
                    "Prolonged server silence"
                    f" · no observable events for {_format_elapsed(seconds)}"
                    f" · last visible tool failed: {tool_error.strip()}"
                    " · turn may still recover"
                )
            else:
                self.warning(
                    "Prolonged server silence"
                    f" · no observable events for {_format_elapsed(seconds)}"
                    " · turn may still be working invisibly"
                    " · do not resend yet"
                )
            return
        if event_type == "stream_handoff_server_resumed":
            silent = event.get("silent_seconds")
            seconds = float(silent) if isinstance(silent, (int, float)) else 0.0
            self.info(f"Visibility restored after {_format_elapsed(seconds)}")
            return
        previous_text = self._answer_state.text
        previous_message_id = self._answer_state.message_id
        if self._answer_state.apply(event):
            self._render_stream_answer_event(
                event,
                previous_text=previous_text,
                previous_message_id=previous_message_id,
            )
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

        if kind == "commentary":
            rendered = text or label
            if rendered:
                self.commentary(rendered)
            return

        if kind == "tool_call" and self.settings.tools != "hidden":
            name, detail = render_tool_call_parts(tool=tool_name, text=text, label=label)
            if name:
                self.tool(name, detail)
            return

        if kind == "tool_result":
            error = _tool_result_error(text)
            if error:
                self.warning(f"{tool_name or 'tool'} failed · {error}")
            elif label and self.settings.tools != "hidden":
                # ChatGPT Web exposes concise successful tool/activity status text.
                # Preserve that user-visible information without dumping raw results.
                self.activity(label)
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

    def commentary(self, text: str) -> None:
        self._gap_before("commentary")
        self.console.print(Text(text))
        self.state.last_block = "commentary"

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

    def _start_stream_answer(self) -> None:
        if self._answer_stream_started:
            return
        self.finish_elapsed()
        self.console.print()
        self._rule("answer")
        self.console.print()
        self._answer_stream_started = True
        self.state.last_block = "answer"
        self.state.turn_active = True

    def _suppress_revised_stream(self) -> None:
        if self._answer_stream_suppressed:
            return
        if self._answer_stream_started:
            self.console.print()
        self.console.print(Text("Response revised while streaming; canonical final follows.", style="dim"))
        self._answer_stream_suppressed = True

    def _write_answer_fragment(self, text: str) -> None:
        writer = getattr(self.stdout, "write_stream_fragment", None)
        if callable(writer):
            writer(text)
            return
        self.console.print(Text(text), end="")

    def _render_stream_answer_event(
        self,
        event: dict[str, Any],
        *,
        previous_text: str,
        previous_message_id: str | None,
    ) -> None:
        event_type = event.get("type")
        message_id = self._answer_state.message_id
        if self._answer_stream_suppressed:
            return
        if event_type == "assistant_text_revision":
            self._suppress_revised_stream()
            return

        self._start_stream_answer()
        if event_type == "assistant_text_delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                self._write_answer_fragment(delta)
            return

        text = self._answer_state.text
        if not previous_text or previous_message_id is None:
            if text:
                self._write_answer_fragment(text)
            return
        if message_id == previous_message_id and text.startswith(previous_text):
            suffix = text[len(previous_text) :]
            if suffix:
                self._write_answer_fragment(suffix)
            return
        self._suppress_revised_stream()

    def answer(self, text: str) -> None:
        self.finish_elapsed()
        if self._answer_stream_started and not self._answer_stream_suppressed:
            streamed_text = self._answer_state.text
            if streamed_text == text:
                self.console.print()
                self._answer_stream_started = False
                self.state.last_block = "answer"
                self.state.turn_active = False
                return
            if streamed_text and text.startswith(streamed_text):
                suffix = text[len(streamed_text) :]
                if suffix:
                    self._write_answer_fragment(suffix)
                self.console.print()
                self._answer_stream_started = False
                self.state.last_block = "answer"
                self.state.turn_active = False
                return
        if self._answer_stream_started:
            if not self._answer_stream_suppressed:
                self.console.print()
                self.console.print(Text("Stream differed from canonical final; corrected answer follows.", style="dim"))
            self._answer_stream_started = False
            self._answer_stream_suppressed = False
            self.console.print()
            self._rule("answer · final")
            self.console.print()
            if self.settings.markdown and text:
                self._markdown(text)
            else:
                self.console.print(text)
            self.state.last_block = "answer"
            self.state.turn_active = False
            return
        self.console.print()
        self._rule("answer")
        self.console.print()
        if self.settings.markdown and text:
            self._markdown(text)
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
            if message.role == "user":
                self.console.print(_user_message_text(message.text))
                continue
            self.console.print(Text(message.role, style="bold"))
            if self.settings.markdown and message.text:
                self._markdown(message.text)
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


def _user_message_text(value: str) -> Text:
    lines = value.rstrip().splitlines() or [""]
    result = Text()
    result.append(" YOU ", style="bold reverse")
    result.append("❯ ")
    result.append(lines[0])
    for line in lines[1:]:
        result.append("\n       ")
        result.append(line)
    return result


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _conversation_url(ref: str) -> str:
    value = ref.strip()
    if value.startswith(("https://", "http://")):
        return value
    return f"https://chatgpt.com/c/{value}"


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
