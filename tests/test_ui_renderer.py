from __future__ import annotations

from io import StringIO

from gptty.output import OutputMessage
from gptty.ui.renderer import PrettyRenderer, _format_elapsed, _user_message_text
from gptty.ui.state import UISettings


class _SemanticMarkdownSink(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.markdown_blocks: list[str] = []

    def write_markdown(self, text: str) -> None:
        self.markdown_blocks.append(text)


def test_renderer_routes_markdown_to_semantic_transcript_sink() -> None:
    out = _SemanticMarkdownSink()
    renderer = PrettyRenderer(out, UISettings(markdown=True))

    renderer.messages(
        [
            OutputMessage(role="assistant", text="**history**"),
        ]
    )
    renderer.answer("**final**")

    assert out.markdown_blocks == ["**history**", "**final**"]
    assert "**history**" not in out.getvalue()
    assert "**final**" not in out.getvalue()


def test_renderer_separates_thinking_and_groups_tools() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.turn_start()
    renderer.thinking("Inspecting the repository.")
    renderer.tool("api_tool.call_tool", "Reading git status...")
    renderer.tool("api_tool.call_tool", "Reading README.md...")
    renderer.thinking("The issue is isolated.")
    renderer.answer("Final answer")

    text = out.getvalue()
    assert "working" in text
    assert "answer" in text
    assert "Thinking\nInspecting the repository." in text
    assert "Reading git status...\n◇ api_tool.call_tool  Reading README.md..." in text
    assert "Reading README.md...\n\nThinking\nThe issue is isolated." in text
    assert text.rstrip().endswith("Final answer")


def test_renderer_live_tool_calls_use_compact_formatter() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    for tool_name, label in (
        ("api_tool.list_resources", "Using tool..."),
        ("api_tool.call_tool", "Searching LOCAL_QUIT_CODE..."),
        ("api_tool.call_tool", "Calling list workspaces..."),
        ("api_tool.call_tool", "Opening current CodexPro workspace..."),
        ("api_tool.call_tool", "Using tool..."),
    ):
        renderer.live_event(
            {
                "type": "canonical_intermediate_message",
                "message_kind": "tool_call",
                "tool_name": tool_name,
                "label": label,
            }
        )

    text = out.getvalue()
    assert "◇ list_resources" in text
    assert "◇ search  LOCAL_QUIT_CODE" in text
    assert "◇ list_workspaces" in text
    assert "◇ open_current_workspace" in text
    assert "api_tool.call_tool" not in text
    assert "call_tool" not in text
    assert "Using tool" not in text
    assert "Searching LOCAL_QUIT_CODE" not in text


def test_renderer_surfaces_tool_result_labels_and_errors_without_raw_success_payload() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_result",
            "tool_name": "api_tool.call_tool",
            "label": "Workspace inspection complete",
            "text": '{"ok":true,"message":"raw success payload"}',
        }
    )
    renderer.live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_result",
            "tool_name": "api_tool.call_tool",
            "text": '{"ok":false,"error":"Workspace not found"}',
        }
    )

    text = out.getvalue()
    assert "Workspace inspection complete" in text
    assert "raw success payload" not in text
    assert "api_tool.call_tool failed · Workspace not found" in text


def test_user_message_badge_is_high_contrast_and_multiline_aligned() -> None:
    rendered = _user_message_text("hello\nworld")

    assert rendered.plain == " YOU ❯ hello\n       world"
    assert any(span.style == "bold reverse" for span in rendered.spans)


def test_renderer_resume_messages_highlights_user_turns() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.messages(
        [
            OutputMessage(role="user", text="question\ncontinued"),
            OutputMessage(role="assistant", text="answer"),
        ]
    )

    text = out.getvalue()
    assert " YOU ❯ question\n       continued" in text
    assert "assistant\nanswer" in text


def test_renderer_header_shows_full_chat_link() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.header(conversation="conv-123", model="latest frontier · High")

    text = out.getvalue()
    assert "https://chatgpt.com/c/conv-123" in text
    assert "latest frontier · High" in text


def test_renderer_answer_model_shows_observed_model_and_mismatch() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.answer_model(
        "gpt-5-4-thinking",
        requested_model="gpt-5-6-thinking",
        sent_model="gpt-5-6-thinking",
    )

    assert out.getvalue() == "model: gpt-5-4-thinking · requested: gpt-5-6-thinking\n"


def test_renderer_answer_model_is_explicit_when_observation_is_missing() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.answer_model(None, requested_model="gpt-5-6-thinking")

    assert out.getvalue() == "model: unknown · requested: gpt-5-6-thinking\n"


def test_elapsed_format_scales_to_hours() -> None:
    assert _format_elapsed(0) == "00:00"
    assert _format_elapsed(65.9) == "01:05"
    assert _format_elapsed(3661) == "01:01:01"


def test_renderer_clear_context_resets_spacing() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))
    cleared: list[bool] = []
    renderer.console.clear = lambda: cleared.append(True)

    renderer.info("old")
    renderer.clear_context()
    renderer.info("new")

    assert cleared == [True]
    assert out.getvalue() == "old\nnew\n"


def test_renderer_can_hide_thinking_and_tools() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False, thinking=False, tools="hidden"))

    renderer.turn_start()
    renderer.live_event(
        {"type": "canonical_intermediate_message", "message_kind": "assistant_progress", "text": "hidden"}
    )
    renderer.live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "tool_call",
            "tool_name": "tool",
            "label": "hidden",
        }
    )
    renderer.answer("visible")

    text = out.getvalue()
    assert "hidden" not in text
    assert "visible" in text


def test_renderer_keeps_commentary_visible_when_thinking_is_hidden() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False, thinking=False, tools="hidden"))

    renderer.turn_start()
    renderer.live_event(
        {
            "type": "canonical_intermediate_message",
            "message_kind": "commentary",
            "text": "Visible progress update.",
        }
    )

    text = out.getvalue()
    assert "Visible progress update." in text
    assert "Thinking" not in text


def test_renderer_surfaces_stream_health_transitions() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.live_event(
        {
            "type": "stream_handoff_server_quiet",
            "server_idle_seconds": 125.0,
        }
    )
    renderer.live_event(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 305.0,
        }
    )
    renderer.live_event(
        {
            "type": "stream_handoff_delivery_recovered",
            "catchup_count": 7,
        }
    )
    renderer.live_event(
        {
            "type": "stream_handoff_server_resumed",
            "silent_seconds": 306.0,
        }
    )
    renderer.live_event(
        {
            "type": "stream_handoff_server_quiet",
            "server_idle_seconds": 125.0,
            "final_text_seen": True,
        }
    )
    renderer.live_event(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 305.0,
            "final_text_seen": True,
        }
    )
    renderer.live_event(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 605.0,
            "last_tool_error": "corrupt patch at line 13",
        }
    )

    text = out.getvalue()
    assert "Server quiet" in text
    assert "Prolonged server silence" in text
    assert "Delivery recovered · replayed 7 events" in text
    assert "Visibility restored after 05:06" in text
    assert "Answer text received · finality still unconfirmed after 02:05" in text
    assert "Finality unconfirmed · answer text received" in text
    assert "05:05" in text
    assert "do not resend yet" in text
    assert "Prolonged server silence" in text
    assert "corrupt patch at line 13" in text
    assert "turn may still recover" in text


def test_renderer_streams_append_only_answer_without_duplicate_final() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.turn_start(show_elapsed=False)
    renderer.live_event(
        {"type": "assistant_text_snapshot", "sequence": 1, "message_id": "m1", "text": "hello"}
    )
    renderer.live_event(
        {"type": "assistant_text_delta", "sequence": 2, "message_id": "m1", "delta": " world"}
    )
    renderer.answer("hello world")

    text = out.getvalue()
    assert text.count("hello world") == 1
    assert "corrected answer follows" not in text


def test_renderer_stop_readback_does_not_duplicate_matching_stream() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.turn_start(show_elapsed=False)
    renderer.live_event(
        {"type": "assistant_text_snapshot", "sequence": 1, "message_id": "m1", "text": "partial"}
    )
    renderer.turn_stop_pending()
    renderer.info("ChatGPT stopped; finalizing local readback…")
    renderer.answer("partial")

    text = out.getvalue()
    assert text.count("partial") == 1
    assert "answer · final" not in text
    assert "corrected answer follows" not in text


def test_renderer_stop_readback_appends_only_canonical_suffix() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.turn_start(show_elapsed=False)
    renderer.live_event(
        {"type": "assistant_text_snapshot", "sequence": 1, "message_id": "m1", "text": "partial"}
    )
    renderer.turn_stop_pending()
    renderer.info("ChatGPT stopped; finalizing local readback…")
    renderer.answer("partial final")

    text = out.getvalue()
    assert text.count("partial") == 1
    assert "partial final" not in text
    assert " final" in text
    assert "answer · final" not in text
    assert "corrected answer follows" not in text


def test_renderer_revision_falls_back_to_canonical_final() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.turn_start(show_elapsed=False)
    renderer.live_event(
        {"type": "assistant_text_snapshot", "sequence": 1, "message_id": "m1", "text": "draft"}
    )
    renderer.live_event(
        {"type": "assistant_text_revision", "sequence": 2, "message_id": "m1", "text": "revised"}
    )
    renderer.answer("revised final")

    text = out.getvalue()
    assert "Response revised while streaming; canonical final follows." in text
    assert "answer · final" in text
    assert text.rstrip().endswith("revised final")


def test_renderer_never_recommends_destructive_action_from_silence_alone() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.live_event(
        {
            "type": "stream_handoff_server_stalled",
            "server_idle_seconds": 1238.736,
            "last_tool_error": "corrupt patch at line 13",
        }
    )

    rendered = out.getvalue()
    assert "turn may still recover" in rendered
    assert "Ctrl-C" not in rendered
    assert "new turn" not in rendered
    assert "resend" not in rendered.lower()
