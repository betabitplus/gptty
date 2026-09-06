from __future__ import annotations

from io import StringIO

from gptty.ui.renderer import PrettyRenderer, _format_elapsed
from gptty.ui.state import UISettings


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


def test_renderer_header_shows_full_chat_link() -> None:
    out = StringIO()
    renderer = PrettyRenderer(out, UISettings(markdown=False))

    renderer.header(conversation="conv-123", model="latest frontier · High")

    text = out.getvalue()
    assert "https://chatgpt.com/c/conv-123" in text
    assert "latest frontier · High" in text


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
