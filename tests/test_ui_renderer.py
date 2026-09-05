from __future__ import annotations

from io import StringIO

from gptty.ui.renderer import PrettyRenderer
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
