from __future__ import annotations

from typing import Any

import pytest

import gptty.commands.ask as ask_command
import gptty.commands.send as send_command
from gptty import cli
from gptty.media import collect_media_inputs


def test_ask_routes_image_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_read_stdin_text(mode: str) -> None:
        calls["mode"] = mode
        return None

    def fake_run_ask(args: Any, *, stdin_text: str | None = None) -> int:
        calls["prompt"] = args.prompt
        calls["image"] = args.image
        calls["stdin_text"] = stdin_text
        return 0

    monkeypatch.setattr(cli, "read_stdin_text", fake_read_stdin_text)
    monkeypatch.setattr(ask_command, "run_ask", fake_run_ask)

    assert cli.main([
        "ask",
        "--image",
        "before.png",
        "--image",
        "https://example.com/after.webp",
        "compare",
    ]) == 0
    assert calls == {
        "mode": "auto",
        "prompt": ["compare"],
        "image": ["before.png", "https://example.com/after.webp"],
        "stdin_text": None,
    }


def test_send_routes_image_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_read_stdin_text(mode: str) -> str:
        calls["mode"] = mode
        return "stdin context"

    def fake_run_send(args: Any, *, stdin_text: str | None = None) -> int:
        calls["prompt"] = args.prompt
        calls["to"] = args.to
        calls["image"] = args.image
        calls["stdin_text"] = stdin_text
        return 0

    monkeypatch.setattr(cli, "read_stdin_text", fake_read_stdin_text)
    monkeypatch.setattr(send_command, "run_send", fake_run_send)

    assert cli.main([
        "send",
        "--to",
        "abc",
        "--image",
        "diagram.png",
        "review",
    ]) == 0
    assert calls == {
        "mode": "auto",
        "prompt": ["review"],
        "to": "abc",
        "image": ["diagram.png"],
        "stdin_text": "stdin context",
    }


def test_ask_preserves_mixed_image_file_flag_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    monkeypatch.setattr(cli, "read_stdin_text", lambda _mode: None)

    def fake_run_ask(args: Any, *, stdin_text: str | None = None) -> int:
        calls["image"] = args.image
        calls["file"] = args.file
        calls["ordered"] = args._media_inputs
        calls["media"] = collect_media_inputs(args)
        return 0

    monkeypatch.setattr(ask_command, "run_ask", fake_run_ask)

    assert cli.main([
        "ask",
        "--file",
        "https://example.com/notes.pdf",
        "--image",
        "https://example.com/plot.png",
        "--file",
        "https://example.com/data.csv",
        "review",
    ]) == 0
    assert calls == {
        "image": ["https://example.com/plot.png"],
        "file": [
            "https://example.com/notes.pdf",
            "https://example.com/data.csv",
        ],
        "ordered": [
            ("file", "https://example.com/notes.pdf"),
            ("image", "https://example.com/plot.png"),
            ("file", "https://example.com/data.csv"),
        ],
        "media": [
            "https://example.com/notes.pdf",
            "https://example.com/plot.png",
            "https://example.com/data.csv",
        ],
    }
