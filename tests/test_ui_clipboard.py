from __future__ import annotations

from types import SimpleNamespace

import pytest

from gptty.ui import clipboard


def test_capture_clipboard_image_uses_native_macos_png_coercion(tmp_path, monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        output = kwargs["env"]["GPTTY_CLIPBOARD_IMAGE_PATH"]
        with open(output, "wb") as stream:
            stream.write(b"png")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)

    path = clipboard.capture_clipboard_image(tmp_path)

    assert path.read_bytes() == b"png"
    assert calls[0][0][:2] == ["osascript", "-e"]
    assert "clipboard as «class PNGf»" in calls[0][0][2]


def test_capture_clipboard_image_reports_non_image_clipboard(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "darwin")
    monkeypatch.setattr(
        clipboard.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="execution error: Clipboard does not contain an image.",
        ),
    )

    with pytest.raises(clipboard.ClipboardImageError, match="clipboard does not contain an image"):
        clipboard.capture_clipboard_image(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_capture_clipboard_image_is_macos_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(clipboard.sys, "platform", "linux")

    with pytest.raises(clipboard.ClipboardImageError, match="macOS only"):
        clipboard.capture_clipboard_image(tmp_path)
