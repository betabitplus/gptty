from __future__ import annotations

import subprocess
import sys

_NOTIFICATION_SCRIPT = r'''
on run argv
    set notificationBody to item 1 of argv
    set notificationTitle to item 2 of argv
    display notification notificationBody with title notificationTitle sound name "Glass"
end run
'''.strip()


def notify_response_complete(*, chat_title: str | None = None, final_response: str | None = None) -> bool:
    """Show a best-effort native notification after a complete interactive reply."""
    if sys.platform != "darwin":
        return False

    title = _notification_title(chat_title)
    body = _notification_response(final_response)

    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                _NOTIFICATION_SCRIPT,
                body,
                title,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _notification_title(chat_title: str | None, *, max_chars: int = 64) -> str:
    text = " ".join(str(chat_title or "").split()) or "ChatGPT"
    return _bounded_preview(text, max_chars=max_chars)


def _notification_response(final_response: str | None, *, max_chars: int = 240) -> str:
    if not final_response:
        return "Response complete."
    text = " ".join(str(final_response).split())
    if not text:
        return "Response complete."
    return _bounded_preview(text, max_chars=max_chars)


def _bounded_preview(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rstrip()
    word_boundary = clipped.rfind(" ")
    if word_boundary >= max_chars // 2:
        clipped = clipped[:word_boundary].rstrip()
    return f"{clipped}…"
