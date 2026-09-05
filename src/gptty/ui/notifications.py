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


def notify_response_complete(*, conversation: str | None = None, prompt: str | None = None) -> bool:
    """Show a best-effort native notification after a complete interactive reply."""
    if sys.platform != "darwin":
        return False

    title = "gptty"
    if conversation:
        ref = conversation.strip()
        if ref:
            title = f"gptty · chat …{ref[-8:]}"
    body = _notification_prompt(prompt)

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


def _notification_prompt(prompt: str | None, *, max_chars: int = 180) -> str:
    if not prompt:
        return "ChatGPT response is ready."
    text = " ".join(str(prompt).split())
    if not text:
        return "ChatGPT response is ready."
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"
