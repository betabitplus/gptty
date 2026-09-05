from __future__ import annotations

import subprocess
import sys


def notify_response_complete() -> bool:
    """Show a best-effort native notification after a complete interactive reply."""
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'display notification "ChatGPT response is ready." with title "gptty"',
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
