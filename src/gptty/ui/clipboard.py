from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


class ClipboardImageError(RuntimeError):
    """Raised when an image cannot be captured from the system clipboard."""


_MACOS_CLIPBOARD_PNG_SCRIPT = r'''
set outputPath to system attribute "GPTTY_CLIPBOARD_IMAGE_PATH"
try
    set imageData to the clipboard as «class PNGf»
on error
    error "Clipboard does not contain an image."
end try
set outputFile to missing value
try
    set outputFile to open for access POSIX file outputPath with write permission
    set eof outputFile to 0
    write imageData to outputFile
    close access outputFile
on error errorMessage number errorNumber
    try
        if outputFile is not missing value then close access outputFile
    end try
    error errorMessage number errorNumber
end try
'''.strip()


def capture_clipboard_image(directory: str | Path) -> Path:
    """Materialize the current macOS clipboard image as a temporary PNG."""
    if sys.platform != "darwin":
        raise ClipboardImageError("clipboard image paste is currently supported on macOS only")

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix="clipboard-", suffix=".png", dir=target_dir)
    os.close(fd)
    path = Path(raw_path)

    env = os.environ.copy()
    env["GPTTY_CLIPBOARD_IMAGE_PATH"] = str(path)
    try:
        result = subprocess.run(
            ["osascript", "-e", _MACOS_CLIPBOARD_PNG_SCRIPT],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        path.unlink(missing_ok=True)
        raise ClipboardImageError(f"could not read image from clipboard: {exc}") from exc

    if result.returncode != 0 or not path.exists() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        detail = (result.stderr or "").strip()
        if "Clipboard does not contain an image" in detail:
            raise ClipboardImageError("clipboard does not contain an image")
        raise ClipboardImageError("could not read image from clipboard")
    return path
