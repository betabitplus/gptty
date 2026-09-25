from __future__ import annotations

import sys
from typing import Literal, TextIO

StdinMode = Literal["auto", "always", "never"]

DEFAULT_STDIN_MAX_BYTES = 4 * 1024 * 1024
_STDIN_READ_CHARS = 64 * 1024


class StdinReadError(RuntimeError):
    """Raised when stdin should be read but cannot be read safely."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str = "stdin_read",
        exit_code: int = 1,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.exit_code = exit_code


def read_stdin_text(
    mode: StdinMode = "auto",
    stdin: TextIO = sys.stdin,
    *,
    max_bytes: int = DEFAULT_STDIN_MAX_BYTES,
) -> str | None:
    """Read text stdin with a bounded, shell-friendly policy.

    `auto` reads only from a pipe or redirected stdin.
    `always` forces a read even if stdin reports itself as a TTY.
    `never` ignores stdin and returns None.
    """

    if mode == "never":
        return None
    if mode not in {"auto", "always"}:
        raise ValueError(f"Unsupported stdin mode: {mode}")
    if isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    try:
        is_tty = stdin.isatty()
    except OSError as exc:
        raise StdinReadError(f"failed to inspect stdin: {exc}") from exc

    if mode == "auto" and is_tty:
        return None

    chunks: list[str] = []
    total_bytes = 0
    try:
        while True:
            # Character count is also bounded so a multibyte chunk can exceed
            # the byte limit only by one bounded read, never by an unbounded read().
            remaining = max_bytes - total_bytes
            read_chars = min(_STDIN_READ_CHARS, max(1, remaining + 1))
            chunk = stdin.read(read_chars)
            if not chunk:
                break
            if "\x00" in chunk:
                raise StdinReadError(
                    "stdin looks binary because it contains a NUL byte; "
                    "use text stdin or an attachment-supported workflow",
                    error_class="stdin_binary",
                    exit_code=2,
                )
            total_bytes += len(chunk.encode("utf-8"))
            if total_bytes > max_bytes:
                raise StdinReadError(
                    f"stdin exceeds the {max_bytes}-byte safety limit; "
                    "pass smaller stdin or use an attachment-supported workflow",
                    error_class="stdin_too_large",
                    exit_code=2,
                )
            chunks.append(chunk)
    except StdinReadError:
        raise
    except OSError as exc:
        raise StdinReadError(f"failed to read stdin: {exc}") from exc
    return "".join(chunks)
