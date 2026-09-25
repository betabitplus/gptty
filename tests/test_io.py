from __future__ import annotations

import pytest

from gptty.io import StdinReadError, read_stdin_text


class FakeStdin:
    def __init__(
        self,
        text: str = "",
        *,
        is_tty: bool = False,
        isatty_error: OSError | None = None,
        read_error: OSError | None = None,
    ) -> None:
        self.text = text
        self.is_tty = is_tty
        self.isatty_error = isatty_error
        self.read_error = read_error
        self.reads = 0
        self.offset = 0

    def isatty(self) -> bool:
        if self.isatty_error is not None:
            raise self.isatty_error
        return self.is_tty

    def read(self, size: int = -1) -> str:
        self.reads += 1
        if self.read_error is not None:
            raise self.read_error
        if self.offset >= len(self.text):
            return ""
        if size < 0:
            chunk = self.text[self.offset :]
            self.offset = len(self.text)
            return chunk
        end = min(len(self.text), self.offset + size)
        chunk = self.text[self.offset : end]
        self.offset = end
        return chunk


def test_read_stdin_auto_reads_when_not_tty() -> None:
    stdin = FakeStdin("piped text", is_tty=False)

    assert read_stdin_text("auto", stdin=stdin) == "piped text"
    assert stdin.reads == 2


def test_read_stdin_auto_ignores_tty() -> None:
    stdin = FakeStdin("interactive text", is_tty=True)

    assert read_stdin_text("auto", stdin=stdin) is None
    assert stdin.reads == 0


def test_read_stdin_always_reads_even_tty() -> None:
    stdin = FakeStdin("forced text", is_tty=True)

    assert read_stdin_text("always", stdin=stdin) == "forced text"
    assert stdin.reads == 2


def test_read_stdin_never_ignores_pipe() -> None:
    stdin = FakeStdin("piped text", is_tty=False)

    assert read_stdin_text("never", stdin=stdin) is None
    assert stdin.reads == 0


def test_read_stdin_empty_text_is_returned() -> None:
    stdin = FakeStdin("", is_tty=False)

    assert read_stdin_text("auto", stdin=stdin) == ""
    assert stdin.reads == 1


def test_read_stdin_wraps_isatty_errors() -> None:
    stdin = FakeStdin(isatty_error=OSError("bad fd"))

    with pytest.raises(StdinReadError, match="failed to inspect stdin"):
        read_stdin_text("auto", stdin=stdin)


def test_read_stdin_wraps_read_errors() -> None:
    stdin = FakeStdin(read_error=OSError("broken pipe"))

    with pytest.raises(StdinReadError, match="failed to read stdin"):
        read_stdin_text("auto", stdin=stdin)



def test_read_stdin_rejects_input_over_byte_limit() -> None:
    stdin = FakeStdin("abcdef", is_tty=False)

    with pytest.raises(StdinReadError, match="exceeds the 5-byte safety limit") as error:
        read_stdin_text("auto", stdin=stdin, max_bytes=5)

    assert error.value.error_class == "stdin_too_large"
    assert error.value.exit_code == 2


def test_read_stdin_counts_utf8_bytes_not_characters() -> None:
    stdin = FakeStdin("ééé", is_tty=False)

    with pytest.raises(StdinReadError, match="exceeds the 5-byte safety limit"):
        read_stdin_text("auto", stdin=stdin, max_bytes=5)


def test_read_stdin_accepts_text_exactly_at_byte_limit() -> None:
    stdin = FakeStdin("éé", is_tty=False)

    assert read_stdin_text("auto", stdin=stdin, max_bytes=4) == "éé"


def test_read_stdin_rejects_nul_as_binary_looking_input() -> None:
    stdin = FakeStdin("text\x00binary", is_tty=False)

    with pytest.raises(StdinReadError, match="looks binary") as error:
        read_stdin_text("auto", stdin=stdin)

    assert error.value.error_class == "stdin_binary"
    assert error.value.exit_code == 2


def test_read_stdin_rejects_non_positive_max_bytes() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        read_stdin_text("always", stdin=FakeStdin("x"), max_bytes=0)


def test_read_stdin_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported stdin mode"):
        read_stdin_text("sometimes")  # type: ignore[arg-type]
