from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="PTY smoke requires a Unix pseudo-terminal")


def _read_until(fd: int, needle: bytes, *, timeout: float = 5.0) -> bytes:
    deadline = time.monotonic() + timeout
    data = bytearray()
    while needle not in data and time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def test_real_pty_help_and_exit(tmp_path) -> None:
    master, slave = pty.openpty()
    env = os.environ.copy()
    env.pop("NO_COLOR", None)
    env["TERM"] = "xterm-256color"
    env["GPTTY_CONFIG_HOME"] = str(tmp_path / "config")
    env["GPTTY_DATA_HOME"] = str(tmp_path / "data")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gptty",
            "chat",
            "--state",
            str(tmp_path / "state.json"),
            "--auth",
            str(tmp_path / "auth.json"),
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    try:
        startup = _read_until(master, "❯ ".encode(), timeout=5.0)
        assert b"ChatGPT" in startup

        os.write(master, b"/help\r")
        help_output = _read_until(master, b"/settings", timeout=5.0)
        assert b"/switch" in help_output
        assert b"/export" in help_output

        os.write(master, b"/\r")
        menu = _read_until(master, b"Actions", timeout=5.0)
        assert b"Actions" in menu
        os.write(master, b"\x1b")
        cancelled = _read_until(master, "❯".encode(), timeout=5.0)
        assert "❯".encode() in cancelled

        os.write(master, b"/\r")
        menu = _read_until(master, b"Actions", timeout=5.0)
        assert b"Actions" in menu
        os.write(master, b"\x1b[B\r")
        switched = _read_until(master, b"No local recent conversations yet.", timeout=5.0)
        assert b"No local recent conversations yet." in switched

        os.write(master, b"/\r")
        menu = _read_until(master, b"Actions", timeout=5.0)
        assert b"Actions" in menu
        os.write(master, b"\r")
        selected = _read_until(master, b"Started a new conversation.", timeout=5.0)
        assert b"Started a new conversation." in selected

        os.write(master, b"/exit\r")
        assert process.wait(timeout=5.0) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5.0)
        os.close(master)
