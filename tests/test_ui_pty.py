from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="PTY smoke requires a Unix pseudo-terminal")


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _read_for(fd: int, *, duration: float = 0.2) -> bytes:
    deadline = time.monotonic() + duration
    data = bytearray()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        if b"\x1b[6n" in chunk:
            os.write(fd, b"\x1b[1;1R")
        data.extend(chunk)
    return bytes(data)


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
        if b"\x1b[6n" in chunk:
            os.write(fd, b"\x1b[1;1R")
        data.extend(chunk)
    return bytes(data)


def test_real_pty_action_menu_and_exit(tmp_path) -> None:
    master, slave = pty.openpty()
    _set_winsize(master, 30, 100)
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
        startup += _read_for(master, duration=0.2)
        assert b"ChatGPT" in startup
        assert b"\x1b[?1049h" in startup
        assert b"\x1b[?1007h" in startup
        assert b"\x1b[?1000h" not in startup
        assert b"\x1b[?1003h" not in startup
        assert b"\x1b[?1006h" not in startup

        for rows, cols in ((18, 60), (35, 120)):
            _set_winsize(master, rows, cols)
            os.kill(process.pid, signal.SIGWINCH)
            resized = _read_until(master, "❯".encode(), timeout=5.0)
            assert "❯".encode() in resized
            assert b"\x1b[?1049l" not in resized
            assert b"\x1b[?1000h" not in resized
            assert b"\x1b[?1003h" not in resized
            assert b"\x1b[?1006h" not in resized
            assert process.poll() is None
            # SIGWINCH triggers a CPR query; let the fake terminal answer it
            # before the next resize/command.
            _read_for(master, duration=0.25)

        # Command completion is an in-place overlay: typing slash must not leave
        # the alternate screen or restart the application.
        os.write(master, b"/")
        menu = _read_until(master, b"/exit", timeout=5.0)
        assert b"/new" in menu
        assert b"/resume" in menu
        assert b"/detach" in menu
        assert b"/stop" in menu
        assert b"/goal" in menu
        assert b"/image" in menu
        assert b"Actions" not in menu
        assert b"/help" not in menu
        assert b"\x1b[?1049l" not in menu
        os.write(master, b"\x7f")
        _read_for(master, duration=0.15)

        os.write(master, b"/new\r")
        selected = _read_until(master, b"Started a new conversation.", timeout=5.0)
        assert b"Started a new conversation." in selected
        assert b"\x1b[?1049l" not in selected
        assert b"\x1b[?1049h" not in selected

        image = tmp_path / "screen shot.png"
        image.write_bytes(b"png")
        os.write(master, f'/image "{image}"\r'.encode())
        attached = _read_until(master, b"Attached for next prompt", timeout=5.0)
        assert b"Attached for next prompt" in attached
        assert b"pending: 1" in attached

        os.write(master, b"/image clear\r")
        cleared = _read_until(master, b"Cleared 1 pending image.", timeout=5.0)
        assert b"Cleared 1 pending image." in cleared

        os.write(master, b"/exit\r")
        assert process.wait(timeout=5.0) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5.0)
        os.close(master)
