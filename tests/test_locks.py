from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import pytest

from gptty.locks import (
    ConversationLockError,
    acquire_conversation_lock,
    conversation_lock_is_held,
    conversation_lock_path,
    read_conversation_lock,
    render_lock_error,
    render_lock_timeout,
)


def test_conversation_lock_retains_diagnostic_sidecar_but_kernel_owns_state(
    tmp_path,
) -> None:
    lock = acquire_conversation_lock(
        conversation_ref="conv-1",
        lock_dir=tmp_path,
        profile="work",
        command="send",
    )
    path = lock.info.lock_path

    assert path.exists()
    assert conversation_lock_is_held(tmp_path, "conv-1") is True
    assert lock.info.profile == "work"
    assert lock.info.command == "send"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    lock.release()

    assert path.exists(), "sidecar is retained only as diagnostic metadata"
    assert conversation_lock_is_held(tmp_path, "conv-1") is False
    metadata = read_conversation_lock(path, fallback_conversation="conv-1")
    assert metadata.command == "send"
    assert metadata.profile == "work"


def test_acquire_conversation_lock_fails_when_kernel_lock_is_active(tmp_path) -> None:
    first = acquire_conversation_lock(
        conversation_ref="conv-1",
        lock_dir=tmp_path,
        profile="work",
        command="send",
    )

    try:
        with pytest.raises(ConversationLockError) as exc_info:
            acquire_conversation_lock(
                conversation_ref="conv-1",
                lock_dir=tmp_path,
                command="chat",
                timeout=0,
            )
    finally:
        first.release()

    assert exc_info.value.info.conversation_ref == "conv-1"
    assert exc_info.value.info.command == "send"
    assert exc_info.value.info.profile == "work"


def test_old_mtime_cannot_steal_a_live_conversation_lock(tmp_path) -> None:
    first = acquire_conversation_lock(
        conversation_ref="conv-old",
        lock_dir=tmp_path,
        command="send",
    )
    path = conversation_lock_path(tmp_path, "conv-old")
    os.utime(path, (1, 1))

    try:
        with pytest.raises(ConversationLockError):
            acquire_conversation_lock(
                conversation_ref="conv-old",
                lock_dir=tmp_path,
                command="chat",
                timeout=0,
            )
        assert conversation_lock_is_held(tmp_path, "conv-old") is True
    finally:
        first.release()


def test_conversation_kernel_lock_is_released_by_process_death(tmp_path) -> None:
    ready = tmp_path / "ready"
    script = r"""
import os
import sys
import time
from pathlib import Path
from gptty.locks import acquire_conversation_lock

root = Path(sys.argv[1])
ready = Path(sys.argv[2])
lock = acquire_conversation_lock(
    conversation_ref="conv-crash",
    lock_dir=root,
    command="send",
)
ready.write_text(str(os.getpid()), encoding="utf-8")
time.sleep(60)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(ready)],
        env=env,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        assert conversation_lock_is_held(tmp_path, "conv-crash") is True

        child.kill()
        child.wait(timeout=5)

        deadline = time.monotonic() + 2
        while (
            conversation_lock_is_held(tmp_path, "conv-crash")
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert conversation_lock_is_held(tmp_path, "conv-crash") is False

        second = acquire_conversation_lock(
            conversation_ref="conv-crash",
            lock_dir=tmp_path,
            command="chat",
            timeout=0,
        )
        second.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_render_lock_error_uses_in_progress_copy(tmp_path) -> None:
    first = acquire_conversation_lock(
        conversation_ref="conv-1",
        lock_dir=tmp_path,
        profile="work",
        command="send",
    )

    try:
        with pytest.raises(ConversationLockError) as exc_info:
            acquire_conversation_lock(
                conversation_ref="conv-1", lock_dir=tmp_path, timeout=0
            )
    finally:
        first.release()

    stderr = StringIO()
    render_lock_error(exc_info.value, stderr=stderr)

    output = stderr.getvalue()
    assert "gptty: conversation in progress" in output
    assert "This conversation is already waiting for a reply." in output
    assert "Profile: work" in output
    assert "Conversation: conv-1" in output


def test_render_lock_timeout_includes_waited_time(tmp_path) -> None:
    first = acquire_conversation_lock(
        conversation_ref="conv-1", lock_dir=tmp_path, command="send"
    )

    try:
        with pytest.raises(ConversationLockError) as exc_info:
            acquire_conversation_lock(
                conversation_ref="conv-1", lock_dir=tmp_path, timeout=0
            )
    finally:
        first.release()

    stderr = StringIO()
    render_lock_timeout(exc_info.value, stderr=stderr)

    output = stderr.getvalue()
    assert "gptty: conversation still in progress" in output
    assert "Waited:" in output
