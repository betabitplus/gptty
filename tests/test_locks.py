from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from gptty.locks import (
    acquire_conversation_lock,
    conversation_lock_is_held,
)


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
    env["PYTHONPATH"] = (
        str(Path(__file__).parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
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
