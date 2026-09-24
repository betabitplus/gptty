from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from gptty.goal_lock import (
    goal_lock_is_held,
    read_goal_lock_metadata,
    try_acquire_goal_lock,
)


def test_goal_kernel_lock_excludes_second_owner_and_releases_cleanly(tmp_path) -> None:
    first = try_acquire_goal_lock(
        tmp_path, "goal-lock", runner_id="runner-a", pid=os.getpid()
    )
    assert first is not None
    assert goal_lock_is_held(tmp_path, "goal-lock") is True
    assert (
        try_acquire_goal_lock(
            tmp_path, "goal-lock", runner_id="runner-b", pid=os.getpid()
        )
        is None
    )
    assert read_goal_lock_metadata(tmp_path, "goal-lock")["runner_id"] == "runner-a"

    first.release()
    assert goal_lock_is_held(tmp_path, "goal-lock") is False
    second = try_acquire_goal_lock(tmp_path, "goal-lock", runner_id="runner-b")
    assert second is not None
    second.release()


def test_goal_kernel_lock_is_released_by_sigkill_without_pid_heuristics(tmp_path) -> None:
    ready = tmp_path / "ready"
    script = r"""
import os, sys, time
from pathlib import Path
from gptty.goal_lock import try_acquire_goal_lock
root=Path(sys.argv[1]); ready=Path(sys.argv[2])
lock=try_acquire_goal_lock(root, "goal-crash", runner_id="child")
assert lock is not None
ready.write_text(str(os.getpid()))
time.sleep(60)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(ready)], env=env
    )
    try:
        deadline = time.time() + 5
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        assert goal_lock_is_held(tmp_path, "goal-crash") is True
        child.kill()
        child.wait(timeout=5)
        deadline = time.time() + 2
        while goal_lock_is_held(tmp_path, "goal-crash") and time.time() < deadline:
            time.sleep(0.02)
        assert goal_lock_is_held(tmp_path, "goal-crash") is False
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
