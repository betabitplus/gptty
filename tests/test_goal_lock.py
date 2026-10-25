from __future__ import annotations

import os

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
