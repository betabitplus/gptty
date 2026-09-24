from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .file_lock import KernelFileLock


@dataclass
class GoalRunLock:
    """Process-lifetime kernel ownership for one active Goal."""

    goal_id: str
    path: Path
    _kernel_lock: KernelFileLock
    runner_id: str
    pid: int
    released: bool = False

    @property
    def fd(self) -> int:
        return self._kernel_lock.fd

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self._kernel_lock.release()

    def __enter__(self) -> "GoalRunLock":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def goal_lock_path(root: str | Path, goal_id: str) -> Path:
    token = "".join(ch for ch in str(goal_id) if ch.isalnum() or ch in {"-", "_"})
    if not token:
        raise ValueError("goal_id is required")
    return Path(root) / "locks" / f"goal-{token}.lock"


def try_acquire_goal_lock(
    root: str | Path,
    goal_id: str,
    *,
    runner_id: str,
    pid: int | None = None,
) -> GoalRunLock | None:
    """Acquire Goal ownership without stale-file/PID heuristics."""

    path = goal_lock_path(root, goal_id)
    kernel_lock = KernelFileLock(path)
    if not kernel_lock.try_acquire():
        return None

    owner_pid = int(pid if pid is not None else os.getpid())
    payload = {
        "goal_id": goal_id,
        "runner_id": runner_id,
        "pid": owner_pid,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        kernel_lock.write_metadata(encoded)
    except Exception:
        kernel_lock.release()
        raise
    return GoalRunLock(
        goal_id=goal_id,
        path=path,
        _kernel_lock=kernel_lock,
        runner_id=runner_id,
        pid=owner_pid,
    )


def read_goal_lock_metadata(root: str | Path, goal_id: str) -> dict[str, object]:
    path = goal_lock_path(root, goal_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def goal_lock_is_held(root: str | Path, goal_id: str) -> bool:
    """Probe kernel lock state without trusting metadata or PID reuse."""

    return KernelFileLock.is_held(goal_lock_path(root, goal_id))
