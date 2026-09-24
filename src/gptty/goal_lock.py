from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class GoalRunLock:
    """Process-lifetime kernel ownership for one active Goal."""

    goal_id: str
    path: Path
    fd: int
    runner_id: str
    pid: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self.fd)
            except OSError:
                pass

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
    """Acquire Goal ownership without stale-file/PID heuristics.

    flock is held by the open file description and the kernel releases it on
    process death, including SIGKILL. The file is deliberately retained as
    diagnostic metadata; its existence never means the Goal is locked.
    """

    path = goal_lock_path(root, goal_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except Exception:
        os.close(fd)
        raise

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
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, encoded)
    os.fsync(fd)
    return GoalRunLock(
        goal_id=goal_id,
        path=path,
        fd=fd,
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

    path = goal_lock_path(root, goal_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
