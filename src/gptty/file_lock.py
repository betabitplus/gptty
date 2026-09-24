from __future__ import annotations

import errno
import os
import time
from pathlib import Path


_CONTENTION_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}


class KernelFileLock:
    """Cross-process exclusive lock whose authority is held by the kernel."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def fd(self) -> int:
        if self._fd is None:
            raise RuntimeError(f"lock is not held: {self.path}")
        return self._fd

    def try_acquire(self) -> bool:
        if self._fd is not None:
            raise RuntimeError(f"lock is already held: {self.path}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name != "nt":
                os.fchmod(fd, 0o600)
            self._ensure_lock_byte(fd)
            try:
                self._lock_fd(fd)
            except OSError as exc:
                if self._is_contention(exc):
                    os.close(fd)
                    return False
                raise
            self._fd = fd
            return True
        except BaseException:
            if self._fd is None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    def acquire(self, *, timeout: float, poll_interval: float = 0.05) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        started = time.monotonic()
        while True:
            if self.try_acquire():
                return
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"lock is busy: {self.path}")
            time.sleep(min(poll_interval, max(0.0, timeout - (time.monotonic() - started))))

    def write_metadata(self, payload: bytes) -> None:
        fd = self.fd
        os.lseek(fd, 0, os.SEEK_SET)
        view = memoryview(payload)
        remaining = len(view)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short lock metadata write: {written}/{remaining} bytes")
            view = view[written:]
            remaining -= written
        os.ftruncate(fd, len(payload))
        os.fsync(fd)

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            self._unlock_fd(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> "KernelFileLock":
        if not self.try_acquire():
            raise TimeoutError(f"lock is busy: {self.path}")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

    @classmethod
    def is_held(cls, path: str | Path) -> bool:
        lock_path = Path(path)
        if not lock_path.exists():
            return False
        lock = cls(lock_path)
        if not lock.try_acquire():
            return True
        lock.release()
        return False

    @staticmethod
    def _ensure_lock_byte(fd: int) -> None:
        if os.name != "nt":
            return
        if os.fstat(fd).st_size > 0:
            return
        os.write(fd, b"0")
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)

    @staticmethod
    def _is_contention(exc: OSError) -> bool:
        return isinstance(exc, BlockingIOError) or exc.errno in _CONTENTION_ERRNOS

    @staticmethod
    def _lock_fd(fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return

        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock_fd(fd: int) -> None:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
