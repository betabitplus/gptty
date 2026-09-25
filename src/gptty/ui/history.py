from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Iterable

from prompt_toolkit.history import FileHistory

from ..file_lock import KernelFileLock


DEFAULT_HISTORY_LIMIT = 2_000


class PrivatePromptHistory(FileHistory):
    """Prompt history with private permissions and mode-aware persistence."""

    def __init__(self, path: str | Path, *, limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        self.path = Path(path)
        self.limit = max(0, int(limit))
        self._persistent = self.limit > 0
        self._ephemeral_count = 0
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._harden_existing_file()
        super().__init__(str(self.path))
        self._compact_existing_file()

    @property
    def persistent(self) -> bool:
        return self._persistent and self.limit > 0

    def set_persistent(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self._persistent and self._ephemeral_count:
            del self._loaded_strings[: self._ephemeral_count]
            self._ephemeral_count = 0
        self._persistent = enabled

    def store_string(self, string: str) -> None:
        if not self.persistent:
            self._ephemeral_count += 1
            return
        lock = self._acquire_lock()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                if os.name != "nt":
                    os.fchmod(fd, 0o600)
                with os.fdopen(fd, "ab", closefd=False) as handle:
                    handle.write(self._encode_entry(string))
                    handle.flush()
            finally:
                os.close(fd)
            self._trim_persisted_history_unlocked()
        finally:
            lock.release()

    def clear(self) -> None:
        lock = self._acquire_lock()
        try:
            self._loaded_strings.clear()
            self._loaded = True
            self._ephemeral_count = 0
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        finally:
            lock.release()

    def load_history_strings(self) -> Iterable[str]:
        lock = self._acquire_lock()
        try:
            strings = list(super().load_history_strings())
        finally:
            lock.release()
        if self.limit <= 0:
            return iter(())
        return iter(strings[: self.limit])

    def _acquire_lock(self) -> KernelFileLock:
        lock = KernelFileLock(self._lock_path)
        lock.acquire(timeout=2.0)
        return lock

    def _harden_existing_file(self) -> None:
        if os.name == "nt" or not self.path.exists():
            return
        self.path.chmod(0o600)

    def _compact_existing_file(self) -> None:
        if not self.path.exists():
            return
        lock = self._acquire_lock()
        try:
            persisted = list(super().load_history_strings())
            if self.limit <= 0:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                return
            if len(persisted) > self.limit:
                self._rewrite_persisted_unlocked(persisted[: self.limit])
        finally:
            lock.release()

    def _trim_persisted_history_unlocked(self) -> None:
        persisted = list(super().load_history_strings())
        if len(persisted) <= self.limit:
            return
        self._rewrite_persisted_unlocked(persisted[: self.limit])

    def _rewrite_persisted_unlocked(self, newest_first: list[str]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        try:
            if os.name != "nt":
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=False) as handle:
                for string in reversed(newest_first):
                    handle.write(self._encode_entry(string))
                handle.flush()
        finally:
            os.close(fd)
        os.replace(temporary, self.path)
        self._harden_existing_file()

    @staticmethod
    def _encode_entry(string: str) -> bytes:
        payload = [f"\n# {datetime.datetime.now()}\n"]
        payload.extend(f"+{line}\n" for line in string.split("\n"))
        return "".join(payload).encode("utf-8")
