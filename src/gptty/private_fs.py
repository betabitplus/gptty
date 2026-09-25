from __future__ import annotations

import os
from pathlib import Path
import uuid


PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
PRIVATE_MODES_SUPPORTED = os.name != "nt"


def ensure_private_dir(path: str | Path) -> Path:
    """Create or harden an application-owned directory as owner-only on POSIX."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    if PRIVATE_MODES_SUPPORTED:
        target.chmod(PRIVATE_DIR_MODE)
    return target


def atomic_write_private_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    sync: bool = False,
) -> Path:
    """Atomically replace a text file and clamp it to owner-only permissions."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            PRIVATE_FILE_MODE,
        )
        if PRIVATE_MODES_SUPPORTED:
            os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            fd = None
            handle.write(text)
            handle.flush()
            if sync:
                os.fsync(handle.fileno())
        os.replace(temporary, target)
        harden_private_file(target)
        return target
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def create_private_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Create a new owner-only text file without overwriting an existing path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        PRIVATE_FILE_MODE,
    )
    try:
        if PRIVATE_MODES_SUPPORTED:
            os.fchmod(fd, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            fd = -1
            handle.write(text)
            handle.flush()
    finally:
        if fd >= 0:
            os.close(fd)
    harden_private_file(target)
    return target


def harden_private_file(path: str | Path) -> Path:
    target = Path(path)
    if PRIVATE_MODES_SUPPORTED:
        target.chmod(PRIVATE_FILE_MODE)
    return target
