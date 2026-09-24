from __future__ import annotations

import os
import stat

import pytest

from gptty.file_lock import KernelFileLock


def test_kernel_lock_probe_does_not_create_missing_sidecar(tmp_path) -> None:
    path = tmp_path / "missing.lock"

    assert KernelFileLock.is_held(path) is False
    assert not path.exists()


def test_kernel_lock_metadata_rewrite_truncates_previous_payload(tmp_path) -> None:
    path = tmp_path / "metadata.lock"
    lock = KernelFileLock(path)
    assert lock.try_acquire() is True
    try:
        lock.write_metadata(b'{"long":"value-that-must-disappear"}\n')
        lock.write_metadata(b'{"short":1}\n')
        assert path.read_bytes() == b'{"short":1}\n'
    finally:
        lock.release()

    assert path.exists()
    assert KernelFileLock.is_held(path) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits do not apply on Windows")
def test_kernel_lock_repairs_sidecar_to_owner_only_mode(tmp_path) -> None:
    path = tmp_path / "mode.lock"
    path.write_bytes(b"0")
    path.chmod(0o666)

    lock = KernelFileLock(path)
    assert lock.try_acquire() is True
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        lock.release()
