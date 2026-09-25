from __future__ import annotations

import os
from pathlib import Path

import pytest

from gptty import private_fs
from gptty.private_fs import (
    PRIVATE_DIR_MODE,
    PRIVATE_FILE_MODE,
    atomic_write_private_text,
    create_private_text,
    ensure_private_dir,
    harden_private_file,
)


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are unavailable on Windows")
def test_ensure_private_dir_creates_and_hardens_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "private"
    path.mkdir(mode=0o755)
    path.chmod(0o755)

    assert ensure_private_dir(path) == path

    assert _mode(path) == PRIVATE_DIR_MODE


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are unavailable on Windows")
def test_atomic_write_private_text_creates_owner_only_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"

    assert atomic_write_private_text(path, "one\n") == path

    assert path.read_text(encoding="utf-8") == "one\n"
    assert _mode(path) == PRIVATE_FILE_MODE


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are unavailable on Windows")
def test_atomic_write_private_text_replaces_permissive_file_privately(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o666)

    atomic_write_private_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert _mode(path) == PRIVATE_FILE_MODE
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_write_private_text_cleans_temp_and_preserves_target_on_replace_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    path.write_text("old\n", encoding="utf-8")

    def fail_replace(source: object, target: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(private_fs.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_private_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are unavailable on Windows")
def test_create_private_text_is_owner_only_and_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"

    assert create_private_text(path, "secret\n") == path

    assert path.read_text(encoding="utf-8") == "secret\n"
    assert _mode(path) == PRIVATE_FILE_MODE
    with pytest.raises(FileExistsError):
        create_private_text(path, "replacement\n")
    assert path.read_text(encoding="utf-8") == "secret\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are unavailable on Windows")
def test_harden_private_file_clamps_existing_mode(tmp_path: Path) -> None:
    path = tmp_path / "existing.txt"
    path.write_text("secret\n", encoding="utf-8")
    path.chmod(0o644)

    assert harden_private_file(path) == path

    assert _mode(path) == PRIVATE_FILE_MODE


def test_mode_operations_are_skipped_when_platform_does_not_support_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    directory = tmp_path / "private"
    monkeypatch.setattr(private_fs, "PRIVATE_MODES_SUPPORTED", False)

    def fail_chmod(*args: object, **kwargs: object) -> None:
        raise AssertionError("chmod/fchmod should not be used")

    monkeypatch.setattr(Path, "chmod", fail_chmod)
    monkeypatch.setattr(private_fs.os, "fchmod", fail_chmod)

    ensure_private_dir(directory)
    atomic_write_private_text(path, "one\n")
    harden_private_file(path)

    assert path.read_text(encoding="utf-8") == "one\n"
