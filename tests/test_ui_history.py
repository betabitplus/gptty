from __future__ import annotations

import multiprocessing
import os
import stat
from pathlib import Path

from gptty.ui.history import PrivatePromptHistory


def persisted(path, *, limit=100):
    history = PrivatePromptHistory(path, limit=limit)
    return list(history.load_history_strings())


def _append_history_worker(path: str, prefix: str, count: int, limit: int) -> None:
    history = PrivatePromptHistory(Path(path), limit=limit)
    for index in range(count):
        history.append_string(f"{prefix}-{index}")


def test_temporary_prompt_never_leaks_into_persisted_history_during_retention(tmp_path) -> None:
    path = tmp_path / "history"
    history = PrivatePromptHistory(path, limit=2)

    history.append_string("normal-one")
    history.set_persistent(False)
    history.append_string("temporary-secret")
    history.set_persistent(True)
    assert "temporary-secret" not in history.get_strings()
    assert history.get_strings() == ["normal-one"]
    history.append_string("normal-two")
    history.append_string("normal-three")

    assert persisted(path, limit=2) == ["normal-three", "normal-two"]
    assert "temporary-secret" not in path.read_text(encoding="utf-8")


def test_history_file_is_owner_only_and_existing_mode_is_hardened(tmp_path) -> None:
    path = tmp_path / "history"
    path.write_text("\n# old\n+old prompt\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o644)

    history = PrivatePromptHistory(path)
    history.append_string("new prompt")

    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_history_limit_compacts_existing_and_new_entries(tmp_path) -> None:
    path = tmp_path / "history"
    history = PrivatePromptHistory(path, limit=3)
    for index in range(6):
        history.append_string(f"prompt-{index}")

    assert persisted(path, limit=3) == ["prompt-5", "prompt-4", "prompt-3"]


def test_clear_removes_persisted_and_loaded_history(tmp_path) -> None:
    path = tmp_path / "history"
    history = PrivatePromptHistory(path)
    history.append_string("persisted")
    assert history.get_strings() == ["persisted"]

    history.clear()

    assert history.get_strings() == []
    assert not path.exists()


def test_zero_history_limit_never_creates_persistent_history(tmp_path) -> None:
    path = tmp_path / "history"
    history = PrivatePromptHistory(path, limit=0)

    history.append_string("memory only")

    assert history.get_strings() == ["memory only"]
    assert not path.exists()


def test_multiprocess_appends_remain_parseable_and_bounded(tmp_path) -> None:
    path = tmp_path / "history"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_append_history_worker,
            args=(str(path), f"client-{index}", 20, 50),
        )
        for index in range(5)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    entries = persisted(path, limit=50)
    assert len(entries) == 50
    assert len(set(entries)) == 50
    assert all(entry.startswith("client-") for entry in entries)
