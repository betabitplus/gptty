from __future__ import annotations

import os
import pty
import sqlite3
from pathlib import Path

import pytest

from gptty.local_store import local_store_path
from gptty.session_state import SessionStateError, session_handle
from gptty.state import (
    ChatState,
    GoalState,
    save_chat_state,
    session_chat_state_path,
)


def test_default_session_imports_legacy_state_once_and_then_is_authoritative(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(
        state_path,
        ChatState(current_conversation="legacy-conv", model="legacy-model"),
    )

    handle = session_handle(
        state_path=state_path,
        profile=None,
        environ={},
    )
    state = handle.load()
    assert handle.session_id == "default"
    assert handle.kind == "default"
    assert state.current_conversation == "legacy-conv"
    assert state.model == "legacy-model"

    state.current_conversation = "sqlite-conv"
    handle.save(state)

    save_chat_state(
        state_path,
        ChatState(current_conversation="changed-legacy", model="changed-legacy"),
    )
    reloaded = session_handle(
        state_path=state_path,
        profile=None,
        environ={},
    ).load()
    assert reloaded.current_conversation == "sqlite-conv"
    assert reloaded.model == "legacy-model"


def test_explicit_session_imports_old_sibling_and_reuses_intentionally(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gptty_state.json"
    explicit = session_handle(
        state_path=state_path,
        profile=None,
        explicit_session="named-session",
        environ={},
    )
    assert explicit.legacy_path is not None
    save_chat_state(
        explicit.legacy_path,
        ChatState(current_conversation="old-explicit", model="old-model"),
    )

    first = explicit.load()
    assert explicit.kind == "explicit"
    assert first.current_conversation == "old-explicit"

    first.current_conversation = "updated-explicit"
    explicit.save(first)

    second_handle = session_handle(
        state_path=state_path,
        profile=None,
        explicit_session="named-session",
        environ={},
    )
    second = second_handle.load()
    assert second_handle.session_id == explicit.session_id
    assert second.current_conversation == "updated-explicit"


def test_interactive_runtime_session_is_random_and_seeded_from_default(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gptty_state.json"
    default = session_handle(state_path=state_path, profile=None, environ={})
    default_state = default.load()
    default_state.current_conversation = "shared-seed"
    default.save(default_state)

    first = session_handle(
        state_path=state_path,
        profile=None,
        interactive=True,
        input_stream=None,
        environ={},
    )
    second = session_handle(
        state_path=state_path,
        profile=None,
        interactive=True,
        input_stream=None,
        environ={},
    )

    assert first.kind == second.kind == "runtime"
    assert first.session_id != second.session_id
    assert first.load().current_conversation == "shared-seed"
    assert second.load().current_conversation == "shared-seed"


def test_terminal_identity_is_discovery_hint_not_session_identity(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    master_fd, slave_fd = pty.openpty()
    try:
        with os.fdopen(slave_fd, "r", encoding="utf-8", closefd=True) as stream:
            first = session_handle(
                state_path=state_path,
                profile=None,
                interactive=True,
                input_stream=stream,
                environ={"CMUX_SURFACE_ID": "surface-A"},
            )
            second = session_handle(
                state_path=state_path,
                profile=None,
                interactive=True,
                input_stream=stream,
                environ={"CMUX_SURFACE_ID": "surface-A"},
            )
    finally:
        os.close(master_fd)

    assert first.session_id != second.session_id
    assert first.discovery_hint == second.discovery_hint
    assert first.discovery_hint is not None
    assert first.discovery_hint.startswith("cmux:")




def test_interactive_runtime_imports_old_terminal_state_only_once(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    save_chat_state(
        state_path,
        ChatState(current_conversation="default-conv", model="default-model"),
    )
    master_fd, slave_fd = pty.openpty()
    env = {"CMUX_SURFACE_ID": "surface-migration"}
    try:
        with os.fdopen(slave_fd, "r", encoding="utf-8", closefd=True) as stream:
            legacy_path = session_chat_state_path(
                state_path,
                input_stream=stream,
                environ=env,
            )
            assert legacy_path != state_path
            save_chat_state(
                legacy_path,
                ChatState(current_conversation="old-surface", model="old-surface-model"),
            )

            first = session_handle(
                state_path=state_path,
                profile=None,
                interactive=True,
                input_stream=stream,
                environ=env,
            )
            first_state = first.load()
            assert first_state.current_conversation == "old-surface"
            assert first_state.model == "old-surface-model"
            assert first.store.session_imported_from(legacy_path) is True

            first_state.current_conversation = "first-runtime-updated"
            first.save(first_state)

            second = session_handle(
                state_path=state_path,
                profile=None,
                interactive=True,
                input_stream=stream,
                environ=env,
            )
            second_state = second.load()
    finally:
        os.close(master_fd)

    assert second.session_id != first.session_id
    assert second_state.current_conversation == "default-conv"
    assert second_state.model == "default-model"
    assert legacy_path.exists()




def test_corrupt_legacy_json_does_not_block_new_transactional_session(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    state_path.write_text("{broken", encoding="utf-8")

    handle = session_handle(
        state_path=state_path,
        profile=None,
        environ={},
    )
    state = handle.load()

    assert state.current_conversation is None
    assert state.model is None
    assert handle.migration_warning is not None
    assert handle.store.get_session(handle.session_id) is not None


def test_existing_session_can_recover_legacy_goal_after_import_crash(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    legacy_goal = GoalState(goal_id="goal-crash-recovery", objective="recover me")
    save_chat_state(
        state_path,
        ChatState(goal=legacy_goal, goal_id=legacy_goal.goal_id),
    )

    first = session_handle(state_path=state_path, profile=None, environ={})
    first_state = first.load()
    assert first_state.goal is not None
    assert first_state.goal.goal_id == legacy_goal.goal_id

    # Simulate process death after the session row was committed but before the
    # legacy Goal payload was copied into GoalStore.
    second = session_handle(state_path=state_path, profile=None, environ={})
    second_state = second.load()
    assert second_state.goal is not None
    assert second_state.goal.goal_id == legacy_goal.goal_id
    assert second_state.goal.objective == "recover me"


def test_session_handle_never_age_evicts_unleased_runtime_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gptty_state.json"
    handle = session_handle(state_path=state_path, profile=None, environ={})
    monkeypatch.setattr("gptty.local_store.time.time", lambda: 1.0)
    handle.store.create_session(
        "runtime-long-idle",
        kind="runtime",
        discovery_hint="cmux:old",
        current_conversation="conv-old",
    )

    monkeypatch.setattr("gptty.local_store.time.time", lambda: 10_000_000.0)
    session_handle(state_path=state_path, profile=None, environ={})

    preserved = handle.store.get_session("runtime-long-idle")
    assert preserved is not None
    assert preserved["current_conversation"] == "conv-old"


def test_explicit_environment_session_is_stable_even_without_tty(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    first = session_handle(
        state_path=state_path,
        profile=None,
        interactive=False,
        environ={"GPTTY_SESSION_ID": "automation-A"},
    )
    second = session_handle(
        state_path=state_path,
        profile=None,
        interactive=False,
        environ={"GPTTY_SESSION_ID": "automation-A"},
    )

    assert first.kind == second.kind == "explicit"
    assert first.session_id == second.session_id


def test_session_handle_fails_closed_on_stale_revision(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    first = session_handle(state_path=state_path, profile=None, environ={})
    second = session_handle(state_path=state_path, profile=None, environ={})
    first_state = first.load()
    second_state = second.load()

    first_state.current_conversation = "winner"
    first.save(first_state)

    second_state.current_conversation = "stale"
    with pytest.raises(SessionStateError, match="changed concurrently"):
        second.save(second_state)

    assert session_handle(
        state_path=state_path,
        profile=None,
        environ={},
    ).load().current_conversation == "winner"


def test_session_persists_only_goal_reference_not_goal_payload(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    handle = session_handle(state_path=state_path, profile=None, environ={})
    state = handle.load()
    state.goal = GoalState(goal_id="goal-123", objective="large durable objective")
    handle.save(state)

    row = handle.store.get_session(handle.session_id)
    assert row is not None
    assert row["goal_id"] == "goal-123"

    reloaded = session_handle(
        state_path=state_path,
        profile=None,
        environ={},
    ).load()
    assert reloaded.goal_id == "goal-123"
    assert reloaded.goal is None


def test_session_handle_wraps_corrupt_sqlite_database(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    db_path = local_store_path(profile=None, state_path=state_path)
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"not-a-sqlite-database")

    with pytest.raises(SessionStateError, match="failed to open local session database"):
        session_handle(state_path=state_path, profile=None, environ={})


def test_session_handle_wraps_future_local_store_schema(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    db_path = local_store_path(profile=None, state_path=state_path)
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA user_version=999")
        db.commit()

    with pytest.raises(
        SessionStateError,
        match="local state schema is newer than this gptty build",
    ):
        session_handle(state_path=state_path, profile=None, environ={})


def test_transactional_session_round_trips_reasoning_effort(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    handle = session_handle(state_path=state_path, profile=None, environ={})
    state = handle.load()
    state.current_conversation = "conv-effort"
    state.reasoning_effort = "instant"
    handle.save(state)

    reloaded = session_handle(state_path=state_path, profile=None, environ={}).load()
    assert reloaded.current_conversation == "conv-effort"
    assert reloaded.reasoning_effort == "instant"
