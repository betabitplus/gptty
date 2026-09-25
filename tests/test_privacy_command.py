from __future__ import annotations

import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import gptty.commands.privacy as privacy_command
from gptty.commands.privacy import run_privacy_prune, run_privacy_status
from gptty.local_store import LocalEventStore, local_store_path, local_store_root


def _args(state_path: Path, **overrides):
    data = {
        "profile": None,
        "state": str(state_path),
        "older_than_days": 30,
        "include_archives": False,
        "include_exports": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _create_run(
    store: LocalEventStore,
    *,
    run_id: str,
    status: str,
    timestamp: str,
) -> None:
    store.create_run(
        run_id=run_id,
        summary={
            "run_id": run_id,
            "status": status,
            "updated_at": timestamp,
            "last_event": "run_started",
        },
        first_event={"type": "run_started", "timestamp": timestamp},
    )


def test_privacy_status_reports_content_free_inventory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gptty_state.json"
    store_path = local_store_path(profile=None, state_path=state_path)
    store = LocalEventStore(store_path)
    _create_run(
        store,
        run_id="run-1",
        status="completed",
        timestamp="2026-09-25T00:00:00+00:00",
    )
    store.put_pending_tui_event(
        "turn-1",
        {
            "event_id": "turn-1:user",
            "text": "PRIVATE_PROMPT_MARKER",
            "observed_at": "2026-09-25T00:00:00+00:00",
        },
    )
    store.import_tui_conversation(
        "conv-12345678",
        events=[],
        title="PRIVATE_TITLE_MARKER",
        imported_at="2026-09-25T00:00:00+00:00",
    )
    export_root = tmp_path / "exports"
    export_root.mkdir()
    generated = export_root / "2026-09-25_12-00-00 - chat.md"
    generated.write_text("PRIVATE_EXPORT_MARKER\n", encoding="utf-8")
    monkeypatch.setattr(privacy_command, "DEFAULT_EXPORT_DIRECTORY", export_root)
    stdout = StringIO()

    assert run_privacy_status(_args(state_path), stdout=stdout) == 0

    output = stdout.getvalue()
    assert "Runs: 1" in output
    assert "Orphan-pending candidates: 1" in output
    assert "Archived conversations: 1" in output
    assert "Default generated exports: 1" in output
    assert "PRIVATE_PROMPT_MARKER" not in output
    assert "PRIVATE_TITLE_MARKER" not in output
    assert "PRIVATE_EXPORT_MARKER" not in output


def test_privacy_prune_respects_lifecycle_and_user_owned_exports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "gptty_state.json"
    run_root = local_store_root(profile=None, state_path=state_path)
    store_path = local_store_path(profile=None, state_path=state_path)
    store = LocalEventStore(store_path)
    old = "2026-01-01T00:00:00+00:00"
    recent = "2026-09-20T00:00:00+00:00"
    _create_run(store, run_id="old-completed", status="completed", timestamp=old)
    _create_run(store, run_id="old-running", status="running", timestamp=old)
    _create_run(store, run_id="recent-completed", status="completed", timestamp=recent)
    for run_id in ("old-completed", "old-running", "recent-completed"):
        (run_root / f"{run_id}.json").write_text("{}\n", encoding="utf-8")
        (run_root / f"{run_id}.jsonl").write_text("{}\n", encoding="utf-8")

    store.put_pending_tui_event(
        "old-pending",
        {"event_id": "old-pending:user", "observed_at": old, "text": "old"},
    )
    store.put_pending_tui_event(
        "recent-pending",
        {
            "event_id": "recent-pending:user",
            "observed_at": recent,
            "text": "recent",
        },
    )

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conversations_root = archive_root / "conversations"
    conversations_root.mkdir()
    pending_root = archive_root / "pending"
    pending_root.mkdir()
    store.import_tui_conversation(
        "conv-old1234",
        events=[
            {
                "event_id": "old:user",
                "role": "user",
                "text": "old archive",
                "observed_at": old,
            }
        ],
        title="Old archive",
        imported_at=old,
    )
    old_conversation_dir = conversations_root / "conv-old1234"
    old_conversation_dir.mkdir()
    (old_conversation_dir / "transcript.md").write_text("old\n", encoding="utf-8")
    old_epoch = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(old_conversation_dir, (old_epoch, old_epoch))
    monkeypatch.setattr(privacy_command, "archive_root", lambda: archive_root)

    export_root = tmp_path / "exports"
    export_root.mkdir()
    old_generated = export_root / "2026-01-01_00-00-00 - old.md"
    recent_generated = export_root / "2026-09-20_00-00-00 - recent.md"
    user_owned = export_root / "my-explicit-export.md"
    for path in (old_generated, recent_generated, user_owned):
        path.write_text("copy\n", encoding="utf-8")
    recent_epoch = datetime(2026, 9, 20, tzinfo=timezone.utc).timestamp()
    os.utime(old_generated, (old_epoch, old_epoch))
    os.utime(user_owned, (old_epoch, old_epoch))
    os.utime(recent_generated, (recent_epoch, recent_epoch))
    monkeypatch.setattr(privacy_command, "DEFAULT_EXPORT_DIRECTORY", export_root)
    stdout = StringIO()

    assert run_privacy_prune(
        _args(state_path, include_archives=True, include_exports=True),
        stdout=stdout,
        now=datetime(2026, 9, 25, tzinfo=timezone.utc),
    ) == 0

    assert store.run_summary("old-completed") is None
    assert not (run_root / "old-completed.json").exists()
    assert not (run_root / "old-completed.jsonl").exists()
    assert store.run_summary("old-running") is not None
    assert (run_root / "old-running.json").exists()
    assert store.run_summary("recent-completed") is not None
    assert (run_root / "recent-completed.json").exists()
    assert store.pop_pending_tui_event("old-pending") is None
    assert store.pop_pending_tui_event("recent-pending") is not None
    assert store.tui_events("conv-old1234") == []
    assert not old_conversation_dir.exists()
    assert not old_generated.exists()
    assert recent_generated.exists()
    assert user_owned.exists()
    output = stdout.getvalue()
    assert "Runs removed: 1" in output
    assert "Orphan pending prompts removed: 1" in output
    assert "Archived conversations removed: 1" in output
    assert "Default generated exports removed: 1" in output
