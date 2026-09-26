from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

from ..local_store import LocalEventStore, local_store_path, local_store_root
from ..tui_archive import TUIArchive, archive_root
from .export import DEFAULT_EXPORT_DIRECTORY

_GENERATED_EXPORT_RE = re.compile(
    r"^(?P<stem>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2} - .+)\.md$"
)
_GENERATED_EXPORT_SIDECAR_RE = re.compile(
    r"^(?P<stem>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2} - .+)\.(?:context|manifest)\.json$"
)


def _store_path(args: Any) -> Path:
    return local_store_path(
        profile=getattr(args, "profile", None),
        state_path=Path(getattr(args, "state", "gptty_state.json")),
    )


def _run_root(args: Any) -> Path:
    return local_store_root(
        profile=getattr(args, "profile", None),
        state_path=Path(getattr(args, "state", "gptty_state.json")),
    )


def _generated_export_bundles() -> list[tuple[Path, ...]]:
    root = DEFAULT_EXPORT_DIRECTORY
    if not root.is_dir():
        return []
    grouped: dict[str, set[Path]] = {}
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        match = _GENERATED_EXPORT_RE.match(path.name)
        if match is None:
            match = _GENERATED_EXPORT_SIDECAR_RE.match(path.name)
        if match is None:
            continue
        grouped.setdefault(match.group("stem"), set()).add(path)
    return [
        tuple(sorted(paths, key=lambda item: item.name))
        for _, paths in sorted(grouped.items())
    ]


def _generated_exports() -> list[Path]:
    result: list[Path] = []
    for bundle in _generated_export_bundles():
        markdown = next((path for path in bundle if path.suffix == ".md"), None)
        result.append(markdown or bundle[0])
    return result


def run_privacy_status(
    args: Any,
    *,
    stdout: TextIO = sys.stdout,
) -> int:
    store_path = _store_path(args)
    inventory = {
        "runs": 0,
        "pending_prompts": 0,
        "archived_conversations": 0,
        "delivery_events": 0,
    }
    if store_path.exists():
        inventory = LocalEventStore(store_path).privacy_inventory()

    print("gptty local privacy status", file=stdout)
    print(f"Store: {store_path}", file=stdout)
    print(f"Runs: {inventory['runs']}", file=stdout)
    print(f"Orphan-pending candidates: {inventory['pending_prompts']}", file=stdout)
    print(f"Archived conversations: {inventory['archived_conversations']}", file=stdout)
    print(f"Delivery evidence events: {inventory['delivery_events']}", file=stdout)
    print(f"Default generated exports: {len(_generated_exports())}", file=stdout)
    print("Pending prompt TTL: 24 hours on TUI startup", file=stdout)
    print(
        "Explicit --output files are user-owned and are never auto-pruned.",
        file=stdout,
    )
    return 0


def _remove_run_projections(root: Path, run_ids: list[str]) -> list[str]:
    removable: list[str] = []
    for run_id in run_ids:
        failed = False
        for suffix in (".json", ".jsonl"):
            path = root / f"{run_id}{suffix}"
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failed = True
        if not failed:
            removable.append(run_id)
    return removable


def _prune_generated_exports(cutoff: datetime) -> int:
    removed = 0
    cutoff_epoch = cutoff.timestamp()
    for bundle in _generated_export_bundles():
        try:
            stale = max(path.stat().st_mtime for path in bundle) <= cutoff_epoch
        except OSError:
            continue
        if not stale:
            continue
        failed = False
        for path in bundle:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                failed = True
        if not failed:
            removed += 1
    return removed


def run_privacy_prune(
    args: Any,
    *,
    stdout: TextIO = sys.stdout,
    now: datetime | None = None,
) -> int:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    days = max(0, int(getattr(args, "older_than_days", 0)))
    cutoff = current - timedelta(days=days)

    store_path = _store_path(args)
    root = _run_root(args)
    store = LocalEventStore(store_path)

    run_candidates = store.run_ids_before(cutoff.isoformat())
    removable_runs = _remove_run_projections(root, run_candidates)
    removed_runs = store.delete_runs(removable_runs)

    archive = TUIArchive(
        root=archive_root(),
        db_path=store_path,
        reconcile_pending=False,
    )
    removed_pending = archive.prune_orphan_pending(
        max_age_seconds=days * 24 * 60 * 60,
        now=current,
    )
    removed_archives = (
        archive.prune_conversations_before(cutoff)
        if bool(getattr(args, "include_archives", False))
        else 0
    )
    removed_exports = (
        _prune_generated_exports(cutoff)
        if bool(getattr(args, "include_exports", False))
        else 0
    )

    print(f"Runs removed: {removed_runs}", file=stdout)
    print(f"Orphan pending prompts removed: {removed_pending}", file=stdout)
    print(f"Archived conversations removed: {removed_archives}", file=stdout)
    print(f"Default generated exports removed: {removed_exports}", file=stdout)
    return 0


def run_privacy(args: Any, *, stdout: TextIO = sys.stdout) -> int:
    command = getattr(args, "privacy_command", None)
    if command == "status":
        return run_privacy_status(args, stdout=stdout)
    if command == "prune":
        return run_privacy_prune(args, stdout=stdout)
    return 2
