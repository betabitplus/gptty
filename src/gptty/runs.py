from __future__ import annotations

import json
import os
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .local_store import DB_FILENAME, LocalEventStore, local_store_root


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_file: Path
    events_file: Path
    store_file: Path


class RunRecorder:
    def __init__(
        self,
        paths: RunPaths,
        summary: dict[str, Any],
        store: LocalEventStore,
    ) -> None:
        self.paths = paths
        self.summary = summary
        self.store = store

    @property
    def run_id(self) -> str:
        return self.paths.run_id

    @property
    def run_file(self) -> Path:
        return self.paths.run_file

    @property
    def events_file(self) -> Path:
        return self.paths.events_file

    @property
    def store_file(self) -> Path:
        return self.paths.store_file

    def event(self, event_type: str, **data: Any) -> None:
        event = {
            "type": event_type,
            "timestamp": utc_now(),
            **data,
        }
        self.summary["last_event"] = event_type
        self.summary["updated_at"] = event["timestamp"]
        self.store.append_run_event(
            run_id=self.run_id,
            summary=self.summary,
            event=event,
        )
        self._project(event)

    def complete(self) -> None:
        self.summary["status"] = "completed"
        self.summary["completed_at"] = utc_now()
        self.event("completed")

    def fail(self, message: str, *, traceback_text: str | None = None) -> None:
        self.summary["status"] = "failed"
        self.summary["error"] = message
        if isinstance(traceback_text, str) and traceback_text.strip():
            self.summary["traceback"] = traceback_text
        self.summary["completed_at"] = utc_now()
        event_data: dict[str, Any] = {"message": message}
        if isinstance(traceback_text, str) and traceback_text.strip():
            event_data["traceback"] = traceback_text
        self.event("failed", **event_data)

    def _project(self, event: dict[str, Any]) -> None:
        try:
            _append_event_projection(self.events_file, event)
            write_run_summary(self.run_file, self.summary)
        except OSError as exc:
            self.summary.setdefault(
                "projection_error",
                f"{type(exc).__name__}: {exc}",
            )
            try:
                self.store.replace_run_summary(
                    self.run_id,
                    self.summary,
                    updated_at=str(self.summary.get("updated_at") or event["timestamp"]),
                )
            except Exception:
                # The event transaction already committed. A secondary failure
                # while annotating the derived projection must not rewrite turn
                # outcome or fabricate a failed ChatGPT operation.
                pass


def run_dir(*, profile: str | None, state_path: str | Path) -> Path:
    return local_store_root(profile=profile, state_path=state_path)


def start_run(
    *,
    profile: str | None,
    state_path: str | Path,
    command: str,
    conversation_ref: str,
) -> RunRecorder:
    root = run_dir(profile=profile, state_path=state_path)
    run_id = uuid.uuid4().hex
    paths = RunPaths(
        run_id=run_id,
        run_file=root / f"{run_id}.json",
        events_file=root / f"{run_id}.jsonl",
        store_file=root / DB_FILENAME,
    )
    started_at = utc_now()
    first_event = {
        "type": "run_started",
        "timestamp": started_at,
    }
    summary: dict[str, Any] = {
        "run_id": run_id,
        "profile": profile,
        "command": command,
        "conversation_ref": conversation_ref,
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "last_event": "run_started",
        "events_file": str(paths.events_file),
        "store_file": str(paths.store_file),
    }
    store = LocalEventStore(paths.store_file)
    store.create_run(
        run_id=run_id,
        summary=summary,
        first_event=first_event,
    )
    recorder = RunRecorder(paths, summary, store)
    recorder._project(first_event)
    return recorder


def write_run_summary(path: str | Path, summary: dict[str, Any]) -> None:
    run_path = Path(path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = run_path.with_name(
        f".{run_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    payload = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        if os.name != "nt":
            tmp_path.chmod(0o600)
        os.replace(tmp_path, run_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_run_summary(path: str | Path) -> dict[str, Any]:
    run_path = Path(path)
    store = _store_for_run_path(run_path)
    if store is not None:
        summary = store.run_summary(run_path.stem)
        if summary is not None:
            return summary

    try:
        data = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_run_events(
    path: str | Path,
    *,
    from_start: bool = False,
) -> list[dict[str, Any]]:
    events_path = Path(path)
    store = _store_for_run_path(events_path)
    if store is not None and store.run_summary(events_path.stem) is not None:
        return store.run_events(
            events_path.stem,
            limit=None if from_start else 20,
        )

    return _read_legacy_run_events(events_path, from_start=from_start)


def _store_for_run_path(path: Path) -> LocalEventStore | None:
    store_path = path.parent / DB_FILENAME
    if not store_path.is_file():
        return None
    return LocalEventStore(store_path)


def _read_legacy_run_events(
    path: Path,
    *,
    from_start: bool,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] | deque[dict[str, Any]]
    events = [] if from_start else deque(maxlen=20)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return []
    return list(events)


def _append_event_projection(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short run projection write: {written}/{len(view)} bytes")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def render_run_status(
    *,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    stdout: TextIO,
    status_only: bool = False,
) -> None:
    conversation = _optional_str(summary.get("conversation_ref")) or "unknown"
    profile = _optional_str(summary.get("profile")) or "local files"
    elapsed = format_elapsed(_optional_str(summary.get("started_at")))
    status = _optional_str(summary.get("status")) or "unknown"
    last_event = _optional_str(summary.get("last_event")) or "unknown"

    if status == "running":
        print("gptty: conversation in progress", file=stdout)
    elif status == "failed":
        print("gptty: previous command failed", file=stdout)
    else:
        print("gptty: conversation run status", file=stdout)
    print(file=stdout)
    print(f"Profile: {profile}", file=stdout)
    print(f"Conversation: {conversation}", file=stdout)
    print(f"Elapsed: {elapsed}", file=stdout)
    print(f"Status: {status}", file=stdout)
    print(f"Last event: {last_event}", file=stdout)

    if status_only:
        return

    token_text = "".join(
        str(event.get("text", ""))
        for event in events
        if event.get("type") == "token_delta"
    )
    required_action = next(
        (
            event
            for event in reversed(events)
            if event.get("type") == "required_action"
        ),
        None,
    )
    failure = next(
        (event for event in reversed(events) if event.get("type") == "failed"),
        None,
    )

    if token_text:
        print(file=stdout)
        print("Assistant:", file=stdout)
        print(token_text, file=stdout)
    elif required_action:
        print(file=stdout)
        print("Action needed:", file=stdout)
        print(
            str(
                required_action.get("message")
                or "ChatGPT is waiting for a web UI action."
            ),
            file=stdout,
        )
    elif failure:
        print(file=stdout)
        print(str(failure.get("message") or "The command failed."), file=stdout)
    elif status == "running":
        print(file=stdout)
        print("Waiting for ChatGPT...", file=stdout)


def format_elapsed(started_at: str | None) -> str:
    started = parse_time(started_at)
    if started is None:
        return "unknown"
    seconds = max(
        0,
        int(datetime.now(timezone.utc).timestamp() - started.timestamp()),
    )
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
