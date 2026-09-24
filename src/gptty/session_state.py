from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

from .local_store import LocalEventStore, LocalSessionConflictError, local_store_path
from .state import ChatState, GoalState, StateError, load_chat_state, session_chat_state_path

SESSION_ENV = "GPTTY_SESSION_ID"


class SessionStateError(StateError):
    """Raised when transactional local session state cannot be loaded or saved."""


class SessionStateConflictError(SessionStateError):
    """Raised when a stale session writer would overwrite a newer revision."""


@dataclass
class SessionStateHandle:
    base_state_path: Path
    store: LocalEventStore
    session_id: str
    kind: str
    discovery_hint: str | None
    legacy_path: Path | None
    revision: int = 0
    legacy_goal: GoalState | None = None
    migration_warning: StateError | None = None

    def load(self) -> ChatState:
        try:
            row = self.store.get_session(self.session_id)
            if row is None:
                row, legacy_goal = self._create_seeded_session()
                self.legacy_goal = legacy_goal
            else:
                self.store.touch_session(
                    self.session_id,
                    discovery_hint=self.discovery_hint,
                )
                self.legacy_goal = self._legacy_goal_for_row(row)
                self.migration_warning = None
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            raise SessionStateError(
                f"failed to load local session {self.session_id}: {exc}"
            ) from exc

        self.revision = int(row["revision"])
        return ChatState(
            current_conversation=_optional_str(row.get("current_conversation")),
            model=_optional_str(row.get("model")),
            goal_id=_optional_str(row.get("goal_id")),
            goal=self.legacy_goal,
        )

    def save(self, state: ChatState) -> None:
        goal_id = (
            state.goal.goal_id
            if state.goal is not None and state.goal.goal_id
            else state.goal_id
        )
        try:
            self.revision = self.store.save_session(
                self.session_id,
                expected_revision=self.revision,
                current_conversation=state.current_conversation,
                model=state.model,
                goal_id=goal_id,
                discovery_hint=self.discovery_hint,
            )
        except LocalSessionConflictError as exc:
            raise SessionStateConflictError(str(exc)) from exc
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            raise SessionStateError(
                f"failed to save local session {self.session_id}: {exc}"
            ) from exc
        state.goal_id = goal_id

    def reload(self) -> ChatState:
        return self.load()

    def _create_seeded_session(self) -> tuple[dict[str, object], GoalState | None]:
        if self.legacy_path is not None and self.legacy_path.exists():
            try:
                seed_state = load_chat_state(self.legacy_path)
            except StateError as exc:
                self.migration_warning = exc
            else:
                legacy_goal = seed_state.goal
                goal_id = (
                    legacy_goal.goal_id
                    if legacy_goal is not None and legacy_goal.goal_id
                    else seed_state.goal_id
                )
                row, imported = self.store.create_session_claiming_import(
                    self.session_id,
                    source_path=self.legacy_path,
                    kind=self.kind,
                    discovery_hint=self.discovery_hint,
                    current_conversation=seed_state.current_conversation,
                    model=seed_state.model,
                    goal_id=goal_id,
                )
                if row is not None:
                    return row, legacy_goal if imported else self._legacy_goal_for_row(row)

        seed_state = ChatState()
        legacy_goal: GoalState | None = None
        if self.session_id != "default":
            default, legacy_goal = self._ensure_default_session()
            seed_state = ChatState(
                current_conversation=_optional_str(default.get("current_conversation")),
                model=_optional_str(default.get("model")),
                goal_id=_optional_str(default.get("goal_id")),
            )

        row = self.store.create_session(
            self.session_id,
            kind=self.kind,
            discovery_hint=self.discovery_hint,
            current_conversation=seed_state.current_conversation,
            model=seed_state.model,
            goal_id=seed_state.goal_id,
        )
        return row, legacy_goal

    def _ensure_default_session(self) -> tuple[dict[str, object], GoalState | None]:
        current = self.store.get_session("default")
        if current is not None:
            return current, self._legacy_goal_for_row(current)

        if self.base_state_path.exists():
            try:
                seed = load_chat_state(self.base_state_path)
            except StateError as exc:
                self.migration_warning = exc
            else:
                legacy_goal = seed.goal
                goal_id = (
                    legacy_goal.goal_id
                    if legacy_goal is not None and legacy_goal.goal_id
                    else seed.goal_id
                )
                row, imported = self.store.create_session_claiming_import(
                    "default",
                    source_path=self.base_state_path,
                    kind="default",
                    discovery_hint=None,
                    current_conversation=seed.current_conversation,
                    model=seed.model,
                    goal_id=goal_id,
                )
                if row is not None:
                    return row, legacy_goal if imported else self._legacy_goal_for_row(row)

        row = self.store.create_session(
            "default",
            kind="default",
            discovery_hint=None,
        )
        return row, None

    def _legacy_goal_for_row(self, row: dict[str, object]) -> GoalState | None:
        goal_id = _optional_str(row.get("goal_id"))
        if not goal_id:
            return None
        candidates: list[Path] = []
        imported_from = _optional_str(row.get("imported_from"))
        if imported_from:
            candidates.append(Path(imported_from))
        if self.base_state_path not in candidates:
            candidates.append(self.base_state_path)
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                legacy = load_chat_state(candidate)
            except StateError:
                continue
            goal = legacy.goal
            if goal is not None and goal.goal_id == goal_id:
                return goal
        return None


def session_handle(
    *,
    state_path: str | Path,
    profile: str | None,
    explicit_session: str | None = None,
    interactive: bool = False,
    input_stream: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> SessionStateHandle:
    base = Path(state_path).expanduser()
    env = os.environ if environ is None else environ
    requested = str(explicit_session or env.get(SESSION_ENV) or "").strip()
    hint = _terminal_discovery_hint(input_stream, env) if interactive else None

    if requested:
        digest = hashlib.sha256(requested.encode("utf-8")).hexdigest()[:32]
        session_id = f"explicit-{digest}"
        kind = "explicit"
        discovery_hint = f"explicit:{digest[:16]}"
        legacy_path = _legacy_explicit_session_path(base, requested)
    elif interactive:
        session_id = f"runtime-{uuid.uuid4().hex}"
        kind = "runtime"
        discovery_hint = hint
        legacy_candidate = session_chat_state_path(
            base,
            input_stream=input_stream,
            environ=env,
        )
        legacy_path = legacy_candidate if legacy_candidate != base else None
    else:
        session_id = "default"
        kind = "default"
        discovery_hint = None
        legacy_path = base

    try:
        store = LocalEventStore(local_store_path(profile=profile, state_path=base))
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        raise SessionStateError(f"failed to open local session database: {exc}") from exc
    return SessionStateHandle(
        base_state_path=base,
        store=store,
        session_id=session_id,
        kind=kind,
        discovery_hint=discovery_hint,
        legacy_path=legacy_path,
    )



def session_handle_for_args(
    args: Any,
    *,
    interactive: bool = False,
    input_stream: TextIO | None = None,
) -> SessionStateHandle:
    """Resolve the transactional local session selected by CLI/profile options."""

    return session_handle(
        state_path=Path(getattr(args, "state", "gptty_state.json")),
        profile=getattr(args, "profile", None),
        explicit_session=getattr(args, "session", None),
        interactive=interactive,
        input_stream=input_stream,
    )

def _legacy_explicit_session_path(base: Path, requested: str) -> Path:
    digest = hashlib.sha256(requested.encode("utf-8")).hexdigest()[:12]
    suffix = base.suffix
    stem = base.stem if suffix else base.name
    return base.with_name(f"{stem}.session-custom-{digest}{suffix}")


def _terminal_discovery_hint(
    input_stream: TextIO | None,
    environ: Mapping[str, str],
) -> str | None:
    if input_stream is None:
        return None
    try:
        fd = input_stream.fileno()
        if not os.isatty(fd):
            return None
        tty_name = os.ttyname(fd).strip()
    except (AttributeError, OSError, ValueError):
        return None

    candidates = (
        ("cmux", environ.get("CMUX_SURFACE_ID")),
        ("term", environ.get("TERM_SESSION_ID")),
        ("wezterm", environ.get("WEZTERM_PANE")),
        ("kitty", environ.get("KITTY_WINDOW_ID")),
        ("tty", tty_name),
    )
    for source, value in candidates:
        normalized = str(value or "").strip()
        if normalized:
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
            return f"{source}:{digest}"
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
