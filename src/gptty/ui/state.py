from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class UIStateError(RuntimeError):
    """Raised when interactive UI state cannot be read or written."""


@dataclass
class UISettings:
    pretty: str = "auto"
    markdown: bool = True
    thinking: bool = True
    tools: str = "compact"
    editor: str = "emacs"


@dataclass(frozen=True)
class RecentConversation:
    ref: str
    label: str
    last_used: float


class RecentStore:
    def __init__(self, path: str | Path, *, limit: int = 30) -> None:
        self.path = Path(path)
        self.limit = max(1, int(limit))

    def list(self) -> list[RecentConversation]:
        if not self.path.exists():
            return []
        data = _read_json_object(self.path)
        raw_items = data.get("conversations", [])
        if not isinstance(raw_items, list):
            raise UIStateError(f"failed to load recent conversations from {self.path}: expected list")
        items: list[RecentConversation] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            ref = str(raw.get("ref") or "").strip()
            if not ref:
                continue
            label = str(raw.get("label") or ref).strip() or ref
            try:
                last_used = float(raw.get("last_used") or 0.0)
            except (TypeError, ValueError):
                last_used = 0.0
            items.append(RecentConversation(ref=ref, label=label, last_used=last_used))
        return sorted(items, key=lambda item: item.last_used, reverse=True)[: self.limit]

    def remember(self, ref: str, *, label: str | None = None) -> None:
        normalized = str(ref).strip()
        if not normalized:
            return
        items = self.list()
        existing = next((item for item in items if item.ref == normalized), None)
        chosen_label = (label or (existing.label if existing else normalized)).strip() or normalized
        updated = RecentConversation(ref=normalized, label=chosen_label, last_used=time.time())
        merged = [updated, *(item for item in items if item.ref != normalized)][: self.limit]
        _write_json(
            self.path,
            {"version": 1, "conversations": [asdict(item) for item in merged]},
        )


def ui_settings_path(state_path: str | Path) -> Path:
    return Path(state_path).with_name("ui.json")


def history_path(state_path: str | Path) -> Path:
    return Path(state_path).with_name("history")


def recent_path(state_path: str | Path) -> Path:
    return Path(state_path).with_name("recent_conversations.json")


def load_ui_settings(path: str | Path) -> UISettings:
    settings_path = Path(path)
    if not settings_path.exists():
        return UISettings()
    data = _read_json_object(settings_path)
    pretty = str(data.get("pretty", "auto")).strip().lower()
    tools = str(data.get("tools", "compact")).strip().lower()
    editor = str(data.get("editor", "emacs")).strip().lower()
    if pretty not in {"auto", "on", "off"}:
        pretty = "auto"
    if tools not in {"compact", "hidden"}:
        tools = "compact"
    if editor not in {"emacs", "vi"}:
        editor = "emacs"
    return UISettings(
        pretty=pretty,
        markdown=bool(data.get("markdown", True)),
        thinking=bool(data.get("thinking", True)),
        tools=tools,
        editor=editor,
    )


def save_ui_settings(path: str | Path, settings: UISettings) -> None:
    _write_json(Path(path), {"version": 1, **asdict(settings)})


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UIStateError(f"failed to read UI state from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise UIStateError(f"failed to read UI state from {path}: expected JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise UIStateError(f"failed to write UI state to {path}: {exc}") from exc
