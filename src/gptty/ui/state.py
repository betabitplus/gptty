from __future__ import annotations

import json
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
    history_limit: int = 2_000
    notifications: bool = True
    notification_preview: bool = False
    notification_sound: bool = True


def ui_settings_path(state_path: str | Path) -> Path:
    return Path(state_path).with_name("ui.json")


def history_path(state_path: str | Path) -> Path:
    return Path(state_path).with_name("history")


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
    history_limit = data.get("history_limit", 2_000)
    if isinstance(history_limit, bool) or not isinstance(history_limit, int):
        history_limit = 2_000
    history_limit = min(100_000, max(0, history_limit))
    notifications = data.get("notifications", True)
    if not isinstance(notifications, bool):
        notifications = True
    notification_preview = data.get("notification_preview", False)
    if not isinstance(notification_preview, bool):
        notification_preview = False
    notification_sound = data.get("notification_sound", True)
    if not isinstance(notification_sound, bool):
        notification_sound = True
    return UISettings(
        pretty=pretty,
        markdown=bool(data.get("markdown", True)),
        thinking=bool(data.get("thinking", True)),
        tools=tools,
        editor=editor,
        history_limit=history_limit,
        notifications=notifications,
        notification_preview=notification_preview,
        notification_sound=notification_sound,
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
