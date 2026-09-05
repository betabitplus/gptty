from __future__ import annotations

import shlex
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..commands.export import save_markdown_export
from ..media import MediaInputError, normalize_media_input
from ..output import OutputMessage, normalize_messages
from ..state import ChatState, StateError, save_chat_state
from .clipboard import ClipboardImageError, capture_clipboard_image
from .notifications import notify_response_complete
from .renderer import PrettyRenderer
from .session import InteractiveSession
from .signals import turn_control_signals

FOLLOW_INTERVAL_SECONDS = 15.0
FOLLOW_TIMEOUT_SECONDS = 2 * 60 * 60
ACTIVE_STATUSES = {
    "running",
    "streaming",
    "tool_running",
    "tool_calling",
    "user_last_message",
}


class InteractiveCommands:
    def __init__(
        self,
        *,
        state: ChatState,
        state_path: Path,
        get_client: Callable[[], Any],
        ui: InteractiveSession,
        renderer: PrettyRenderer,
    ) -> None:
        self.state = state
        self.state_path = state_path
        self.get_client = get_client
        self.ui = ui
        self.renderer = renderer
        self._pending_media: list[str] = []
        self._owned_media: set[Path] = set()
        self._clipboard_dir: Path | None = None
        self._conversation_titles: dict[str, str] = {}
        self._conversation_mode = "normal"
        self._temporary_conversation: str | None = None
        self._temporary_messages: list[OutputMessage] = []
        self._temporary_title: str | None = None

    def handle(self, raw: str) -> int | None:
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            self.renderer.warning(f"Invalid command: {exc}")
            return None
        if not parts:
            return None
        name = parts[0].lstrip("/").lower()
        method = getattr(self, f"_cmd_{name}", None)
        if not callable(method):
            self.renderer.warning(f"Unknown command: /{name}. Press / for actions.")
            return None
        return method(parts[1:])

    @property
    def conversation_mode(self) -> str:
        return self._conversation_mode

    @property
    def conversation_ref(self) -> str | None:
        if self._conversation_mode == "temporary":
            return self._temporary_conversation
        return self.state.current_conversation

    @property
    def pending_media(self) -> list[str]:
        return list(self._pending_media)

    @property
    def pending_media_count(self) -> int:
        return len(self._pending_media)

    def record_temporary_turn(
        self,
        *,
        prompt: str,
        answer: str,
        conversation_ref: str | None,
        title: str | None,
    ) -> None:
        if self._conversation_mode != "temporary":
            return
        if conversation_ref:
            self._temporary_conversation = conversation_ref
        if title:
            self._temporary_title = title
        self._temporary_messages.append(OutputMessage(role="user", text=prompt))
        if answer:
            self._temporary_messages.append(OutputMessage(role="assistant", text=answer))

    def clear_pending_media(self) -> None:
        self._pending_media.clear()
        for path in list(self._owned_media):
            path.unlink(missing_ok=True)
        self._owned_media.clear()
        if self._clipboard_dir is not None:
            shutil.rmtree(self._clipboard_dir, ignore_errors=True)
            self._clipboard_dir = None

    def close(self) -> None:
        self.clear_pending_media()

    def _reset_temporary_context(self) -> None:
        self._temporary_conversation = None
        self._temporary_messages.clear()
        self._temporary_title = None

    def _leave_temporary_mode(self) -> None:
        if self._conversation_mode != "temporary":
            return
        try:
            client = self.get_client()
            snapshot = client.temporary_lifecycle_snapshot()
            if snapshot.get("state") == "LIVE":
                client.end_temporary_chat()
        except Exception as exc:  # noqa: BLE001 - context switch must remain usable.
            self.renderer.warning(f"Temporary chat cleanup failed: {exc}")
        self._conversation_mode = "normal"
        self._reset_temporary_context()

    def _cmd_exit(self, argv: list[str]) -> int:
        self._leave_temporary_mode()
        self.close()
        return 0

    def _cmd_new(self, argv: list[str]) -> None:
        self._leave_temporary_mode()
        previous = self.state.current_conversation
        self.state.current_conversation = None
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(model=self.state.model or "latest frontier · High")
        self.renderer.info("Started a new conversation.")

    def _cmd_temporary(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning("/temporary takes no arguments.")
            return
        self._leave_temporary_mode()
        previous = self.state.current_conversation
        self.state.current_conversation = None
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self._conversation_mode = "temporary"
        self._reset_temporary_context()
        self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(model=self.state.model or "latest frontier · High", temporary=True)
        self.renderer.info("Started a new Temporary ChatGPT conversation.")

    def _cmd_temp(self, argv: list[str]) -> None:
        self._cmd_temporary(argv)

    def _cmd_detach(self, argv: list[str]) -> None:
        if self._conversation_mode == "temporary":
            self._leave_temporary_mode()
            self.clear_pending_media()
            self.renderer.clear_context()
            self.renderer.header(model=self.state.model or "latest frontier · High")
            self.renderer.info("Detached from the Temporary ChatGPT conversation.")
            return
        if not self.state.current_conversation:
            self.renderer.info("No conversation is attached.")
            return
        previous = self.state.current_conversation
        self.state.current_conversation = None
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(model=self.state.model or "latest frontier · High")
        self.renderer.info("Detached locally. The ChatGPT conversation was not changed.")

    def _cmd_stop(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning("/stop takes no arguments.")
            return
        ref = self.conversation_ref
        if not ref:
            self.renderer.info("No conversation is attached.")
            return
        client = self.get_client()
        if self._request_stop_generation(client, ref):
            self.renderer.turn_abort()
            self.renderer.info("Stop requested.")

    def _cmd_export(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning("/export takes no arguments.")
            return
        ref = self.conversation_ref
        if not ref:
            self.renderer.info("No conversation is attached.")
            return
        title = self._temporary_title if self._conversation_mode == "temporary" else self._conversation_titles.get(ref)
        try:
            if self._conversation_mode == "temporary":
                messages = list(self._temporary_messages)
            else:
                messages = normalize_messages(self.get_client().get_messages(ref))
            path = save_markdown_export(messages, title=title)
        except Exception as exc:  # noqa: BLE001 - interactive export boundary.
            self.renderer.warning(f"Export failed: {exc}")
            return
        self.renderer.info(f"Exported Markdown: {path}")

    def _request_stop_generation(self, client: Any, ref: str) -> bool:
        try:
            result = client.stop_generation(ref, timeout=30.0)
        except Exception as exc:  # noqa: BLE001 - interactive command boundary.
            self.renderer.warning(f"Stop failed: {exc}")
            return False
        stopped = (
            bool(result.get("stopped"))
            if isinstance(result, dict)
            else bool(getattr(result, "stopped", False))
        )
        if not stopped:
            self.renderer.info("No active ChatGPT response to stop.")
            return False
        return True

    def _cmd_image(self, argv: list[str]) -> None:
        if argv and argv[0].strip().lower() == "clear":
            count = self.pending_media_count
            self.clear_pending_media()
            self.renderer.info(f"Cleared {count} pending image{'s' if count != 1 else ''}.")
            return

        raw = " ".join(argv).strip() if argv else self.ui.read_image_path()
        if not raw:
            return
        if not argv:
            try:
                parsed = shlex.split(raw)
            except ValueError as exc:
                self.renderer.warning(f"Invalid image path: {exc}")
                return
            raw = " ".join(parsed)
        try:
            media = normalize_media_input(raw)
        except MediaInputError as exc:
            self.renderer.warning(str(exc))
            return
        if media not in self._pending_media:
            self._pending_media.append(media)
        self.renderer.info(f"Attached for next prompt: {Path(media).name or media} · pending: {self.pending_media_count}")

    def _cmd_paste(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning("/paste takes no arguments; it attaches the current clipboard image.")
            return
        if self._clipboard_dir is None:
            self._clipboard_dir = Path(tempfile.mkdtemp(prefix="gptty-clipboard-"))
        try:
            path = capture_clipboard_image(self._clipboard_dir)
        except ClipboardImageError as exc:
            self.renderer.warning(str(exc))
            return
        self._owned_media.add(path)
        self._pending_media.append(str(path))
        self.renderer.info(f"Attached clipboard image for next prompt · pending: {self.pending_media_count}")

    def _cmd_resume(self, argv: list[str]) -> int | None:
        client = self.get_client()
        ref = argv[0] if argv else self._choose_conversation(client)
        if not ref:
            return

        attached_ref = _canonical_conversation_ref(str(ref))
        self._leave_temporary_mode()
        try:
            snapshot = client.conversation_snapshot(attached_ref)
        except Exception as exc:  # noqa: BLE001 - interactive command boundary.
            self.renderer.warning(f"Resume failed: {exc}")
            return

        previous = self.state.current_conversation
        self.state.current_conversation = attached_ref
        if not self._save_state():
            self.state.current_conversation = previous
            return

        self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(
            conversation=attached_ref,
            model=self.state.model or "latest frontier · High",
        )
        self.renderer.info(f"Resumed: {_short_ref(attached_ref)}")
        messages = _snapshot_messages(snapshot)
        self.renderer.messages(normalize_messages(messages))
        if self._follow_if_active(
            client,
            attached_ref,
            snapshot,
            messages,
            chat_title=self._conversation_titles.get(attached_ref),
        ):
            return 0
        return None

    def _choose_conversation(self, client: Any) -> str | None:
        try:
            conversations = client.list_conversations()
        except Exception as exc:  # noqa: BLE001 - interactive command boundary.
            self.renderer.warning(f"Conversation list failed: {exc}")
            return None
        current_ref = _canonical_conversation_ref(self.state.current_conversation or "")
        options: list[tuple[str, str]] = []
        for item in conversations:
            conversation_id = _catalog_conversation_id(item)
            if conversation_id is None:
                continue
            title = _field_text(item, "title")
            if title:
                self._conversation_titles[conversation_id] = title
            options.append(
                (
                    conversation_id,
                    _conversation_label(
                        item,
                        current=conversation_id == current_ref,
                    ),
                )
            )
        if not options:
            self.renderer.info("No ChatGPT conversations found.")
            return None
        selected = self.ui.choose_searchable(
            "Resume conversation",
            options,
        )
        return str(selected) if selected else None

    def _cmd_model(self, argv: list[str]) -> None:
        if argv and argv[0].strip().lower() == "default":
            previous = self.state.model
            self.state.model = None
            if not self._save_state():
                self.state.model = previous
                return
            self.renderer.info("Model: latest frontier · High")
            return

        try:
            models = self.get_client().list_models()
        except Exception as exc:  # noqa: BLE001 - interactive command boundary.
            self.renderer.warning(f"Model list failed: {exc}")
            return

        available = [model for model in models if _model_slug(model) is not None and _model_available(model)]
        by_slug = {_model_slug(model): model for model in available}
        by_slug = {slug: model for slug, model in by_slug.items() if slug is not None}

        if argv:
            selected = argv[0].strip()
            if selected not in by_slug:
                self.renderer.warning("Unknown model slug. Run /model and choose from the live ChatGPT list.")
                return
        else:
            options: list[tuple[Any, str]] = [
                (
                    "",
                    "Default · latest frontier · High"
                    + (" · current" if self.state.model is None else ""),
                )
            ]
            options.extend(
                (
                    slug,
                    _model_label(model, current=slug == self.state.model),
                )
                for slug, model in by_slug.items()
            )
            value = self.ui.choose_searchable("ChatGPT model", options)
            if value is None:
                return
            selected = str(value)

        previous = self.state.model
        self.state.model = selected or None
        if not self._save_state():
            self.state.model = previous
            return
        self.renderer.info(f"Model: {self.state.model or 'latest frontier · High'}")

    def _follow_if_active(
        self,
        client: Any,
        ref: str,
        snapshot: Any,
        messages: list[Any],
        *,
        chat_title: str | None = None,
    ) -> bool:
        status = _snapshot_status(snapshot)
        if status == "awaiting_tool_approval":
            self.renderer.warning("Conversation is waiting for tool approval.")
            return False
        if status not in ACTIVE_STATUSES:
            return False

        seen = {_message_identity(message): _message_text(message) for message in messages}
        deadline = time.monotonic() + FOLLOW_TIMEOUT_SECONDS
        stopped_by_user = False
        with turn_control_signals(enabled=True) as controls:
            self.renderer.info("Following active response… Ctrl-C stops ChatGPT · Ctrl-\\ exits gptty only.")
            self.renderer.start_elapsed(initial_elapsed=_active_elapsed_seconds(messages))
            while time.monotonic() < deadline:
                woke = controls.wake.wait(timeout=FOLLOW_INTERVAL_SECONDS)
                controls.wake.clear()
                if controls.quit_requested.is_set():
                    self.renderer.turn_abort()
                    self.renderer.info("Exited gptty; ChatGPT response continues in browser.")
                    return True
                if controls.stop_requested.is_set():
                    controls.stop_requested.clear()
                    if stopped_by_user:
                        self.renderer.info("Stop already requested; waiting for ChatGPT to save the partial response…")
                        continue
                    self.renderer.info("Stopping ChatGPT…")
                    if self._request_stop_generation(client, ref):
                        stopped_by_user = True
                        self.renderer.turn_abort()
                        self.renderer.info("Stop requested; waiting for ChatGPT to save the partial response…")
                        time.sleep(1.0)
                    else:
                        continue
                elif woke:
                    continue

                snapshot = client.conversation_snapshot(ref)
                current = _snapshot_messages(snapshot)
                changed: list[Any] = []
                for message in current:
                    identity = _message_identity(message)
                    text = _message_text(message)
                    if seen.get(identity) == text:
                        continue
                    seen[identity] = text
                    changed.append(message)
                if changed:
                    self.renderer.messages(normalize_messages(changed))

                status = _snapshot_status(snapshot)
                if status == "completed":
                    self.renderer.finish_elapsed()
                    if stopped_by_user:
                        self.renderer.info("Stopped by user.")
                    self.renderer.chat_link(ref)
                    if not stopped_by_user:
                        notify_response_complete(
                            chat_title=chat_title,
                            prompt=_last_user_message_text(current),
                        )
                    return False
                if status == "awaiting_tool_approval":
                    self.renderer.turn_abort()
                    self.renderer.warning("Conversation is waiting for tool approval.")
                    return False
                if status not in ACTIVE_STATUSES:
                    self.renderer.turn_abort()
                    self.renderer.info(f"Follow stopped: status={status or 'unknown'}")
                    return False
        self.renderer.turn_abort()
        self.renderer.info("Stopped following after 2 hours; conversation remains attached.")
        return False

    def _save_state(self) -> bool:
        try:
            save_chat_state(self.state_path, self.state)
        except StateError as exc:
            self.renderer.warning(str(exc))
            return False
        return True


def _canonical_conversation_ref(value: str) -> str:
    raw = value.strip()
    if not raw:
        return raw
    parsed = urlparse(raw)
    supported_hosts = {"chatgpt.com", "www.chatgpt.com", "chat.openai.com"}
    if parsed.scheme in {"http", "https"} and parsed.hostname in supported_hosts:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "c":
            return parts[1]
    return raw


def _catalog_conversation_id(item: Any) -> str | None:
    value = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _conversation_label(item: Any, *, current: bool = False) -> str:
    conversation_id = _catalog_conversation_id(item) or "unknown"
    title = _field_text(item, "title") or "Untitled"
    updated = _format_update_time(_field(item, "update_time"))
    starred = "★ " if _field(item, "is_starred") is True else ""
    archived = " · archived" if _field(item, "is_archived") is True else ""
    updated_part = f" · {updated}" if updated else ""
    current_part = " · current" if current else ""
    return f"{starred}{title}{updated_part}{archived}{current_part} · {_short_ref(conversation_id, max_len=18)}"


def _format_update_time(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))
        except (OverflowError, OSError, ValueError):
            return ""
    if isinstance(value, str):
        return value.replace("T", " ")[:16]
    return ""


def _model_slug(model: Any) -> str | None:
    value = _field(model, "slug")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _model_available(model: Any) -> bool:
    if _field(model, "enabled") is False:
        return False
    if _field(model, "is_disabled") is True:
        return False
    if _field(model, "is_work_mode_model") is True:
        return False
    return _model_slug(model) != "research"


def _model_label(model: Any, *, current: bool = False) -> str:
    slug = _model_slug(model) or "unknown"
    title = (
        _field_text(model, "title")
        or _field_text(model, "display_name")
        or _field_text(model, "name")
        or slug
    )
    label = title if title == slug else f"{title} · {slug}"
    return f"{label} · current" if current else label


def _snapshot_status(snapshot: Any) -> str:
    status = _field(snapshot, "status")
    if status is None and isinstance(snapshot, dict):
        status = snapshot.get("status")
    value = _field(status, "status") if status is not None else None
    if value is None and isinstance(status, str):
        value = status
    return str(value).strip().lower() if value else ""


def _snapshot_messages(snapshot: Any) -> list[Any]:
    messages = _field(snapshot, "messages")
    if isinstance(messages, list):
        items = messages
    elif messages is not None:
        try:
            items = list(messages)
        except TypeError:
            items = []
    else:
        items = []
    return [message for message in items if _message_is_user_visible(message)]


def _message_is_user_visible(message: Any) -> bool:
    role = _field_text(message, "role")
    if role == "user":
        return True
    if role != "assistant":
        return False
    recipient = _field(message, "recipient")
    return recipient in {None, "", "all"}


def _message_identity(message: Any) -> str:
    for field in ("message_id", "id", "node_id"):
        value = _field(message, field)
        if value:
            return str(value)
    return f"{_field_text(message, 'role')}:{_message_text(message)}"


def _message_text(message: Any) -> str:
    for field in ("text", "content", "message"):
        value = _field(message, field)
        if value is not None:
            return str(value)
    return ""


def _last_user_message_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if _field_text(message, "role") == "user":
            return _message_text(message)
    return ""


def _active_elapsed_seconds(messages: list[Any]) -> float:
    for message in reversed(messages):
        if _field_text(message, "role") != "user":
            continue
        created = _field(message, "create_time")
        if isinstance(created, (int, float)) and not isinstance(created, bool):
            return max(0.0, time.time() - float(created))
    return 0.0


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _field_text(value: Any, name: str) -> str:
    raw = _field(value, name)
    return raw.strip() if isinstance(raw, str) else ""


def _short_ref(ref: str, *, max_len: int = 44) -> str:
    if len(ref) <= max_len:
        return ref
    return f"…{ref[-(max_len - 1):]}"
