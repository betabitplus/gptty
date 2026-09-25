from __future__ import annotations

from dataclasses import dataclass
import queue
import subprocess
import sys
import threading
from typing import Any


_NOTIFICATION_SCRIPT = r'''
on run argv
    set notificationBody to item 1 of argv
    set notificationTitle to item 2 of argv
    set notificationSound to item 3 of argv
    if notificationSound is "1" then
        display notification notificationBody with title notificationTitle sound name "Glass"
    else
        display notification notificationBody with title notificationTitle
    end if
end run
'''.strip()

_GENERIC_TITLE = "ChatGPT"
_GENERIC_BODY = "ChatGPT response complete."


@dataclass(frozen=True)
class NotificationPolicy:
    enabled: bool = True
    preview: bool = False
    sound: bool = True


_policy = NotificationPolicy()


def configure_notifications(settings: Any | None) -> NotificationPolicy:
    """Configure per-process interactive notification privacy from UI settings."""

    global _policy
    _policy = NotificationPolicy(
        enabled=bool(getattr(settings, "notifications", True)),
        preview=bool(getattr(settings, "notification_preview", False)),
        sound=bool(getattr(settings, "notification_sound", True)),
    )
    return _policy


class _NotificationDispatcher:
    """One daemon worker with a bounded queue so notifications never block the TUI."""

    def __init__(self, *, max_pending: int = 2) -> None:
        self._queue: queue.Queue[tuple[str, ...]] = queue.Queue(maxsize=max_pending)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, argv: tuple[str, ...]) -> bool:
        self._ensure_started()
        try:
            self._queue.put_nowait(argv)
        except queue.Full:
            return False
        return True

    def _ensure_started(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        with self._lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker,
                name="gptty-notifications",
                daemon=True,
            )
            self._thread.start()

    def _worker(self) -> None:
        while True:
            argv = self._queue.get()
            try:
                _run_notification(argv)
            except Exception:  # noqa: BLE001 - notification delivery is best-effort.
                pass
            finally:
                self._queue.task_done()


_dispatcher = _NotificationDispatcher()


def notify_response_complete(
    *,
    chat_title: str | None = None,
    final_response: str | None = None,
    private: bool = False,
) -> bool:
    """Schedule a best-effort native completion notification without blocking."""

    if sys.platform != "darwin" or not _policy.enabled:
        return False

    preview = _policy.preview and not private
    title = _notification_title(chat_title, preview=preview)
    body = _notification_response(final_response, preview=preview)
    command = (
        "osascript",
        "-e",
        _NOTIFICATION_SCRIPT,
        body,
        title,
        "1" if _policy.sound else "0",
    )
    return _dispatcher.submit(command)


def _run_notification(argv: tuple[str, ...]) -> None:
    try:
        subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return


def _notification_title(
    chat_title: str | None,
    *,
    preview: bool = False,
    max_chars: int = 64,
) -> str:
    if not preview:
        return _GENERIC_TITLE
    text = " ".join(str(chat_title or "").split()) or _GENERIC_TITLE
    return _bounded_preview(text, max_chars=max_chars)


def _notification_response(
    final_response: str | None,
    *,
    preview: bool = False,
    max_chars: int = 240,
) -> str:
    if not preview:
        return _GENERIC_BODY
    if not final_response:
        return _GENERIC_BODY
    text = " ".join(str(final_response).split())
    if not text:
        return _GENERIC_BODY
    return _bounded_preview(text, max_chars=max_chars)


def _bounded_preview(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 1].rstrip()
    word_boundary = clipped.rfind(" ")
    if word_boundary >= max_chars // 2:
        clipped = clipped[:word_boundary].rstrip()
    return f"{clipped}…"
