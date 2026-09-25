from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import pytest

from gptty.ui import notifications


class _FakeDispatcher:
    def __init__(self, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls: list[tuple[str, ...]] = []

    def submit(self, argv: tuple[str, ...]) -> bool:
        self.calls.append(argv)
        return self.accepted


@pytest.fixture(autouse=True)
def _reset_notification_policy():
    notifications.configure_notifications(None)
    yield
    notifications.configure_notifications(None)


def test_notification_default_is_generic_and_uses_sound(monkeypatch) -> None:
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_dispatcher", dispatcher)

    assert notifications.notify_response_complete(
        chat_title="  Sensitive   Project  ",
        final_response='  secret\nanswer with "context"  ',
    ) is True

    argv = dispatcher.calls[0]
    assert argv[:3] == ("osascript", "-e", notifications._NOTIFICATION_SCRIPT)
    assert argv[3] == "ChatGPT response complete."
    assert argv[4] == "ChatGPT"
    assert argv[5] == "1"


def test_notification_preview_is_explicit_opt_in_and_sound_can_be_disabled(monkeypatch) -> None:
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_dispatcher", dispatcher)
    notifications.configure_notifications(
        SimpleNamespace(
            notifications=True,
            notification_preview=True,
            notification_sound=False,
        )
    )

    assert notifications.notify_response_complete(
        chat_title="  Inspect   Image Bands  ",
        final_response='  This is   the final\nanswer with "context"  ',
    ) is True

    argv = dispatcher.calls[0]
    assert argv[3] == 'This is the final answer with "context"'
    assert argv[4] == "Inspect Image Bands"
    assert argv[5] == "0"


def test_temporary_private_notification_never_previews_even_when_opted_in(monkeypatch) -> None:
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_dispatcher", dispatcher)
    notifications.configure_notifications(
        SimpleNamespace(
            notifications=True,
            notification_preview=True,
            notification_sound=True,
        )
    )

    assert notifications.notify_response_complete(
        chat_title="Temporary secret title",
        final_response="Temporary secret answer",
        private=True,
    ) is True

    argv = dispatcher.calls[0]
    assert argv[3] == "ChatGPT response complete."
    assert argv[4] == "ChatGPT"


def test_notification_can_be_disabled(monkeypatch) -> None:
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications, "_dispatcher", dispatcher)
    notifications.configure_notifications(
        SimpleNamespace(
            notifications=False,
            notification_preview=True,
            notification_sound=True,
        )
    )

    assert notifications.notify_response_complete(
        chat_title="Example Chat",
        final_response="hello",
    ) is False
    assert dispatcher.calls == []


def test_notification_response_preview_is_bounded() -> None:
    body = notifications._notification_response("x" * 400, preview=True)

    assert len(body) == 240
    assert body.endswith("…")


def test_notification_title_preview_is_bounded_and_never_falls_back_to_prompt() -> None:
    assert notifications._notification_title(None, preview=True) == "ChatGPT"
    title = notifications._notification_title("word " * 30, preview=True)
    assert len(title) <= 64
    assert title.endswith("…")


def test_notification_is_noop_off_macos(monkeypatch) -> None:
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(notifications, "_dispatcher", dispatcher)

    assert notifications.notify_response_complete(
        chat_title="Example Chat",
        final_response="hello",
    ) is False
    assert dispatcher.calls == []


def test_notification_worker_failure_is_best_effort(monkeypatch) -> None:
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )

    notifications._run_notification(("osascript", "-e", "script", "body", "title", "1"))


def test_notification_worker_uses_short_timeout(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    notifications._run_notification(("osascript", "-e", "script", "body", "title", "1"))

    assert calls[0][1]["timeout"] == 2
    assert calls[0][1]["check"] is False


def test_notification_dispatcher_survives_unexpected_worker_error(monkeypatch) -> None:
    attempts: list[tuple[str, ...]] = []
    completed = threading.Event()

    def flaky_run(argv: tuple[str, ...]) -> None:
        attempts.append(argv)
        if len(attempts) == 1:
            raise RuntimeError("unexpected provider failure")
        completed.set()

    monkeypatch.setattr(notifications, "_run_notification", flaky_run)
    dispatcher = notifications._NotificationDispatcher(max_pending=2)

    assert dispatcher.submit(("first",)) is True
    assert dispatcher.submit(("second",)) is True

    assert completed.wait(timeout=1)
    dispatcher._queue.join()
    assert attempts == [("first",), ("second",)]


def test_notification_dispatcher_is_non_blocking_and_bounded(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()
    processed: list[tuple[str, ...]] = []

    def fake_run(argv: tuple[str, ...]) -> None:
        processed.append(argv)
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(notifications, "_run_notification", fake_run)
    dispatcher = notifications._NotificationDispatcher(max_pending=1)

    before = time.monotonic()
    assert dispatcher.submit(("first",)) is True
    assert time.monotonic() - before < 0.1
    assert started.wait(timeout=1)

    assert dispatcher.submit(("second",)) is True
    assert dispatcher.submit(("third",)) is False

    release.set()
    dispatcher._queue.join()
    assert processed == [("first",), ("second",)]
