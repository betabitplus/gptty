from __future__ import annotations

from types import SimpleNamespace

from gptty.ui import notifications


def test_notification_uses_native_macos_osascript(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    assert notifications.notify_response_complete() is True
    assert calls[0][0][0] == "osascript"
    assert "display notification" in calls[0][0][2]
    assert calls[0][1]["check"] is False


def test_notification_is_noop_off_macos(monkeypatch) -> None:
    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert notifications.notify_response_complete() is False


def test_notification_failure_is_best_effort(monkeypatch) -> None:
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )

    assert notifications.notify_response_complete() is False
