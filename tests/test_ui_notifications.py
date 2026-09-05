from __future__ import annotations

from types import SimpleNamespace

from gptty.ui import notifications


def test_notification_uses_chat_title_last_prompt_and_sound(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    assert notifications.notify_response_complete(
        chat_title="  Inspect   Image Bands  ",
        prompt='  Inspect   "this"\nimage please  ',
    ) is True

    argv, kwargs = calls[0]
    assert argv[0] == "osascript"
    assert "display notification" in argv[2]
    assert 'sound name "Glass"' in argv[2]
    assert argv[3] == 'Inspect "this" image please'
    assert argv[4] == "Inspect Image Bands"
    assert len(argv) == 5
    assert kwargs["check"] is False


def test_notification_prompt_is_bounded() -> None:
    body = notifications._notification_prompt("x" * 400)

    assert len(body) == 180
    assert body.endswith("…")


def test_notification_is_noop_off_macos(monkeypatch) -> None:
    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert notifications.notify_response_complete(chat_title="Example Chat", prompt="hello") is False


def test_notification_failure_is_best_effort(monkeypatch) -> None:
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )

    assert notifications.notify_response_complete(chat_title="Example Chat", prompt="hello") is False
