from __future__ import annotations

from types import SimpleNamespace

from gptty.ui import notifications


def test_notification_uses_chat_title_final_response_and_sound(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(notifications.subprocess, "run", fake_run)

    assert notifications.notify_response_complete(
        chat_title="  Inspect   Image Bands  ",
        final_response='  This is   the final\nanswer with "context"  ',
    ) is True

    argv, kwargs = calls[0]
    assert argv[0] == "osascript"
    assert "display notification" in argv[2]
    assert 'sound name "Glass"' in argv[2]
    assert argv[3] == 'This is the final answer with "context"'
    assert argv[4] == "Inspect Image Bands"
    assert len(argv) == 5
    assert kwargs["check"] is False


def test_notification_response_is_bounded() -> None:
    body = notifications._notification_response("x" * 400)

    assert len(body) == 160
    assert body.endswith("…")


def test_notification_title_is_bounded_and_never_falls_back_to_prompt() -> None:
    assert notifications._notification_title(None) == "ChatGPT"
    title = notifications._notification_title("word " * 30)
    assert len(title) <= 64
    assert title.endswith("…")


def test_notification_is_noop_off_macos(monkeypatch) -> None:
    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert notifications.notify_response_complete(chat_title="Example Chat", final_response="hello") is False


def test_notification_failure_is_best_effort(monkeypatch) -> None:
    monkeypatch.setattr(notifications.sys, "platform", "darwin")
    monkeypatch.setattr(
        notifications.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )

    assert notifications.notify_response_complete(chat_title="Example Chat", final_response="hello") is False
