from __future__ import annotations

from gptty.ui.state import UISettings, load_ui_settings, save_ui_settings


def test_ui_settings_round_trip(tmp_path) -> None:
    path = tmp_path / "ui.json"
    settings = UISettings(
        pretty="on",
        markdown=False,
        thinking=False,
        tools="hidden",
        editor="vi",
        history_limit=321,
        notifications=False,
        notification_preview=True,
        notification_sound=False,
    )

    save_ui_settings(path, settings)

    assert load_ui_settings(path) == settings


def test_ui_settings_history_limit_is_bounded_and_type_checked(tmp_path) -> None:
    path = tmp_path / "ui.json"

    path.write_text('{"history_limit": -3}\n', encoding="utf-8")
    assert load_ui_settings(path).history_limit == 0

    path.write_text('{"history_limit": 200000}\n', encoding="utf-8")
    assert load_ui_settings(path).history_limit == 100000

    path.write_text('{"history_limit": true}\n', encoding="utf-8")
    assert load_ui_settings(path).history_limit == 2000


def test_ui_settings_notification_privacy_defaults_fail_closed(tmp_path) -> None:
    path = tmp_path / "ui.json"
    path.write_text(
        '{"notifications":"yes","notification_preview":"yes","notification_sound":"no"}\n',
        encoding="utf-8",
    )

    settings = load_ui_settings(path)

    assert settings.notifications is True
    assert settings.notification_preview is False
    assert settings.notification_sound is True
