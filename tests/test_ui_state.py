from __future__ import annotations

from gptty.ui.state import UISettings, load_ui_settings, save_ui_settings


def test_ui_settings_round_trip(tmp_path) -> None:
    path = tmp_path / "ui.json"
    settings = UISettings(pretty="on", markdown=False, thinking=False, tools="hidden", editor="vi")

    save_ui_settings(path, settings)

    assert load_ui_settings(path) == settings
