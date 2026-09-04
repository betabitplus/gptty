from __future__ import annotations

from gptty.ui.state import RecentStore, UISettings, load_ui_settings, save_ui_settings


def test_ui_settings_round_trip(tmp_path) -> None:
    path = tmp_path / "ui.json"
    settings = UISettings(pretty="on", markdown=False, thinking=False, tools="hidden", editor="vi")

    save_ui_settings(path, settings)

    assert load_ui_settings(path) == settings


def test_recent_store_deduplicates_and_moves_latest_to_front(tmp_path) -> None:
    store = RecentStore(tmp_path / "recent.json")

    store.remember("conv-1", label="First")
    store.remember("conv-2", label="Second")
    store.remember("conv-1")

    items = store.list()
    assert [item.ref for item in items] == ["conv-1", "conv-2"]
    assert items[0].label == "First"
