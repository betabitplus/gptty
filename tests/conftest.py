from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_gptty_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test out of the real ~/.config/gptty and ~/.local/share/gptty."""
    monkeypatch.setenv("GPTTY_CONFIG_HOME", str(tmp_path / "gptty-config"))
    monkeypatch.setenv("GPTTY_DATA_HOME", str(tmp_path / "gptty-data"))
    monkeypatch.setenv("GPTTY_ARCHIVE_HOME", str(tmp_path / "gptty-archive"))
