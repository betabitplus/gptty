from __future__ import annotations

import importlib.metadata

import gptty


def test_package_version_matches_distribution_metadata() -> None:
    assert gptty.__version__ == importlib.metadata.version("gptty-web")
