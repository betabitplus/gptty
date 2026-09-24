from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERIFIED_CWA_REQUIREMENT = ">=0.3.1,<0.4.0"
REQUIRED_CWA_MODULES = (
    "chatgpt_web_adapter.browser_authority_backend",
    "chatgpt_web_adapter.wkwebview_provider",
)


def _declared_cwa_requirement() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'"chatgpt-web-adapter([^"]+)"', text)
    assert match is not None, "pyproject.toml must declare chatgpt-web-adapter"
    return match.group(1)


def test_cwa_dependency_floor_matches_verified_runtime_contract() -> None:
    assert _declared_cwa_requirement() == VERIFIED_CWA_REQUIREMENT


def test_required_cwa_runtime_modules_are_importable() -> None:
    missing = [
        module
        for module in REQUIRED_CWA_MODULES
        if importlib.util.find_spec(module) is None
    ]
    assert not missing, f"missing required CWA runtime modules: {missing}"
