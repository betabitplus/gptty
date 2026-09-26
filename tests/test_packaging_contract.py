from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERIFIED_CWA_REQUIREMENT = ">=0.3.2,<0.4.0"
VERIFIED_EXPORTER_REQUIREMENT = ">=0.1.0,<0.2.0"
REQUIRED_CWA_MODULES = (
    "chatgpt_web_adapter.browser_authority_backend",
    "chatgpt_web_adapter.wkwebview_provider",
    "chatgpt_web_adapter.artifact_manifest",
)
REQUIRED_GPTTY_MODULES = (
    "gptty.sdk_client",
    "gptty.exporter_bridge",
)


def _declared_requirement(distribution: str) -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(rf'"{re.escape(distribution)}([^"]+)"', text)
    assert match is not None, f"pyproject.toml must declare {distribution}"
    return match.group(1)


def test_cwa_dependency_floor_matches_verified_runtime_contract() -> None:
    assert _declared_requirement("chatgpt-web-adapter") == VERIFIED_CWA_REQUIREMENT


def test_exporter_dependency_floor_matches_visible_graph_contract() -> None:
    assert (
        _declared_requirement("chatgpt-conversation-exporter")
        == VERIFIED_EXPORTER_REQUIREMENT
    )


def test_required_runtime_modules_are_importable() -> None:
    missing = [
        module
        for module in (*REQUIRED_CWA_MODULES, *REQUIRED_GPTTY_MODULES)
        if importlib.util.find_spec(module) is None
    ]
    assert not missing, f"missing required runtime modules: {missing}"
