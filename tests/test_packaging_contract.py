from __future__ import annotations

import importlib.util

REQUIRED_CWA_MODULES = (
    "chatgpt_web_adapter.browser_authority_backend",
    "chatgpt_web_adapter.wkwebview_provider",
    "chatgpt_web_adapter.artifact_manifest",
)
REQUIRED_GPTTY_MODULES = (
    "gptty.sdk_client",
    "gptty.exporter_bridge",
)


def test_required_runtime_modules_are_importable() -> None:
    missing = [
        module
        for module in (*REQUIRED_CWA_MODULES, *REQUIRED_GPTTY_MODULES)
        if importlib.util.find_spec(module) is None
    ]
    assert not missing, f"missing required runtime modules: {missing}"
