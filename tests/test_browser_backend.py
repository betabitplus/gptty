from __future__ import annotations

from gptty.sdk_client import GpttyClient, _ProductRuntimeClient


def test_gptty_client_exposes_effective_browser_backend() -> None:
    from chatgpt_web_adapter.browser_authority_backend import (
        resolve_browser_authority_backend,
    )

    client = GpttyClient(sdk_client=object())

    assert client.browser_authority_backend is None
    assert client.effective_browser_authority_backend == resolve_browser_authority_backend(None)


def test_product_runtime_client_uses_cwa_wk_provider_when_selected(monkeypatch) -> None:
    import chatgpt_web_adapter
    from chatgpt_web_adapter.wkwebview_provider import WKWebViewTurnProvider

    calls: list[dict] = []

    def fake_assemble(**kwargs):
        calls.append(dict(kwargs))
        return object()

    monkeypatch.setattr(chatgpt_web_adapter, "assemble_product_runtime", fake_assemble)

    default_client = _ProductRuntimeClient(auth_file="auth.json", timeout=10)
    wk_client = _ProductRuntimeClient(
        auth_file="auth.json",
        timeout=10,
        browser_authority_backend="wkwebview",
    )

    assert default_client.runtime is not None
    assert wk_client.runtime is not None
    assert "browser_authority_backend" not in calls[0]
    assert isinstance(calls[1]["provider"], WKWebViewTurnProvider)
    assert calls[1]["browser_authority_policy"] == "TURN_SCOPED"
    assert "browser_authority_backend" not in calls[1]
