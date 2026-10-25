from __future__ import annotations

from gptty.sdk_client import _ProductRuntimeClient


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
