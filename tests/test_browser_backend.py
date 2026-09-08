from __future__ import annotations

from types import SimpleNamespace

from gptty.commands._client import build_client
from gptty.sdk_client import _ProductRuntimeClient


def test_command_client_boundary_omits_backend_when_not_selected() -> None:
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    build_client(
        factory,
        SimpleNamespace(auth="auth.json", timeout=12, backend=None),
    )

    assert captured == {"auth_file": "auth.json", "timeout": 12}


def test_command_client_boundary_forwards_selected_backend() -> None:
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    build_client(
        factory,
        SimpleNamespace(auth="auth.json", timeout=12, backend="wkwebview"),
    )

    assert captured == {
        "auth_file": "auth.json",
        "timeout": 12,
        "browser_authority_backend": "wkwebview",
    }


def test_product_runtime_client_only_uses_new_cwa_kwarg_when_selected(monkeypatch) -> None:
    import chatgpt_web_adapter

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
    assert calls[1]["browser_authority_backend"] == "wkwebview"


