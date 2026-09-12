from __future__ import annotations

import json
import sys
import tempfile
from importlib import metadata
from pathlib import Path


def _requires(distribution: str) -> tuple[str, ...]:
    return tuple(metadata.requires(distribution) or ())


def _is_site_package(path: Path) -> bool:
    return "site-packages" in path.parts


def main() -> int:
    if sys.platform != "darwin":
        raise RuntimeError("installed WK smoke requires macOS")

    import chatgpt_web_adapter
    import gptty
    from gptty.sdk_client import GpttyClient

    gptty_path = Path(gptty.__file__).resolve()
    cwa_path = Path(chatgpt_web_adapter.__file__).resolve()
    if not _is_site_package(gptty_path) or not _is_site_package(cwa_path):
        raise RuntimeError(
            f"installed packages required: gptty={gptty_path} cwa={cwa_path}"
        )

    gptty_requires = _requires("gptty-web")
    cwa_requires = _requires("chatgpt-web-adapter")
    if any(
        requirement.lower().startswith(("curl-cffi", "websockets"))
        for requirement in gptty_requires
    ):
        raise RuntimeError("gptty must not own WK transport dependencies")
    for required_prefix in ("curl-cffi==0.16.3", "websockets==16.1.1"):
        if not any(
            requirement.lower().startswith(required_prefix)
            and "darwin" in requirement.lower()
            for requirement in cwa_requires
        ):
            raise RuntimeError(f"CWA Darwin dependency missing: {required_prefix}")

    with tempfile.TemporaryDirectory(prefix="gptty-wk-installed-") as temp_dir:
        auth_path = Path(temp_dir) / "auth.json"
        auth_field = "".join(("access", "Token"))
        auth_path.write_text(
            json.dumps({auth_field: "ci-placeholder"}),
            encoding="utf-8",
        )

        client = GpttyClient(
            auth_file=auth_path,
            browser_authority_backend="wkwebview",
        )
        runtime = client._client.runtime
        governance = runtime.governance()
        provider = runtime.write_transport.provider

        if governance.get("browser_authority_backend") != "wkwebview":
            raise RuntimeError("gptty did not assemble the WK browser authority backend")
        if type(provider).__name__ != "WKWebViewTurnProvider":
            raise RuntimeError(f"unexpected WK provider: {type(provider).__name__}")

        status = provider.status()
        if not status.available or not status.extension_connected:
            raise RuntimeError("installed CWA WK helper did not build successfully")

    helper_dir = cwa_path.parent / "wkwebview_helper"
    expected_helpers = {
        "WKChatGPTAuthority.m",
        "Info.plist",
        "minimal_security_shell.js",
    }
    present_helpers = {path.name for path in helper_dir.iterdir()} if helper_dir.is_dir() else set()
    if not expected_helpers.issubset(present_helpers):
        missing = sorted(expected_helpers - present_helpers)
        raise RuntimeError(f"installed CWA WK helper resources missing: {missing}")

    print("GPTTY_WK_INSTALLED_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
