from __future__ import annotations

from pathlib import Path


def test_wkwebview_transport_dependencies_are_owned_by_cwa() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert '"chatgpt-web-adapter>=0.3.0,<0.4.0"' in pyproject
    assert "chatgpt-web-adapter[wkwebview]" not in pyproject
    assert '"curl-cffi' not in pyproject
    assert '"websockets' not in pyproject


def test_macos_wk_installed_artifact_ci_is_blocking_and_candidate_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    smoke = root / "tools" / "installed_wk_smoke.py"

    assert "macos-wk-installed:" in workflow
    assert "runs-on: macos-latest" in workflow
    assert "repository: betabitplus/chatgpt-web-adapter" in workflow
    assert "CWA_CANDIDATE_REF: prerelease/wkwebview-main-hardening" in workflow
    assert "python -m build --wheel --outdir artifacts/cwa .cwa-candidate" in workflow
    assert "python -m build --wheel --outdir artifacts/gptty ." in workflow
    assert (
        "python -m pip install --force-reinstall artifacts/cwa/*.whl artifacts/gptty/*.whl"
        in workflow
    )
    assert "python tools/installed_wk_smoke.py" in workflow
    assert smoke.is_file()


def test_release_candidate_version_and_publish_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    release_docs = (root / "docs" / "release.md").read_text(encoding="utf-8")

    assert 'version = "0.1.2"' in pyproject
    assert "## Unreleased" in changelog
    assert "## 0.1.1 - 2026-06-24" in changelog
    assert "workflow_dispatch" not in workflow
    assert "macos-wk-publish:" in workflow
    assert "runs-on: macos-latest" in workflow
    assert "      - macos-wk-publish" in workflow
    assert "Verify release tag and changelog" in workflow
    assert "Verify promoted CWA dependency floor" in workflow
    assert "chatgpt-web-adapter>=0.3.1,<0.4.0" in workflow
    assert "ref: ${{ github.event.release.tag_name }}" in workflow
    assert "python tools/installed_wk_smoke.py" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "staged gptty 0.1.2" in release_docs
    assert "waiting for CWA 0.3.1" in release_docs
    assert "git tag v0.1.2" in release_docs
