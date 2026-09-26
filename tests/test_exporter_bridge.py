from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gptty.exporter_bridge as bridge
from chatgpt_web_adapter.doctor import DoctorCheckStatus


def _completed(payload: dict, *, returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


def _write_result_files(root: Path) -> tuple[Path, Path, Path]:
    markdown = root / "conversation.md"
    context = root / "conversation.context.json"
    manifest = root / "conversation.manifest.json"
    markdown.write_text("# Visible graph\n", encoding="utf-8")
    context.write_text('{"scope":"canonical-web-visible"}\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema": 2,
                "artifact_kind": "conversation_visible_graph_export",
                "contract": "canonical_visible_graph_export_v1",
                "conversation_id": "conversation-123",
                "index": 1,
                "format": "markdown",
                "representations": ["canonical_visible_graph"],
                "content_sha256": "0" * 64,
                "provenance": {
                    "producer": "chatgpt-conversation-exporter",
                    "producer_version": "0.1.0",
                    "source": "chatgpt-canonical-visible-graph",
                    "fetched_at": "2026-09-26T12:00:00.000Z",
                    "source_revision": "rev-1",
                    "projection_version": "canonical_visible_graph_export_v1",
                },
                "storage": {
                    "privacy": "owner_only",
                    "creation": "private_atomic_replace",
                    "completion_marker": "manifest_last",
                },
                "files": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return markdown.resolve(), context.resolve(), manifest.resolve()


def test_resolve_exporter_command_prefers_explicit_then_env(monkeypatch) -> None:
    monkeypatch.setenv(bridge.EXPORTER_COMMAND_ENV, "python -m export_chatgpt_chat")

    assert bridge.resolve_exporter_command(("custom-exporter", "--flag")) == (
        "custom-exporter",
        "--flag",
    )
    assert bridge.resolve_exporter_command() == (
        "python",
        "-m",
        "export_chatgpt_chat",
    )


def test_export_persistent_conversation_validates_identity_paths_and_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    markdown, context, manifest = _write_result_files(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, "kwargs": kwargs})
        return _completed(
            {
                "conversation_id": "conversation-123",
                "path": str(markdown),
                "context_path": str(context),
                "manifest_path": str(manifest),
                "title": "Visible",
                "messages": 3,
                "branch_points": 1,
                "leaf_branches": 2,
            }
        )

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    monkeypatch.setattr(
        bridge,
        "verify_artifact_manifest",
        lambda path: SimpleNamespace(
            status=DoctorCheckStatus.PASS,
            evidence={"conversation_id": "conversation-123", "errors": []},
        ),
    )

    artifact = bridge.export_persistent_conversation(
        "conversation-123",
        markdown,
        auth_file="/private/auth.json",
        timeout=33,
        exporter_command=("chatgpt-export-one",),
    )

    assert artifact.conversation_id == "conversation-123"
    assert artifact.markdown_path == markdown
    assert artifact.context_path == context
    assert artifact.manifest_path == manifest
    assert calls[0]["command"] == [
        "chatgpt-export-one",
        "conversation-123",
        str(markdown),
        "--auth",
        "/private/auth.json",
    ]
    assert calls[0]["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 33.0,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("conversation_id", "wrong-conversation", "conversation identity mismatch"),
        ("path", "/tmp/wrong.md", "output path mismatch"),
        ("context_path", "/tmp/wrong.context.json", "context path mismatch"),
        ("manifest_path", "/tmp/wrong.manifest.json", "manifest path mismatch"),
    ],
)
def test_export_persistent_conversation_rejects_result_identity_drift(
    monkeypatch,
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    markdown, context, manifest = _write_result_files(tmp_path)
    payload = {
        "conversation_id": "conversation-123",
        "path": str(markdown),
        "context_path": str(context),
        "manifest_path": str(manifest),
    }
    if field != "conversation_id":
        wrong = Path(value)
        wrong.parent.mkdir(parents=True, exist_ok=True)
        wrong.write_text("{}\n", encoding="utf-8")
        payload[field] = str(wrong.resolve())
    else:
        payload[field] = value

    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: _completed(payload),
    )

    with pytest.raises(bridge.ExporterBridgeError, match=message):
        bridge.export_persistent_conversation(
            "conversation-123",
            markdown,
            exporter_command=("chatgpt-export-one",),
        )


def test_export_persistent_conversation_rejects_failed_cwa_verification(
    monkeypatch, tmp_path: Path
) -> None:
    markdown, context, manifest = _write_result_files(tmp_path)
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {
                "conversation_id": "conversation-123",
                "path": str(markdown),
                "context_path": str(context),
                "manifest_path": str(manifest),
            }
        ),
    )
    monkeypatch.setattr(
        bridge,
        "verify_artifact_manifest",
        lambda path: SimpleNamespace(
            status=DoctorCheckStatus.FAIL,
            evidence={"conversation_id": "conversation-123", "errors": ["sha256 mismatch"]},
        ),
    )

    with pytest.raises(bridge.ExporterBridgeError, match="CWA verification"):
        bridge.export_persistent_conversation(
            "conversation-123",
            markdown,
            exporter_command=("chatgpt-export-one",),
        )


def test_resolve_exporter_command_fails_closed_when_unavailable(monkeypatch) -> None:
    monkeypatch.delenv(bridge.EXPORTER_COMMAND_ENV, raising=False)
    monkeypatch.setattr(bridge.shutil, "which", lambda name: None)

    with pytest.raises(bridge.ExporterBridgeError, match="chatgpt-export-one is required"):
        bridge.resolve_exporter_command()


def test_export_persistent_conversation_rejects_non_visible_graph_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    markdown, context, manifest = _write_result_files(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact_kind"] = "conversation_export"
    payload["contract"] = "normalized_current_branch_export_v1"
    payload["representations"] = ["current_branch"]
    payload["provenance"]["projection_version"] = "normalized_current_branch_export_v1"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {
                "conversation_id": "conversation-123",
                "path": str(markdown),
                "context_path": str(context),
                "manifest_path": str(manifest),
            }
        ),
    )
    monkeypatch.setattr(
        bridge,
        "verify_artifact_manifest",
        lambda path: SimpleNamespace(
            status=DoctorCheckStatus.PASS,
            evidence={"conversation_id": "conversation-123", "errors": []},
        ),
    )

    with pytest.raises(bridge.ExporterBridgeError, match="artifact kind"):
        bridge.export_persistent_conversation(
            "conversation-123",
            markdown,
            exporter_command=("chatgpt-export-one",),
        )


def test_exporter_failure_redacts_sensitive_diagnostics(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "conversation.md"
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {},
            returncode=1,
            stderr=(
                "Authorization: Bearer secret-token "+ str(Path.home()) + "/auth_data.json"
            ),
        ),
    )

    with pytest.raises(bridge.ExporterBridgeError) as exc_info:
        bridge.export_persistent_conversation(
            "conversation-123",
            output,
            exporter_command=("chatgpt-export-one",),
        )

    message = str(exc_info.value)
    assert "secret-token" not in message
    assert str(Path.home()) not in message
    assert "[REDACTED]" in message
    assert "~/auth_data.json" in message
