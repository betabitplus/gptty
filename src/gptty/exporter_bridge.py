from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from chatgpt_web_adapter.artifact_manifest import (
    CANONICAL_VISIBLE_GRAPH_REPRESENTATION,
    VISIBLE_GRAPH_ARTIFACT_KIND,
    VISIBLE_GRAPH_CONTRACT,
)
from chatgpt_web_adapter.doctor import DoctorCheckStatus, verify_artifact_manifest
from chatgpt_web_adapter.types import ConversationRef

from .privacy import redact_diagnostic_text

EXPORTER_COMMAND_ENV = "GPTTY_EXPORTER_COMMAND"


class ExporterBridgeError(RuntimeError):
    """Raised when the visible-graph exporter cannot produce a verified artifact."""


@dataclass(frozen=True)
class PersistentExportArtifact:
    conversation_id: str
    markdown_path: Path
    context_path: Path
    manifest_path: Path
    title: str | None = None
    messages: int | None = None
    branch_points: int | None = None
    leaf_branches: int | None = None


def resolve_exporter_command(
    explicit: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    if explicit is not None:
        parts = shlex.split(explicit) if isinstance(explicit, str) else list(explicit)
        if not parts:
            raise ExporterBridgeError("exporter command cannot be empty")
        return tuple(str(part) for part in parts)

    configured = os.environ.get(EXPORTER_COMMAND_ENV)
    if configured:
        parts = shlex.split(configured)
        if not parts:
            raise ExporterBridgeError(f"{EXPORTER_COMMAND_ENV} cannot be empty")
        return tuple(parts)

    executable = shutil.which("chatgpt-export-one")
    if executable:
        return (executable,)
    raise ExporterBridgeError(
        "chatgpt-export-one is required for persistent conversation export; "
        f"install chatgpt-conversation-exporter or set {EXPORTER_COMMAND_ENV}"
    )


def export_persistent_conversation(
    conversation_ref: str,
    output_path: str | Path,
    *,
    auth_file: str | Path | None = None,
    timeout: float = 120.0,
    exporter_command: str | Sequence[str] | None = None,
) -> PersistentExportArtifact:
    expected_conversation_id = ConversationRef.from_any(conversation_ref).conversation_id
    expected_markdown_path = Path(output_path).expanduser().resolve()
    command = [
        *resolve_exporter_command(exporter_command),
        str(conversation_ref),
        str(expected_markdown_path),
    ]
    if auth_file:
        command.extend(["--auth", str(Path(auth_file).expanduser())])

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExporterBridgeError(f"visible-graph exporter failed to start: {exc}") from exc

    if completed.returncode != 0:
        detail = redact_diagnostic_text((completed.stderr or completed.stdout or "").strip())
        raise ExporterBridgeError(
            "visible-graph exporter failed"
            + (f": {detail}" if detail else f" with exit code {completed.returncode}")
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExporterBridgeError("visible-graph exporter returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ExporterBridgeError("visible-graph exporter returned a non-object result")

    artifact = _artifact_from_payload(payload)
    if artifact.conversation_id != expected_conversation_id:
        raise ExporterBridgeError("visible-graph exporter conversation identity mismatch")
    if artifact.markdown_path != expected_markdown_path:
        raise ExporterBridgeError("visible-graph exporter output path mismatch")
    expected_context_path = expected_markdown_path.with_suffix(".context.json")
    expected_manifest_path = expected_markdown_path.with_suffix(".manifest.json")
    if artifact.context_path != expected_context_path:
        raise ExporterBridgeError("visible-graph exporter context path mismatch")
    if artifact.manifest_path != expected_manifest_path:
        raise ExporterBridgeError("visible-graph exporter manifest path mismatch")

    check = verify_artifact_manifest(artifact.manifest_path)
    if check.status is not DoctorCheckStatus.PASS:
        errors = check.evidence.get("errors") if isinstance(check.evidence, dict) else None
        raise ExporterBridgeError(
            "visible-graph artifact failed CWA verification"
            + (f": {errors}" if errors else "")
        )
    if check.evidence.get("conversation_id") != artifact.conversation_id:
        raise ExporterBridgeError("visible-graph manifest conversation identity mismatch")
    _validate_visible_graph_manifest(artifact.manifest_path, artifact.conversation_id)
    return artifact


def _validate_visible_graph_manifest(manifest_path: Path, conversation_id: str) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExporterBridgeError("visible-graph manifest cannot be read") from exc
    if not isinstance(manifest, dict):
        raise ExporterBridgeError("visible-graph manifest root must be an object")
    if manifest.get("artifact_kind") != VISIBLE_GRAPH_ARTIFACT_KIND:
        raise ExporterBridgeError("exporter artifact kind is not canonical visible graph")
    if manifest.get("contract") != VISIBLE_GRAPH_CONTRACT:
        raise ExporterBridgeError("exporter artifact contract is not canonical visible graph")
    if manifest.get("conversation_id") != conversation_id:
        raise ExporterBridgeError("visible-graph manifest conversation identity mismatch")
    representations = manifest.get("representations")
    if representations != [CANONICAL_VISIBLE_GRAPH_REPRESENTATION]:
        raise ExporterBridgeError("exporter artifact representation is not canonical visible graph")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ExporterBridgeError("visible-graph manifest provenance is missing")
    if provenance.get("projection_version") != VISIBLE_GRAPH_CONTRACT:
        raise ExporterBridgeError("visible-graph manifest projection version mismatch")


def _artifact_from_payload(payload: dict[str, Any]) -> PersistentExportArtifact:
    conversation_id = _required_text(payload.get("conversation_id"), "conversation_id")
    markdown_path = _required_file(payload.get("path"), "path")
    context_path = _required_file(payload.get("context_path"), "context_path")
    manifest_path = _required_file(payload.get("manifest_path"), "manifest_path")
    return PersistentExportArtifact(
        conversation_id=conversation_id,
        markdown_path=markdown_path,
        context_path=context_path,
        manifest_path=manifest_path,
        title=_optional_text(payload.get("title")),
        messages=_optional_int(payload.get("messages")),
        branch_points=_optional_int(payload.get("branch_points")),
        leaf_branches=_optional_int(payload.get("leaf_branches")),
    )


def _required_file(value: Any, field: str) -> Path:
    text = _required_text(value, field)
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise ExporterBridgeError(f"visible-graph exporter result {field} is missing: {path}")
    return path


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExporterBridgeError(f"visible-graph exporter result {field} is required")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
