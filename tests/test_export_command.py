from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import gptty.commands.export as export_command
from gptty.commands.export import run_export, save_markdown_export
from gptty.exporter_bridge import PersistentExportArtifact
from gptty.output import OutputMessage
from gptty.private_fs import atomic_write_private_text
from gptty.session_state import SessionStateError
from gptty.state import ChatState, save_chat_state


class FakeGpttyClient:
    instances: list["FakeGpttyClient"] = []

    def __init__(self, auth_file: str = "auth_data.json", timeout: int = 90) -> None:
        self.auth_file = auth_file
        self.timeout = timeout
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.instances.append(self)

    def get_messages(self, url_or_id: str, **options: Any) -> list[dict[str, str]]:
        self.calls.append(("get_messages", (url_or_id,), options))
        return [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "content": "hi"},
        ]


class RaisingGpttyClient(FakeGpttyClient):
    def get_messages(self, url_or_id: str, **options: Any) -> list[dict[str, str]]:
        raise RuntimeError("backend unavailable")


def make_args(tmp_path: Path, **overrides: Any) -> Namespace:
    values: dict[str, Any] = {
        "url_or_id": None,
        "state": str(tmp_path / "gptty_state.json"),
        "auth": "auth_data.json",
        "timeout": 90,
        "last": None,
        "format": "markdown",
        "output": None,
        "overwrite": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _install_fake_visible_export(monkeypatch, calls, *, markdown="# Visible graph\n"):
    def fake_export(conversation_ref, output_path, *, auth_file=None, timeout=120.0):
        path = Path(output_path)
        context_path = path.with_suffix(".context.json")
        manifest_path = path.with_suffix(".manifest.json")
        calls.append(
            {
                "conversation_ref": conversation_ref,
                "output_path": path,
                "auth_file": auth_file,
                "timeout": timeout,
            }
        )
        atomic_write_private_text(path, markdown)
        atomic_write_private_text(
            context_path,
            json.dumps(
                {
                    "schema": 1,
                    "conversation_id": str(conversation_ref),
                    "scope": "canonical-web-visible",
                    "messages": [],
                }
            )
            + "\n",
        )
        atomic_write_private_text(manifest_path, "{}\n")
        return PersistentExportArtifact(
            conversation_id=str(conversation_ref),
            markdown_path=path.resolve(),
            context_path=context_path.resolve(),
            manifest_path=manifest_path.resolve(),
            title="Visible graph",
            messages=3,
            branch_points=1,
            leaf_branches=2,
        )

    monkeypatch.setattr(export_command, "export_persistent_conversation", fake_export)


def test_export_prints_visible_graph_markdown_without_sdk_read(
    monkeypatch, tmp_path: Path
) -> None:
    FakeGpttyClient.instances.clear()
    calls = []
    _install_fake_visible_export(monkeypatch, calls)
    stdout = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123"),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    assert result == 0
    assert stdout.getvalue() == "# Visible graph\n"
    assert calls[0]["conversation_ref"] == "conversation-123"
    assert calls[0]["auth_file"] == "auth_data.json"
    assert FakeGpttyClient.instances == []


def test_export_rejects_last_limit_for_persistent_artifact(tmp_path: Path) -> None:
    save_chat_state(
        tmp_path / "gptty_state.json",
        ChatState(current_conversation="attached-456"),
    )
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path, last=5),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert result == 2
    assert "complete visible-graph artifact" in stderr.getvalue()
    assert "gptty messages --last N" in stderr.getvalue()


def test_export_requires_explicit_or_attached_conversation(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path),
        client_factory=FakeGpttyClient,
        stderr=stderr,
    )

    assert result == 2
    assert "gptty export requires a conversation URL/id" in stderr.getvalue()
    assert FakeGpttyClient.instances == []


def test_export_prints_visible_graph_context_json(monkeypatch, tmp_path: Path) -> None:
    calls = []
    _install_fake_visible_export(monkeypatch, calls)
    stdout = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123", format="json"),
        stdout=stdout,
    )

    assert result == 0
    payload = json.loads(stdout.getvalue())
    assert payload["scope"] == "canonical-web-visible"
    assert payload["conversation_id"] == "conversation-123"


def test_export_rejects_plain_persistent_projection(tmp_path: Path) -> None:
    stderr = StringIO()
    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123", format="plain"),
        stderr=stderr,
    )
    assert result == 2
    assert "gptty messages --format plain" in stderr.getvalue()


def test_export_writes_reserved_visible_graph_bundle(monkeypatch, tmp_path: Path) -> None:
    calls = []
    _install_fake_visible_export(monkeypatch, calls)
    output_path = tmp_path / "conversation.md"

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123", output=str(output_path)),
        stdout=StringIO(),
    )

    assert result == 0
    assert output_path.read_text(encoding="utf-8") == "# Visible graph\n"
    assert output_path.with_suffix(".context.json").is_file()
    assert output_path.with_suffix(".manifest.json").is_file()
    if os.name != "nt":
        for path in (
            output_path,
            output_path.with_suffix(".context.json"),
            output_path.with_suffix(".manifest.json"),
        ):
            assert path.stat().st_mode & 0o777 == 0o600


def test_export_refuses_existing_bundle_by_default(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "conversation.md"
    output_path.with_suffix(".context.json").write_text("existing\n", encoding="utf-8")
    calls = []
    _install_fake_visible_export(monkeypatch, calls)
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123", output=str(output_path)),
        stderr=stderr,
    )

    assert result == 1
    assert calls == []
    assert "export artifact already exists" in stderr.getvalue()


def test_export_allows_bundle_overwrite(monkeypatch, tmp_path: Path) -> None:
    calls = []
    _install_fake_visible_export(monkeypatch, calls)
    output_path = tmp_path / "conversation.md"
    output_path.write_text("existing\n", encoding="utf-8")

    result = run_export(
        make_args(
            tmp_path,
            url_or_id="conversation-123",
            output=str(output_path),
            overwrite=True,
        ),
        stdout=StringIO(),
    )

    assert result == 0
    assert output_path.read_text(encoding="utf-8") == "# Visible graph\n"
    assert len(calls) == 1


def test_export_bridge_failure_is_reported(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise export_command.ExporterBridgeError("backend unavailable")

    monkeypatch.setattr(export_command, "export_persistent_conversation", fail)
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123"),
        stderr=stderr,
    )

    assert result == 1
    assert "export failed: backend unavailable" in stderr.getvalue()


def test_export_bridge_failure_cleans_new_reserved_bundle(
    monkeypatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "conversation.md"

    def fail(*args, **kwargs):
        raise export_command.ExporterBridgeError("failed after reservation")

    monkeypatch.setattr(export_command, "export_persistent_conversation", fail)
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123", output=str(output_path)),
        stderr=stderr,
    )

    assert result == 1
    assert not output_path.exists()
    assert not output_path.with_suffix(".context.json").exists()
    assert not output_path.with_suffix(".manifest.json").exists()


def test_export_recovers_from_corrupt_legacy_state_with_warning(tmp_path: Path) -> None:
    state_path = tmp_path / "gptty_state.json"
    state_path.write_text("[]", encoding="utf-8")
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path, state=str(state_path)),
        client_factory=FakeGpttyClient,
        stderr=stderr,
    )

    assert result == 2
    assert "legacy chat state could not be imported" in stderr.getvalue()
    assert "requires a conversation" in stderr.getvalue()


def test_export_returns_1_on_transactional_state_error(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise SessionStateError("local session database failed")

    monkeypatch.setattr(export_command, "resolve_conversation_ref", fail)
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path),
        client_factory=FakeGpttyClient,
        stderr=stderr,
    )

    assert result == 1
    assert "local session database failed" in stderr.getvalue()


def test_save_markdown_export_creates_timestamped_readable_file(tmp_path: Path) -> None:
    path = save_markdown_export(
        [
            OutputMessage(role="user", text="hello"),
            OutputMessage(role="assistant", text="hi"),
        ],
        directory=tmp_path,
        title='My / unsafe: chat?',
        now=datetime(2026, 9, 5, 19, 20, 30, tzinfo=timezone.utc),
    )

    assert path == (tmp_path / "2026-09-05_19-20-30 - My - unsafe- chat.md").resolve()
    assert path.read_text(encoding="utf-8") == "### user\n\nhello\n\n### assistant\n\nhi\n"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_default_markdown_export_directory_is_owner_only(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "default-exports"
    monkeypatch.setattr(export_command, "DEFAULT_EXPORT_DIRECTORY", root)

    path = save_markdown_export(
        [OutputMessage(role="user", text="private")],
        title="Private chat",
        now=datetime(2026, 9, 5, 19, 20, 30, tzinfo=timezone.utc),
    )

    assert path.parent == root.resolve()
    if os.name != "nt":
        assert root.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600


def test_save_markdown_export_never_overwrites_previous_export(tmp_path: Path) -> None:
    now = datetime(2026, 9, 5, 19, 20, 30, tzinfo=timezone.utc)
    first = save_markdown_export(
        [OutputMessage(role="user", text="first")],
        directory=tmp_path,
        title="Same chat",
        now=now,
    )
    second = save_markdown_export(
        [OutputMessage(role="user", text="second")],
        directory=tmp_path,
        title="Same chat",
        now=now,
    )

    assert first.name == "2026-09-05_19-20-30 - Same chat.md"
    assert second.name == "2026-09-05_19-20-30 - Same chat (2).md"
    assert "first" in first.read_text(encoding="utf-8")
    assert "second" in second.read_text(encoding="utf-8")


def test_save_markdown_export_appends_typed_sources_without_using_citation_offsets(
    tmp_path: Path,
) -> None:
    path = save_markdown_export(
        [OutputMessage(role="assistant", text="answer")],
        directory=tmp_path,
        title="Sources",
        now=datetime(2026, 9, 26, 10, 0, 0, tzinfo=timezone.utc),
        observations={
            "sources": [
                {
                    "kind": "source",
                    "source_id": "source-1",
                    "url": "https://example.com/source",
                    "title": "Typed Source",
                }
            ],
            "citations": [
                {
                    "kind": "citation",
                    "citation_id": "citation-1",
                    "source_id": "source-1",
                    "start_index": 123456,
                    "end_index": 234567,
                    "range_coordinate_space": "unknown",
                }
            ],
        },
    )

    rendered = path.read_text(encoding="utf-8")
    assert "### Sources" in rendered
    assert "Typed Source" in rendered
    assert "https://example.com/source" in rendered
    assert "123456" not in rendered
    assert "234567" not in rendered


def test_save_markdown_export_marks_reduced_artifact_scope(tmp_path: Path) -> None:
    path = save_markdown_export(
        [OutputMessage(role="assistant", text="temporary answer")],
        directory=tmp_path,
        title="Temporary",
        now=datetime(2026, 9, 26, 14, 45, 0, tzinfo=timezone.utc),
        artifact_scope="temporary_in_memory_current_branch",
        artifact_provenance="gptty_temporary_transcript",
    )

    rendered = path.read_text(encoding="utf-8")
    assert rendered.startswith(
        "> Export scope: `temporary_in_memory_current_branch`\n"
        "> Provenance: `gptty_temporary_transcript`\n\n"
    )
    assert "temporary answer" in rendered
