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
from gptty.output import OutputMessage
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


def test_export_prints_explicit_conversation_as_markdown(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    stdout = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123"),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    assert result == 0
    assert stdout.getvalue() == "### user\n\nhello\n\n### assistant\n\nhi\n"
    assert FakeGpttyClient.instances[0].calls == [
        ("get_messages", ("conversation-123",), {}),
    ]


def test_export_uses_attached_conversation_and_last_limit(tmp_path: Path) -> None:
    FakeGpttyClient.instances.clear()
    save_chat_state(tmp_path / "gptty_state.json", ChatState(current_conversation="attached-456"))

    result = run_export(
        make_args(
            tmp_path,
            last=5,
            auth="custom_auth.json",
            timeout=12,
        ),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    assert result == 0
    client = FakeGpttyClient.instances[0]
    assert client.auth_file == "custom_auth.json"
    assert client.timeout == 12
    assert client.calls == [("get_messages", ("attached-456",), {"limit": 5})]


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


def test_export_supports_json_output(tmp_path: Path) -> None:
    stdout = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123", format="json"),
        client_factory=FakeGpttyClient,
        stdout=stdout,
    )

    assert result == 0
    assert json.loads(stdout.getvalue()) == {
        "messages": [
            {"created_at": None, "role": "user", "text": "hello"},
            {"created_at": None, "role": "assistant", "text": "hi"},
        ]
    }


def test_export_writes_markdown_to_file(tmp_path: Path) -> None:
    output_path = tmp_path / "conversation.md"

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123", output=str(output_path)),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    assert result == 0
    assert output_path.read_text(encoding="utf-8") == "### user\n\nhello\n\n### assistant\n\nhi\n"
    if os.name != "nt":
        assert output_path.stat().st_mode & 0o777 == 0o600


def test_export_refuses_to_overwrite_existing_file_by_default(tmp_path: Path) -> None:
    output_path = tmp_path / "conversation.md"
    output_path.write_text("existing\n", encoding="utf-8")
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123", output=str(output_path)),
        client_factory=FakeGpttyClient,
        stderr=stderr,
    )

    assert result == 1
    assert output_path.read_text(encoding="utf-8") == "existing\n"
    assert "output file already exists" in stderr.getvalue()


def test_export_allows_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "conversation.md"
    output_path.write_text("existing\n", encoding="utf-8")
    if os.name != "nt":
        output_path.chmod(0o666)

    result = run_export(
        make_args(
            tmp_path,
            url_or_id="conversation-123",
            output=str(output_path),
            overwrite=True,
        ),
        client_factory=FakeGpttyClient,
        stdout=StringIO(),
    )

    assert result == 0
    assert output_path.read_text(encoding="utf-8") == "### user\n\nhello\n\n### assistant\n\nhi\n"
    if os.name != "nt":
        assert output_path.stat().st_mode & 0o777 == 0o600


def test_export_returns_1_on_sdk_error(tmp_path: Path) -> None:
    stderr = StringIO()

    result = run_export(
        make_args(tmp_path, url_or_id="conversation-123"),
        client_factory=RaisingGpttyClient,
        stderr=stderr,
    )

    assert result == 1
    assert "export request failed: backend unavailable" in stderr.getvalue()


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


def test_export_returns_1_on_file_write_error(tmp_path: Path) -> None:
    stderr = StringIO()

    result = run_export(
        make_args(
            tmp_path,
            url_or_id="conversation-123",
            output=str(tmp_path),
            overwrite=True,
        ),
        client_factory=FakeGpttyClient,
        stderr=stderr,
    )

    assert result == 1
    assert "failed to write export" in stderr.getvalue()


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
