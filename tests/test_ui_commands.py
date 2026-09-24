from __future__ import annotations

import asyncio
from types import SimpleNamespace

from gptty.goal import MAX_ROLLOVERS
from gptty.goal_store import GoalStore
from gptty.state import (
    ChatState,
    GoalState,
    StateError,
    load_chat_state,
    save_chat_state,
)
from gptty.tui_archive import TUIArchive
from gptty.ui.commands import InteractiveCommands


class FakeUI:
    def __init__(self, *, choices=None, image_paths=None) -> None:
        self.choices = list(choices or [])
        self.image_paths = list(image_paths or [])
        self.seen: list[tuple[str, list[tuple[object, str]]]] = []

    def choose_searchable(self, message, options, *, default=None):
        self.seen.append((message, list(options)))
        return self.choices.pop(0) if self.choices else default

    async def choose_searchable_async(self, message, options):
        self.seen.append((message, list(options)))
        return self.choices.pop(0) if self.choices else None

    def read_image_path(self):
        return self.image_paths.pop(0) if self.image_paths else None

    async def read_image_path_async(self):
        return self.image_paths.pop(0) if self.image_paths else None


class FakeRenderer:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def clear_context(self):
        self.events.append(("clear_context", None))

    def header(self, **kwargs):
        self.events.append(("header", kwargs))

    def start_elapsed(self, *, initial_elapsed=0.0):
        self.events.append(("start_elapsed", initial_elapsed))

    def finish_elapsed(self):
        self.events.append(("finish_elapsed", None))

    def chat_link(self, ref):
        self.events.append(("chat_link", ref))

    def turn_abort(self):
        self.events.append(("turn_abort", None))

    def info(self, text):
        self.events.append(("info", text))

    def warning(self, text):
        self.events.append(("warning", text))

    def turn_marker(self, label, status, message):
        self.events.append(
            (
                "turn_marker",
                {"label": label, "status": status, "message": message},
            )
        )

    def messages(self, messages):
        self.events.append(("messages", messages))


class FakeClient:
    def __init__(self, *, snapshots=None) -> None:
        self.calls: list[tuple[str, object]] = []
        self.snapshots = list(
            snapshots
            or [
                {
                    "status": SimpleNamespace(status="completed"),
                    "messages": [
                        {"message_id": "u1", "role": "user", "text": "question"},
                        {"message_id": "t1", "role": "tool", "text": "raw tool result"},
                        {
                            "message_id": "call1",
                            "role": "assistant",
                            "recipient": "api_tool.call_tool",
                            "text": "raw tool call",
                        },
                        {"message_id": "a1", "role": "assistant", "text": "answer"},
                    ],
                }
            ]
        )

    def list_conversations(self):
        self.calls.append(("list_conversations", None))
        return [
            {"id": "conv-2", "title": "Second chat", "update_time": 2.0},
            {"id": "conv-1", "title": "First chat", "update_time": 1.0},
        ]

    def conversation_snapshot(self, ref, **options):
        self.calls.append(("snapshot", ref))
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]

    def stop_generation(self, ref, **options):
        self.calls.append(("stop_generation", (ref, options)))
        return {"ok": True, "stopped": True, "conversationId": ref}

    def get_messages(self, ref):
        self.calls.append(("get_messages", ref))
        return [
            {"role": "user", "text": "question"},
            {"role": "assistant", "text": "answer"},
        ]

    def temporary_lifecycle_snapshot(self):
        self.calls.append(("temporary_lifecycle_snapshot", None))
        return {"state": "LIVE", "conversation_id": "temp-1"}

    def end_temporary_chat(self):
        self.calls.append(("end_temporary_chat", None))
        return True

    def list_models(self):
        self.calls.append(("list_models", None))
        return [
            {"slug": "gpt-real-a", "title": "Real A"},
            {"slug": "gpt-real-b", "title": "Real B"},
            {"slug": "disabled", "title": "Disabled", "is_disabled": True},
            {"slug": "work-mode", "title": "Work Mode", "is_work_mode_model": True},
            {"slug": "research", "title": "Deep Research"},
        ]


def make_commands(
    tmp_path, *, state=None, ui=None, client=None, tui_archive=None, runner_id=None
):
    state = state or ChatState()
    ui = ui or FakeUI()
    client = client or FakeClient()
    renderer = FakeRenderer()
    state_path = tmp_path / "gptty_state.json"
    commands = InteractiveCommands(
        state=state,
        state_path=state_path,
        get_client=lambda: client,
        ui=ui,
        renderer=renderer,
        tui_archive=tui_archive,
        runner_id=runner_id,
    )
    return commands, renderer, client, state_path


def finish_pending_resume(commands, client):
    request = commands.take_pending_resume()
    assert request is not None
    snapshot = client.conversation_snapshot(request.conversation_ref)
    commands.complete_resume(request, snapshot)
    return request


def test_resume_picker_prefers_bounded_recent_catalog(tmp_path) -> None:
    class RecentClient(FakeClient):
        def list_recent_conversations(self, *, limit=100):
            self.calls.append(("list_recent_conversations", limit))
            return [
                {"id": "conv-2", "title": "Second chat", "update_time": 2.0},
                {"id": "conv-1", "title": "First chat", "update_time": 1.0},
            ]

        def list_conversations(self):
            raise AssertionError("full catalog must not be used by /resume picker")

    ui = FakeUI(choices=["conv-2"])
    client = RecentClient()
    commands, _renderer, _client, _state_path = make_commands(
        tmp_path,
        ui=ui,
        client=client,
    )

    assert commands.handle("/resume") is None
    assert client.calls == [("list_recent_conversations", 100)]


def test_resume_async_picker_stays_in_enhanced_ui(tmp_path) -> None:
    ui = FakeUI(choices=["conv-2"])
    commands, _renderer, client, _state_path = make_commands(tmp_path, ui=ui)

    assert asyncio.run(commands.handle_async("/resume")) is None

    request = commands.take_pending_resume()
    assert request is not None
    assert request.conversation_ref == "conv-2"
    assert client.calls == [("list_conversations", None)]
    assert ui.seen[-1][0] == "Resume conversation"


def test_resume_lists_real_conversations_and_renders_full_history(tmp_path) -> None:
    ui = FakeUI(choices=["conv-2"])
    commands, renderer, client, state_path = make_commands(tmp_path, ui=ui)

    assert commands.handle("/resume") is None
    assert client.calls == [("list_conversations", None)]
    assert load_chat_state(state_path).current_conversation is None

    finish_pending_resume(commands, client)

    assert client.calls[:2] == [
        ("list_conversations", None),
        ("snapshot", "conv-2"),
    ]
    assert load_chat_state(state_path).current_conversation == "conv-2"
    clear_index = next(
        index
        for index, event in enumerate(renderer.events)
        if event[0] == "clear_context"
    )
    assert clear_index > 0
    rendered = [event for event in renderer.events if event[0] == "messages"][-1][1]
    assert [message.text for message in rendered] == ["question", "answer"]


def test_resume_warns_when_history_comes_from_rate_limit_cache(tmp_path) -> None:
    commands, renderer, _client, _state_path = make_commands(tmp_path)

    commands.handle("/resume conv-cache")
    request = commands.take_pending_resume()
    assert request is not None
    commands.complete_resume(
        request,
        {
            "status": SimpleNamespace(status="completed"),
            "messages": [
                {"message_id": "u1", "role": "user", "text": "cached question"},
                {"message_id": "a1", "role": "assistant", "text": "cached answer"},
            ],
            "canonical_cache_stale": True,
            "canonical_cache_age_seconds": 12.75,
        },
    )

    warnings = [event[1] for event in renderer.events if event[0] == "warning"]
    assert (
        "Canonical history is rate-limited; showing cached history (12s old)."
        in warnings
    )


def test_resume_terminal_backend_override_opens_idle_and_warns_about_missing_final_text(
    tmp_path,
) -> None:
    commands, renderer, _client, _state_path = make_commands(tmp_path)

    commands.handle("/resume conv-stale")
    request = commands.take_pending_resume()
    assert request is not None
    commands.complete_resume(
        request,
        {
            "status": SimpleNamespace(status="completed"),
            "messages": [
                {"message_id": "u1", "role": "user", "text": "question"},
                {"message_id": "t1", "role": "tool", "text": "tool output"},
            ],
            "backend_stream_status": "COMPLETE",
            "backend_terminal_status_proven": True,
            "canonical_status_overridden": True,
            "canonical_status_before_override": "tool_running",
            "canonical_terminal_text_missing": True,
        },
    )

    markers = [event[1] for event in renderer.events if event[0] == "turn_marker"]
    assert markers == [
        {
            "label": "turn",
            "status": "unresolved",
            "message": "ChatGPT is terminal, but canonical history contains no final assistant response.",
        }
    ]
    warnings = [event[1] for event in renderer.events if event[0] == "warning"]
    assert not any("unfinished turn" in warning for warning in warnings)


def test_resume_completed_user_tail_marks_historical_turn_unresolved(tmp_path) -> None:
    commands, renderer, _client, _state_path = make_commands(tmp_path)

    commands.handle("/resume conv-user-tail")
    request = commands.take_pending_resume()
    assert request is not None
    commands.complete_resume(
        request,
        {
            "status": SimpleNamespace(status="completed"),
            "messages": [
                {"message_id": "a1", "role": "assistant", "text": "previous answer"},
                {"message_id": "u2", "role": "user", "text": "unanswered question"},
            ],
        },
    )

    markers = [event[1] for event in renderer.events if event[0] == "turn_marker"]
    assert markers == [
        {
            "label": "turn",
            "status": "unresolved",
            "message": "Canonical history ends after a user message; no final assistant response is recorded.",
        }
    ]


def test_resume_prefers_persistent_chat_terminal_marker_over_generic_unresolved(
    tmp_path,
) -> None:
    archive = TUIArchive(tmp_path / "archive")
    turn_id = archive.record_user(
        "old question",
        conversation_ref="conv-12345678",
        model=None,
    )
    archive.record_terminal(
        turn_id,
        conversation_ref="conv-12345678",
        label="chat",
        status="limit-reached",
        text="This conversation reached its maximum length; start a new chat to continue.",
        source="stream",
    )
    commands, renderer, _client, _state_path = make_commands(
        tmp_path,
        tui_archive=archive,
    )

    commands.handle("/resume conv-12345678")
    request = commands.take_pending_resume()
    assert request is not None
    commands.complete_resume(
        request,
        {
            "status": SimpleNamespace(status="completed"),
            "messages": [
                {"message_id": "u1", "role": "user", "text": "question"},
            ],
            "backend_stream_status": "COMPLETE",
            "backend_terminal_status_proven": True,
            "canonical_status_overridden": True,
            "canonical_status_before_override": "user_last_message",
            "canonical_terminal_text_missing": True,
        },
    )

    markers = [event[1] for event in renderer.events if event[0] == "turn_marker"]
    assert markers == [
        {
            "label": "chat",
            "status": "limit-reached",
            "message": (
                "This conversation reached its maximum length; "
                "start a new chat to continue."
            ),
        }
    ]


def test_resume_terminal_backend_override_with_recovered_final_is_informational(
    tmp_path,
) -> None:
    commands, renderer, _client, _state_path = make_commands(tmp_path)

    commands.handle("/resume conv-stale")
    request = commands.take_pending_resume()
    assert request is not None
    commands.complete_resume(
        request,
        {
            "status": SimpleNamespace(status="completed"),
            "messages": [
                {"message_id": "u1", "role": "user", "text": "question"},
                {"message_id": "a1", "role": "assistant", "text": "final"},
            ],
            "backend_stream_status": "COMPLETE",
            "backend_terminal_status_proven": True,
            "canonical_status_overridden": True,
            "canonical_status_before_override": "tool_running",
            "canonical_terminal_text_missing": False,
        },
    )

    infos = [event[1] for event in renderer.events if event[0] == "info"]
    assert (
        "Backend reports COMPLETE; ignored stale canonical status=tool_running."
        in infos
    )
    warnings = [event[1] for event in renderer.events if event[0] == "warning"]
    assert not any("unfinished turn" in warning for warning in warnings)


def test_resume_switches_while_already_attached_without_detach(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, _, client, state_path = make_commands(
        tmp_path,
        state=state,
        ui=FakeUI(choices=["conv-2"]),
    )

    commands.handle("/resume")
    assert state.current_conversation == "conv-1"

    finish_pending_resume(commands, client)

    assert client.calls[:2] == [
        ("list_conversations", None),
        ("snapshot", "conv-2"),
    ]
    assert load_chat_state(state_path).current_conversation == "conv-2"


def test_resume_direct_ref_skips_catalog_picker(tmp_path) -> None:
    commands, _, client, state_path = make_commands(tmp_path)

    commands.handle("/resume https://chatgpt.com/c/direct")

    assert client.calls == []
    assert load_chat_state(state_path).current_conversation is None
    finish_pending_resume(commands, client)
    assert client.calls == [("snapshot", "direct")]
    assert load_chat_state(state_path).current_conversation == "direct"


def test_resume_renders_and_persists_historical_web_ui_error(tmp_path) -> None:
    archive = TUIArchive(tmp_path / "archive")
    commands, renderer, _client, _state_path = make_commands(
        tmp_path,
        tui_archive=archive,
    )
    commands.handle("/resume conv-direct")
    request = commands.take_pending_resume()
    assert request is not None

    snapshot = {
        "status": SimpleNamespace(status="completed"),
        "messages": [
            {"message_id": "u1", "role": "user", "text": "question"},
        ],
        "backend_terminal_status_proven": True,
        "canonical_status_overridden": True,
        "canonical_terminal_text_missing": True,
        "historical_ui_state": {
            "code": "response_error",
            "scope": "turn",
            "status": "abnormal",
            "detail": "ChatGPT web UI reports an error for the last turn; Retry may be available.",
            "source": "web-ui",
        },
    }

    assert commands.complete_resume(request, snapshot) is True

    assert (
        "turn_marker",
        {
            "label": "turn",
            "status": "abnormal",
            "message": "ChatGPT web UI reports an error for the last turn; Retry may be available.",
        },
    ) in renderer.events
    assert not any(
        event[0] == "turn_marker" and event[1].get("status") == "unresolved"
        for event in renderer.events
    )
    transcript = archive.conversation_paths("conv-direct")["transcript"].read_text(
        encoding="utf-8"
    )
    assert "## TURN — abnormal" in transcript
    assert "Retry may be available." in transcript


def test_reload_refreshes_same_chat_without_detach_or_goal_pause(tmp_path) -> None:
    image = tmp_path / "queued.png"
    image.write_bytes(b"png")
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(
            conversation_ref="conv-1",
            status="active",
            objective="finish it",
        ),
    )
    commands, renderer, client, state_path = make_commands(tmp_path, state=state)
    commands.handle(f"/image {image}")
    assert commands.pending_media == [str(image)]

    commands.handle("/reload")

    request = commands.take_pending_resume()
    assert request is not None
    assert request.conversation_ref == "conv-1"
    assert request.reload is True
    assert state.current_conversation == "conv-1"
    assert state.goal is not None and state.goal.status == "active"
    assert commands.pending_media == [str(image)]
    assert client.calls == []
    assert ("info", "Reloading: conv-1") in renderer.events

    snapshot = client.conversation_snapshot(request.conversation_ref)
    assert commands.complete_resume(request, snapshot) is True

    assert load_chat_state(state_path).current_conversation == "conv-1"
    assert state.goal is not None and state.goal.status == "active"
    assert commands.pending_media == [str(image)]
    assert ("info", "Reloaded: conv-1") in renderer.events


def test_reload_failure_keeps_current_attachment(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, state_path = make_commands(tmp_path, state=state)
    save_chat_state(state_path, state)

    commands.handle("/reload")
    request = commands.take_pending_resume()
    assert request is not None and request.reload is True
    commands.fail_resume(request, RuntimeError("snapshot failed"))

    assert state.current_conversation == "conv-1"
    assert load_chat_state(state_path).current_conversation == "conv-1"
    assert any(
        event[0] == "warning"
        and event[1] == "Reload failed for conv-1: snapshot failed"
        for event in renderer.events
    )


def test_reload_rejects_args_missing_attachment_and_temporary_chat(tmp_path) -> None:
    commands, renderer, _, _ = make_commands(tmp_path)

    commands.handle("/reload extra")
    assert renderer.events[-1] == ("warning", "/reload takes no arguments.")
    assert commands.take_pending_resume() is None

    commands.handle("/reload")
    assert renderer.events[-1] == ("info", "No conversation is attached.")
    assert commands.take_pending_resume() is None

    commands.handle("/temporary")
    commands.handle("/reload")
    assert renderer.events[-1] == (
        "warning",
        "/reload is unavailable for Temporary ChatGPT conversations.",
    )
    assert commands.take_pending_resume() is None


def test_detach_is_local_only(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, client, state_path = make_commands(tmp_path, state=state)

    commands.handle("/detach")

    assert state.current_conversation is None
    assert load_chat_state(state_path).current_conversation is None
    assert client.calls == []
    assert renderer.events[0] == ("clear_context", None)
    assert "not changed" in renderer.events[-1][1]


def test_stop_command_stops_current_chat_without_detaching(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, client, _ = make_commands(tmp_path, state=state)

    commands.handle("/stop")

    assert state.current_conversation == "conv-1"
    assert ("stop_generation", ("conv-1", {"timeout": 2.0})) in client.calls
    assert ("turn_abort", None) in renderer.events
    assert ("info", "Stop requested.") in renderer.events


def test_temporary_command_clears_persistent_attachment_without_persisting_temp_id(
    tmp_path,
) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, state_path = make_commands(tmp_path, state=state)

    commands.handle("/temporary")

    assert commands.conversation_mode == "temporary"
    assert commands.conversation_ref is None
    assert state.current_conversation is None
    assert load_chat_state(state_path).current_conversation is None
    assert any(
        event == ("header", {"model": "latest frontier · High", "temporary": True})
        for event in renderer.events
    )


def test_temporary_export_uses_live_transcript_and_prints_exact_path(
    tmp_path, monkeypatch
) -> None:
    commands, renderer, client, _ = make_commands(tmp_path)
    exported: list[tuple[list[object], str | None]] = []
    export_path = tmp_path / "temporary.md"

    def fake_export(messages, *, title=None):
        exported.append((list(messages), title))
        return export_path

    monkeypatch.setattr("gptty.ui.commands.save_markdown_export", fake_export)
    commands.handle("/temporary")
    commands.record_temporary_turn(
        prompt="hello",
        answer="hi",
        conversation_ref="temp-1",
        title="Temporary title",
    )
    commands.handle("/export")

    assert [message.role for message in exported[0][0]] == ["user", "assistant"]
    assert [message.text for message in exported[0][0]] == ["hello", "hi"]
    assert exported[0][1] == "Temporary title"
    assert ("get_messages", "temp-1") not in client.calls
    assert renderer.events[-1] == ("info", f"Exported Markdown: {export_path}")


def test_normal_export_reads_complete_attached_history_from_cwa(
    tmp_path, monkeypatch
) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, client, _ = make_commands(tmp_path, state=state)
    exported: list[list[object]] = []
    export_path = tmp_path / "normal.md"

    monkeypatch.setattr(
        "gptty.ui.commands.save_markdown_export",
        lambda messages, *, title=None: exported.append(list(messages)) or export_path,
    )
    commands.handle("/export")

    assert ("get_messages", "conv-1") in client.calls
    assert [message.text for message in exported[0]] == ["question", "answer"]
    assert renderer.events[-1] == ("info", f"Exported Markdown: {export_path}")


def test_new_ends_live_temporary_lifecycle(tmp_path) -> None:
    commands, _, client, _ = make_commands(tmp_path)
    commands.handle("/temporary")
    commands.record_temporary_turn(
        prompt="hello",
        answer="hi",
        conversation_ref="temp-1",
        title=None,
    )

    commands.handle("/new")

    assert commands.conversation_mode == "normal"
    assert ("temporary_lifecycle_snapshot", None) in client.calls
    assert ("end_temporary_chat", None) in client.calls


def test_image_command_queues_real_file_for_next_prompt(tmp_path) -> None:
    image = tmp_path / "screen shot.png"
    image.write_bytes(b"png")
    commands, renderer, _, _ = make_commands(tmp_path)

    commands.handle(f'/image "{image}"')

    assert commands.pending_media == [str(image)]
    assert commands.pending_media_count == 1
    assert "Attached for next prompt" in renderer.events[-1][1]


def test_image_command_without_argument_uses_path_prompt(tmp_path) -> None:
    image = tmp_path / "picked image.png"
    image.write_bytes(b"png")
    dragged_path = str(image).replace(" ", "\\ ")
    commands, _, _, _ = make_commands(tmp_path, ui=FakeUI(image_paths=[dragged_path]))

    commands.handle("/image")

    assert commands.pending_media == [str(image)]


def test_async_image_command_uses_persistent_path_prompt(tmp_path) -> None:
    image = tmp_path / "picked async image.png"
    image.write_bytes(b"png")
    dragged_path = str(image).replace(" ", "\\ ")
    commands, _, _, _ = make_commands(tmp_path, ui=FakeUI(image_paths=[dragged_path]))

    asyncio.run(commands.handle_async("/image"))

    assert commands.pending_media == [str(image)]


def test_image_clear_removes_pending_clipboard_temp_file(tmp_path, monkeypatch) -> None:
    clipboard_image = tmp_path / "clipboard.png"
    clipboard_image.write_bytes(b"png")
    commands, renderer, _, _ = make_commands(tmp_path)
    monkeypatch.setattr(
        "gptty.ui.commands.tempfile.mkdtemp", lambda **_kwargs: str(tmp_path)
    )
    monkeypatch.setattr(
        "gptty.ui.commands.capture_clipboard_image", lambda _directory: clipboard_image
    )

    commands.handle("/paste")
    assert commands.pending_media == [str(clipboard_image)]

    commands.handle("/image clear")

    assert commands.pending_media == []
    assert not clipboard_image.exists()
    assert "Cleared 1 pending image" in renderer.events[-1][1]


def test_resume_clears_pending_images_before_switching_context(tmp_path) -> None:
    image = tmp_path / "queued.png"
    image.write_bytes(b"png")
    commands, _, client, _ = make_commands(tmp_path)
    commands.handle(f"/image {image}")

    commands.handle("/resume conv-1")
    assert commands.pending_media == [str(image)]

    finish_pending_resume(commands, client)
    assert commands.pending_media == []


def test_model_uses_live_catalog_slug(tmp_path) -> None:
    state = ChatState(model="old")
    commands, renderer, client, state_path = make_commands(
        tmp_path,
        state=state,
        ui=FakeUI(choices=["gpt-real-b"]),
    )

    commands.handle("/model")

    assert client.calls == [("list_models", None)]
    assert state.model == "gpt-real-b"
    assert load_chat_state(state_path).model == "gpt-real-b"
    assert renderer.events[-1] == ("info", "Model: gpt-real-b")


def test_model_async_picker_stays_in_enhanced_ui(tmp_path) -> None:
    state = ChatState(model="old")
    ui = FakeUI(choices=["gpt-real-b"])
    commands, renderer, client, state_path = make_commands(
        tmp_path,
        state=state,
        ui=ui,
    )

    assert asyncio.run(commands.handle_async("/model")) is None

    assert client.calls == [("list_models", None)]
    assert ui.seen[-1][0] == "ChatGPT model"
    assert state.model == "gpt-real-b"
    assert load_chat_state(state_path).model == "gpt-real-b"
    assert renderer.events[-1] == ("info", "Model: gpt-real-b")


def test_model_picker_excludes_non_chat_modes(tmp_path) -> None:
    ui = FakeUI(choices=[None])
    commands, _, client, _ = make_commands(tmp_path, ui=ui)

    commands.handle("/model")

    assert client.calls == [("list_models", None)]
    _message, options = ui.seen[-1]
    values = [value for value, _label in options]
    assert options[0][1].startswith("Default · latest frontier · High")
    assert "gpt-real-a" in values
    assert "gpt-real-b" in values
    assert "disabled" not in values
    assert "work-mode" not in values
    assert "research" not in values


def test_model_picker_can_reset_to_default(tmp_path) -> None:
    state = ChatState(model="gpt-real-a")
    commands, renderer, client, state_path = make_commands(
        tmp_path,
        state=state,
        ui=FakeUI(choices=[""]),
    )

    commands.handle("/model")

    assert client.calls == [("list_models", None)]
    assert state.model is None
    assert load_chat_state(state_path).model is None
    assert renderer.events[-1] == ("info", "Model: latest frontier · High")


def test_model_default_is_local_only(tmp_path) -> None:
    state = ChatState(model="gpt-real-a")
    commands, renderer, client, state_path = make_commands(tmp_path, state=state)

    commands.handle("/model default")

    assert client.calls == []
    assert state.model is None
    assert load_chat_state(state_path).model is None
    assert renderer.events[-1] == ("info", "Model: latest frontier · High")


def test_model_rejects_slug_not_in_live_catalog(tmp_path) -> None:
    state = ChatState(model="gpt-real-a")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    commands.handle("/model invented")

    assert state.model == "gpt-real-a"
    assert renderer.events[-1][0] == "warning"
    assert "live ChatGPT list" in renderer.events[-1][1]


def test_resume_unfinished_snapshot_attaches_without_follow_polling(tmp_path) -> None:
    unfinished = {
        "status": SimpleNamespace(status="tool_running"),
        "messages": [
            {"message_id": "u1", "role": "user", "text": "question"},
            {"message_id": "t1", "role": "tool", "text": "tool output"},
        ],
    }
    client = FakeClient(snapshots=[unfinished])
    commands, renderer, _, state_path = make_commands(
        tmp_path,
        ui=FakeUI(choices=["conv-1"]),
        client=client,
    )

    commands.handle("/resume")
    finish_pending_resume(commands, client)

    assert load_chat_state(state_path).current_conversation == "conv-1"
    assert [call[0] for call in client.calls].count("snapshot") == 1
    assert not any(call[0] == "stop_generation" for call in client.calls)
    assert any(
        event[0] == "warning" and "unfinished turn (status=tool_running)" in event[1]
        for event in renderer.events
    )


def test_resume_failure_keeps_previous_attachment(tmp_path) -> None:
    state = ChatState(current_conversation="conv-old")
    commands, renderer, _, state_path = make_commands(tmp_path, state=state)

    commands.handle("/resume conv-new")
    request = commands.take_pending_resume()
    assert request is not None
    commands.fail_resume(request, RuntimeError("snapshot failed"))

    assert state.current_conversation == "conv-old"
    assert any(
        event[0] == "warning" and "snapshot failed" in event[1]
        for event in renderer.events
    )


def test_new_clears_current_conversation(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, state_path = make_commands(tmp_path, state=state)

    commands.handle("/new")

    assert state.current_conversation is None
    assert load_chat_state(state_path).current_conversation is None
    assert renderer.events[0] == ("clear_context", None)


def test_state_save_failure_rolls_back_interactive_change(
    tmp_path, monkeypatch
) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    def fail_save(*args, **kwargs) -> None:
        raise StateError("disk failed")

    monkeypatch.setattr("gptty.ui.commands.save_chat_state", fail_save)
    commands.handle("/new")

    assert state.current_conversation == "conv-1"
    assert renderer.events[-1] == ("warning", "disk failed")


def test_goal_command_starts_on_attached_conversation_and_queues_activation(
    tmp_path,
) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, state_path = make_commands(tmp_path, state=state)

    commands.handle("/goal")

    assert state.goal is not None
    assert state.goal.status == "active"
    assert state.goal.conversation_ref == "conv-1"
    assert state.goal.conversations == ["conv-1"]
    assert state.goal.context_seed == [
        "user: question",
        "assistant: answer",
    ]
    assert state.goal.goal_id is not None
    assert load_chat_state(state_path).goal == state.goal
    goal_path = tmp_path / "goals" / state.goal.goal_id / "goal.json"
    checkpoint_path = tmp_path / "goals" / state.goal.goal_id / "checkpoint.md"
    assert goal_path.exists()
    assert checkpoint_path.exists()
    prompt = commands.pop_automatic_prompt()
    assert prompt is not None
    assert "GPTTY Goal mode is now active" in prompt
    assert any(kind == "info" and str(message).startswith("Goal · active · ") and str(message).endswith(" · starting") for kind, message in renderer.events)
    assert ("info", f"Goal state: {goal_path}") in renderer.events


def test_goal_start_keeps_authoritative_new_goal_if_local_chat_state_save_fails(
    tmp_path, monkeypatch
) -> None:
    previous_goal = GoalState(
        goal_id="goal-old",
        conversation_ref="conv-1",
        conversations=["conv-1"],
        status="complete",
        objective="Old finished goal",
    )
    state = ChatState(current_conversation="conv-1", goal=previous_goal)
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    commands.goal_store.save(previous_goal)

    def fail_save(*args, **kwargs) -> None:
        raise StateError("disk failed")

    monkeypatch.setattr("gptty.ui.commands.save_chat_state", fail_save)
    commands.handle('/goal "New goal"')

    assert state.goal is not None
    assert state.goal.goal_id != "goal-old"
    assert state.goal.objective == "New goal"
    assert commands.goal_store.load(state.goal.goal_id) == state.goal
    assert commands.goal_store.load_for_conversation("conv-1").goal_id == state.goal.goal_id
    assert commands.goal_store.load("goal-old").status == "complete"
    assert ("warning", "disk failed") in renderer.events


def test_goal_clear_keeps_authoritative_unbind_if_local_chat_state_save_fails(
    tmp_path, monkeypatch
) -> None:
    goal = GoalState(
        goal_id="goal-clear",
        conversation_ref="conv-1",
        conversations=["conv-1"],
        status="paused",
    )
    state = ChatState(current_conversation="conv-1", goal=goal)
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    commands.goal_store.save(goal)

    def fail_save(*args, **kwargs) -> None:
        raise StateError("disk failed")

    monkeypatch.setattr("gptty.ui.commands.save_chat_state", fail_save)
    commands.handle("/goal clear")

    assert state.goal is None
    retained = commands.goal_store.load("goal-clear")
    assert retained is not None
    assert retained.status == "interrupted"
    assert retained.reason == "cleared by user"
    assert commands.goal_store.load_for_conversation("conv-1") is None
    assert commands.goal_store.list_goals(statuses={"active", "paused", "blocked"}) == []
    assert renderer.events[-1] == ("warning", "disk failed")


def test_goal_rollover_safety_limit_interrupts_instead_of_looping(
    tmp_path, monkeypatch
) -> None:
    goal = GoalState(
        goal_id="goal-loop",
        conversation_ref="conv-1",
        conversations=["conv-1"],
        status="active",
        rollover_count=MAX_ROLLOVERS,
    )
    state = ChatState(current_conversation="conv-1", goal=goal)
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    commands.handle_goal_turn_result(
        {
            "text": "",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
            "terminal_marker": (
                "chat",
                "limit-reached",
                "This conversation reached its maximum length.",
            ),
        }
    )

    assert goal.status == "interrupted"
    assert goal.rollover_count == MAX_ROLLOVERS
    assert commands.has_automatic_prompt is False
    assert "rollover safety limit reached" in (goal.reason or "")
    assert any(
        event[0] == "warning" and "Goal · interrupted" in str(event[1])
        for event in renderer.events
    )
    assert notified


def test_goal_command_can_start_new_chat_with_explicit_objective(tmp_path) -> None:
    commands, _, _, _ = make_commands(tmp_path)

    commands.handle('/goal "Finish the exact agreed task"')

    assert commands.goal_active is True
    assert commands.state.goal is not None
    assert commands.state.goal.conversation_ref is None
    assert commands.state.goal.objective == "Finish the exact agreed task"
    assert "Finish the exact agreed task" in (commands.pop_automatic_prompt() or "")


def test_goal_command_requires_objective_when_no_chat_context_exists(tmp_path) -> None:
    commands, renderer, _, _ = make_commands(tmp_path)

    commands.handle("/goal")

    assert commands.state.goal is None
    assert renderer.events[-1] == (
        "warning",
        "No conversation is attached. Use /goal <objective> to start a Goal in a new chat.",
    )


def test_goal_continue_queues_next_turn_without_notification(
    tmp_path, monkeypatch
) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    commands.handle_goal_turn_result(
        {
            "text": "GPTTY_GOAL: CONTINUE\nImplemented half; tests remain.",
            "title": "Goal chat",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
        }
    )

    assert state.goal is not None
    assert state.goal.status == "active"
    assert state.goal.turn_count == 1
    assert state.goal.protocol_failures == 0
    assert commands.has_automatic_prompt is True
    assert "Continue pursuing the active goal" in (
        commands.pop_automatic_prompt() or ""
    )
    assert notified == []
    assert ("info", "Goal · continuing · next turn 2") in renderer.events


def test_goal_complete_stops_loop_and_sends_single_clean_notification(
    tmp_path, monkeypatch
) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active", turn_count=2),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    commands.handle_goal_turn_result(
        {
            "text": "GPTTY_GOAL: COMPLETE\n"
                'GPTTY_CHECKPOINT: {"summary":"done","completed":["verified"],"decisions":[],"pending":[],"next":"none"}\n'
                "Everything is implemented and verified.",
            "title": "Goal chat",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
        }
    )

    assert state.goal is not None
    assert state.goal.status == "complete"
    assert state.goal.turn_count == 3
    assert commands.has_automatic_prompt is False
    assert notified == [
        {
            "chat_title": "Goal chat",
            "final_response": "Everything is implemented and verified.",
        }
    ]
    assert ("info", "Goal · complete · 3 turns") in renderer.events


def test_goal_blocked_stops_loop_and_notifies_for_user_action(
    tmp_path, monkeypatch
) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    commands.handle_goal_turn_result(
        {
            "text": "GPTTY_GOAL: BLOCKED\nPlease log in to the provider account.",
            "title": "Goal chat",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
        }
    )

    assert state.goal is not None
    assert state.goal.status == "blocked"
    assert commands.has_automatic_prompt is False
    assert notified == [
        {
            "chat_title": "Goal chat",
            "final_response": "Goal blocked. Please log in to the provider account.",
        }
    ]
    assert ("warning", "Goal · blocked · user action required") in renderer.events


def test_goal_missing_status_recovers_twice_then_rolls_over(
    tmp_path, monkeypatch
) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    for expected_failures in (1, 2):
        commands.handle_goal_turn_result(
            {
                "text": "Turn ended without the protocol marker.",
                "title": "Goal chat",
                "conversation_ref": "conv-1",
                "stopped_by_user": False,
            }
        )
        assert state.goal is not None
        assert state.goal.status == "active"
        assert state.goal.protocol_failures == expected_failures
        recovery = commands.pop_automatic_prompt() or ""
        assert "previous turn ended without a valid GPTTY_GOAL status line" in recovery

    commands.handle_goal_turn_result(
        {
            "text": "Still no marker.",
            "title": "Goal chat",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
        }
    )

    assert state.goal is not None
    assert state.goal.status == "active"
    assert state.goal.protocol_failures == 0
    assert state.goal.rollover_count == 1
    assert state.goal.conversation_ref is None
    assert state.goal.conversations == ["conv-1"]
    assert state.current_conversation is None
    assert commands.goal_bootstrap_pending is True
    handoff = commands.pop_automatic_prompt() or ""
    assert "fresh ChatGPT conversation" in handoff
    assert "missing valid GPTTY_GOAL status for 3 consecutive turns" in handoff
    assert notified == []


def test_goal_complete_survives_post_final_chat_limit_without_rollover(
    tmp_path, monkeypatch
) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(
            goal_id="goal-complete",
            conversation_ref="conv-1",
            conversations=["conv-1"],
            status="active",
        ),
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    commands.handle_goal_turn_result(
        {
            "text": (
                "GPTTY_GOAL: COMPLETE\n"
                'GPTTY_CHECKPOINT: {"summary":"done","completed":["verified"],'
                '"decisions":[],"pending":[],"next":"none"}\n'
                "Everything is finished."
            ),
            "title": "Goal chat",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
            "terminal_marker": (
                "chat",
                "limit-reached",
                "This conversation reached its maximum length; start a new chat to continue.",
            ),
        }
    )

    assert state.goal is not None
    assert state.goal.status == "complete"
    assert state.goal.rollover_count == 0
    assert state.goal.checkpoint.completed == ["verified"]
    assert state.current_conversation == "conv-1"
    assert commands.has_automatic_prompt is False
    assert notified == [
        {
            "chat_title": "Goal chat",
            "final_response": "Everything is finished.",
        }
    ]


def test_goal_complete_survives_post_final_chat_unavailable_without_rollover(
    tmp_path, monkeypatch
) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(
            goal_id="goal-complete-unavailable",
            conversation_ref="conv-1",
            conversations=["conv-1"],
            status="active",
        ),
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    commands.handle_goal_turn_result(
        {
            "text": (
                "GPTTY_GOAL: COMPLETE\n"
                'GPTTY_CHECKPOINT: {"summary":"done","completed":["verified"],'
                '"decisions":[],"pending":[],"next":"none"}\n'
                "Everything is finished."
            ),
            "title": "Goal chat",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
            "terminal_marker": (
                "chat",
                "unavailable",
                "This conversation is no longer available; continue in a new chat.",
            ),
        }
    )

    assert state.goal is not None
    assert state.goal.status == "complete"
    assert state.goal.rollover_count == 0
    assert state.goal.checkpoint.completed == ["verified"]
    assert commands.has_automatic_prompt is False
    assert notified[0]["final_response"] == "Everything is finished."


def test_goal_truncated_turn_continues_same_chat_without_recovery_escalation(
    tmp_path,
) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(
            goal_id="goal-truncated",
            conversation_ref="conv-1",
            conversations=["conv-1"],
            status="active",
        ),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    for _ in range(4):
        commands.handle_goal_turn_result(
            {
                "text": "Partial answer",
                "conversation_ref": "conv-1",
                "terminal_marker": (
                    "turn",
                    "truncated",
                    "ChatGPT ended the response at an output-length limit.",
                ),
            }
        )

    assert state.goal is not None
    assert state.goal.status == "active"
    assert state.goal.recovery_count == 0
    assert state.goal.rollover_count == 0
    assert state.goal.conversation_ref == "conv-1"
    assert commands.has_automatic_prompt is True
    assert (
        "info",
        "Goal · continuing · response was truncated",
    ) in renderer.events


def test_goal_abnormal_turn_preserves_structured_checkpoint_before_recovery(
    tmp_path,
) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(
            goal_id="goal-checkpoint-recovery",
            conversation_ref="conv-1",
            conversations=["conv-1"],
            status="active",
        ),
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)

    commands.handle_goal_turn_result(
        {
            "text": (
                "GPTTY_GOAL: CONTINUE\n"
                'GPTTY_CHECKPOINT: {"summary":"write landed","completed":["commit A pushed"],'
                '"decisions":["keep old API"],"pending":["verify"],"next":"inspect repo"}\n'
                "The web turn ended strangely after the write."
            ),
            "conversation_ref": "conv-1",
            "terminal_marker": (
                "turn",
                "abnormal",
                "ChatGPT web UI reports an error for the last turn.",
            ),
        }
    )

    assert state.goal is not None
    assert state.goal.checkpoint.completed == ["commit A pushed"]
    assert state.goal.checkpoint.decisions == ["keep old API"]
    assert state.goal.checkpoint.next_step == "inspect repo"
    assert state.goal.checkpoint.updated_turn == 1
    assert state.goal.recovery_count == 1
    assert commands.has_automatic_prompt is True


def test_goal_repeated_abnormal_turns_reconcile_then_roll_over(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(
            goal_id="goal-recover",
            conversation_ref="conv-1",
            conversations=["conv-1"],
            status="active",
        ),
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)
    result = {
        "text": "",
        "conversation_ref": "conv-1",
        "stopped_by_user": False,
        "terminal_marker": (
            "turn",
            "abnormal",
            "ChatGPT web UI reports an error for the last turn.",
        ),
    }

    for expected_attempt in (1, 2):
        commands.handle_goal_turn_result(result)
        assert state.goal is not None
        assert state.goal.recovery_count == expected_attempt
        recovery = commands.pop_automatic_prompt() or ""
        assert "Do not blindly repeat the previous action" in recovery
        assert state.current_conversation == "conv-1"

    commands.handle_goal_turn_result(result)

    assert state.goal is not None
    assert state.goal.rollover_count == 1
    assert state.goal.recovery_count == 0
    assert state.goal.conversation_ref is None
    assert state.current_conversation is None
    assert commands.goal_bootstrap_pending is True
    handoff = commands.pop_automatic_prompt() or ""
    assert "repeated non-standard turns" in handoff


def test_goal_rate_limit_pauses_instead_of_spamming_requests(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(
            goal_id="goal-rate",
            conversation_ref="conv-1",
            conversations=["conv-1"],
            status="active",
        ),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    commands.handle_goal_turn_result(
        {
            "text": "",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
            "terminal_marker": (
                "turn",
                "rate-limited",
                "ChatGPT rate-limited this request.",
            ),
        }
    )

    assert state.goal is not None
    assert state.goal.status == "paused"
    assert state.goal.rollover_count == 0
    assert commands.has_automatic_prompt is False
    assert any(
        event[0] == "warning" and "service backoff required" in str(event[1])
        for event in renderer.events
    )


def test_goal_user_stop_pauses_and_never_auto_continues(tmp_path, monkeypatch) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr(
        "gptty.ui.commands.notify_response_complete",
        lambda **kwargs: notified.append(kwargs),
    )

    commands.pause_goal_after_user_stop("conv-1")

    assert state.goal is not None
    assert state.goal.status == "paused"
    assert state.goal.reason == "stopped by user"
    assert state.goal.turn_count == 1
    assert commands.has_automatic_prompt is False
    assert notified == []
    assert renderer.events.count(("info", "Goal · paused · stopped by user")) == 1

    commands.handle_goal_turn_result(
        {
            "text": "GPTTY_GOAL: CONTINUE\nPartial response",
            "title": "Goal chat",
            "conversation_ref": "conv-1",
            "stopped_by_user": True,
        }
    )
    assert state.goal.turn_count == 1
    assert renderer.events.count(("info", "Goal · paused · stopped by user")) == 1


def test_goal_pause_resume_and_context_switch_keep_goal_routed_to_original_chat(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, _, client, state_path = make_commands(tmp_path, state=state)
    commands.handle("/goal important work")
    assert state.goal is not None
    goal_id = state.goal.goal_id
    assert commands.pop_automatic_prompt() is not None

    commands.handle("/goal pause")
    assert state.goal is not None and state.goal.status == "paused"
    commands.handle("/goal resume")
    assert state.goal is not None and state.goal.status == "active"
    assert commands.pop_automatic_prompt() is not None

    commands.handle("/new")
    assert state.goal is None
    assert state.current_conversation is None
    routed = commands.goal_store.load_for_conversation("conv-1")
    assert routed is not None and routed.goal_id == goal_id and routed.status == "paused"
    assert load_chat_state(state_path).goal is None

    commands._begin_resume("conv-1")
    finish_pending_resume(commands, client)
    assert state.goal is not None and state.goal.goal_id == goal_id
    commands.handle("/goal clear")
    assert state.goal is None
    assert commands.goal_store.load_for_conversation("conv-1") is None


def test_goal_resume_refuses_to_continue_in_different_conversation(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-2",
        goal=GoalState(conversation_ref="conv-1", status="paused", turn_count=4),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    commands.handle("/goal resume")

    assert state.goal is not None and state.goal.status == "paused"
    assert commands.has_automatic_prompt is False
    assert renderer.events[-1] == (
        "warning",
        "Goal belongs to conv-1. Resume that conversation before /goal resume.",
    )


def test_goal_is_rejected_in_temporary_chat(tmp_path) -> None:
    commands, renderer, _, _ = make_commands(tmp_path)
    commands.handle("/temporary")

    commands.handle("/goal should not run here")

    assert commands.state.goal is None
    assert commands.has_automatic_prompt is False
    assert renderer.events[-1] == (
        "warning",
        "Goal mode is only available for normal ChatGPT conversations.",
    )


def test_goal_complete_without_structured_checkpoint_is_rejected(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    commands.handle_goal_turn_result(
        {
            "text": "GPTTY_GOAL: COMPLETE\nI think it is done.",
            "conversation_ref": "conv-1",
            "stopped_by_user": False,
        }
    )

    assert state.goal is not None
    assert state.goal.status == "active"
    assert state.goal.protocol_failures == 1
    assert commands.has_automatic_prompt is True
    assert "missing status" in renderer.events[-1][1]
    assert any(
        kind == "warning" and "rejected COMPLETE" in str(message)
        for kind, message in renderer.events
    )


def test_goal_complete_with_pending_work_is_rejected(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    commands.handle_goal_turn_result(
        {
            "text": (
                "GPTTY_GOAL: COMPLETE\n"
                'GPTTY_CHECKPOINT: {"summary":"almost","completed":["code"],'
                '"decisions":[],"pending":["live test"],"next":"run live test"}\n'
                "Done."
            ),
            "conversation_ref": "conv-1",
        }
    )

    assert state.goal is not None
    assert state.goal.status == "active"
    assert state.goal.checkpoint.pending == ["live test"]
    assert commands.has_automatic_prompt is True
    assert any(
        kind == "warning" and "still lists pending work" in str(message)
        for kind, message in renderer.events
    )


def test_goal_operation_identity_and_tool_evidence_survive_pause_resume(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, _, _, _ = make_commands(tmp_path, state=state)
    commands.handle('/goal "durable side effect test"')
    activation = commands.pop_automatic_prompt()
    assert activation is not None

    outgoing = commands.mark_goal_turn_started(activation, automatic=True)
    assert outgoing is not None
    assert state.goal is not None
    operation_id = state.goal.active_operation_id
    assert operation_id
    assert operation_id in outgoing

    call_event = {
        "type": "canonical_intermediate_message",
        "message_kind": "tool_call",
        "message_id": "tool-call-1",
        "tool_name": "api_tool.call_tool",
        "label": "write marker",
        "text": '{"path":"/CodexTool/link/bash","args":{"command":"touch marker"}}',
    }
    commands.record_goal_tool_event(call_event)
    commands.record_goal_tool_event(call_event)
    commands.record_goal_tool_event(
        {
            **call_event,
            "message_kind": "tool_result",
            "message_id": "tool-result-1",
            "label": "marker written",
            "text": '{"exitCode":0}',
        }
    )

    commands.handle("/goal pause")
    assert state.goal.status == "paused"
    assert state.goal.active_operation_id == operation_id
    commands.handle("/goal resume")
    recovery = commands.pop_automatic_prompt() or ""
    assert operation_id in recovery
    assert "Do not blindly repeat" in recovery

    events = commands.goal_store.events(state.goal)
    kinds = [event["type"] for event in events]
    assert kinds.count("tool_call_observed") == 1
    assert kinds.count("tool_result_observed") == 1
    assert "operation_started" in kinds
    assert "goal_paused" in kinds
    assert "goal_resumed" in kinds


def test_goal_rollover_is_continue_as_new_generation_with_machine_journal(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-old",
        goal=GoalState(
            goal_id="goal-generation",
            conversation_ref="conv-old",
            conversations=["conv-old"],
            status="active",
        ),
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)
    prepared = commands.prepare_goal_user_prompt(
        "Keep the public API unchanged even after rollover."
    )
    assert prepared is not None

    assert commands._rollover_goal("conversation exhausted") is True

    assert state.goal is not None
    assert state.goal.generation == 2
    assert state.goal.rollover_count == 1
    assert state.goal.conversation_ref is None
    handoff = commands.pop_automatic_prompt() or ""
    assert "Goal generation: 2" in handoff
    assert "Keep the public API unchanged even after rollover." in handoff
    assert "rollover: generation 1 -> 2" in handoff


def test_goal_start_journals_full_visible_context_while_seed_stays_compact(tmp_path) -> None:
    class LongContextClient(FakeClient):
        def get_messages(self, ref):
            self.calls.append(("get_messages", ref))
            return [
                {"role": "user" if index % 2 == 0 else "assistant", "text": f"message-{index}-" + "x" * 200}
                for index in range(20)
            ]

    client = LongContextClient()
    state = ChatState(current_conversation="conv-context")
    commands, _, _, _ = make_commands(tmp_path, state=state, client=client)

    commands.handle('/goal "preserve all context"')

    assert state.goal is not None
    assert len(state.goal.context_seed) == 12
    events = commands.goal_store.events(state.goal)
    created = next(event for event in events if event["type"] == "goal_created")
    snapshot = created["payload"]["context_snapshot"]
    assert len(snapshot) == 20
    assert "message-0-" in snapshot[0]
    assert not any("message-0-" in item for item in state.goal.context_seed)
    portable_journal = commands.goal_store.goal_dir(state.goal) / "events.jsonl"
    assert portable_journal.exists()
    assert "message-0-" in portable_journal.read_text(encoding="utf-8")


def test_remote_live_goal_owner_is_read_only_in_second_commands_instance(tmp_path) -> None:
    import os

    owner = InteractiveCommands(
        state=ChatState(current_conversation="conv-owner"),
        state_path=tmp_path / "gptty_state.json",
        get_client=lambda: FakeClient(),
        ui=FakeUI(),
        renderer=FakeRenderer(),
        runner_id="owner-runner",
    )
    owner.handle('/goal "owned work"')
    assert owner.state.goal is not None
    goal_id = owner.state.goal.goal_id
    owner.state.goal.runner_pid = os.getpid()
    owner._save_state(event_type="owner_heartbeat")

    remote_state = ChatState(
        current_conversation="conv-owner",
        goal=owner.goal_store.load_for_conversation("conv-owner"),
    )
    remote, renderer, _, _ = make_commands(
        tmp_path, state=remote_state, runner_id="other-runner"
    )
    assert remote.goal_active is False
    remote.handle("/goal pause")
    remote.handle("/goal clear")

    authoritative = remote.goal_store.load(goal_id)
    assert authoritative is not None
    assert authoritative.status == "active"
    assert authoritative.runner_id == "owner-runner"
    warnings = [str(message) for kind, message in renderer.events if kind == "warning"]
    assert any("another live gptty process" in message for message in warnings)


def test_remote_live_goal_does_not_block_local_context_switches(tmp_path) -> None:
    import os

    owner_state = ChatState(
        current_conversation="conv-owner",
        goal=GoalState(
            goal_id="goal-owner-guard",
            conversation_ref="conv-owner",
            conversations=["conv-owner"],
            status="active",
            runner_id="owner-runner",
            runner_pid=os.getpid(),
        ),
    )
    owner, _, _, _ = make_commands(tmp_path, state=owner_state, runner_id="owner-runner")
    owner._save_state(event_type="owner_ready")

    remote_state = ChatState(
        current_conversation="conv-owner",
        goal=owner.goal_store.load_for_conversation("conv-owner"),
    )
    remote, _, _, _ = make_commands(tmp_path, state=remote_state, runner_id="remote-runner")

    remote.handle("/new")
    assert remote.state.current_conversation is None
    assert remote.state.goal is None
    remote._begin_resume("conv-other")
    assert remote.has_pending_resume is True
    remote._apply_model("different-model")
    assert remote.state.model == "different-model"

    authoritative = remote.goal_store.load("goal-owner-guard")
    assert authoritative is not None
    assert authoritative.status == "active"
    assert authoritative.runner_id == "owner-runner"


def test_unresolved_machine_observed_tool_call_forces_reconciliation_before_complete(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-evidence",
        goal=GoalState(
            goal_id="goal-evidence-veto",
            conversation_ref="conv-evidence",
            conversations=["conv-evidence"],
            status="active",
            active_operation_id="goal-evidence-veto:g1:t1",
            active_operation_turn=1,
        ),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    assert commands._save_state(event_type="operation_started") is True
    commands.goal_store.record_observed_event(
        state.goal,
        "tool_call_observed",
        {
            "operation_id": state.goal.active_operation_id,
            "label": "write marker",
            "text": "touch marker.txt",
        },
        event_key="unresolved-call",
    )
    complete = {
        "text": (
            "GPTTY_GOAL: COMPLETE\n"
            'GPTTY_CHECKPOINT: {"summary":"done","completed":["marker written"],'
            '"decisions":[],"pending":[],"next":"none"}\n'
            "Everything is done."
        ),
        "conversation_ref": "conv-evidence",
    }

    commands.handle_goal_turn_result(complete)

    assert state.goal.status == "active"
    assert state.goal.active_operation_id == "goal-evidence-veto:g1:t1"
    assert state.goal.recovery_count == 1
    recovery = commands.pop_automatic_prompt() or ""
    assert "machine journal has 1 observed tool call" in recovery
    assert "touch marker.txt" in recovery
    assert any(
        kind == "warning" and "rejected COMPLETE" in str(message)
        for kind, message in renderer.events
    )

    commands.goal_store.record_observed_event(
        state.goal,
        "tool_result_observed",
        {
            "operation_id": state.goal.active_operation_id,
            "label": "marker written",
            "text": "exit 0",
        },
        event_key="resolved-result",
    )
    commands.handle_goal_turn_result(complete)
    assert state.goal.status == "complete"
    assert state.goal.active_operation_id is None


def test_two_goal_runners_racing_resume_have_single_authoritative_owner(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    state_path = tmp_path / "gptty_state.json"
    store = GoalStore(state_path)
    paused = GoalState(
        goal_id="goal-resume-race",
        conversation_ref="conv-race",
        conversations=["conv-race"],
        status="paused",
        objective="resume exactly once",
    )
    store.save(paused, event_type="goal_created")

    first_state = ChatState(
        current_conversation="conv-race",
        goal=store.load("goal-resume-race"),
    )
    second_state = ChatState(
        current_conversation="conv-race",
        goal=store.load("goal-resume-race"),
    )
    first, _, _, _ = make_commands(
        tmp_path, state=first_state, runner_id="runner-A"
    )
    second, _, _, _ = make_commands(
        tmp_path, state=second_state, runner_id="runner-B"
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda command: command._resume_goal(), (first, second)))

    authoritative = store.load("goal-resume-race")
    assert authoritative is not None
    assert authoritative.status == "active"
    assert authoritative.runner_id in {"runner-A", "runner-B"}
    winners = [
        command
        for command in (first, second)
        if command.state.goal is not None
        and command.state.goal.runner_id == authoritative.runner_id
        and command.has_automatic_prompt
    ]
    assert len(winners) == 1
    losers = [command for command in (first, second) if command not in winners]
    assert len(losers) == 1
    assert losers[0].has_automatic_prompt is False
    assert losers[0].state.goal is not None
    assert losers[0].state.goal.runner_id == authoritative.runner_id


def test_accepted_complete_journals_machine_validation_evidence(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-validation",
        goal=GoalState(
            goal_id="goal-validation",
            conversation_ref="conv-validation",
            conversations=["conv-validation"],
            status="active",
            active_operation_id="goal-validation:g1:t1",
            active_operation_turn=1,
        ),
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)
    assert commands._save_state(event_type="operation_started") is True
    commands.goal_store.record_observed_event(
        state.goal,
        "tool_call_observed",
        {
            "operation_id": state.goal.active_operation_id,
            "tool_name": "verify_api",
            "text": "verify state",
        },
        event_key="validation-call",
    )
    commands.goal_store.record_observed_event(
        state.goal,
        "tool_result_observed",
        {
            "operation_id": state.goal.active_operation_id,
            "tool_name": "verify_api",
            "text": "verified ok",
        },
        event_key="validation-result",
    )

    commands.handle_goal_turn_result(
        {
            "text": (
                "GPTTY_GOAL: COMPLETE\n"
                'GPTTY_CHECKPOINT: {"summary":"verified complete","completed":["state verified"],'
                '"decisions":[],"pending":[],"next":"none"}\n'
                "Done."
            ),
            "conversation_ref": "conv-validation",
        }
    )

    assert state.goal is not None and state.goal.status == "complete"
    terminal = commands.goal_store.events(state.goal)[-1]
    assert terminal["type"] == "turn_terminal"
    validation = terminal["payload"]["machine_validation"]
    assert validation["structured_checkpoint"] is True
    assert validation["completed_count"] == 1
    assert validation["pending_count"] == 0
    assert validation["operation_evidence"]["unresolved_tool_calls"] == 0


def test_goal_transport_write_completion_is_durably_journaled(tmp_path) -> None:
    operation_id = "goal-route-ui:g2:t3"
    state = ChatState(
        goal=GoalState(
            goal_id="goal-route-ui",
            generation=2,
            status="active",
            active_operation_id=operation_id,
            active_operation_turn=3,
        )
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)
    assert commands._save_state(event_type="operation_resumed") is True

    event = {
        "type": "browser_native_write_completed",
        "conversation_id": "conv-route-ui-12345678",
        "submission_id": "submit-1",
        "turn_exchange_id": "exchange-1",
    }
    commands.record_goal_tool_event(event)
    commands.record_goal_tool_event(event)

    events = commands.goal_store.events(state.goal)
    committed = [event for event in events if event["type"] == "conversation_write_committed"]
    assert len(committed) == 1
    assert committed[0]["payload"]["operation_id"] == operation_id
    assert committed[0]["payload"]["conversation_ref"] == "conv-route-ui-12345678"


def test_goal_list_shows_multiple_unfinished_and_all_terminal_goals(tmp_path) -> None:
    commands, renderer, _, _ = make_commands(tmp_path, runner_id="viewer")
    active = GoalState(
        goal_id="aaaaaaaa11111111",
        conversation_ref="conv-a",
        conversations=["conv-a"],
        status="active",
        objective="active objective",
        runner_id="runner-a",
        runner_pid=12345,
        turn_count=2,
    )
    paused = GoalState(
        goal_id="bbbbbbbb22222222",
        conversation_ref="conv-b",
        conversations=["conv-b"],
        status="paused",
        objective="paused objective",
        turn_count=4,
    )
    complete = GoalState(
        goal_id="cccccccc33333333",
        conversation_ref="conv-c",
        conversations=["conv-c"],
        status="complete",
        objective="finished objective",
        turn_count=6,
    )
    for goal in (active, paused, complete):
        commands.goal_store.save(goal, event_type="goal_created")
    commands.state.current_conversation = "conv-b"
    commands.state.goal = commands.goal_store.load_for_conversation("conv-b")

    commands.handle("/goal list")
    infos = [str(message) for kind, message in renderer.events if kind == "info"]
    assert "Goals · 2 unfinished" in infos
    assert any("aaaaaaaa" in line and "active" in line and "conv-a" in line for line in infos)
    assert any(line.startswith("* paused") and "bbbbbbbb" in line for line in infos)
    assert not any("cccccccc" in line for line in infos)

    renderer.events.clear()
    commands.handle("/goal list all")
    infos = [str(message) for kind, message in renderer.events if kind == "info"]
    assert "Goals · 3 all" in infos
    assert any("cccccccc" in line and "complete" in line for line in infos)


def test_goal_open_switches_to_target_goal_by_unique_prefix(tmp_path) -> None:
    client = FakeClient()
    commands, renderer, _, _ = make_commands(
        tmp_path,
        state=ChatState(current_conversation="conv-a"),
        client=client,
        runner_id="runner-view",
    )
    left = GoalState(
        goal_id="aaaabbbb11112222",
        conversation_ref="conv-a",
        conversations=["conv-a"],
        status="paused",
        objective="left",
    )
    right = GoalState(
        goal_id="ccccdddd33334444",
        conversation_ref="conv-b",
        conversations=["conv-b"],
        status="paused",
        objective="right",
    )
    commands.goal_store.save(left, event_type="goal_created")
    commands.goal_store.save(right, event_type="goal_created")
    commands.state.goal = commands.goal_store.load_for_conversation("conv-a")

    commands.handle("/goal open cccc")
    request = commands.take_pending_resume()
    assert request is not None
    assert request.conversation_ref == "conv-b"
    snapshot = client.conversation_snapshot("conv-b")
    assert commands.complete_resume(request, snapshot) is True

    assert commands.state.current_conversation == "conv-b"
    assert commands.state.goal is not None
    assert commands.state.goal.goal_id == "ccccdddd33334444"
    assert any(
        kind == "info" and "Goal · paused" in str(message)
        for kind, message in renderer.events
    )


def test_goal_open_current_goal_is_instant_and_does_not_reload(tmp_path) -> None:
    client = FakeClient()
    goal = GoalState(
        goal_id="aaaabbbb11112222",
        conversation_ref="conv-a",
        conversations=["conv-a"],
        status="complete",
        objective="already here",
        turn_count=3,
    )
    state = ChatState(current_conversation="conv-a", goal=goal)
    commands, renderer, _, _ = make_commands(
        tmp_path,
        state=state,
        client=client,
        runner_id="runner-view",
    )
    commands.goal_store.save(goal, event_type="goal_created")

    commands.handle("/goal open aaaa")

    assert commands.has_pending_resume is False
    assert commands.state.current_conversation == "conv-a"
    assert commands.state.goal is not None
    assert commands.state.goal.goal_id == "aaaabbbb11112222"
    assert any(
        kind == "info" and "Goal · complete" in str(message)
        for kind, message in renderer.events
    )


def test_resume_picker_labels_conversations_with_goal_status(tmp_path) -> None:
    commands, _, client, _ = make_commands(tmp_path)
    commands.goal_store.save(
        GoalState(
            goal_id="goal-picker-active",
            conversation_ref="conv-2",
            conversations=["conv-2"],
            status="active",
            objective="picker active",
            runner_id="other",
            runner_pid=999,
        ),
        event_type="goal_created",
    )
    commands.goal_store.save(
        GoalState(
            goal_id="goal-picker-paused",
            conversation_ref="conv-1",
            conversations=["conv-1"],
            status="paused",
            objective="picker paused",
        ),
        event_type="goal_created",
    )

    options = commands._conversation_options(client)
    labels = dict(options)
    assert "Goal active goal-pic" in labels["conv-2"]
    assert "Goal paused goal-pic" in labels["conv-1"]


def test_two_different_goals_can_resume_concurrently_with_independent_owners(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store = GoalStore(tmp_path / "gptty_state.json")
    goal_a = GoalState(
        goal_id="goal-concurrent-a",
        conversation_ref="conv-a",
        conversations=["conv-a"],
        status="paused",
        objective="run A",
    )
    goal_b = GoalState(
        goal_id="goal-concurrent-b",
        conversation_ref="conv-b",
        conversations=["conv-b"],
        status="paused",
        objective="run B",
    )
    store.save(goal_a, event_type="goal_created")
    store.save(goal_b, event_type="goal_created")

    first, _, _, _ = make_commands(
        tmp_path,
        state=ChatState(
            current_conversation="conv-a",
            goal=store.load_for_conversation("conv-a"),
        ),
        runner_id="runner-A",
    )
    second, _, _, _ = make_commands(
        tmp_path,
        state=ChatState(
            current_conversation="conv-b",
            goal=store.load_for_conversation("conv-b"),
        ),
        runner_id="runner-B",
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda command: command._resume_goal(), (first, second)))

    authoritative_a = store.load("goal-concurrent-a")
    authoritative_b = store.load("goal-concurrent-b")
    assert authoritative_a is not None and authoritative_a.status == "active"
    assert authoritative_b is not None and authoritative_b.status == "active"
    assert authoritative_a.runner_id == "runner-A"
    assert authoritative_b.runner_id == "runner-B"
    assert first.has_automatic_prompt is True
    assert second.has_automatic_prompt is True
    assert store.goal_id_for_conversation("conv-a") == "goal-concurrent-a"
    assert store.goal_id_for_conversation("conv-b") == "goal-concurrent-b"


def test_goal_resume_history_hides_internal_protocol_and_preserves_steering(tmp_path) -> None:
    goal = GoalState(
        goal_id="goal-history-clean",
        conversation_ref="conv-goal-history",
        conversations=["conv-goal-history"],
        status="complete",
        objective="clean history",
        turn_count=2,
    )
    state = ChatState(current_conversation=None)
    commands, renderer, client, _ = make_commands(tmp_path, state=state)
    commands.goal_store.save(goal, event_type="goal_created")
    snapshot = {
        "status": SimpleNamespace(status="completed"),
        "messages": [
            {
                "role": "user",
                "text": (
                    "GPTTY Goal mode is now active. Pursue the task.\n\n"
                    "GPTTY_GOAL: CONTINUE"
                ),
            },
            {
                "role": "assistant",
                "text": (
                    "GPTTY_GOAL: CONTINUE\n"
                    'GPTTY_CHECKPOINT: {"summary":"half","completed":["one"],'
                    '"decisions":[],"pending":["two"],"next":"continue"}\n'
                    "VISIBLE_STAGE"
                ),
            },
            {
                "role": "user",
                "text": (
                    "Keep the CLI stable.\n\n"
                    "[GPTTY Goal mode remains active. Treat the user message above as steering/refinement "
                    "of the existing goal.]\nGPTTY_GOAL: CONTINUE"
                ),
            },
            {
                "role": "assistant",
                "text": (
                    "GPTTY_GOAL: COMPLETE\n"
                    'GPTTY_CHECKPOINT: {"summary":"done","completed":["verified"],'
                    '"decisions":[],"pending":[],"next":"none"}\n'
                    "VISIBLE_DONE"
                ),
            },
        ],
    }

    commands._begin_resume("conv-goal-history")
    request = commands.take_pending_resume()
    assert request is not None
    assert commands.complete_resume(request, snapshot) is True

    rendered = next(value for kind, value in renderer.events if kind == "messages")
    assert [(message.role, message.text) for message in rendered] == [
        ("assistant", "VISIBLE_STAGE"),
        ("user", "Keep the CLI stable."),
        ("assistant", "VISIBLE_DONE"),
    ]
    assert all("GPTTY_" not in message.text for message in rendered)


def test_goal_open_does_not_follow_stale_historical_conversation_binding(tmp_path) -> None:
    commands, renderer, _, _ = make_commands(
        tmp_path,
        state=ChatState(current_conversation="conv-shared"),
        runner_id="viewer",
    )
    old = GoalState(
        goal_id="oldgoal1111111111",
        conversation_ref="conv-shared",
        conversations=["conv-shared"],
        status="complete",
        objective="old complete goal",
    )
    new = GoalState(
        goal_id="newgoal2222222222",
        conversation_ref="conv-shared",
        conversations=["conv-shared"],
        status="paused",
        objective="new current goal",
    )
    commands.goal_store.save(old, event_type="goal_created")
    commands.goal_store.save(new, event_type="goal_created")
    commands.state.goal = commands.goal_store.load_for_conversation("conv-shared")

    assert commands.goal_store.bindings_for_goal(old) == []
    assert commands.goal_store.goal_id_for_conversation("conv-shared") == new.goal_id

    commands.handle("/goal open oldgoal")

    assert commands.has_pending_resume is False
    assert commands.state.goal is not None
    assert commands.state.goal.goal_id == new.goal_id
    warnings = [str(message) for kind, message in renderer.events if kind == "warning"]
    assert any("history only" in message for message in warnings)

    renderer.events.clear()
    commands.handle("/goal list all")
    infos = [str(message) for kind, message in renderer.events if kind == "info"]
    assert any("oldgoal1" in line and "history only" in line for line in infos)
    assert any("newgoal2" in line and "conv-shared" in line for line in infos)

def test_goal_acceptance_criteria_veto_complete_until_evidence_exists(tmp_path) -> None:
    state = ChatState(current_conversation="conv-criteria")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    commands.handle(
        '/goal --accept "full gate passes" --accept "visual cmux passes" "Ship safely"'
    )
    assert state.goal is not None
    assert [item.criterion_id for item in state.goal.acceptance_criteria] == ["A1", "A2"]
    assert "A1: full gate passes" in (commands.pop_automatic_prompt() or "")

    commands.handle_goal_turn_result(
        {
            "text": (
                "GPTTY_GOAL: COMPLETE\n"
                'GPTTY_CHECKPOINT: {"summary":"done","completed":["implementation"],'
                '"decisions":[],"pending":[],"next":"none"}\n'
                "Done."
            ),
            "conversation_ref": "conv-criteria",
        }
    )
    assert state.goal.status == "active"
    assert any(
        kind == "warning" and "acceptance criteria" in str(message)
        for kind, message in renderer.events
    )

    commands.handle("/goal criteria attest A1 pytest-500-pass")
    commands.handle("/goal criteria attest A2 cmux-visual-pass")
    assert all(item.satisfied for item in state.goal.acceptance_criteria)
    assert all(item.evidence_source == "human" for item in state.goal.acceptance_criteria)

    commands.handle_goal_turn_result(
        {
            "text": (
                "GPTTY_GOAL: COMPLETE\n"
                'GPTTY_CHECKPOINT: {"summary":"done","completed":["implementation"],'
                '"decisions":[],"pending":[],"next":"none"}\n'
                "Done."
            ),
            "conversation_ref": "conv-criteria",
        }
    )
    assert state.goal.status == "complete"


def test_goal_criteria_list_is_readable_and_definitions_are_creation_only(tmp_path) -> None:
    commands, renderer, _, _ = make_commands(
        tmp_path, state=ChatState(current_conversation="conv-criteria-list")
    )
    commands.handle('/goal --accept "test gate" "Objective"')
    renderer.events.clear()
    commands.handle("/goal criteria")
    infos = [str(message) for kind, message in renderer.events if kind == "info"]
    assert "Goal acceptance criteria · 0/1 satisfied" in infos
    assert any("A1 · PENDING" in message and "test gate" in message for message in infos)

def test_goal_resume_explicitly_migrates_legacy_runtime_before_dispatch(tmp_path) -> None:
    state = ChatState(
        current_conversation="conv-runtime-old",
        goal=GoalState(
            goal_id="goal-runtime-ui",
            runtime_version=1,
            protocol_version=1,
            conversation_ref="conv-runtime-old",
            conversations=["conv-runtime-old"],
            status="paused",
        ),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    commands.goal_store.save(state.goal, event_type="goal_created")

    commands.handle("/goal resume")

    assert state.goal is not None
    assert state.goal.status == "active"
    assert state.goal.runtime_version > 1
    assert state.goal.protocol_version > 1
    assert any(
        kind == "info" and "migrated to runtime/protocol" in str(message)
        for kind, message in renderer.events
    )
    assert any(
        event["type"] == "goal_runtime_migrated"
        for event in commands.goal_store.events(state.goal)
    )


def test_goal_resume_rejects_future_runtime_without_mutating_it(tmp_path) -> None:
    from gptty.state import CURRENT_GOAL_RUNTIME_VERSION

    state = ChatState(
        current_conversation="conv-runtime-future",
        goal=GoalState(
            goal_id="goal-runtime-future-ui",
            runtime_version=CURRENT_GOAL_RUNTIME_VERSION + 1,
            conversation_ref="conv-runtime-future",
            conversations=["conv-runtime-future"],
            status="paused",
        ),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    commands.goal_store.save(state.goal, event_type="goal_created")
    revision = state.goal.revision

    commands.handle("/goal resume")

    assert state.goal.status == "paused"
    assert state.goal.revision == revision
    assert commands.has_automatic_prompt is False
    assert any(
        kind == "warning" and "newer runtime/protocol" in str(message)
        for kind, message in renderer.events
    )

def test_goal_doctor_and_trace_are_read_only_and_visible(tmp_path) -> None:
    state = ChatState(current_conversation="conv-doctor-ui")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    commands.handle('/goal "diagnose active goal"')
    assert state.goal is not None
    revision = state.goal.revision

    renderer.events.clear()
    commands.handle("/goal doctor")
    infos = [str(message) for kind, message in renderer.events if kind == "info"]
    assert any("Goal doctor · PASS" in message for message in infos)
    assert any("replay · PASS" in message for message in infos)
    assert any("kernel lock · PASS · held" in message for message in infos)
    assert state.goal.revision == revision

    renderer.events.clear()
    commands.handle("/goal trace 5")
    infos = [str(message) for kind, message in renderer.events if kind == "info"]
    assert any(message.startswith("Goal trace · ") for message in infos)
    assert any("goal_created" in message for message in infos)
    assert state.goal.revision == revision
