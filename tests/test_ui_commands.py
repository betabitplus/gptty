from __future__ import annotations

from types import SimpleNamespace

from gptty.state import ChatState, GoalState, StateError, load_chat_state
from gptty.ui.commands import InteractiveCommands


class FakeUI:
    def __init__(self, *, choices=None, image_paths=None) -> None:
        self.choices = list(choices or [])
        self.image_paths = list(image_paths or [])
        self.seen: list[tuple[str, list[tuple[object, str]]]] = []

    def choose_searchable(self, message, options, *, default=None):
        self.seen.append((message, list(options)))
        return self.choices.pop(0) if self.choices else default

    def read_image_path(self):
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


def make_commands(tmp_path, *, state=None, ui=None, client=None):
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
    )
    return commands, renderer, client, state_path


def finish_pending_resume(commands, client):
    request = commands.take_pending_resume()
    assert request is not None
    snapshot = client.conversation_snapshot(request.conversation_ref)
    commands.complete_resume(request, snapshot)
    return request


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
    clear_index = next(index for index, event in enumerate(renderer.events) if event[0] == "clear_context")
    assert clear_index > 0
    rendered = [event for event in renderer.events if event[0] == "messages"][-1][1]
    assert [message.text for message in rendered] == ["question", "answer"]


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


def test_temporary_command_clears_persistent_attachment_without_persisting_temp_id(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, state_path = make_commands(tmp_path, state=state)

    commands.handle("/temporary")

    assert commands.conversation_mode == "temporary"
    assert commands.conversation_ref is None
    assert state.current_conversation is None
    assert load_chat_state(state_path).current_conversation is None
    assert any(event == ("header", {"model": "latest frontier · High", "temporary": True}) for event in renderer.events)


def test_temporary_export_uses_live_transcript_and_prints_exact_path(tmp_path, monkeypatch) -> None:
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


def test_normal_export_reads_complete_attached_history_from_cwa(tmp_path, monkeypatch) -> None:
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


def test_image_clear_removes_pending_clipboard_temp_file(tmp_path, monkeypatch) -> None:
    clipboard_image = tmp_path / "clipboard.png"
    clipboard_image.write_bytes(b"png")
    commands, renderer, _, _ = make_commands(tmp_path)
    monkeypatch.setattr("gptty.ui.commands.tempfile.mkdtemp", lambda **_kwargs: str(tmp_path))
    monkeypatch.setattr("gptty.ui.commands.capture_clipboard_image", lambda _directory: clipboard_image)

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


def test_state_save_failure_rolls_back_interactive_change(tmp_path, monkeypatch) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, _ = make_commands(tmp_path, state=state)

    def fail_save(*args, **kwargs) -> None:
        raise StateError("disk failed")

    monkeypatch.setattr("gptty.ui.commands.save_chat_state", fail_save)
    commands.handle("/new")

    assert state.current_conversation == "conv-1"
    assert renderer.events[-1] == ("warning", "disk failed")


def test_goal_command_starts_on_attached_conversation_and_queues_activation(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, renderer, _, state_path = make_commands(tmp_path, state=state)

    commands.handle("/goal")

    assert state.goal is not None
    assert state.goal.status == "active"
    assert state.goal.conversation_ref == "conv-1"
    assert load_chat_state(state_path).goal == state.goal
    prompt = commands.pop_automatic_prompt()
    assert prompt is not None
    assert "GPTTY Goal mode is now active" in prompt
    assert ("info", "Goal · active · starting") in renderer.events


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
        "No conversation is attached. Use /goal <objective> to start a goal in a new chat.",
    )


def test_goal_continue_queues_next_turn_without_notification(tmp_path, monkeypatch) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr("gptty.ui.commands.notify_response_complete", lambda **kwargs: notified.append(kwargs))

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
    assert "Continue pursuing the active goal" in (commands.pop_automatic_prompt() or "")
    assert notified == []
    assert ("info", "Goal · continuing · next turn 2") in renderer.events


def test_goal_complete_stops_loop_and_sends_single_clean_notification(tmp_path, monkeypatch) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active", turn_count=2),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr("gptty.ui.commands.notify_response_complete", lambda **kwargs: notified.append(kwargs))

    commands.handle_goal_turn_result(
        {
            "text": "GPTTY_GOAL: COMPLETE\nEverything is implemented and verified.",
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
        {"chat_title": "Goal chat", "final_response": "Everything is implemented and verified."}
    ]
    assert ("info", "Goal · complete · 3 turns") in renderer.events


def test_goal_blocked_stops_loop_and_notifies_for_user_action(tmp_path, monkeypatch) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr("gptty.ui.commands.notify_response_complete", lambda **kwargs: notified.append(kwargs))

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
        {"chat_title": "Goal chat", "final_response": "Goal blocked. Please log in to the provider account."}
    ]
    assert ("warning", "Goal · blocked · user action required") in renderer.events


def test_goal_missing_status_recovers_twice_then_interrupts(tmp_path, monkeypatch) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, _, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr("gptty.ui.commands.notify_response_complete", lambda **kwargs: notified.append(kwargs))

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
    assert state.goal.status == "interrupted"
    assert state.goal.protocol_failures == 3
    assert commands.has_automatic_prompt is False
    assert notified == [
        {
            "chat_title": "Goal chat",
            "final_response": "Goal interrupted. missing valid GPTTY_GOAL status for 3 consecutive turns",
        }
    ]


def test_goal_user_stop_pauses_and_never_auto_continues(tmp_path, monkeypatch) -> None:
    state = ChatState(
        current_conversation="conv-1",
        goal=GoalState(conversation_ref="conv-1", status="active"),
    )
    commands, renderer, _, _ = make_commands(tmp_path, state=state)
    notified: list[dict[str, object]] = []
    monkeypatch.setattr("gptty.ui.commands.notify_response_complete", lambda **kwargs: notified.append(kwargs))

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


def test_goal_pause_resume_clear_and_context_switch_are_safe(tmp_path) -> None:
    state = ChatState(current_conversation="conv-1")
    commands, _, _, state_path = make_commands(tmp_path, state=state)
    commands.handle("/goal important work")
    assert commands.pop_automatic_prompt() is not None

    commands.handle("/goal pause")
    assert state.goal is not None and state.goal.status == "paused"
    commands.handle("/goal resume")
    assert state.goal is not None and state.goal.status == "active"
    assert commands.pop_automatic_prompt() is not None

    commands.handle("/new")
    assert state.goal is not None and state.goal.status == "paused"
    assert state.current_conversation is None
    assert load_chat_state(state_path).goal is not None

    commands.handle("/goal clear")
    assert state.goal is None
    assert load_chat_state(state_path).goal is None


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
