from __future__ import annotations

import inspect

from gptty.ui.command_registry import (
    COMMANDS,
    INTERNAL_COMMAND_HANDLERS,
    canonical_command_name,
    command_allowed,
    command_context_tokens,
    commands_for_context,
    resolve_command,
)
from gptty.ui.commands import InteractiveCommands
from gptty.ui.session import COMMANDS as SESSION_COMMANDS


def test_session_completion_reexports_the_canonical_registry() -> None:
    assert SESSION_COMMANDS is COMMANDS
    assert len({spec.name for spec in COMMANDS}) == len(COMMANDS)


def test_command_contexts_are_canonical_and_complete() -> None:
    all_names = {spec.name for spec in COMMANDS}

    assert {spec.name for spec in commands_for_context("idle")} == all_names
    assert {spec.name for spec in commands_for_context("follow")} == all_names
    assert {spec.name for spec in commands_for_context("working")} == {
        "queue",
        "stop",
        "exit",
        "goal",
        "image",
        "paste",
    }
    assert {spec.name for spec in commands_for_context("resume_loading")} == {
        "queue",
        "exit",
    }


def test_context_help_tokens_come_from_same_availability_metadata() -> None:
    assert command_context_tokens("working") == (
        "/queue",
        "/stop",
        "/exit",
        "/goal pause|status|list",
        "/image PATH|clear",
        "/paste",
    )
    assert command_context_tokens("resume_loading") == ("/queue", "/exit")


def test_aliases_resolve_to_one_canonical_command() -> None:
    assert canonical_command_name("/temp") == "temporary"
    assert canonical_command_name("temporary") == "temporary"
    assert canonical_command_name("/quit") == "exit"
    assert canonical_command_name("exit") == "exit"
    assert command_allowed("quit", "working") is True
    assert command_allowed("temp", "working") is False


def test_registry_owner_matches_dispatch_surface() -> None:
    command_handlers = {
        name.removeprefix("_cmd_")
        for name, value in vars(InteractiveCommands).items()
        if name.startswith("_cmd_")
        and not name.endswith("_async")
        and inspect.isfunction(value)
    }
    registered_handlers = {
        spec.name for spec in COMMANDS if spec.owner == "commands"
    }

    assert command_handlers == registered_handlers | INTERNAL_COMMAND_HANDLERS
    assert resolve_command("queue").owner == "chat"
    assert "queue" not in command_handlers
    assert not any(resolve_command(name) for name in INTERNAL_COMMAND_HANDLERS)


def test_async_picker_metadata_matches_async_handlers() -> None:
    async_handlers = {
        name.removeprefix("_cmd_").removesuffix("_async")
        for name, value in vars(InteractiveCommands).items()
        if name.startswith("_cmd_")
        and name.endswith("_async")
        and inspect.iscoroutinefunction(value)
    }
    registered_async = {
        spec.name for spec in COMMANDS if spec.async_picker
    }

    assert async_handlers == registered_async == {"resume", "model", "image", "effort"}


class _Renderer:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_dispatch_canonicalizes_aliases_and_rejects_internal_helpers() -> None:
    commands = object.__new__(InteractiveCommands)
    commands.renderer = _Renderer()
    calls: list[tuple[str, list[str]]] = []
    commands._cmd_temporary = lambda argv: calls.append(("temporary", argv))
    commands._cmd_exit = lambda argv: calls.append(("exit", argv)) or 0

    assert commands.handle("/temp") is None
    assert commands.handle("/quit") == 0
    assert calls == [("temporary", []), ("exit", [])]

    assert commands.handle("/goal_doctor") is None
    assert commands.renderer.warnings[-1].startswith("Unknown command: /goal_doctor")


def test_every_alias_is_unique_and_maps_back_to_its_spec() -> None:
    seen: set[str] = set()
    for spec in COMMANDS:
        for name in (spec.name, *spec.aliases):
            assert name not in seen
            seen.add(name)
            assert resolve_command(name) is spec
