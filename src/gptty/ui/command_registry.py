from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CommandContext = Literal["idle", "working", "follow", "resume_loading"]
CommandOwner = Literal["commands", "chat"]

ALL_COMMAND_CONTEXTS: frozenset[CommandContext] = frozenset(
    {"idle", "working", "follow", "resume_loading"}
)
IDLE_FOLLOW_CONTEXTS: frozenset[CommandContext] = frozenset({"idle", "follow"})
WORKING_CONTEXTS: frozenset[CommandContext] = frozenset(
    {"idle", "working", "follow"}
)


@dataclass(frozen=True)
class CommandOptionSpec:
    value: str
    description: str


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    usage: str | None = None
    options: tuple[CommandOptionSpec, ...] = ()
    aliases: tuple[str, ...] = ()
    owner: CommandOwner = "commands"
    contexts: frozenset[CommandContext] = IDLE_FOLLOW_CONTEXTS
    async_picker: bool = False
    working_hint: str | None = None


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("new", "Start a new ChatGPT conversation"),
    CommandSpec(
        "temporary",
        "Start a new Temporary ChatGPT conversation",
        aliases=("temp",),
    ),
    CommandSpec(
        "resume",
        "Resume a real ChatGPT conversation",
        "Enter: choose chat · or <conversation-id>",
        async_picker=True,
    ),
    CommandSpec("detach", "Detach locally from the current conversation"),
    CommandSpec("reload", "Refresh the currently attached conversation"),
    CommandSpec(
        "stop",
        "Stop the active ChatGPT response",
        contexts=WORKING_CONTEXTS,
    ),
    CommandSpec(
        "queue",
        "Inspect or manage queued turns",
        "send | remove <index|id> | clear",
        (
            CommandOptionSpec("send", "Release held turns into the current context"),
            CommandOptionSpec("remove", "Remove one queued turn by position or id prefix"),
            CommandOptionSpec("clear", "Discard all queued turns"),
        ),
        owner="chat",
        contexts=ALL_COMMAND_CONTEXTS,
    ),
    CommandSpec(
        "history",
        "Manage local prompt history",
        "clear",
        (CommandOptionSpec("clear", "Delete persisted and in-memory prompt history"),),
    ),
    CommandSpec(
        "goal",
        "Run the current task until complete or blocked",
        "<objective> | list [all] | open <id> | pause | resume | status | doctor | trace [N] | criteria | clear",
        (
            CommandOptionSpec("list", "List unfinished goals (add 'all' for terminal)"),
            CommandOptionSpec("open", "Switch to a goal by id prefix"),
            CommandOptionSpec("pause", "Pause the attached goal"),
            CommandOptionSpec("resume", "Resume the attached paused or blocked goal"),
            CommandOptionSpec("status", "Show attached goal state and turn count"),
            CommandOptionSpec("doctor", "Verify Goal durability and recovery invariants"),
            CommandOptionSpec("trace", "Show recent typed Goal journal events"),
            CommandOptionSpec("criteria", "Show or attest Goal acceptance criteria"),
            CommandOptionSpec("clear", "Unbind the attached goal; retain history"),
        ),
        contexts=WORKING_CONTEXTS,
        working_hint="/goal pause|status|list",
    ),
    CommandSpec("export", "Export the attached conversation to Markdown"),
    CommandSpec(
        "image",
        "Attach an image to the next prompt",
        "Enter: choose path · or clear",
        (CommandOptionSpec("clear", "Remove pending images"),),
        contexts=WORKING_CONTEXTS,
        async_picker=True,
        working_hint="/image PATH|clear",
    ),
    CommandSpec(
        "paste",
        "Attach the clipboard image to the next prompt",
        contexts=WORKING_CONTEXTS,
    ),
    CommandSpec(
        "model",
        "Choose a real ChatGPT model",
        "Enter: choose model · or default | <slug>",
        (CommandOptionSpec("default", "Use latest frontier · High"),),
        async_picker=True,
    ),
    CommandSpec(
        "exit",
        "Exit gptty chat",
        aliases=("quit",),
        contexts=ALL_COMMAND_CONTEXTS,
    ),
)

INTERNAL_COMMAND_HANDLERS: frozenset[str] = frozenset(
    {"goal_doctor", "goal_trace", "goal_criteria"}
)

_BY_NAME: dict[str, CommandSpec] = {}
for _spec in COMMANDS:
    for _name in (_spec.name, *_spec.aliases):
        if _name in _BY_NAME:
            raise RuntimeError(f"duplicate interactive command name: {_name}")
        _BY_NAME[_name] = _spec


def resolve_command(name: str) -> CommandSpec | None:
    return _BY_NAME.get(str(name).strip().lower().lstrip("/"))


def canonical_command_name(name: str) -> str | None:
    spec = resolve_command(name)
    return spec.name if spec is not None else None


def command_allowed(name: str, context: CommandContext) -> bool:
    spec = resolve_command(name)
    return bool(spec is not None and context in spec.contexts)


def commands_for_context(context: CommandContext) -> tuple[CommandSpec, ...]:
    return tuple(spec for spec in COMMANDS if context in spec.contexts)


_CONTEXT_ORDER: dict[CommandContext, tuple[str, ...]] = {
    "working": ("queue", "stop", "exit", "goal", "image", "paste"),
    "resume_loading": ("queue", "exit"),
}


def command_context_tokens(context: CommandContext) -> tuple[str, ...]:
    ordered = _CONTEXT_ORDER.get(context)
    if ordered is None:
        specs = commands_for_context(context)
    else:
        specs = tuple(_BY_NAME[name] for name in ordered)
    return tuple(
        spec.working_hint
        if context == "working" and spec.working_hint
        else f"/{spec.name}"
        for spec in specs
        if context in spec.contexts
    )


__all__ = [
    "ALL_COMMAND_CONTEXTS",
    "COMMANDS",
    "CommandContext",
    "CommandOptionSpec",
    "CommandOwner",
    "CommandSpec",
    "INTERNAL_COMMAND_HANDLERS",
    "canonical_command_name",
    "command_allowed",
    "command_context_tokens",
    "commands_for_context",
    "resolve_command",
]
