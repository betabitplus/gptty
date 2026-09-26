from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .io import StdinReadError, read_stdin_text
from .profiles import ProfileError, resolve_auth_path, resolve_session_paths
from .reasoning import EFFORT_VALUES

DEFAULT_TURN_TIMEOUT_SECONDS = 7200
BROWSER_BACKEND_CHOICES = ("chrome-native", "wkwebview")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_backend_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=BROWSER_BACKEND_CHOICES,
        default=None,
        help=(
            "Browser authority backend for ChatGPT web-session writes. "
            "Defaults to CWA's production backend."
        ),
    )


def _add_profile_option(parser: argparse.ArgumentParser, *, suppress_default: bool = True) -> None:
    default = argparse.SUPPRESS if suppress_default else None
    parser.add_argument(
        "--profile",
        default=default,
        help="Profile name to use for this command. Overrides GPTTY_PROFILE and the active profile.",
    )


def _add_session_identity_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--session",
        default=None,
        metavar="NAME",
        help=(
            "Reuse an explicit local gptty session. Without this option, interactive "
            "chat creates an independent runtime session and scripted commands use "
            "the default session."
        ),
    )


def _add_session_options(parser: argparse.ArgumentParser) -> None:
    _add_profile_option(parser)
    _add_backend_option(parser)
    _add_session_identity_option(parser)
    parser.add_argument(
        "--auth",
        default=None,
        help="Path to auth_data.json.",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Path to the local gptty state file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Request timeout in seconds.",
    )


def _add_auth_file_option(parser: argparse.ArgumentParser) -> None:
    _add_profile_option(parser)
    parser.add_argument(
        "--auth",
        default=None,
        help="Path to auth_data.json.",
    )


def _add_lock_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wait-lock",
        action="store_true",
        help="Wait longer when the target conversation is already waiting for a reply.",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds to wait for a conversation lock before failing.",
    )


def _add_stdin_options(parser: argparse.ArgumentParser) -> None:
    stdin_group = parser.add_mutually_exclusive_group()
    stdin_group.add_argument(
        "--stdin",
        dest="stdin_mode",
        action="store_const",
        const="always",
        default="auto",
        help="Force reading stdin, even when stdin does not look piped.",
    )
    stdin_group.add_argument(
        "--no-stdin",
        dest="stdin_mode",
        action="store_const",
        const="never",
        help="Ignore stdin, even when input is piped.",
    )
    parser.add_argument(
        "--stdin-max-bytes",
        type=_positive_int,
        default=None,
        metavar="BYTES",
        help=(
            "Override the stdin safety limit in bytes. "
            "The default is 4 MiB."
        ),
    )


class _MediaFlagAction(argparse.Action):
    """Preserve mixed --image/--file CLI order while keeping legacy attrs."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        current = list(getattr(namespace, self.dest, None) or ())
        current.append(values)
        setattr(namespace, self.dest, current)
        ordered = list(getattr(namespace, "_media_inputs", None) or ())
        ordered.append((self.dest, values))
        setattr(namespace, "_media_inputs", ordered)


def _add_media_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image",
        action=_MediaFlagAction,
        default=[],
        metavar="PATH_OR_URL",
        help="Attach an image path, URL, or data URI. Can be used more than once.",
    )
    parser.add_argument(
        "--file",
        action=_MediaFlagAction,
        default=[],
        metavar="PATH_OR_URL",
        help="Attach a general file path, URL, or data URI. Can be used more than once.",
    )


def _add_effort_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--effort",
        choices=("default", *EFFORT_VALUES),
        default=None,
        help=(
            "Reasoning effort intent: instant, medium, or high. "
            "Stored independently from the model for session commands."
        ),
    )


def _add_output_format_option(
    parser: argparse.ArgumentParser,
    *,
    default: str = "plain",
    jsonl: bool = False,
) -> None:
    choices = ("plain", "json", "jsonl", "markdown") if jsonl else ("plain", "json", "markdown")
    parser.add_argument(
        "--format",
        choices=choices,
        default=default,
        help="Output format.",
    )


def _add_auth_status_format_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("plain", "json"),
        default="plain",
        help="Output format.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gptty",
        description="Terminal client for existing ChatGPT web sessions.",
    )
    _add_profile_option(parser, suppress_default=False)
    parser.add_argument(
        "--version",
        action="version",
        version=f"gptty {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage gptty profiles and profile paths.",
    )
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command")
    profile_subparsers.add_parser("list", help="List available profiles.")
    profile_subparsers.add_parser("current", help="Show the profile used by default.")
    profile_create_parser = profile_subparsers.add_parser("create", help="Create a profile.")
    profile_create_parser.add_argument("name", help="Profile name to create.")
    profile_use_parser = profile_subparsers.add_parser("use", help="Set the active profile.")
    profile_use_parser.add_argument("name", help="Profile name to use by default.")
    profile_paths_parser = profile_subparsers.add_parser("paths", help="Show profile config/auth/state paths.")
    profile_paths_parser.add_argument("name", nargs="?", help="Optional profile name to inspect.")

    privacy_parser = subparsers.add_parser(
        "privacy",
        help="Inspect and prune gptty-owned local sensitive data.",
    )
    privacy_subparsers = privacy_parser.add_subparsers(dest="privacy_command")
    privacy_status_parser = privacy_subparsers.add_parser(
        "status",
        help="Show content-free counts for gptty-owned local privacy surfaces.",
    )
    _add_profile_option(privacy_status_parser)
    privacy_status_parser.add_argument(
        "--state",
        default=None,
        help="Path to the local gptty state file used to locate the local store.",
    )
    privacy_prune_parser = privacy_subparsers.add_parser(
        "prune",
        help="Prune old auto-generated local run data and orphan pending prompts.",
    )
    _add_profile_option(privacy_prune_parser)
    privacy_prune_parser.add_argument(
        "--state",
        default=None,
        help="Path to the local gptty state file used to locate the local store.",
    )
    privacy_prune_parser.add_argument(
        "--older-than-days",
        type=_positive_int,
        required=True,
        metavar="DAYS",
        help="Remove eligible gptty-owned data older than this many days.",
    )
    privacy_prune_parser.add_argument(
        "--include-archives",
        action="store_true",
        help="Also remove old local TUI conversation archive copies.",
    )
    privacy_prune_parser.add_argument(
        "--include-exports",
        action="store_true",
        help="Also remove old timestamped files from gptty's default /export directory.",
    )

    auth_parser = subparsers.add_parser(
        "auth",
        help="Inspect or refresh ChatGPT web-session auth data.",
    )
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command")

    auth_status_parser = auth_subparsers.add_parser(
        "status",
        help="Inspect auth_data.json without opening a browser.",
    )
    _add_auth_file_option(auth_status_parser)
    _add_auth_status_format_option(auth_status_parser)

    auth_refresh_parser = auth_subparsers.add_parser(
        "refresh",
        help="Refresh auth_data.json through the browser auth capture flow.",
    )
    _add_auth_file_option(auth_refresh_parser)
    auth_refresh_parser.add_argument(
        "--mode",
        choices=("auto", "wait"),
        default="auto",
        help="auto sends a probe prompt; wait lets you log in and send a message manually.",
    )
    auth_refresh_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for auth capture after the trigger action starts.",
    )
    auth_refresh_parser.add_argument(
        "--ready-timeout",
        type=float,
        default=0.0,
        help="Only for wait mode: seconds to wait for login/chat readiness. 0 waits indefinitely.",
    )
    auth_refresh_parser.add_argument(
        "--probe-prompt",
        default="Hello",
        help="Prompt text to send once in auto mode to trigger auth capture.",
    )
    auth_refresh_parser.add_argument(
        "--credential-store",
        choices=("auto", "keyring", "file"),
        default="auto",
        help="Reusable auth backend; auto prefers the OS credential store.",
    )

    auth_migrate_parser = auth_subparsers.add_parser(
        "migrate",
        help="Migrate reusable auth between the OS credential store and secure file.",
    )
    _add_auth_file_option(auth_migrate_parser)
    auth_migrate_parser.add_argument(
        "--backend",
        choices=("keyring", "file"),
        default="keyring",
        help="Target credential backend; keyring uses the OS credential store.",
    )

    auth_logout_parser = auth_subparsers.add_parser(
        "logout",
        help="Remove reusable local ChatGPT authorization material.",
    )
    _add_auth_file_option(auth_logout_parser)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Send a one-shot prompt through the SDK-backed ChatGPT web-session client.",
    )
    ask_parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. If omitted, gptty reads the prompt from piped stdin.",
    )
    _add_stdin_options(ask_parser)
    _add_media_options(ask_parser)
    _add_auth_file_option(ask_parser)
    _add_backend_option(ask_parser)
    _add_effort_option(ask_parser)
    ask_parser.add_argument(
        "--model",
        default=None,
        help="Model name to pass through to the SDK.",
    )
    ask_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full response before printing output.",
    )
    ask_parser.add_argument(
        "--plain",
        action="store_true",
        help="Print plain response text. Currently this is the default output mode.",
    )
    _add_output_format_option(ask_parser, jsonl=True)
    ask_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TURN_TIMEOUT_SECONDS,
        help="Request timeout in seconds.",
    )

    send_parser = subparsers.add_parser(
        "send",
        help="Send a prompt to the attached conversation, an explicit conversation, or a new chat.",
    )
    send_parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. If omitted, gptty reads the prompt from piped stdin.",
    )
    destination_group = send_parser.add_mutually_exclusive_group()
    destination_group.add_argument(
        "--to",
        default=None,
        help="Conversation URL or id to send to instead of the attached conversation.",
    )
    destination_group.add_argument(
        "--new",
        action="store_true",
        help="Start a new conversation instead of using an attached conversation.",
    )
    _add_stdin_options(send_parser)
    _add_media_options(send_parser)
    _add_session_options(send_parser)
    _add_effort_option(send_parser)
    send_parser.set_defaults(timeout=DEFAULT_TURN_TIMEOUT_SECONDS)
    _add_output_format_option(send_parser, jsonl=True)
    _add_lock_options(send_parser)
    send_parser.add_argument(
        "--model",
        default=None,
        help="Model name to pass through to the SDK and store in state.",
    )
    send_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full response before printing output.",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="Start an interactive SDK-backed chat loop.",
    )
    _add_profile_option(chat_parser)
    _add_backend_option(chat_parser)
    _add_session_identity_option(chat_parser)
    _add_effort_option(chat_parser)
    chat_parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run the legacy main.py interactive chat runtime.",
    )
    chat_parser.add_argument(
        "--state",
        default=None,
        help="Path to the local chat state file.",
    )
    chat_parser.add_argument(
        "--auth",
        default=None,
        help="Path to auth_data.json.",
    )
    chat_parser.add_argument(
        "--model",
        default=None,
        help="Model name to store in state and pass through to the SDK.",
    )
    chat_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for each full response before printing output.",
    )
    chat_parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable the enhanced interactive UI and use the legacy line-oriented output.",
    )
    chat_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TURN_TIMEOUT_SECONDS,
        help="Request timeout in seconds.",
    )
    _add_lock_options(chat_parser)

    attach_parser = subparsers.add_parser(
        "attach",
        help="Attach an existing ChatGPT conversation and save it in gptty state.",
    )
    attach_parser.add_argument("url_or_id", help="Conversation URL or id to attach.")
    _add_session_options(attach_parser)

    messages_parser = subparsers.add_parser(
        "messages",
        help="Print messages from an explicit or attached ChatGPT conversation.",
    )
    messages_parser.add_argument(
        "url_or_id",
        nargs="?",
        help="Optional conversation URL or id. Defaults to the attached conversation.",
    )
    messages_parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Limit output to the last N messages when supported by the SDK.",
    )
    _add_session_options(messages_parser)
    _add_output_format_option(messages_parser)

    status_parser = subparsers.add_parser(
        "status",
        help="Print status for an explicit or attached ChatGPT conversation.",
    )
    status_parser.add_argument(
        "url_or_id",
        nargs="?",
        help="Optional conversation URL or id. Defaults to the attached conversation.",
    )
    _add_session_options(status_parser)
    _add_output_format_option(status_parser)

    observe_parser = subparsers.add_parser(
        "observe",
        help="Show the local live status for an active gptty run.",
    )
    observe_parser.add_argument(
        "url_or_id",
        nargs="?",
        help="Optional conversation URL or id. Defaults to the attached conversation.",
    )
    observe_parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only show run status metadata, without recent text/events.",
    )
    observe_parser.add_argument(
        "--from-start",
        action="store_true",
        help="Show the run event text from the start instead of only recent events.",
    )
    _add_session_options(observe_parser)

    export_parser = subparsers.add_parser(
        "export",
        help="Export the complete canonical-visible conversation graph.",
        description=(
            "Export a complete canonical-visible graph artifact. Persistent export "
            "requires the companion chatgpt-export-one executable (or "
            "GPTTY_EXPORTER_COMMAND)."
        ),
    )
    export_parser.add_argument(
        "url_or_id",
        nargs="?",
        help="Optional conversation URL or id. Defaults to the attached conversation.",
    )
    export_parser.add_argument(
        "--last",
        type=int,
        default=None,
        help=(
            "Not supported for visible-graph artifacts; accepted for compatibility "
            "so gptty can direct you to gptty messages --last N."
        ),
    )
    export_parser.add_argument(
        "--output",
        default=None,
        help=(
            "Write a Markdown visible-graph bundle; .context.json and "
            ".manifest.json sidecars are written beside it."
        ),
    )
    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the complete --output artifact bundle if it exists.",
    )
    export_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Print Markdown or the visible-graph context JSON when --output is omitted.",
    )
    _add_session_options(export_parser)

    return parser


def _run_legacy_chat(state_path: str | Path, auth_file: str | Path) -> int:
    try:
        import main as legacy_main
    except ImportError as exc:
        print(
            "gptty could not import the legacy chat entrypoint. "
            "Run `python main.py` from the repository checkout, or reinstall the package.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    return int(legacy_main.main(state_path=state_path, auth_file=auth_file))


def _profile_arg(args: argparse.Namespace) -> str | None:
    return getattr(args, "profile", None)


def _apply_auth_path(args: argparse.Namespace) -> bool:
    try:
        resolved = resolve_auth_path(auth_file=getattr(args, "auth", None), profile=_profile_arg(args))
    except ProfileError as exc:
        print(f"gptty: {exc}", file=sys.stderr)
        return False
    args.auth = str(resolved.auth_file)
    args.profile = resolved.profile
    return True


def _apply_session_paths(args: argparse.Namespace, *, state_filename: str = "gptty_state.json") -> bool:
    try:
        resolved = resolve_session_paths(
            auth_file=getattr(args, "auth", None),
            state_file=getattr(args, "state", None),
            profile=_profile_arg(args),
            state_filename=state_filename,
        )
    except ProfileError as exc:
        print(f"gptty: {exc}", file=sys.stderr)
        return False
    args.auth = str(resolved.auth_file)
    args.state = str(resolved.state_file)
    args.profile = resolved.profile
    return True


def _read_command_stdin(args: argparse.Namespace) -> str | None:
    mode = getattr(args, "stdin_mode", "auto")
    max_bytes = getattr(args, "stdin_max_bytes", None)
    if max_bytes is None:
        return read_stdin_text(mode)
    return read_stdin_text(mode, max_bytes=max_bytes)


def _report_stdin_error(args: argparse.Namespace, exc: StdinReadError) -> int:
    exit_code = int(getattr(exc, "exit_code", 1))
    if getattr(args, "format", None) == "jsonl":
        from .output import normalize_turn_failure, render_jsonl_event

        print(
            render_jsonl_event(
                normalize_turn_failure(
                    {
                        "status": "usage-error" if exit_code == 2 else "failed",
                        "message": str(exc),
                        "source": "stdin",
                    },
                    exit_code=exit_code,
                    error_class=getattr(exc, "error_class", "stdin_read"),
                )
            )
        )
    print(f"gptty: {exc}", file=sys.stderr)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "profile":
        from .commands.profile import run_profile

        return run_profile(args)

    if args.command == "privacy":
        from .commands.privacy import run_privacy

        if args.privacy_command not in {"status", "prune"}:
            parser.print_help()
            return 2
        if not _apply_session_paths(args):
            return 2
        return run_privacy(args)

    if args.command == "auth":
        from .commands.auth import (
            run_auth_logout,
            run_auth_migrate,
            run_auth_refresh,
            run_auth_status,
        )

        if args.auth_command == "status":
            if not _apply_auth_path(args):
                return 2
            return run_auth_status(args)
        if args.auth_command == "refresh":
            if not _apply_auth_path(args):
                return 2
            return run_auth_refresh(args)
        if args.auth_command == "migrate":
            if not _apply_auth_path(args):
                return 2
            return run_auth_migrate(args)
        if args.auth_command == "logout":
            if not _apply_auth_path(args):
                return 2
            return run_auth_logout(args)
        parser.print_help()
        return 2

    if args.command == "ask":
        from .commands.ask import run_ask

        if not _apply_auth_path(args):
            return 2
        try:
            stdin_text = _read_command_stdin(args)
        except StdinReadError as exc:
            return _report_stdin_error(args, exc)
        return run_ask(args, stdin_text=stdin_text)

    if args.command == "send":
        from .commands.send import run_send

        if not _apply_session_paths(args):
            return 2
        try:
            stdin_text = _read_command_stdin(args)
        except StdinReadError as exc:
            return _report_stdin_error(args, exc)
        return run_send(args, stdin_text=stdin_text)

    if args.command == "attach":
        from .commands.attach import run_attach

        if not _apply_session_paths(args):
            return 2
        return run_attach(args)

    if args.command == "messages":
        from .commands.messages import run_messages

        if not _apply_session_paths(args):
            return 2
        return run_messages(args)

    if args.command == "status":
        from .commands.status import run_status

        if not _apply_session_paths(args):
            return 2
        return run_status(args)

    if args.command == "observe":
        from .commands.observe import run_observe

        if not _apply_session_paths(args):
            return 2
        return run_observe(args)

    if args.command == "export":
        from .commands.export import run_export

        if not _apply_session_paths(args):
            return 2
        return run_export(args)

    if args.command in {None, "chat"}:
        if bool(getattr(args, "legacy", False)):
            if not _apply_session_paths(args, state_filename="webchat_state.json"):
                return 2
            return _run_legacy_chat(state_path=args.state, auth_file=args.auth)

        from .commands.chat import run_chat

        if not _apply_session_paths(args):
            return 2
        return run_chat(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
