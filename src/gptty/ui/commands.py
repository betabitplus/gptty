from __future__ import annotations

import shlex
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ..auth_inspect import inspect_auth_file
from ..output import normalize_messages, normalize_status, render_messages
from ..profiles import list_profiles, set_active_profile
from ..state import ChatState, StateError, save_chat_state
from .renderer import PrettyRenderer
from .session import InteractiveSession, command_help
from .state import RecentStore, UIStateError


class InteractiveCommands:
    def __init__(
        self,
        *,
        args: Any,
        state: ChatState,
        state_path: Path,
        get_client: Callable[[], Any],
        reset_client: Callable[[], None],
        ui: InteractiveSession,
        renderer: PrettyRenderer,
        recent: RecentStore,
        stdout: TextIO,
        stderr: TextIO,
    ) -> None:
        self.args = args
        self.state = state
        self.state_path = state_path
        self.get_client = get_client
        self.reset_client = reset_client
        self.ui = ui
        self.renderer = renderer
        self.recent = recent
        self.stdout = stdout
        self.stderr = stderr

    def handle(self, raw: str) -> int | None:
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            self.renderer.warning(f"Invalid command: {exc}")
            return None
        if not parts:
            return None
        name = parts[0].lstrip("/").lower()
        argv = parts[1:]
        method = getattr(self, f"_cmd_{name}", None)
        if not callable(method):
            self.renderer.warning(f"Unknown command: /{name}. Type /help.")
            return None
        return method(argv)

    def _cmd_exit(self, argv: list[str]) -> int:
        return 0

    _cmd_quit = _cmd_exit

    def _cmd_help(self, argv: list[str]) -> None:
        self.renderer.commands(command_help())

    def _cmd_new(self, argv: list[str]) -> None:
        previous = self.state.current_conversation
        self.state.current_conversation = None
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self.renderer.info("Started a new conversation.")

    def _cmd_attach(self, argv: list[str]) -> None:
        ref = argv[0] if argv else self.ui.ask("ChatGPT URL or conversation ID")
        if not ref:
            return
        try:
            response = self.get_client().attach_conversation(ref)
        except Exception as exc:  # noqa: BLE001
            self.renderer.warning(f"Attach failed: {exc}")
            return
        attached = _conversation_ref(response) or ref
        previous = self.state.current_conversation
        self.state.current_conversation = attached
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self._remember(attached)
        self.renderer.info(f"Attached: {attached}")
        return

    def _cmd_switch(self, argv: list[str]) -> None:
        try:
            items = self.recent.list()
        except UIStateError as exc:
            self.renderer.warning(str(exc))
            return
        if not items:
            self.renderer.info("No local recent conversations yet. Use /attach or start a chat first.")
            return
        selected = self.ui.choose(
            "Recent conversations",
            [(item.ref, item.label) for item in items],
            default=self.state.current_conversation,
        )
        if not selected:
            return
        previous = self.state.current_conversation
        self.state.current_conversation = str(selected)
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self._remember(str(selected))
        self.renderer.info(f"Switched to: {selected}")
        return

    def _cmd_model(self, argv: list[str]) -> None:
        value = argv[0].lower() if argv else self.ui.choose(
            "Model effort",
            [("default", "Default"), ("fast", "Fast"), ("balanced", "Balanced"), ("deep", "Deep")],
            default=self.state.model or "default",
        )
        if value is None:
            return
        if value not in {"default", "fast", "balanced", "deep"}:
            self.renderer.warning("Usage: /model [default|fast|balanced|deep]")
            return
        previous = self.state.model
        self.state.model = None if value == "default" else str(value)
        if not self._save_state():
            self.state.model = previous
            return
        self.renderer.info(f"Model effort: {self.state.model or 'default'}")

    def _cmd_messages(self, argv: list[str]) -> None:
        ref = self._require_conversation()
        if not ref:
            return
        if argv:
            try:
                limit = max(1, int(argv[0]))
            except ValueError:
                self.renderer.warning("Usage: /messages [count]")
                return
        else:
            raw = self.ui.ask("How many messages", default="20")
            if raw is None:
                return
            try:
                limit = max(1, int(raw or "20"))
            except ValueError:
                self.renderer.warning("Message count must be a number.")
                return
        try:
            response = self.get_client().get_messages(ref, limit=limit)
        except Exception as exc:  # noqa: BLE001
            self.renderer.warning(f"Messages failed: {exc}")
            return
        self.renderer.messages(normalize_messages(response))
        return

    def _cmd_status(self, argv: list[str]) -> None:
        ref = self._require_conversation()
        if not ref:
            return
        try:
            response = self.get_client().get_status(ref)
        except Exception as exc:  # noqa: BLE001
            self.renderer.warning(f"Status failed: {exc}")
            return
        self.renderer.mapping(normalize_status(response, conversation=ref))
        return

    def _cmd_export(self, argv: list[str]) -> None:
        ref = self._require_conversation()
        if not ref:
            return
        fmt = argv[0] if argv else self.ui.choose(
            "Export format",
            [("markdown", "Markdown"), ("json", "JSON"), ("plain", "Plain text")],
            default="markdown",
        )
        if not fmt:
            return
        try:
            response = self.get_client().get_messages(ref)
            rendered = render_messages(normalize_messages(response), str(fmt))
        except Exception as exc:  # noqa: BLE001
            self.renderer.warning(f"Export failed: {exc}")
            return
        destination = self.ui.ask("Output file (blank = terminal)", default="")
        if destination is None:
            return
        if not destination:
            self.renderer.info(rendered)
            return
        path = Path(destination).expanduser()
        if path.exists():
            overwrite = self.ui.choose("File exists", [(False, "Cancel"), (True, "Overwrite")], default=False)
            if not overwrite:
                return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
        except OSError as exc:
            self.renderer.warning(f"Export failed: {exc}")
            return
        self.renderer.info(f"Exported to {path}")
        return

    def _cmd_profile(self, argv: list[str]) -> None:
        try:
            profiles = list_profiles()
        except Exception as exc:  # noqa: BLE001
            self.renderer.warning(f"Profile list failed: {exc}")
            return
        if not profiles:
            self.renderer.info("No profiles found.")
            return
        selected = argv[0] if argv else self.ui.choose("Active profile", [(name, name) for name in profiles])
        if not selected:
            return
        try:
            set_active_profile(str(selected))
        except Exception as exc:  # noqa: BLE001
            self.renderer.warning(f"Profile switch failed: {exc}")
            return
        self.renderer.info(f"Active profile set to {selected}. Restart chat to use it in this session.")
        return

    def _cmd_auth(self, argv: list[str]) -> None:
        action = argv[0] if argv else self.ui.choose("Auth", [("status", "Status"), ("refresh", "Refresh")], default="status")
        if not action:
            return
        if action == "status":
            self.renderer.mapping(inspect_auth_file(getattr(self.args, "auth", "auth_data.json")))
            return
        if action != "refresh":
            self.renderer.warning("Usage: /auth [status|refresh]")
            return
        from ..commands.auth import run_auth_refresh

        refresh_args = Namespace(
            auth=getattr(self.args, "auth", "auth_data.json"),
            timeout=120.0,
            mode="auto",
            ready_timeout=0.0,
            probe_prompt="Hello",
        )
        code = run_auth_refresh(refresh_args, stdout=self.stdout, stderr=self.stderr)
        if code == 0:
            self.reset_client()
        return

    def _cmd_settings(self, argv: list[str]) -> None:
        changed = self.ui.edit_settings()
        if changed:
            self.renderer.info("UI settings saved. Pretty on/off takes effect on the next chat start.")

    def _require_conversation(self) -> str | None:
        if not self.state.current_conversation:
            self.renderer.info("No conversation attached. Use /new, /attach, or /switch.")
            return None
        return self.state.current_conversation

    def _save_state(self) -> bool:
        try:
            save_chat_state(self.state_path, self.state)
        except StateError as exc:
            self.renderer.warning(str(exc))
            return False
        return True

    def _remember(self, ref: str, *, label: str | None = None) -> None:
        try:
            self.recent.remember(ref, label=label)
        except UIStateError as exc:
            self.renderer.warning(str(exc))


def _conversation_ref(response: Any) -> str | None:
    for field in ("conversation_url", "conversation_id", "conversation_ref", "url", "id"):
        value = response.get(field) if isinstance(response, dict) else getattr(response, field, None)
        if value:
            return str(value)
    return None
