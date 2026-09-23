from __future__ import annotations

import shlex
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..commands.export import save_markdown_export
from ..goal import (
    MAX_PROTOCOL_FAILURES,
    MAX_RECOVERY_ATTEMPTS,
    MAX_ROLLOVERS,
    GoalSignal,
    abnormal_recovery_prompt,
    activation_prompt,
    continuation_prompt,
    parse_goal_response,
    rollover_prompt,
    steering_prompt,
)
from ..goal_store import GoalStore, ensure_goal_id
from ..media import MediaInputError, normalize_media_input
from ..output import OutputMessage, normalize_messages
from ..state import ChatState, GoalState, StateError, save_chat_state
from ..tui_archive import TUIArchive
from .clipboard import ClipboardImageError, capture_clipboard_image
from .notifications import notify_response_complete
from .renderer import PrettyRenderer
from .session import InteractiveSession

UNFINISHED_STATUSES = {
    "running",
    "streaming",
    "tool_running",
    "tool_calling",
    "user_last_message",
}


@dataclass(frozen=True)
class ResumeRequest:
    conversation_ref: str
    reload: bool = False


class InteractiveCommands:
    def __init__(
        self,
        *,
        state: ChatState,
        state_path: Path,
        get_client: Callable[[], Any],
        ui: InteractiveSession,
        renderer: PrettyRenderer,
        tui_archive: TUIArchive | None = None,
    ) -> None:
        self.state = state
        self.state_path = state_path
        self.get_client = get_client
        self.ui = ui
        self.renderer = renderer
        self.tui_archive = tui_archive
        self.goal_store = GoalStore(state_path)
        self._pending_media: list[str] = []
        self._owned_media: set[Path] = set()
        self._clipboard_dir: Path | None = None
        self._conversation_titles: dict[str, str] = {}
        self._conversation_mode = "normal"
        self._temporary_conversation: str | None = None
        self._temporary_messages: list[OutputMessage] = []
        self._temporary_title: str | None = None
        self._automatic_prompts: list[str] = []
        self._goal_bootstrap_pending = False
        self._pending_resume: ResumeRequest | None = None

    def handle(self, raw: str) -> int | None:
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            self.renderer.warning(f"Invalid command: {exc}")
            return None
        if not parts:
            return None
        name = parts[0].lstrip("/").lower()
        method = getattr(self, f"_cmd_{name}", None)
        if not callable(method):
            self.renderer.warning(f"Unknown command: /{name}. Press / for actions.")
            return None
        return method(parts[1:])

    async def handle_async(self, raw: str) -> int | None:
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            self.renderer.warning(f"Invalid command: {exc}")
            return None
        if not parts:
            return None
        name = parts[0].lstrip("/").lower()
        argv = parts[1:]
        if name == "resume" and not argv:
            return await self._cmd_resume_async()
        if name == "model" and not argv:
            return await self._cmd_model_async()
        if name == "image" and not argv:
            return await self._cmd_image_async()
        method = getattr(self, f"_cmd_{name}", None)
        if not callable(method):
            self.renderer.warning(f"Unknown command: /{name}. Press / for actions.")
            return None
        return method(argv)

    @property
    def conversation_mode(self) -> str:
        return self._conversation_mode

    @property
    def conversation_ref(self) -> str | None:
        if self._conversation_mode == "temporary":
            return self._temporary_conversation
        return self.state.current_conversation

    @property
    def pending_media(self) -> list[str]:
        return list(self._pending_media)

    @property
    def pending_media_count(self) -> int:
        return len(self._pending_media)

    @property
    def goal_active(self) -> bool:
        goal = self.state.goal
        if (
            goal is None
            or goal.status != "active"
            or self._conversation_mode != "normal"
        ):
            return False
        if goal.conversation_ref is None:
            return self.state.current_conversation is None
        return goal.conversation_ref == self.state.current_conversation

    @property
    def has_automatic_prompt(self) -> bool:
        return bool(self._automatic_prompts)

    def pop_automatic_prompt(self) -> str | None:
        if not self._automatic_prompts:
            self._goal_bootstrap_pending = False
            return None
        prompt = self._automatic_prompts.pop(0)
        if self._goal_bootstrap_pending:
            self._goal_bootstrap_pending = False
        return prompt

    def clear_automatic_prompts(self) -> None:
        self._automatic_prompts.clear()
        self._goal_bootstrap_pending = False

    @property
    def goal_bootstrap_pending(self) -> bool:
        return self._goal_bootstrap_pending and bool(self._automatic_prompts)

    @property
    def has_pending_resume(self) -> bool:
        return self._pending_resume is not None

    def take_pending_resume(self) -> ResumeRequest | None:
        request = self._pending_resume
        self._pending_resume = None
        return request

    def complete_resume(self, request: ResumeRequest, snapshot: Any) -> bool:
        attached_ref = request.conversation_ref
        previous = self.state.current_conversation
        self.state.current_conversation = attached_ref
        if not self._save_state():
            self.state.current_conversation = previous
            return False

        if not request.reload:
            self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(
            conversation=attached_ref,
            model=self.state.model or "latest frontier · High",
        )
        action = "Reloaded" if request.reload else "Resumed"
        self.renderer.info(f"{action}: {_short_ref(attached_ref)}")
        messages = _snapshot_messages(snapshot)
        self.renderer.messages(normalize_messages(messages))
        historical_ui_marker: tuple[str, str, str, str] | None = None
        if isinstance(snapshot, dict):
            ui_state = snapshot.get("historical_ui_state")
            if isinstance(ui_state, dict):
                label = str(ui_state.get("scope") or "").strip()
                marker_status = str(ui_state.get("status") or "").strip()
                detail = str(ui_state.get("detail") or "").strip()
                source = str(ui_state.get("source") or "web-ui").strip() or "web-ui"
                if label in {"chat", "turn", "session"} and marker_status and detail:
                    historical_ui_marker = (label, marker_status, detail, source)
        persistent_chat_marker: tuple[str, str, str, str | None] | None = None
        if self.tui_archive is not None:
            if historical_ui_marker is not None:
                try:
                    self.tui_archive.record_observed_terminal(
                        conversation_ref=attached_ref,
                        label=historical_ui_marker[0],
                        status=historical_ui_marker[1],
                        text=historical_ui_marker[2],
                        source=historical_ui_marker[3],
                    )
                except Exception:
                    pass
            try:
                persistent_chat_marker = self.tui_archive.conversation_terminal_marker(
                    attached_ref
                )
            except Exception:
                persistent_chat_marker = None
        if persistent_chat_marker is not None:
            self.renderer.turn_marker(*persistent_chat_marker[:3])
        elif historical_ui_marker is not None:
            self.renderer.turn_marker(*historical_ui_marker[:3])
        if isinstance(snapshot, dict) and snapshot.get("canonical_cache_stale") is True:
            age_value = snapshot.get("canonical_cache_age_seconds")
            if isinstance(age_value, (int, float)) and not isinstance(age_value, bool):
                age_seconds = max(0, int(float(age_value)))
                self.renderer.warning(
                    "Canonical history is rate-limited; showing cached history "
                    f"({age_seconds}s old)."
                )
            else:
                self.renderer.warning(
                    "Canonical history is rate-limited; showing cached history."
                )
        if (
            isinstance(snapshot, dict)
            and snapshot.get("backend_terminal_status_proven") is True
            and snapshot.get("canonical_status_overridden") is True
        ):
            backend_status = str(
                snapshot.get("backend_stream_status") or "terminal"
            ).strip()
            canonical_status = str(
                snapshot.get("canonical_status_before_override") or "unfinished"
            ).strip()
            if (
                snapshot.get("canonical_terminal_text_missing") is True
                and persistent_chat_marker is None
                and historical_ui_marker is None
            ):
                self.renderer.turn_marker(
                    "turn",
                    "unresolved",
                    "ChatGPT is terminal, but canonical history contains no final assistant response.",
                )
            else:
                self.renderer.info(
                    f"Backend reports {backend_status}; ignored stale canonical status={canonical_status}."
                )
        status = _snapshot_status(snapshot)
        if (
            status not in UNFINISHED_STATUSES
            and status != "awaiting_tool_approval"
            and messages
            and _field_text(messages[-1], "role") == "user"
            and persistent_chat_marker is None
            and historical_ui_marker is None
            and not (
                isinstance(snapshot, dict)
                and snapshot.get("canonical_terminal_text_missing") is True
                and snapshot.get("backend_terminal_status_proven") is True
            )
        ):
            self.renderer.turn_marker(
                "turn",
                "unresolved",
                "Canonical history ends after a user message; no final assistant response is recorded.",
            )
        if status == "awaiting_tool_approval":
            self.renderer.warning("Conversation is waiting for tool approval.")
        elif status in UNFINISHED_STATUSES:
            self.renderer.warning(
                f"Conversation has an unfinished turn (status={status}); attached without blocking."
            )
        return True

    def fail_resume(self, request: ResumeRequest, error: BaseException) -> None:
        action = "Reload" if request.reload else "Resume"
        self.renderer.warning(
            f"{action} failed for {_short_ref(request.conversation_ref)}: {error}"
        )

    def prepare_goal_user_prompt(self, prompt: str) -> str:
        if not self.goal_active:
            return prompt
        return steering_prompt(prompt)

    def pause_goal_after_user_stop(self, conversation_ref: str | None) -> None:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return
        normalized_ref = str(conversation_ref or "").strip() or None
        if goal.conversation_ref is None and normalized_ref:
            goal.conversation_ref = normalized_ref
        if (
            goal.conversation_ref
            and normalized_ref
            and goal.conversation_ref != normalized_ref
        ):
            self._interrupt_goal(
                "conversation changed while stopping goal", notify=True
            )
            return
        goal.turn_count += 1
        goal.status = "paused"
        goal.reason = "stopped by user"
        self.clear_automatic_prompts()
        self._save_state()
        self.renderer.info("Goal · paused · stopped by user")

    def goal_display_text(self, text: str) -> str:
        parsed = parse_goal_response(text)
        return parsed.body if parsed.signal is not None else text

    def handle_goal_turn_result(self, result: dict[str, Any]) -> None:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return

        conversation_ref = str(result.get("conversation_ref") or "").strip() or None
        if not self._bind_goal_conversation(goal, conversation_ref):
            return

        goal.turn_count += 1
        if bool(result.get("stopped_by_user")):
            goal.status = "paused"
            goal.reason = "stopped by user"
            self.clear_automatic_prompts()
            self._save_state()
            self.renderer.info("Goal · paused · stopped by user")
            return

        parsed = parse_goal_response(str(result.get("text") or ""))
        if parsed.checkpoint is not None:
            self._update_goal_checkpoint(goal, parsed)
        marker = self._goal_terminal_marker(result)
        if marker is not None:
            label, status, detail = marker
            final_goal_signal_survives_dead_chat = (
                label == "chat"
                and status in {"limit-reached", "unavailable"}
                and parsed.signal in {GoalSignal.COMPLETE, GoalSignal.BLOCKED}
            )
            if final_goal_signal_survives_dead_chat:
                marker = None
            elif label == "chat" and status in {"limit-reached", "unavailable"}:
                self._rollover_goal(detail)
                return
            if status in {"filtered", "blocked"}:
                goal.status = "blocked"
                goal.reason = detail
                self.clear_automatic_prompts()
                self._save_state()
                self.renderer.warning("Goal · blocked · user action required")
                notify_response_complete(
                    chat_title=str(result.get("title") or "").strip() or None,
                    final_response=f"Goal blocked. {detail}",
                )
                return
            if status == "rate-limited":
                self._pause_goal_for_service_condition(detail)
                return
            if status in {
                "abnormal",
                "delivery-timeout",
                "failed",
                "incomplete",
                "truncated",
                "unconfirmed",
                "unresolved",
            }:
                self._recover_goal_same_chat(
                    detail,
                    allow_rollover=status != "truncated",
                )
                return

        self._update_goal_checkpoint(goal, parsed)

        if parsed.signal is GoalSignal.COMPLETE:
            goal.status = "complete"
            goal.protocol_failures = 0
            goal.recovery_count = 0
            goal.reason = None
            self.clear_automatic_prompts()
            self._save_state()
            self.renderer.info(
                f"Goal · complete · {goal.turn_count} turn{'s' if goal.turn_count != 1 else ''}"
            )
            notify_response_complete(
                chat_title=str(result.get("title") or "").strip() or None,
                final_response=parsed.body or "Goal complete.",
            )
            return

        if parsed.signal is GoalSignal.BLOCKED:
            goal.status = "blocked"
            goal.protocol_failures = 0
            goal.recovery_count = 0
            goal.reason = parsed.body or "agent reported a blocker"
            self.clear_automatic_prompts()
            self._save_state()
            self.renderer.warning("Goal · blocked · user action required")
            notify_response_complete(
                chat_title=str(result.get("title") or "").strip() or None,
                final_response=f"Goal blocked. {parsed.body}".strip(),
            )
            return

        protocol_recovery = parsed.signal is None
        if protocol_recovery:
            goal.protocol_failures += 1
            if goal.protocol_failures >= MAX_PROTOCOL_FAILURES:
                self._rollover_goal(
                    f"missing valid GPTTY_GOAL status for {goal.protocol_failures} consecutive turns"
                )
                return
        else:
            goal.protocol_failures = 0
            goal.recovery_count = 0

        goal.reason = None
        if not self._save_state():
            self._interrupt_goal("failed to persist goal progress", notify=False)
            return
        self._queue_goal_continuation(protocol_recovery=protocol_recovery)

    def handle_goal_turn_failure(
        self,
        result: dict[str, Any],
        reason: str,
        *,
        chat_title: str | None = None,
    ) -> bool:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return False
        marker = self._goal_terminal_marker(result)
        if marker is None:
            return False
        goal.turn_count += 1
        label, status, detail = marker
        if label == "chat" and status in {"limit-reached", "unavailable"}:
            return self._rollover_goal(detail)
        if status in {"blocked", "filtered"}:
            goal.status = "blocked"
            goal.reason = detail
            self.clear_automatic_prompts()
            self._save_state()
            self.renderer.warning("Goal · blocked · user action required")
            notify_response_complete(
                chat_title=chat_title,
                final_response=f"Goal blocked. {detail}",
            )
            return True
        if status == "rate-limited":
            return self._pause_goal_for_service_condition(detail)
        if status in {
            "abnormal",
            "delivery-timeout",
            "failed",
            "incomplete",
            "truncated",
            "unconfirmed",
            "unresolved",
        }:
            return self._recover_goal_same_chat(
                detail,
                allow_rollover=status != "truncated",
            )
        return False

    @staticmethod
    def _goal_terminal_marker(
        result: dict[str, Any],
    ) -> tuple[str, str, str] | None:
        marker = result.get("terminal_marker")
        if not isinstance(marker, (tuple, list)) or len(marker) != 3:
            return None
        return (str(marker[0]), str(marker[1]), str(marker[2]))

    def _bind_goal_conversation(
        self,
        goal: GoalState,
        conversation_ref: str | None,
    ) -> bool:
        if goal.conversation_ref is None and conversation_ref:
            goal.conversation_ref = conversation_ref
        if (
            goal.conversation_ref
            and conversation_ref
            and goal.conversation_ref != conversation_ref
        ):
            self._interrupt_goal("conversation changed during goal turn", notify=True)
            return False
        if conversation_ref and conversation_ref not in goal.conversations:
            goal.conversations.append(conversation_ref)
        return True

    def _update_goal_checkpoint(self, goal: GoalState, parsed: Any) -> None:
        checkpoint = parsed.checkpoint
        if checkpoint is not None:
            checkpoint.updated_turn = goal.turn_count
            goal.checkpoint = checkpoint
            return
        body = " ".join(str(parsed.body or "").split()).strip()
        if body:
            goal.checkpoint.summary = body[:2400]
            goal.checkpoint.updated_turn = goal.turn_count

    def _recover_goal_same_chat(
        self,
        reason: str,
        *,
        allow_rollover: bool,
    ) -> bool:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return False
        goal.reason = reason
        if not allow_rollover:
            goal.recovery_count = 0
            self.clear_automatic_prompts()
            self._automatic_prompts.append(abnormal_recovery_prompt(reason))
            if not self._save_state():
                self._interrupt_goal("failed to persist goal recovery state", notify=False)
                return False
            self.renderer.info("Goal · continuing · response was truncated")
            return True

        goal.recovery_count += 1
        if goal.recovery_count > MAX_RECOVERY_ATTEMPTS:
            return self._rollover_goal(f"repeated non-standard turns: {reason}")
        self.clear_automatic_prompts()
        self._automatic_prompts.append(abnormal_recovery_prompt(reason))
        if not self._save_state():
            self._interrupt_goal("failed to persist goal recovery state", notify=False)
            return False
        self.renderer.info(
            f"Goal · recovering · attempt {goal.recovery_count}/{MAX_RECOVERY_ATTEMPTS}"
        )
        return True

    def _rollover_goal(self, reason: str) -> bool:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return False
        if goal.rollover_count >= MAX_ROLLOVERS:
            self._interrupt_goal(
                f"rollover safety limit reached after {goal.rollover_count} recoveries: {reason}",
                notify=True,
            )
            return True
        old_ref = goal.conversation_ref or self.state.current_conversation
        if old_ref and old_ref not in goal.conversations:
            goal.conversations.append(old_ref)
        goal.conversation_ref = None
        goal.rollover_count += 1
        goal.recovery_count = 0
        goal.protocol_failures = 0
        goal.reason = f"recovering in a new chat: {reason}"
        self.state.current_conversation = None
        self.clear_automatic_prompts()
        self._automatic_prompts.append(rollover_prompt(goal, reason=reason))
        self._goal_bootstrap_pending = True
        if not self._save_state():
            self._interrupt_goal("failed to persist goal rollover state", notify=False)
            return False
        self.renderer.info(
            f"Goal · recovering · new chat · rollover {goal.rollover_count}"
        )
        return True

    def handle_goal_interruption(
        self, reason: str, *, chat_title: str | None = None
    ) -> None:
        self._interrupt_goal(reason, notify=True, chat_title=chat_title)

    def pause_goal_for_local_quit(self) -> None:
        self._pause_active_goal("local gptty exited while ChatGPT continued")

    def record_temporary_turn(
        self,
        *,
        prompt: str,
        answer: str,
        conversation_ref: str | None,
        title: str | None,
    ) -> None:
        if self._conversation_mode != "temporary":
            return
        if conversation_ref:
            self._temporary_conversation = conversation_ref
        if title:
            self._temporary_title = title
        self._temporary_messages.append(OutputMessage(role="user", text=prompt))
        if answer:
            self._temporary_messages.append(
                OutputMessage(role="assistant", text=answer)
            )

    def take_pending_media(self) -> list[str]:
        media = list(self._pending_media)
        self._pending_media.clear()
        return media

    def release_media(self, media: list[str]) -> None:
        for raw in media:
            path = Path(raw)
            if path not in self._owned_media:
                continue
            path.unlink(missing_ok=True)
            self._owned_media.discard(path)
        if self._clipboard_dir is not None and not self._owned_media:
            shutil.rmtree(self._clipboard_dir, ignore_errors=True)
            self._clipboard_dir = None

    def clear_pending_media(self) -> None:
        self._pending_media.clear()
        for path in list(self._owned_media):
            path.unlink(missing_ok=True)
        self._owned_media.clear()
        if self._clipboard_dir is not None:
            shutil.rmtree(self._clipboard_dir, ignore_errors=True)
            self._clipboard_dir = None

    def close(self) -> None:
        self.clear_pending_media()

    def _reset_temporary_context(self) -> None:
        self._temporary_conversation = None
        self._temporary_messages.clear()
        self._temporary_title = None

    def _leave_temporary_mode(self) -> None:
        if self._conversation_mode != "temporary":
            return
        try:
            client = self.get_client()
            snapshot = client.temporary_lifecycle_snapshot()
            if snapshot.get("state") == "LIVE":
                client.end_temporary_chat()
        except Exception as exc:  # noqa: BLE001 - context switch must remain usable.
            self.renderer.warning(f"Temporary chat cleanup failed: {exc}")
        self._conversation_mode = "normal"
        self._reset_temporary_context()

    def _capture_goal_context_seed(self, conversation_ref: str | None) -> list[str]:
        if not conversation_ref:
            return []
        try:
            messages = list(self.get_client().get_messages(conversation_ref))
        except Exception:
            return []

        captured: list[str] = []
        total = 0
        for message in reversed(messages):
            role = _field_text(message, "role")
            if role not in {"user", "assistant"}:
                continue
            if role == "assistant":
                recipient = _field_text(message, "recipient")
                if recipient and recipient not in {"all", "assistant"}:
                    continue
            text = " ".join(_message_text(message).split()).strip()
            if not text:
                continue
            entry = f"{role}: {text[:1600]}"
            if total + len(entry) > 12000 and captured:
                break
            captured.append(entry)
            total += len(entry)
            if len(captured) >= 12:
                break
        captured.reverse()
        return captured

    def _cmd_goal(self, argv: list[str]) -> None:
        if self._conversation_mode == "temporary":
            self.renderer.warning(
                "Goal mode is only available for normal ChatGPT conversations."
            )
            return

        action = argv[0].strip().lower() if argv else ""
        if action in {"pause", "resume", "clear", "status"} and len(argv) == 1:
            if action == "pause":
                if self._pause_active_goal("paused by user"):
                    self.renderer.info("Goal · paused")
                elif self.state.goal is None:
                    self.renderer.info("No goal is configured.")
                else:
                    self.renderer.info(f"Goal · {self.state.goal.status}")
                return
            if action == "resume":
                self._resume_goal()
                return
            if action == "clear":
                if self.state.goal is None:
                    self.renderer.info("No goal is configured.")
                    return
                previous_goal = self.state.goal
                try:
                    self.goal_store.clear_current()
                except OSError as exc:
                    self.renderer.warning(f"failed to clear Goal pointer: {exc}")
                    return
                self.state.goal = None
                self.clear_automatic_prompts()
                if self._save_state():
                    self.renderer.info("Goal · cleared")
                else:
                    self.state.goal = previous_goal
                    try:
                        self.goal_store.save(previous_goal)
                    except OSError as exc:
                        self.renderer.warning(
                            f"failed to restore Goal pointer after state save failure: {exc}"
                        )
                return
            self._render_goal_status()
            return

        objective = " ".join(argv).strip() or None
        existing = self.state.goal
        if (
            objective is None
            and self.state.current_conversation is None
            and (existing is None or existing.status == "complete")
        ):
            self.renderer.warning(
                "No conversation is attached. Use /goal <objective> to start a goal in a new chat."
            )
            return
        if existing is not None and existing.status not in {"complete"}:
            if objective:
                self.renderer.warning(
                    "An unfinished goal already exists. Use /goal clear before replacing it."
                )
                return
            if existing.status in {"paused", "blocked", "interrupted"}:
                self._resume_goal()
                return
            self._render_goal_status()
            return

        self.state.goal = GoalState(
            conversation_ref=self.state.current_conversation,
            conversations=(
                [self.state.current_conversation]
                if self.state.current_conversation
                else []
            ),
            context_seed=self._capture_goal_context_seed(
                self.state.current_conversation
            ),
            status="active",
            objective=objective,
        )
        ensure_goal_id(self.state.goal)
        if not self._save_state():
            self.state.goal = existing
            try:
                if existing is None:
                    self.goal_store.clear_current()
                else:
                    self.goal_store.save(existing)
            except OSError as exc:
                self.renderer.warning(
                    f"failed to restore Goal pointer after state save failure: {exc}"
                )
            return
        self.clear_automatic_prompts()
        self._automatic_prompts.append(activation_prompt(objective))
        self.renderer.info("Goal · active · starting")
        self.renderer.info(f"Goal state: {self.goal_store.goal_path(self.state.goal)}")

    def _cmd_exit(self, argv: list[str]) -> int:
        self._pause_active_goal("gptty exited")
        self._leave_temporary_mode()
        self.close()
        return 0

    def _cmd_new(self, argv: list[str]) -> None:
        self._pause_active_goal("conversation changed")
        self._leave_temporary_mode()
        previous = self.state.current_conversation
        self.state.current_conversation = None
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(model=self.state.model or "latest frontier · High")
        self.renderer.info("Started a new conversation.")

    def _cmd_temporary(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning("/temporary takes no arguments.")
            return
        self._pause_active_goal("conversation changed")
        self._leave_temporary_mode()
        previous = self.state.current_conversation
        self.state.current_conversation = None
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self._conversation_mode = "temporary"
        self._reset_temporary_context()
        self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(
            model=self.state.model or "latest frontier · High", temporary=True
        )
        self.renderer.info("Started a new Temporary ChatGPT conversation.")

    def _cmd_temp(self, argv: list[str]) -> None:
        self._cmd_temporary(argv)

    def _cmd_detach(self, argv: list[str]) -> None:
        self._pause_active_goal("conversation detached")
        if self._conversation_mode == "temporary":
            self._leave_temporary_mode()
            self.clear_pending_media()
            self.renderer.clear_context()
            self.renderer.header(model=self.state.model or "latest frontier · High")
            self.renderer.info("Detached from the Temporary ChatGPT conversation.")
            return
        if not self.state.current_conversation:
            self.renderer.info("No conversation is attached.")
            return
        previous = self.state.current_conversation
        self.state.current_conversation = None
        if not self._save_state():
            self.state.current_conversation = previous
            return
        self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(model=self.state.model or "latest frontier · High")
        self.renderer.info(
            "Detached locally. The ChatGPT conversation was not changed."
        )

    def _cmd_stop(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning("/stop takes no arguments.")
            return
        ref = self.conversation_ref
        if not ref:
            self.renderer.info("No conversation is attached.")
            return
        client = self.get_client()
        if self._request_stop_generation(client, ref):
            self.renderer.turn_abort()
            self.renderer.info("Stop requested.")

    def _cmd_export(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning("/export takes no arguments.")
            return
        ref = self.conversation_ref
        if not ref:
            self.renderer.info("No conversation is attached.")
            return
        title = (
            self._temporary_title
            if self._conversation_mode == "temporary"
            else self._conversation_titles.get(ref)
        )
        try:
            if self._conversation_mode == "temporary":
                messages = list(self._temporary_messages)
            else:
                messages = normalize_messages(self.get_client().get_messages(ref))
            path = save_markdown_export(messages, title=title)
        except Exception as exc:  # noqa: BLE001 - interactive export boundary.
            self.renderer.warning(f"Export failed: {exc}")
            return
        self.renderer.info(f"Exported Markdown: {path}")

    def _request_stop_generation(self, client: Any, ref: str) -> bool:
        try:
            result = client.stop_generation(ref, timeout=2.0)
        except Exception as exc:  # noqa: BLE001 - interactive command boundary.
            self.renderer.warning(f"Stop failed: {exc}")
            return False
        stopped = (
            bool(result.get("stopped"))
            if isinstance(result, dict)
            else bool(getattr(result, "stopped", False))
        )
        if not stopped:
            self.renderer.info("No active ChatGPT response to stop.")
            return False
        return True

    def _attach_image_input(self, raw: str | None, *, from_prompt: bool) -> None:
        if not raw:
            return
        if from_prompt:
            try:
                parsed = shlex.split(raw)
            except ValueError as exc:
                self.renderer.warning(f"Invalid image path: {exc}")
                return
            raw = " ".join(parsed)
        try:
            media = normalize_media_input(raw)
        except MediaInputError as exc:
            self.renderer.warning(str(exc))
            return
        if media not in self._pending_media:
            self._pending_media.append(media)
        self.renderer.info(
            f"Attached for next prompt: {Path(media).name or media} · pending: {self.pending_media_count}"
        )

    def _cmd_image(self, argv: list[str]) -> None:
        if argv and argv[0].strip().lower() == "clear":
            count = self.pending_media_count
            self.clear_pending_media()
            self.renderer.info(
                f"Cleared {count} pending image{'s' if count != 1 else ''}."
            )
            return

        raw = " ".join(argv).strip() if argv else self.ui.read_image_path()
        self._attach_image_input(raw, from_prompt=not argv)

    async def _cmd_image_async(self) -> None:
        raw = await self.ui.read_image_path_async()
        self._attach_image_input(raw, from_prompt=True)

    def _cmd_paste(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning(
                "/paste takes no arguments; it attaches the current clipboard image."
            )
            return
        if self._clipboard_dir is None:
            self._clipboard_dir = Path(tempfile.mkdtemp(prefix="gptty-clipboard-"))
        try:
            path = capture_clipboard_image(self._clipboard_dir)
        except ClipboardImageError as exc:
            self.renderer.warning(str(exc))
            return
        self._owned_media.add(path)
        self._pending_media.append(str(path))
        self.renderer.info(
            f"Attached clipboard image for next prompt · pending: {self.pending_media_count}"
        )

    def _begin_resume(self, ref: str, *, reload: bool = False) -> None:
        attached_ref = _canonical_conversation_ref(str(ref))
        if (
            self.state.current_conversation
            and attached_ref != self.state.current_conversation
        ):
            self._pause_active_goal("conversation changed")
        self._leave_temporary_mode()
        self._pending_resume = ResumeRequest(
            conversation_ref=attached_ref,
            reload=reload,
        )
        action = "Reloading" if reload else "Loading conversation"
        self.renderer.info(f"{action}: {_short_ref(attached_ref)}")

    def _cmd_reload(self, argv: list[str]) -> None:
        if argv:
            self.renderer.warning("/reload takes no arguments.")
            return
        if self._conversation_mode == "temporary":
            self.renderer.warning(
                "/reload is unavailable for Temporary ChatGPT conversations."
            )
            return
        ref = self.state.current_conversation
        if not ref:
            self.renderer.info("No conversation is attached.")
            return
        self._begin_resume(ref, reload=True)

    def _cmd_resume(self, argv: list[str]) -> None:
        ref = argv[0] if argv else self._choose_conversation(self.get_client())
        if not ref:
            return
        self._begin_resume(str(ref))

    async def _cmd_resume_async(self) -> None:
        options = self._conversation_options(self.get_client())
        if not options:
            return
        selected = await self.ui.choose_searchable_async(
            "Resume conversation",
            options,
        )
        if selected:
            self._begin_resume(str(selected))

    def _conversation_options(self, client: Any) -> list[tuple[str, str]]:
        try:
            recent_catalog = getattr(client, "list_recent_conversations", None)
            conversations = (
                recent_catalog(limit=100)
                if callable(recent_catalog)
                else client.list_conversations()
            )
        except Exception as exc:  # noqa: BLE001 - interactive command boundary.
            self.renderer.warning(f"Conversation list failed: {exc}")
            return []
        current_ref = _canonical_conversation_ref(self.state.current_conversation or "")
        options: list[tuple[str, str]] = []
        for item in conversations:
            conversation_id = _catalog_conversation_id(item)
            if conversation_id is None:
                continue
            title = _field_text(item, "title")
            if title:
                self._conversation_titles[conversation_id] = title
            options.append(
                (
                    conversation_id,
                    _conversation_label(
                        item,
                        current=conversation_id == current_ref,
                    ),
                )
            )
        if not options:
            self.renderer.info("No ChatGPT conversations found.")
        return options

    def _choose_conversation(self, client: Any) -> str | None:
        options = self._conversation_options(client)
        if not options:
            return None
        selected = self.ui.choose_searchable(
            "Resume conversation",
            options,
        )
        return str(selected) if selected else None

    def _available_models(self) -> dict[str, Any] | None:
        try:
            models = self.get_client().list_models()
        except Exception as exc:  # noqa: BLE001 - interactive command boundary.
            self.renderer.warning(f"Model list failed: {exc}")
            return None
        available = [
            model
            for model in models
            if _model_slug(model) is not None and _model_available(model)
        ]
        by_slug = {_model_slug(model): model for model in available}
        return {slug: model for slug, model in by_slug.items() if slug is not None}

    def _model_options(self, by_slug: dict[str, Any]) -> list[tuple[Any, str]]:
        options: list[tuple[Any, str]] = [
            (
                "",
                "Default · latest frontier · High"
                + (" · current" if self.state.model is None else ""),
            )
        ]
        options.extend(
            (
                slug,
                _model_label(model, current=slug == self.state.model),
            )
            for slug, model in by_slug.items()
        )
        return options

    def _apply_model(self, selected: str) -> None:
        previous = self.state.model
        self.state.model = selected or None
        if not self._save_state():
            self.state.model = previous
            return
        self.renderer.info(f"Model: {self.state.model or 'latest frontier · High'}")

    def _cmd_model(self, argv: list[str]) -> None:
        if argv and argv[0].strip().lower() == "default":
            self._apply_model("")
            return

        by_slug = self._available_models()
        if by_slug is None:
            return
        if argv:
            selected = argv[0].strip()
            if selected not in by_slug:
                self.renderer.warning(
                    "Unknown model slug. Run /model and choose from the live ChatGPT list."
                )
                return
        else:
            value = self.ui.choose_searchable(
                "ChatGPT model",
                self._model_options(by_slug),
            )
            if value is None:
                return
            selected = str(value)
        self._apply_model(selected)

    async def _cmd_model_async(self) -> None:
        by_slug = self._available_models()
        if by_slug is None:
            return
        value = await self.ui.choose_searchable_async(
            "ChatGPT model",
            self._model_options(by_slug),
        )
        if value is not None:
            self._apply_model(str(value))

    def _resume_goal(self) -> None:
        goal = self.state.goal
        if goal is None:
            self.renderer.info("No goal is configured.")
            return
        if goal.status == "complete":
            self.renderer.info("Goal · complete")
            return
        if goal.status == "active":
            self._render_goal_status()
            return
        if (
            goal.conversation_ref
            and goal.conversation_ref != self.state.current_conversation
        ):
            self.renderer.warning(
                f"Goal belongs to {_short_ref(goal.conversation_ref)}. Resume that conversation before /goal resume."
            )
            return
        bootstrap_without_chat = (
            goal.conversation_ref is None and self.state.current_conversation is None
        )
        if goal.conversation_ref is None and self.state.current_conversation:
            goal.conversation_ref = self.state.current_conversation
            if self.state.current_conversation not in goal.conversations:
                goal.conversations.append(self.state.current_conversation)
        previous_reason = goal.reason
        goal.status = "active"
        goal.reason = None
        if not self._save_state():
            return
        self.clear_automatic_prompts()
        if bootstrap_without_chat and goal.conversations:
            self._automatic_prompts.append(
                rollover_prompt(
                    goal,
                    reason=previous_reason or "resuming durable Goal state",
                )
            )
            self._goal_bootstrap_pending = True
        else:
            self._automatic_prompts.append(
                activation_prompt(goal.objective)
                if bootstrap_without_chat
                else continuation_prompt()
            )
        self.renderer.info(
            f"Goal · active · resuming after {goal.turn_count} turn{'s' if goal.turn_count != 1 else ''}"
        )

    def _render_goal_status(self) -> None:
        goal = self.state.goal
        if goal is None:
            self.renderer.info("No goal is configured.")
            return
        details = [f"Goal · {goal.status}", f"turns {goal.turn_count}"]
        if goal.goal_id:
            details.append(f"id {goal.goal_id[:8]}")
        if goal.conversation_ref:
            details.append(_short_ref(goal.conversation_ref))
        if goal.rollover_count:
            details.append(f"rollovers {goal.rollover_count}")
        if goal.protocol_failures:
            details.append(
                f"protocol misses {goal.protocol_failures}/{MAX_PROTOCOL_FAILURES}"
            )
        self.renderer.info(" · ".join(details))
        if goal.reason:
            self.renderer.info(f"Goal reason: {goal.reason}")
        if goal.goal_id:
            self.renderer.info(f"Goal state: {self.goal_store.goal_path(goal)}")
            self.renderer.info(
                f"Goal checkpoint: {self.goal_store.checkpoint_path(goal)}"
            )

    def _pause_goal_for_service_condition(self, reason: str) -> bool:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return False
        goal.status = "paused"
        goal.reason = f"service backoff required: {reason}"
        self.clear_automatic_prompts()
        if not self._save_state():
            return False
        self.renderer.warning(
            "Goal · paused · service backoff required · use /goal resume later"
        )
        return True

    def _pause_active_goal(self, reason: str) -> bool:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return False
        goal.status = "paused"
        goal.reason = reason
        self.clear_automatic_prompts()
        self._save_state()
        return True

    def _interrupt_goal(
        self,
        reason: str,
        *,
        notify: bool,
        chat_title: str | None = None,
    ) -> None:
        goal = self.state.goal
        if goal is None:
            return
        goal.status = "interrupted"
        goal.reason = reason
        self.clear_automatic_prompts()
        self._save_state()
        self.renderer.warning(f"Goal · interrupted · {reason}")
        if notify:
            notify_response_complete(
                chat_title=chat_title,
                final_response=f"Goal interrupted. {reason}",
            )

    def _queue_goal_continuation(self, *, protocol_recovery: bool) -> None:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return
        self._automatic_prompts.append(
            continuation_prompt(protocol_recovery=protocol_recovery)
        )
        if protocol_recovery:
            self.renderer.warning(
                f"Goal · continuing · missing status {goal.protocol_failures}/{MAX_PROTOCOL_FAILURES}"
            )
        else:
            self.renderer.info(f"Goal · continuing · next turn {goal.turn_count + 1}")

    def _save_state(self) -> bool:
        try:
            if self.state.goal is not None:
                self.goal_store.save(self.state.goal)
            save_chat_state(self.state_path, self.state)
        except (OSError, StateError) as exc:
            self.renderer.warning(str(exc))
            return False
        return True


def _canonical_conversation_ref(value: str) -> str:
    raw = value.strip()
    if not raw:
        return raw
    parsed = urlparse(raw)
    supported_hosts = {"chatgpt.com", "www.chatgpt.com", "chat.openai.com"}
    if parsed.scheme in {"http", "https"} and parsed.hostname in supported_hosts:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "c":
            return parts[1]
    return raw


def _catalog_conversation_id(item: Any) -> str | None:
    value = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _conversation_label(item: Any, *, current: bool = False) -> str:
    conversation_id = _catalog_conversation_id(item) or "unknown"
    title = _field_text(item, "title") or "Untitled"
    updated = _format_update_time(_field(item, "update_time"))
    starred = "★ " if _field(item, "is_starred") is True else ""
    archived = " · archived" if _field(item, "is_archived") is True else ""
    updated_part = f" · {updated}" if updated else ""
    current_part = " · current" if current else ""
    return f"{starred}{title}{updated_part}{archived}{current_part} · {_short_ref(conversation_id, max_len=18)}"


def _format_update_time(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))
        except (OverflowError, OSError, ValueError):
            return ""
    if isinstance(value, str):
        return value.replace("T", " ")[:16]
    return ""


def _model_slug(model: Any) -> str | None:
    value = _field(model, "slug")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _model_available(model: Any) -> bool:
    if _field(model, "enabled") is False:
        return False
    if _field(model, "is_disabled") is True:
        return False
    if _field(model, "is_work_mode_model") is True:
        return False
    return _model_slug(model) != "research"


def _model_label(model: Any, *, current: bool = False) -> str:
    slug = _model_slug(model) or "unknown"
    title = (
        _field_text(model, "title")
        or _field_text(model, "display_name")
        or _field_text(model, "name")
        or slug
    )
    label = title if title == slug else f"{title} · {slug}"
    return f"{label} · current" if current else label


def _snapshot_status(snapshot: Any) -> str:
    status = _field(snapshot, "status")
    if status is None and isinstance(snapshot, dict):
        status = snapshot.get("status")
    value = _field(status, "status") if status is not None else None
    if value is None and isinstance(status, str):
        value = status
    return str(value).strip().lower() if value else ""


def _snapshot_messages(snapshot: Any) -> list[Any]:
    messages = _field(snapshot, "messages")
    if isinstance(messages, list):
        items = messages
    elif messages is not None:
        try:
            items = list(messages)
        except TypeError:
            items = []
    else:
        items = []
    return [message for message in items if _message_is_user_visible(message)]


def _message_is_user_visible(message: Any) -> bool:
    role = _field_text(message, "role")
    if role == "user":
        return True
    if role != "assistant":
        return False
    recipient = _field(message, "recipient")
    return recipient in {None, "", "all"}


def _message_identity(message: Any) -> str:
    for field in ("message_id", "id", "node_id"):
        value = _field(message, field)
        if value:
            return str(value)
    return f"{_field_text(message, 'role')}:{_message_text(message)}"


def _message_text(message: Any) -> str:
    for field in ("text", "content", "message"):
        value = _field(message, field)
        if value is not None:
            return str(value)
    return ""


def _last_user_message_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if _field_text(message, "role") == "user":
            return _message_text(message)
    return ""


def _last_assistant_message_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if _field_text(message, "role") == "assistant":
            return _message_text(message)
    return ""


def _active_elapsed_seconds(messages: list[Any]) -> float:
    for message in reversed(messages):
        if _field_text(message, "role") != "user":
            continue
        created = _field(message, "create_time")
        if isinstance(created, (int, float)) and not isinstance(created, bool):
            return max(0.0, time.time() - float(created))
    return 0.0


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _field_text(value: Any, name: str) -> str:
    raw = _field(value, name)
    return raw.strip() if isinstance(raw, str) else ""


def _short_ref(ref: str, *, max_len: int = 44) -> str:
    if len(ref) <= max_len:
        return ref
    return f"…{ref[-(max_len - 1) :]}"
