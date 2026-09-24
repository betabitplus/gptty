from __future__ import annotations

import hashlib
import json
import os
import shlex
import sqlite3
import shutil
import tempfile
import time
import uuid
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
    acceptance_criteria_error,
    activation_prompt,
    completion_checkpoint_error,
    continuation_prompt,
    operation_identity_instruction,
    ParsedGoalResponse,
    parse_goal_response,
    rollover_prompt,
    sanitize_goal_history_text,
    steering_prompt,
)
from ..goal_lock import (
    GoalRunLock,
    goal_lock_is_held,
    read_goal_lock_metadata,
    try_acquire_goal_lock,
)
from ..goal_store import GoalCompatibilityError, GoalConflictError, GoalStore, ensure_goal_id
from ..media import MediaInputError, normalize_media_input
from ..output import OutputMessage, normalize_messages
from ..state import ChatState, GoalAcceptanceCriterion, GoalState, StateError, save_chat_state
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
        runner_id: str | None = None,
    ) -> None:
        self.state = state
        self.state_path = state_path
        self.get_client = get_client
        self.ui = ui
        self.renderer = renderer
        self.tui_archive = tui_archive
        self.goal_store = GoalStore(state_path)
        self.runner_id = runner_id or uuid.uuid4().hex
        self._goal_run_lock: GoalRunLock | None = None
        if (
            state.goal is not None
            and state.goal.status == "active"
            and (
                not state.goal.runner_id
                or (
                    state.goal.runner_id == self.runner_id
                    and state.goal.runner_pid in {0, os.getpid()}
                )
            )
        ):
            self._acquire_goal_run_lock(state.goal)
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

    def _acquire_goal_run_lock(self, goal: GoalState) -> bool:
        goal_id = ensure_goal_id(goal)
        current = self._goal_run_lock
        if (
            current is not None
            and not current.released
            and current.goal_id == goal_id
        ):
            goal.runner_id = self.runner_id
            goal.runner_pid = os.getpid()
            return True
        if current is not None and not current.released:
            current.release()
        lease = try_acquire_goal_lock(
            self.goal_store.root,
            goal_id,
            runner_id=self.runner_id,
            pid=os.getpid(),
        )
        self._goal_run_lock = lease
        if lease is None:
            return False
        goal.runner_id = self.runner_id
        goal.runner_pid = os.getpid()
        return True

    def _release_goal_run_lock(self) -> None:
        current = self._goal_run_lock
        self._goal_run_lock = None
        if current is not None:
            current.release()

    def _owns_goal_run(self, goal: GoalState | None = None) -> bool:
        goal = goal or self.state.goal
        lease = self._goal_run_lock
        if goal is None or lease is None or lease.released:
            return False
        return (
            lease.goal_id == (goal.goal_id or "")
            and goal.runner_id == self.runner_id
            and goal.runner_pid == os.getpid()
        )

    @property
    def goal_owned_elsewhere(self) -> bool:
        goal = self.state.goal
        return bool(
            goal is not None
            and goal.status == "active"
            and not self._owns_goal_run(goal)
        )

    def _reject_remote_goal_mutation(self, action: str) -> bool:
        if not self.goal_owned_elsewhere:
            return False
        goal = self.state.goal
        assert goal is not None
        self.renderer.warning(
            f"Goal is active in another live gptty process (owner pid {goal.runner_pid or '?'}); "
            f"{action} is blocked for this shared state/profile."
        )
        return True

    @property
    def goal_active(self) -> bool:
        goal = self.state.goal
        if (
            goal is None
            or goal.status != "active"
            or not self._owns_goal_run(goal)
            or self._conversation_mode != "normal"
        ):
            return False
        if goal.conversation_ref is None:
            return self.state.current_conversation is None
        return goal.conversation_ref == self.state.current_conversation

    def _goal_for_conversation(self, conversation_ref: str | None) -> GoalState | None:
        return self.goal_store.load_for_conversation(conversation_ref)

    def _attach_goal_for_conversation(self, conversation_ref: str | None) -> GoalState | None:
        goal = self._goal_for_conversation(conversation_ref)
        self.state.goal = goal
        return goal

    def _goal_by_prefix(self, prefix: str) -> GoalState | None:
        token = prefix.strip().lower()
        if not token:
            return None
        matches = [
            goal
            for goal in self.goal_store.list_goals(limit=None)
            if (goal.goal_id or "").lower().startswith(token)
        ]
        if not matches:
            self.renderer.warning(f"No Goal matches id prefix {prefix}.")
            return None
        if len(matches) > 1:
            self.renderer.warning(
                f"Goal id prefix {prefix} is ambiguous; use more characters."
            )
            return None
        return matches[0]

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
        previous_ref = self.state.current_conversation
        previous_goal = self.state.goal
        self.state.current_conversation = attached_ref
        try:
            if (
                request.reload
                and previous_goal is not None
                and previous_goal.conversation_ref == attached_ref
            ):
                self.state.goal = previous_goal
            else:
                self._attach_goal_for_conversation(attached_ref)
        except (OSError, sqlite3.Error) as exc:
            self.state.current_conversation = previous_ref
            self.state.goal = previous_goal
            self.renderer.warning(f"Goal routing failed for resumed conversation: {exc}")
            return False
        if not self._save_state(persist_goal=False):
            self.state.current_conversation = previous_ref
            self.state.goal = previous_goal
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
        if self.state.goal is not None:
            self._render_goal_status()
        messages = _snapshot_messages(snapshot)
        normalized_messages = normalize_messages(messages)
        if self.state.goal is not None:
            visible_messages: list[OutputMessage] = []
            for message in normalized_messages:
                visible_text = sanitize_goal_history_text(message.role, message.text)
                if visible_text is None:
                    continue
                visible_messages.append(
                    OutputMessage(
                        role=message.role,
                        text=visible_text,
                        created_at=message.created_at,
                    )
                )
            normalized_messages = visible_messages
        self.renderer.messages(normalized_messages)
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

    def prepare_goal_user_prompt(self, prompt: str) -> str | None:
        if not self.goal_active:
            return prompt
        goal = self.state.goal
        assert goal is not None
        if not self._save_state(
            event_type="user_steering",
            event_payload={
                "text": prompt,
                "conversation_ref": goal.conversation_ref,
                "turn": goal.turn_count + 1,
            },
        ):
            return None
        return steering_prompt(prompt, goal=goal)

    def mark_goal_turn_started(self, prompt: str, *, automatic: bool) -> str | None:
        """Persist an operation boundary before dispatching any Goal model turn."""
        if not self.goal_active:
            return prompt
        goal = self.state.goal
        assert goal is not None
        operation_id = goal.active_operation_id
        event_type = "operation_resumed"
        if not operation_id:
            operation_id = (
                f"{ensure_goal_id(goal)}:g{goal.generation}:t{goal.turn_count + 1}"
            )
            goal.active_operation_id = operation_id
            goal.active_operation_turn = goal.turn_count + 1
            event_type = "operation_started"
        payload = {
            "operation_id": operation_id,
            "turn": goal.active_operation_turn,
            "automatic": bool(automatic),
            "conversation_ref": goal.conversation_ref,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        if not self._save_state(event_type=event_type, event_payload=payload):
            return None
        return f"{prompt.rstrip()}\n\n{operation_identity_instruction(operation_id)}"

    def record_goal_tool_event(self, event: dict[str, Any]) -> None:
        """Persist machine-observed Goal transport/tool evidence without mutating revision.

        Despite the historical method name this also records a committed new-chat
        identity. That event is deliberately independent from the later terminal
        Goal transition so restart can recover a chat created just before a crash.
        """
        goal = self.state.goal
        if goal is None or not goal.goal_id or not goal.active_operation_id:
            return
        event_type = str(event.get("type") or "")
        if event_type == "browser_native_write_completed":
            conversation_ref = str(
                event.get("conversation_id") or event.get("conversationId") or ""
            ).strip()
            if not conversation_ref or conversation_ref.startswith("WEB:"):
                return
            payload = {
                "operation_id": goal.active_operation_id,
                "conversation_ref": conversation_ref,
                "submission_id": event.get("submission_id"),
                "turn_exchange_id": event.get("turn_exchange_id"),
                "source_event": event_type,
            }
            event_key = hashlib.sha256(
                f"conversation-write:{goal.active_operation_id}:{conversation_ref}".encode("utf-8")
            ).hexdigest()
            try:
                self.goal_store.record_observed_event(
                    goal,
                    "conversation_write_committed",
                    payload,
                    event_key=event_key,
                )
            except (GoalConflictError, OSError, sqlite3.Error) as exc:
                self.renderer.warning(f"Goal journal route evidence write failed: {exc}")
            return

        if event_type != "canonical_intermediate_message":
            return
        kind = str(event.get("message_kind") or "")
        if kind not in {"tool_call", "tool_result"}:
            return
        payload = {
            "operation_id": goal.active_operation_id,
            "conversation_ref": goal.conversation_ref or self.state.current_conversation,
            "message_id": event.get("message_id"),
            "tool_call_id": event.get("tool_call_id"),
            "parent_message_id": event.get("parent_message_id"),
            "turn_exchange_id": event.get("turn_exchange_id"),
            "submission_id": event.get("submission_id"),
            "source_offset": event.get("source_offset"),
            "tool_name": event.get("tool_name"),
            "label": event.get("label"),
            "text": str(event.get("text") or ""),
        }
        message_id = str(event.get("message_id") or "").strip()
        if message_id:
            event_key = hashlib.sha256(
                f"{kind}:message:{message_id}".encode("utf-8")
            ).hexdigest()
        else:
            stable = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            event_key = hashlib.sha256(
                f"{kind}:{stable}".encode("utf-8")
            ).hexdigest()
        try:
            self.goal_store.record_observed_event(
                goal,
                "tool_call_observed" if kind == "tool_call" else "tool_result_observed",
                payload,
                event_key=event_key,
            )
        except (OSError, sqlite3.Error) as exc:
            # The turn remains ambiguous. Do not falsify evidence; terminal handling
            # will keep the operation open if delivery itself becomes uncertain.
            self.renderer.warning(f"Goal journal tool evidence write failed: {exc}")

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
        goal.runner_id = None
        goal.runner_pid = 0
        self.clear_automatic_prompts()
        self._save_state(
            event_type="goal_paused",
            event_payload={"reason": goal.reason, "ambiguous_operation": goal.active_operation_id},
        )
        self._release_goal_run_lock()
        self.renderer.info("Goal · paused · stopped by user")

    def goal_display_text(self, text: str) -> str:
        parsed = parse_goal_response(text)
        return parsed.body if parsed.signal is not None else text

    def _goal_result_event_payload(
        self, result: dict[str, Any], parsed: Any | None = None
    ) -> dict[str, Any]:
        goal = self.state.goal
        marker = self._goal_terminal_marker(result)
        if parsed is None:
            parsed = parse_goal_response(str(result.get("text") or ""))
        payload: dict[str, Any] = {
            "operation_id": goal.active_operation_id if goal is not None else None,
            "conversation_ref": str(result.get("conversation_ref") or "").strip() or None,
            "signal": parsed.signal.value if parsed.signal is not None else None,
            "body": parsed.body,
            "text": str(result.get("text") or ""),
        }
        if marker is not None:
            payload["status"] = marker[1]
            payload["detail"] = marker[2]
        if parsed.checkpoint is not None:
            payload["checkpoint"] = {
                "summary": parsed.checkpoint.summary,
                "completed": list(parsed.checkpoint.completed),
                "decisions": list(parsed.checkpoint.decisions),
                "pending": list(parsed.checkpoint.pending),
                "next": parsed.checkpoint.next_step,
            }
        return payload

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
            goal.runner_id = None
            goal.runner_pid = 0
            self.clear_automatic_prompts()
            self._save_state(
                event_type="goal_paused",
                event_payload={"reason": goal.reason, "ambiguous_operation": goal.active_operation_id},
            )
            self._release_goal_run_lock()
            self.renderer.info("Goal · paused · stopped by user")
            return

        parsed = parse_goal_response(str(result.get("text") or ""))
        completion_error = completion_checkpoint_error(parsed)
        if completion_error is None and parsed.signal is GoalSignal.COMPLETE:
            completion_error = acceptance_criteria_error(goal)
        if completion_error is not None:
            self.renderer.warning(f"Goal · rejected COMPLETE · {completion_error}")
            parsed = ParsedGoalResponse(
                signal=None, body=parsed.body, checkpoint=parsed.checkpoint
            )
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
                self._save_state(
                    event_type="turn_abnormal",
                    event_payload=self._goal_result_event_payload(result, parsed),
                )
                self._rollover_goal(detail)
                return
            if status in {"filtered", "blocked"}:
                blocked_payload = self._goal_result_event_payload(result, parsed)
                goal.status = "blocked"
                goal.reason = detail
                goal.runner_id = None
                goal.runner_pid = 0
                self.clear_automatic_prompts()
                if not self._save_state(
                    event_type="turn_terminal",
                    event_payload=blocked_payload,
                ):
                    return
                self._release_goal_run_lock()
                self.renderer.warning("Goal · blocked · user action required")
                notify_response_complete(
                    chat_title=str(result.get("title") or "").strip() or None,
                    final_response=f"Goal blocked. {detail}",
                )
                return
            if status == "rate-limited":
                self._save_state(
                    event_type="turn_abnormal",
                    event_payload=self._goal_result_event_payload(result, parsed),
                )
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
                    event_payload=self._goal_result_event_payload(result, parsed),
                )
                return

        machine_evidence: dict[str, Any] = self.goal_store._tool_evidence_summary([])
        reconciliation_evidence: dict[str, Any] = {
            "verification_calls": 0,
            "verification_results": 0,
            "unresolved_verification_calls": 0,
            "ambiguous_verification_results": 0,
            "proofs": 0,
            "covered_original_calls": [],
            "ready": False,
        }
        reconciled_ambiguity = False
        if goal.active_operation_id:
            machine_evidence = self.goal_store.operation_evidence(
                goal, goal.active_operation_id
            )
            reconciliation_evidence = self.goal_store.operation_reconciliation_evidence(
                goal, goal.active_operation_id
            )
            if machine_evidence["unresolved_tool_calls"]:
                can_accept_reconciliation = bool(
                    parsed.signal is GoalSignal.COMPLETE
                    and parsed.checkpoint is not None
                    and not parsed.checkpoint.pending
                    and reconciliation_evidence["ready"]
                )
                if can_accept_reconciliation:
                    reconciled_ambiguity = True
                    self.goal_store.record_observed_event(
                        goal,
                        "operation_reconciled",
                        {
                            "operation_id": goal.active_operation_id,
                            "original_evidence": machine_evidence,
                            "verification_evidence": reconciliation_evidence,
                            "checkpoint_summary": parsed.checkpoint.summary,
                            "completed": list(parsed.checkpoint.completed),
                        },
                        event_key=f"operation-reconciled:{goal.active_operation_id}",
                    )
                else:
                    detail = (
                        f"machine journal has {machine_evidence['unresolved_tool_calls']} observed tool call(s) "
                        f"without exact result/proof for durable operation {goal.active_operation_id}"
                    )
                    self._update_goal_checkpoint(goal, parsed)
                    payload = self._goal_result_event_payload(result, parsed)
                    payload["machine_evidence"] = machine_evidence
                    payload["reconciliation_evidence"] = reconciliation_evidence

                    verification_finished_without_proof = bool(
                        reconciliation_evidence["verification_results"]
                        and not reconciliation_evidence[
                            "unresolved_verification_calls"
                        ]
                        and not reconciliation_evidence["ready"]
                    )
                    if (
                        parsed.signal is GoalSignal.BLOCKED
                        or verification_finished_without_proof
                    ):
                        if parsed.signal is GoalSignal.COMPLETE:
                            self.renderer.warning(
                                f"Goal · rejected COMPLETE · {detail}"
                            )
                        self._block_goal_for_ambiguous_operation(
                            detail,
                            event_payload=payload,
                        )
                        return

                    if parsed.signal is GoalSignal.COMPLETE:
                        self.renderer.warning(
                            f"Goal · rejected COMPLETE · {detail}"
                        )
                    self._recover_goal_same_chat(
                        detail,
                        allow_rollover=True,
                        event_payload=payload,
                    )
                    return

        terminal_payload = self._goal_result_event_payload(result, parsed)
        terminal_payload["machine_validation"] = {
            "structured_checkpoint": parsed.checkpoint is not None,
            "completed_count": len(parsed.checkpoint.completed) if parsed.checkpoint is not None else 0,
            "pending_count": len(parsed.checkpoint.pending) if parsed.checkpoint is not None else 0,
            "operation_evidence": machine_evidence,
            "reconciliation_evidence": reconciliation_evidence,
            "reconciled_ambiguity": reconciled_ambiguity,
        }
        self._update_goal_checkpoint(goal, parsed)

        if parsed.signal is GoalSignal.COMPLETE:
            goal.active_operation_id = None
            goal.active_operation_turn = 0
            goal.status = "complete"
            goal.runner_id = None
            goal.runner_pid = 0
            goal.protocol_failures = 0
            goal.recovery_count = 0
            goal.reason = None
            self.clear_automatic_prompts()
            if not self._save_state(
                event_type="turn_terminal",
                event_payload=terminal_payload,
            ):
                return
            self._release_goal_run_lock()
            self.renderer.info(
                f"Goal · complete · {goal.turn_count} turn{'s' if goal.turn_count != 1 else ''}"
            )
            notify_response_complete(
                chat_title=str(result.get("title") or "").strip() or None,
                final_response=parsed.body or "Goal complete.",
            )
            return

        if parsed.signal is GoalSignal.BLOCKED:
            if not machine_evidence["unresolved_tool_calls"]:
                goal.active_operation_id = None
                goal.active_operation_turn = 0
            goal.status = "blocked"
            goal.runner_id = None
            goal.runner_pid = 0
            goal.protocol_failures = 0
            goal.recovery_count = 0
            goal.reason = parsed.body or "agent reported a blocker"
            self.clear_automatic_prompts()
            if not self._save_state(
                event_type="turn_terminal",
                event_payload=terminal_payload,
            ):
                return
            self._release_goal_run_lock()
            self.renderer.warning("Goal · blocked · user action required")
            notify_response_complete(
                chat_title=str(result.get("title") or "").strip() or None,
                final_response=f"Goal blocked. {parsed.body}".strip(),
            )
            return

        if not machine_evidence["unresolved_tool_calls"]:
            goal.active_operation_id = None
            goal.active_operation_turn = 0

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
        if not self._save_state(
            event_type="turn_terminal",
            event_payload=terminal_payload,
        ):
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
        failure_payload = self._goal_result_event_payload(result)
        failure_payload["detail"] = detail
        failure_payload["status"] = status
        if label == "chat" and status in {"limit-reached", "unavailable"}:
            self._save_state(event_type="turn_failed", event_payload=failure_payload)
            return self._rollover_goal(detail)
        if status in {"blocked", "filtered"}:
            goal.status = "blocked"
            goal.reason = detail
            goal.runner_id = None
            goal.runner_pid = 0
            self.clear_automatic_prompts()
            goal.active_operation_id = None
            goal.active_operation_turn = 0
            if not self._save_state(event_type="turn_terminal", event_payload=failure_payload):
                return True
            self._release_goal_run_lock()
            self.renderer.warning("Goal · blocked · user action required")
            notify_response_complete(
                chat_title=chat_title,
                final_response=f"Goal blocked. {detail}",
            )
            return True
        if status == "rate-limited":
            self._save_state(event_type="turn_failed", event_payload=failure_payload)
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
                event_payload=failure_payload,
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

    def _block_goal_for_ambiguous_operation(
        self,
        reason: str,
        *,
        event_payload: dict[str, Any],
    ) -> bool:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return False
        goal.status = "blocked"
        goal.reason = reason
        goal.runner_id = None
        goal.runner_pid = 0
        self.clear_automatic_prompts()
        payload = dict(event_payload)
        payload["reason"] = reason
        payload["ambiguous_operation"] = goal.active_operation_id
        if not self._save_state(
            event_type="goal_blocked_ambiguous_operation",
            event_payload=payload,
        ):
            return False
        self._release_goal_run_lock()
        self.renderer.warning(
            "Goal · blocked · ambiguous external side effect requires machine-verifiable reconciliation"
        )
        notify_response_complete(
            final_response=(
                "Goal blocked. An external side effect has no exact result or "
                "trusted reconciliation proof."
            )
        )
        return True

    def _recover_goal_same_chat(
        self,
        reason: str,
        *,
        allow_rollover: bool,
        event_payload: dict[str, Any] | None = None,
    ) -> bool:
        goal = self.state.goal
        if goal is None or goal.status != "active":
            return False
        goal.reason = reason
        if not allow_rollover:
            goal.recovery_count = 0
            self.clear_automatic_prompts()
            self._automatic_prompts.append(
                abnormal_recovery_prompt(
                    reason,
                    goal=goal,
                    journal_context=self.goal_store.recovery_context(goal),
                )
            )
            if not self._save_state(
                event_type="turn_abnormal",
                event_payload=event_payload or {"detail": reason, "operation_id": goal.active_operation_id},
            ):
                self._interrupt_goal("failed to persist goal recovery state", notify=False)
                return False
            self.renderer.info("Goal · continuing · response was truncated")
            return True

        goal.recovery_count += 1
        if goal.recovery_count > MAX_RECOVERY_ATTEMPTS:
            return self._rollover_goal(f"repeated non-standard turns: {reason}")
        self.clear_automatic_prompts()
        self._automatic_prompts.append(
            abnormal_recovery_prompt(
                reason,
                goal=goal,
                journal_context=self.goal_store.recovery_context(goal),
            )
        )
        if not self._save_state(
            event_type="turn_abnormal",
            event_payload=event_payload or {"detail": reason, "operation_id": goal.active_operation_id},
        ):
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
        previous_generation = goal.generation
        goal.generation += 1
        goal.rollover_count += 1
        goal.recovery_count = 0
        goal.protocol_failures = 0
        goal.reason = f"recovering in a new chat: {reason}"
        self.state.current_conversation = None
        self.clear_automatic_prompts()
        if not self._save_state(
            event_type="rollover",
            event_payload={
                "reason": reason,
                "old_conversation": old_ref,
                "from_generation": previous_generation,
                "to_generation": goal.generation,
                "ambiguous_operation": goal.active_operation_id,
            },
        ):
            self._interrupt_goal("failed to persist goal rollover state", notify=False)
            return False
        journal_context = self.goal_store.recovery_context(goal)
        self._automatic_prompts.append(
            rollover_prompt(goal, reason=reason, journal_context=journal_context)
        )
        self._goal_bootstrap_pending = True
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
        self._release_goal_run_lock()
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

    def _capture_goal_context_history(
        self, conversation_ref: str | None
    ) -> list[str]:
        """Capture the full visible user/assistant branch for durable recovery."""
        if not conversation_ref:
            return []
        try:
            messages = list(self.get_client().get_messages(conversation_ref))
        except Exception:
            return []

        captured: list[str] = []
        for message in messages:
            role = _field_text(message, "role")
            if role not in {"user", "assistant"}:
                continue
            if role == "assistant":
                recipient = _field_text(message, "recipient")
                if recipient and recipient not in {"all", "assistant"}:
                    continue
            text = _message_text(message).strip()
            if not text:
                continue
            captured.append(f"{role}: {text}")
        return captured

    @staticmethod
    def _compact_goal_context_seed(history: list[str]) -> list[str]:
        captured: list[str] = []
        total = 0
        for entry in reversed(history):
            role, separator, text = entry.partition(": ")
            compact = " ".join((text if separator else entry).split()).strip()
            if not compact:
                continue
            compact_entry = f"{role}: {compact[:1600]}" if separator else compact[:1600]
            if total + len(compact_entry) > 12000 and captured:
                break
            captured.append(compact_entry)
            total += len(compact_entry)
            if len(captured) >= 12:
                break
        captured.reverse()
        return captured

    def _capture_goal_context_seed(self, conversation_ref: str | None) -> list[str]:
        return self._compact_goal_context_seed(
            self._capture_goal_context_history(conversation_ref)
        )

    def _render_goal_list(self, *, include_terminal: bool) -> None:
        statuses = None if include_terminal else {"active", "paused", "blocked"}
        try:
            goals = self.goal_store.list_goals(statuses=statuses, limit=None)
        except (OSError, sqlite3.Error) as exc:
            self.renderer.warning(f"Goal list failed: {exc}")
            return
        if not goals:
            suffix = "" if include_terminal else " Use /goal list all for completed/archived Goals."
            self.renderer.info(f"No matching Goals.{suffix}")
            return
        label = "all" if include_terminal else "unfinished"
        self.renderer.info(f"Goals · {len(goals)} {label}")
        current_id = self.state.goal.goal_id if self.state.goal is not None else None
        for goal in goals:
            goal_id = goal.goal_id or "?"
            marker = "*" if goal_id == current_id else " "
            try:
                bindings = self.goal_store.bindings_for_goal(goal)
            except (OSError, sqlite3.Error) as exc:
                self.renderer.warning(f"Goal binding lookup failed: {exc}")
                return
            ref = (
                goal.conversation_ref
                if goal.conversation_ref in bindings
                else (bindings[-1] if bindings else None)
            )
            route_label = (
                _short_ref(ref)
                if ref
                else ("history only" if goal.conversation_ref or goal.conversations else "new chat")
            )
            owner = ""
            if goal.status == "active" and goal.runner_pid:
                owner = (
                    " · this process"
                    if self._owns_goal_run(goal)
                    else f" · pid {goal.runner_pid}"
                )
            objective = " ".join((goal.objective or "inherited conversation goal").split())
            if len(objective) > 72:
                objective = objective[:69] + "..."
            self.renderer.info(
                f"{marker} {goal.status:<8} {goal_id[:8]} · "
                f"{route_label} · t{goal.turn_count}{owner} · {objective}"
            )
        self.renderer.info("/goal open <id> switches to a Goal · /goal list all shows terminal Goals")

    def _open_goal(self, prefix: str) -> None:
        try:
            goal = self._goal_by_prefix(prefix)
        except (OSError, sqlite3.Error) as exc:
            self.renderer.warning(f"Goal lookup failed: {exc}")
            return
        if goal is None:
            return
        try:
            bindings = self.goal_store.bindings_for_goal(goal)
        except (OSError, sqlite3.Error) as exc:
            self.renderer.warning(f"Goal binding lookup failed: {exc}")
            return
        ref = (
            goal.conversation_ref
            if goal.conversation_ref in bindings
            else (bindings[-1] if bindings else None)
        )
        if ref is None and (goal.conversation_ref or goal.conversations):
            self.renderer.warning(
                f"Goal {(goal.goal_id or '?')[:8]} is history only; it no longer owns a conversation."
            )
            return
        current_goal_id = self.state.goal.goal_id if self.state.goal is not None else None
        if (
            current_goal_id == goal.goal_id
            and ref is not None
            and self.state.current_conversation == ref
        ):
            # `/goal open` is navigation. Re-opening the already attached Goal
            # must be instantaneous instead of starting a redundant snapshot load
            # that temporarily blocks the next navigation command.
            self.state.goal = goal
            self._render_goal_status()
            return
        if ref:
            self._begin_resume(ref)
            return
        # A Goal created before its first ChatGPT write has no conversation yet.
        # Attaching it locally is enough; /goal resume will safely bootstrap it.
        self._pause_active_goal("conversation changed")
        self.state.current_conversation = None
        self.state.goal = goal
        self.clear_automatic_prompts()
        if self._save_state(persist_goal=False):
            self._render_goal_status()

    def _cmd_goal_doctor(self) -> None:
        goal = self.state.goal
        if goal is None:
            self.renderer.info("No Goal is attached to this conversation.")
            return
        try:
            report = self.goal_store.doctor(goal)
            held = goal_lock_is_held(self.goal_store.root, goal.goal_id or "")
            lock_metadata = read_goal_lock_metadata(
                self.goal_store.root, goal.goal_id or ""
            )
        except (OSError, sqlite3.Error, ValueError) as exc:
            self.renderer.warning(f"Goal doctor failed: {exc}")
            return

        lock_expected = goal.status == "active"
        lock_ok = held if lock_expected else not held
        if not lock_ok:
            report["ok"] = False
            report["safe_to_continue"] = False
            report["warnings"].append(
                "kernel ownership lock does not match durable Goal status"
            )

        verdict = "PASS" if report["ok"] else "FAIL"
        safe = "yes" if report["safe_to_continue"] else "no"
        self.renderer.info(
            f"Goal doctor · {verdict} · safe-to-continue {safe} · "
            f"events {report.get('event_count', 0)} · "
            f"latest {report.get('latest_event') or 'none'}"
        )
        checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
        for name in ("sqlite_integrity", "replay", "event_sequence", "journal_hashes"):
            check = checks.get(name)
            if not isinstance(check, dict):
                continue
            status = "PASS" if check.get("ok") else "FAIL"
            detail = (
                check.get("reason")
                or check.get("detail")
                or (
                    f"count={check.get('count')} last={check.get('last_seq')}"
                    if name == "event_sequence"
                    else ""
                )
            )
            suffix = f" · {detail}" if detail else ""
            self.renderer.info(f"{name} · {status}{suffix}")

        runtime = checks.get("runtime")
        if isinstance(runtime, dict):
            runtime_status = "FAIL" if runtime.get("future") else (
                "MIGRATE" if runtime.get("needs_migration") else "PASS"
            )
            self.renderer.info(
                "runtime · "
                f"{runtime_status} · "
                f"{runtime.get('runtime_version')}/{runtime.get('protocol_version')}"
            )

        bindings = checks.get("bindings")
        if isinstance(bindings, dict):
            self.renderer.info(
                f"bindings · {'PASS' if bindings.get('ok') else 'FAIL'} · "
                f"{bindings.get('count', 0)} conversation(s)"
            )

        acceptance = checks.get("acceptance")
        if isinstance(acceptance, dict):
            self.renderer.info(
                f"acceptance · "
                f"{acceptance.get('satisfied', 0)}/{acceptance.get('required', 0)}"
                " required satisfied"
            )

        operation = checks.get("open_operation")
        if isinstance(operation, dict) and operation.get("operation_id"):
            evidence = operation.get("evidence")
            unresolved = (
                evidence.get("unresolved_tool_calls", 0)
                if isinstance(evidence, dict)
                else 0
            )
            self.renderer.info(
                f"open operation · {'PASS' if operation.get('ok') else 'BLOCKED'} · "
                f"{operation.get('operation_id')} · unresolved={unresolved}"
            )

        owner = str(lock_metadata.get("runner_id") or "").strip()
        pid = lock_metadata.get("pid")
        owner_text = (
            f" · owner={owner[:12]} pid={pid}" if held and owner else ""
        )
        self.renderer.info(
            f"kernel lock · {'PASS' if lock_ok else 'FAIL'} · "
            f"{'held' if held else 'free'}{owner_text}"
        )
        for warning in report.get("warnings") or []:
            self.renderer.warning(f"Goal doctor · {warning}")

    def _cmd_goal_trace(self, argv: list[str]) -> None:
        goal = self.state.goal
        if goal is None:
            self.renderer.info("No Goal is attached to this conversation.")
            return
        if len(argv) > 1:
            self.renderer.warning("Usage: /goal trace [N]")
            return
        limit = 20
        if argv:
            try:
                limit = int(argv[0])
            except ValueError:
                self.renderer.warning("Usage: /goal trace [N]")
                return
            if limit < 1 or limit > 200:
                self.renderer.warning("Goal trace count must be between 1 and 200.")
                return
        try:
            trace = self.goal_store.trace(goal, limit=limit)
        except (OSError, sqlite3.Error) as exc:
            self.renderer.warning(f"Goal trace failed: {exc}")
            return
        self.renderer.info(
            f"Goal trace · {len(trace)} event{'s' if len(trace) != 1 else ''}"
        )
        for event in trace:
            details: list[str] = []
            operation_id = str(event.get("operation_id") or "").strip()
            if operation_id:
                details.append(f"op={operation_id}")
            conversation_ref = str(event.get("conversation_ref") or "").strip()
            if conversation_ref:
                details.append(f"chat={_short_ref(conversation_ref)}")
            tool_name = str(event.get("tool_name") or "").strip()
            if tool_name:
                details.append(f"tool={tool_name}")
            call_id = str(event.get("tool_call_id") or "").strip()
            if call_id:
                details.append(f"call={call_id}")
            criterion_id = str(event.get("criterion_id") or "").strip()
            if criterion_id:
                details.append(f"criterion={criterion_id}")
            reason = " ".join(str(event.get("reason") or "").split()).strip()
            if reason:
                details.append(f"reason={reason[:100]}")
            suffix = " · " + " · ".join(details) if details else ""
            self.renderer.info(
                f"#{event.get('seq')} · g{event.get('generation')} · "
                f"{event.get('type')}{suffix}"
            )

    def _cmd_goal_criteria(self, argv: list[str]) -> None:
        goal = self.state.goal
        if goal is None:
            self.renderer.info("No Goal is attached to this conversation.")
            return
        if not argv:
            if not goal.acceptance_criteria:
                self.renderer.info("Goal acceptance criteria · none configured")
                return
            passed = sum(1 for criterion in goal.acceptance_criteria if criterion.satisfied)
            self.renderer.info(
                f"Goal acceptance criteria · {passed}/{len(goal.acceptance_criteria)} satisfied"
            )
            for criterion in goal.acceptance_criteria:
                status = "PASS" if criterion.satisfied else "PENDING"
                source = (
                    f" · {criterion.evidence_source}"
                    if criterion.evidence_source
                    else ""
                )
                evidence = (
                    f" · {criterion.evidence_refs[0]}"
                    if criterion.evidence_refs
                    else ""
                )
                self.renderer.info(
                    f"{criterion.criterion_id} · {status}{source}{evidence} · "
                    f"{criterion.description}"
                )
            return

        if argv[0].lower() != "attest" or len(argv) < 3:
            self.renderer.warning(
                "Usage: /goal criteria [attest <criterion-id> <evidence-note>]"
            )
            return
        if goal.status == "active" and not self._owns_goal_run(goal):
            self.renderer.warning(
                "Goal is active in another live gptty process; attest from that process."
            )
            return
        criterion_id = argv[1].strip()
        note = " ".join(argv[2:]).strip()
        if not note:
            self.renderer.warning("Human attestation requires an evidence note.")
            return
        evidence_ref = "human:" + hashlib.sha256(
            f"{criterion_id}:{note}".encode("utf-8")
        ).hexdigest()[:16]
        try:
            updated = self.goal_store.record_acceptance_evidence(
                goal,
                criterion_id=criterion_id,
                evidence_ref=evidence_ref,
                source="human",
                details={"note": note},
            )
        except (KeyError, ValueError, GoalConflictError, OSError, sqlite3.Error) as exc:
            self.renderer.warning(f"Acceptance evidence rejected: {exc}")
            return
        self.state.goal = updated
        if not self._save_state(persist_goal=False):
            return
        self.renderer.info(
            f"Goal acceptance · {criterion_id} PASS · human attestation · {evidence_ref}"
        )

    def _cmd_goal(self, argv: list[str]) -> None:
        if self._conversation_mode == "temporary":
            self.renderer.warning(
                "Goal mode is only available for normal ChatGPT conversations."
            )
            return

        action = argv[0].strip().lower() if argv else ""
        if action == "list":
            if len(argv) > 2 or (len(argv) == 2 and argv[1].strip().lower() != "all"):
                self.renderer.warning("Usage: /goal list [all]")
                return
            self._render_goal_list(include_terminal=len(argv) == 2)
            return
        if action == "open":
            if len(argv) != 2:
                self.renderer.warning("Usage: /goal open <id-prefix>")
                return
            self._open_goal(argv[1])
            return
        if action == "criteria":
            self._cmd_goal_criteria(argv[1:])
            return
        if action == "doctor":
            if len(argv) != 1:
                self.renderer.warning("Usage: /goal doctor")
                return
            self._cmd_goal_doctor()
            return
        if action == "trace":
            self._cmd_goal_trace(argv[1:])
            return

        if action in {"pause", "resume", "clear", "status"} and len(argv) == 1:
            if action == "pause":
                if (
                    self.state.goal is not None
                    and self.state.goal.status == "active"
                    and not self._owns_goal_run(self.state.goal)
                ):
                    self.renderer.warning(
                        "Goal is active in another live gptty process; pause it from that process."
                    )
                    return
                if self._pause_active_goal("paused by user"):
                    self.renderer.info("Goal · paused")
                elif self.state.goal is None:
                    self.renderer.info("No Goal is attached to this conversation.")
                else:
                    self.renderer.info(f"Goal · {self.state.goal.status}")
                return
            if action == "resume":
                self._resume_goal()
                return
            if action == "clear":
                if self.state.goal is None:
                    self.renderer.info("No Goal is attached to this conversation.")
                    return
                goal = self.state.goal
                if goal.status == "active" and not self._owns_goal_run(goal):
                    self.renderer.warning(
                        "Goal is active in another live gptty process; it cannot be cleared here."
                    )
                    return
                if goal.status not in {"complete", "interrupted"}:
                    goal.status = "interrupted"
                    goal.reason = "cleared by user"
                    goal.runner_id = None
                    goal.runner_pid = 0
                    try:
                        self.goal_store.save(
                            goal,
                            event_type="goal_interrupted",
                            event_payload={"reason": goal.reason},
                        )
                    except GoalConflictError as exc:
                        authoritative = self.goal_store.load(goal.goal_id or "")
                        if authoritative is not None:
                            self.state.goal = authoritative
                        self.renderer.warning(
                            f"Goal state changed in another process; clear was not applied: {exc}"
                        )
                        return
                    except (OSError, sqlite3.Error) as exc:
                        self.renderer.warning(f"Failed to interrupt Goal before clear: {exc}")
                        return
                    if self.goal_store.last_projection_error is not None:
                        self.renderer.warning(
                            "Goal authoritative state committed, but portable projection "
                            f"could not be refreshed: {self.goal_store.last_projection_error}"
                        )
                    self._release_goal_run_lock()
                try:
                    self.goal_store.unbind_goal(goal)
                except (OSError, sqlite3.Error) as exc:
                    self.renderer.warning(f"Failed to clear Goal bindings: {exc}")
                    return
                self.state.goal = None
                self.clear_automatic_prompts()
                if self._save_state(persist_goal=False):
                    self.renderer.info("Goal · cleared from its conversations; history retained")
                return
            self._render_goal_status()
            return

        criteria_descriptions: list[str] = []
        objective_parts: list[str] = []
        index = 0
        while index < len(argv):
            token = argv[index]
            if token == "--accept":
                if index + 1 >= len(argv):
                    self.renderer.warning(
                        'Usage: /goal [--accept "criterion"]... <objective>'
                    )
                    return
                description = " ".join(argv[index + 1].split()).strip()
                if not description:
                    self.renderer.warning("Acceptance criterion cannot be empty.")
                    return
                criteria_descriptions.append(description)
                index += 2
                continue
            objective_parts.extend(argv[index:])
            break
        objective = " ".join(objective_parts).strip() or None
        existing = self.state.goal
        if (
            objective is None
            and self.state.current_conversation is None
            and (existing is None or existing.status == "complete")
        ):
            self.renderer.warning(
                "No conversation is attached. Use /goal <objective> to start a Goal in a new chat."
            )
            return
        if existing is not None and existing.status not in {"complete"}:
            if objective:
                self.renderer.warning(
                    "This conversation already has an unfinished Goal. Use /goal status, /goal resume, or /goal clear."
                )
                return
            if existing.status in {"paused", "blocked", "interrupted"}:
                self._resume_goal()
                return
            self._render_goal_status()
            return

        context_history = self._capture_goal_context_history(
            self.state.current_conversation
        )
        new_goal = GoalState(
            conversation_ref=self.state.current_conversation,
            conversations=(
                [self.state.current_conversation]
                if self.state.current_conversation
                else []
            ),
            context_seed=self._compact_goal_context_seed(context_history),
            acceptance_criteria=[
                GoalAcceptanceCriterion(
                    criterion_id=f"A{number}",
                    description=description,
                )
                for number, description in enumerate(criteria_descriptions, start=1)
            ],
            status="active",
            objective=objective,
            runner_id=self.runner_id,
            runner_pid=os.getpid(),
        )
        ensure_goal_id(new_goal)
        if not self._acquire_goal_run_lock(new_goal):
            self.state.goal = existing
            self.renderer.warning("Goal ownership lock could not be acquired.")
            return
        self.state.goal = new_goal
        if not self._save_state(
            event_type="goal_created",
            event_payload={
                "objective": objective,
                "conversation_ref": self.state.current_conversation,
                "context_snapshot": context_history,
                "acceptance_criteria": [
                    {
                        "criterion_id": criterion.criterion_id,
                        "description": criterion.description,
                        "required": criterion.required,
                    }
                    for criterion in new_goal.acceptance_criteria
                ],
            },
        ):
            # If the authoritative Goal transaction committed but local chat-state
            # persistence failed, keep the durable Goal visible instead of deleting
            # or rolling it back through a singleton pointer.
            authoritative = self.goal_store.load(new_goal.goal_id or "")
            self._release_goal_run_lock()
            self.state.goal = authoritative if authoritative is not None else existing
            return
        self.clear_automatic_prompts()
        self._automatic_prompts.append(activation_prompt(objective, goal=new_goal))
        self.renderer.info(f"Goal · active · {new_goal.goal_id[:8]} · starting")
        self.renderer.info(f"Goal state: {self.goal_store.goal_path(new_goal)}")

    def _cmd_exit(self, argv: list[str]) -> int:
        self._pause_active_goal("gptty exited")
        self._leave_temporary_mode()
        self.close()
        return 0

    def _cmd_new(self, argv: list[str]) -> None:
        self._pause_active_goal("conversation changed")
        self._leave_temporary_mode()
        previous_ref = self.state.current_conversation
        previous_goal = self.state.goal
        self.state.current_conversation = None
        self.state.goal = None
        if not self._save_state(persist_goal=False):
            self.state.current_conversation = previous_ref
            self.state.goal = previous_goal
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
        previous_ref = self.state.current_conversation
        previous_goal = self.state.goal
        self.state.current_conversation = None
        self.state.goal = None
        if not self._save_state(persist_goal=False):
            self.state.current_conversation = previous_ref
            self.state.goal = previous_goal
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
        previous_ref = self.state.current_conversation
        previous_goal = self.state.goal
        self.state.current_conversation = None
        self.state.goal = None
        if not self._save_state(persist_goal=False):
            self.state.current_conversation = previous_ref
            self.state.goal = previous_goal
            return
        self.clear_pending_media()
        self.renderer.clear_context()
        self.renderer.header(model=self.state.model or "latest frontier · High")
        self.renderer.info(
            "Detached locally. The ChatGPT conversation was not changed."
        )

    def _cmd_stop(self, argv: list[str]) -> None:
        if self._reject_remote_goal_mutation("stopping the active response"):
            return
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
            # Switching local UI context is always allowed. We only pause a Goal
            # owned by this process; a Goal running in another process continues.
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
        try:
            goal_by_conversation = self.goal_store.conversation_goal_map()
        except (OSError, sqlite3.Error):
            goal_by_conversation = {}
        options: list[tuple[str, str]] = []
        for item in conversations:
            conversation_id = _catalog_conversation_id(item)
            if conversation_id is None:
                continue
            title = _field_text(item, "title")
            if title:
                self._conversation_titles[conversation_id] = title
            label = _conversation_label(
                item,
                current=conversation_id == current_ref,
            )
            goal = goal_by_conversation.get(conversation_id)
            if goal is not None and goal.status != "complete":
                label += f" · Goal {goal.status} {(goal.goal_id or '?')[:8]}"
            options.append((conversation_id, label))
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
        # Model selection is local UI/session state. It must not be fenced by an
        # unrelated Goal owner in another process.
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
        compatibility = self.goal_store.runtime_compatibility(goal)
        if compatibility["future"]:
            self.renderer.warning(
                "Goal requires newer runtime/protocol semantics "
                f"{goal.runtime_version}/{goal.protocol_version}; this gptty supports "
                f"{compatibility['current_runtime_version']}/"
                f"{compatibility['current_protocol_version']}."
            )
            return
        if goal.status == "active":
            if not self._owns_goal_run(goal):
                self.renderer.warning(
                    "Goal is active in another live gptty process; this session will not take ownership."
                )
            self._render_goal_status()
            return
        if compatibility["needs_migration"]:
            try:
                goal = self.goal_store.migrate_runtime(goal)
            except GoalConflictError:
                refreshed = self.goal_store.load(goal.goal_id or "")
                if refreshed is None:
                    self.renderer.warning("Goal runtime migration conflicted and reload failed.")
                    return
                goal = refreshed
                compatibility = self.goal_store.runtime_compatibility(goal)
                if compatibility["needs_migration"] or compatibility["future"]:
                    self.renderer.warning(
                        "Goal runtime migration conflicted with another process; retry /goal resume."
                    )
                    return
            except (GoalCompatibilityError, OSError, sqlite3.Error) as exc:
                self.renderer.warning(f"Goal runtime migration failed: {exc}")
                return
            self.state.goal = goal
            self.renderer.info(
                f"Goal · migrated to runtime/protocol "
                f"{goal.runtime_version}/{goal.protocol_version}"
            )

        if (
            goal.conversation_ref
            and goal.conversation_ref != self.state.current_conversation
        ):
            self.renderer.warning(
                f"Goal belongs to {_short_ref(goal.conversation_ref)}. Resume that conversation before /goal resume."
            )
            return
        if goal.conversation_ref is None and goal.active_operation_id:
            try:
                committed_ref = self.goal_store.operation_committed_conversation(
                    goal, goal.active_operation_id
                )
            except (OSError, sqlite3.Error) as exc:
                self.renderer.warning(f"Goal journal recovery failed: {exc}")
                return
            # An open durable operation may only attach to machine-observed write
            # identity. Ignore an arbitrary/stale current chat from gptty_state.json.
            self.state.current_conversation = committed_ref
            if committed_ref:
                goal.conversation_ref = committed_ref
                if committed_ref not in goal.conversations:
                    goal.conversations.append(committed_ref)
        bootstrap_without_chat = (
            goal.conversation_ref is None and self.state.current_conversation is None
        )
        if (
            goal.conversation_ref is None
            and self.state.current_conversation
            and not goal.active_operation_id
        ):
            goal.conversation_ref = self.state.current_conversation
            if self.state.current_conversation not in goal.conversations:
                goal.conversations.append(self.state.current_conversation)
        previous_reason = goal.reason
        if not self._acquire_goal_run_lock(goal):
            authoritative = self.goal_store.load(goal.goal_id or "")
            if authoritative is not None:
                metadata = read_goal_lock_metadata(
                    self.goal_store.root, goal.goal_id or ""
                )
                lock_runner = str(metadata.get("runner_id") or "").strip()
                lock_pid = metadata.get("pid")
                if lock_runner:
                    # The kernel lock can become visible a few microseconds before
                    # its owner commits active metadata. Reflect that ownership
                    # locally without mutating authoritative state.
                    authoritative.status = "active"
                    authoritative.runner_id = lock_runner
                    authoritative.runner_pid = (
                        int(lock_pid) if isinstance(lock_pid, int) else 0
                    )
                self.state.goal = authoritative
            self.renderer.warning(
                "Goal is active in another live gptty process; this session will not take ownership."
            )
            return
        goal.status = "active"
        goal.reason = None
        if not self._save_state(
            event_type="goal_resumed",
            event_payload={"conversation_ref": goal.conversation_ref, "ambiguous_operation": goal.active_operation_id},
        ):
            self._release_goal_run_lock()
            return
        self.clear_automatic_prompts()
        if bootstrap_without_chat and goal.conversations:
            self._automatic_prompts.append(
                rollover_prompt(
                    goal,
                    reason=previous_reason or "resuming durable Goal state",
                    journal_context=self.goal_store.recovery_context(goal),
                )
            )
            self._goal_bootstrap_pending = True
        else:
            if goal.active_operation_id and not bootstrap_without_chat:
                self._automatic_prompts.append(
                    abnormal_recovery_prompt(
                        f"durable operation {goal.active_operation_id} has no confirmed terminal state after restart/pause",
                        goal=goal,
                        journal_context=self.goal_store.recovery_context(goal),
                    )
                )
            else:
                self._automatic_prompts.append(
                    activation_prompt(goal.objective, goal=goal)
                    if bootstrap_without_chat
                    else continuation_prompt(goal=goal)
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
        if goal.status == "active" and not self._owns_goal_run(goal):
            details.append(f"owner pid {goal.runner_pid or '?'}")
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
        goal.runner_id = None
        goal.runner_pid = 0
        self.clear_automatic_prompts()
        if not self._save_state(
            event_type="goal_paused",
            event_payload={"reason": goal.reason, "ambiguous_operation": goal.active_operation_id},
        ):
            return False
        self._release_goal_run_lock()
        self.renderer.warning(
            "Goal · paused · service backoff required · use /goal resume later"
        )
        return True

    def _pause_active_goal(self, reason: str) -> bool:
        goal = self.state.goal
        if goal is None or goal.status != "active" or not self._owns_goal_run(goal):
            return False
        goal.status = "paused"
        goal.reason = reason
        goal.runner_id = None
        goal.runner_pid = 0
        self.clear_automatic_prompts()
        self._save_state(
            event_type="goal_paused",
            event_payload={"reason": reason, "ambiguous_operation": goal.active_operation_id},
        )
        self._release_goal_run_lock()
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
        goal.runner_id = None
        goal.runner_pid = 0
        self.clear_automatic_prompts()
        self._save_state(
            event_type="goal_interrupted",
            event_payload={"reason": reason, "ambiguous_operation": goal.active_operation_id},
        )
        self._release_goal_run_lock()
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
            continuation_prompt(
                protocol_recovery=protocol_recovery, goal=goal
            )
        )
        if protocol_recovery:
            self.renderer.warning(
                f"Goal · continuing · missing status {goal.protocol_failures}/{MAX_PROTOCOL_FAILURES}"
            )
        else:
            self.renderer.info(f"Goal · continuing · next turn {goal.turn_count + 1}")

    def _save_state(
        self,
        *,
        event_type: str = "state_saved",
        event_payload: dict[str, Any] | None = None,
        persist_goal: bool = True,
    ) -> bool:
        try:
            if persist_goal and self.state.goal is not None:
                goal_id = self.state.goal.goal_id or ""
                if (
                    self.state.goal.status == "active"
                    and not self._owns_goal_run(self.state.goal)
                ):
                    authoritative = self.goal_store.load(goal_id)
                    if authoritative is not None:
                        self.state.goal = authoritative
                else:
                    self.goal_store.save(
                        self.state.goal,
                        event_type=event_type,
                        event_payload=event_payload,
                    )
                    if self.goal_store.last_projection_error is not None:
                        self.renderer.warning(
                            "Goal authoritative state committed, but portable projection "
                            f"could not be refreshed: {self.goal_store.last_projection_error}"
                        )
            save_chat_state(self.state_path, self.state)
        except GoalConflictError as exc:
            self._release_goal_run_lock()
            goal_id = self.state.goal.goal_id if self.state.goal is not None else None
            try:
                current = self.goal_store.load(goal_id or "") if goal_id else None
            except (OSError, sqlite3.Error) as reload_exc:
                self.clear_automatic_prompts()
                self.renderer.warning(
                    f"Goal state conflict and authoritative reload failed: {reload_exc}"
                )
                return False
            if current is not None:
                self.state.goal = current
            self.clear_automatic_prompts()
            self.renderer.warning(
                f"Goal state changed in another process; reloaded that Goal: {exc}"
            )
            return False
        except (OSError, sqlite3.Error, StateError) as exc:
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
