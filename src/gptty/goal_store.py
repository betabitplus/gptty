from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .state import GoalState, goal_state_from_dict


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_goal_id(goal: GoalState) -> str:
    if goal.goal_id:
        return goal.goal_id
    goal.goal_id = uuid.uuid4().hex
    return goal.goal_id


class GoalStore:
    """Durable, portable on-disk representation of a /goal run."""

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.root = self.state_path.parent / "goals"

    def goal_dir(self, goal: GoalState | str) -> Path:
        goal_id = goal if isinstance(goal, str) else ensure_goal_id(goal)
        return self.root / goal_id

    def goal_path(self, goal: GoalState | str) -> Path:
        return self.goal_dir(goal) / "goal.json"

    def checkpoint_path(self, goal: GoalState | str) -> Path:
        return self.goal_dir(goal) / "checkpoint.md"

    def current_path(self) -> Path:
        return self.root / "current"

    def clear_current(self) -> None:
        try:
            self.current_path().unlink()
        except FileNotFoundError:
            pass

    def save(self, goal: GoalState) -> Path:
        goal_id = ensure_goal_id(goal)
        directory = self.goal_dir(goal_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "goal_id": goal_id,
            "updated_at": _now_iso(),
            "goal": asdict(goal),
            "files": {
                "checkpoint": "checkpoint.md",
            },
        }
        self._write_json_atomic(directory / "goal.json", payload)
        self._write_text_atomic(
            directory / "checkpoint.md", self._checkpoint_markdown(goal)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(self.current_path(), f"{goal_id}\n")
        return directory / "goal.json"

    def load_current(self) -> GoalState | None:
        try:
            goal_id = self.current_path().read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return self.load(goal_id) if goal_id else None

    def load(self, goal_id: str) -> GoalState | None:
        path = self.goal_path(goal_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        payload = raw.get("goal") if isinstance(raw, dict) else None
        goal = goal_state_from_dict(payload)
        if goal is not None and goal.goal_id is None:
            goal.goal_id = goal_id
        return goal

    @staticmethod
    def _checkpoint_markdown(goal: GoalState) -> str:
        checkpoint = goal.checkpoint
        lines = [
            f"# Goal {goal.goal_id or ''}".rstrip(),
            "",
            f"- Status: `{goal.status}`",
            f"- Turns: {goal.turn_count}",
            f"- Rollovers: {goal.rollover_count}",
            f"- Active conversation: `{goal.conversation_ref or 'none'}`",
            "",
            "## Objective",
            "",
            goal.objective or "_Inherited from the originating conversation._",
            "",
            "## Checkpoint",
            "",
            checkpoint.summary or "_No checkpoint captured yet._",
            "",
        ]
        for heading, values in (
            ("Completed", checkpoint.completed),
            ("Decisions", checkpoint.decisions),
            ("Pending", checkpoint.pending),
        ):
            lines.extend([f"### {heading}", ""])
            if values:
                lines.extend(f"- {value}" for value in values)
            else:
                lines.append("- _none recorded_")
            lines.append("")
        lines.extend(
            [
                "### Next step",
                "",
                checkpoint.next_step or "_not recorded_",
                "",
                "## Recovery context seed",
                "",
            ]
        )
        if goal.context_seed:
            lines.extend(f"- {item}" for item in goal.context_seed)
        else:
            lines.append("- _none captured_")
        lines.extend(["", "## Conversation chain", ""])
        if goal.conversations:
            lines.extend(
                f"- https://chatgpt.com/c/{conversation_id}"
                for conversation_id in goal.conversations
            )
        else:
            lines.append("- _none yet_")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        GoalStore._write_text_atomic(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
