from __future__ import annotations

import json
from pathlib import Path

from gptty.goal_store import GoalStore
from gptty.state import GoalCheckpoint, GoalState


def test_goal_store_writes_portable_json_checkpoint_and_pointer(tmp_path) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(
        goal_id="goal-123",
        conversation_ref="conv-2",
        conversations=["conv-1", "conv-2"],
        context_seed=["user: original request", "assistant: agreed approach"],
        status="active",
        objective="Finish the agreed task",
        turn_count=4,
        rollover_count=1,
        checkpoint=GoalCheckpoint(
            summary="Implementation is half complete.",
            completed=["commit A verified"],
            decisions=["keep API backwards compatible"],
            pending=["finish integration test"],
            next_step="run the live acceptance",
            updated_turn=4,
        ),
    )

    path = store.save(goal)

    assert path == tmp_path / "goals" / "goal-123" / "goal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == 2
    assert payload["goal"]["checkpoint"]["decisions"] == [
        "keep API backwards compatible"
    ]
    assert payload["goal"]["context_seed"] == [
        "user: original request",
        "assistant: agreed approach",
    ]
    checkpoint = store.checkpoint_path(goal).read_text(encoding="utf-8")
    assert "## Recovery context seed" in checkpoint
    assert "commit A verified" in checkpoint
    assert "https://chatgpt.com/c/conv-1" in checkpoint
    assert store.current_path().read_text(encoding="utf-8").strip() == "goal-123"
    assert store.load_current() == goal


def test_goal_store_clear_current_preserves_backupable_goal_directory(tmp_path) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(goal_id="goal-backup", objective="Portable goal")
    store.save(goal)

    store.clear_current()

    assert not store.current_path().exists()
    assert store.goal_path(goal).exists()
    assert store.checkpoint_path(goal).exists()
    assert store.load("goal-backup") == goal


def test_goal_store_records_append_only_events_and_rejects_stale_writer(tmp_path) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(goal_id="goal-journal", status="active", objective="Journal it")
    store.save(goal, event_type="goal_created", event_payload={"source": "test"})
    stale = store.load("goal-journal")
    assert stale is not None

    goal.turn_count = 1
    store.save(goal, event_type="turn_terminal", event_payload={"status": "continue"})

    events = store.events(goal)
    assert [event["type"] for event in events] == ["goal_created", "turn_terminal"]
    assert events[0]["payload"]["source"] == "test"
    assert events[1]["payload"]["revision"] == goal.revision
    assert goal.revision == 2

    stale.turn_count = 99
    import pytest
    from gptty.goal_store import GoalConflictError
    with pytest.raises(GoalConflictError):
        store.save(stale, event_type="stale_write")
    assert store.load("goal-journal").turn_count == 1


def test_goal_store_sqlite_is_authoritative_when_portable_projection_is_stale(tmp_path) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(goal_id="goal-authority", status="active", turn_count=1)
    store.save(goal, event_type="goal_created")
    portable = store.goal_path(goal)
    payload = json.loads(portable.read_text(encoding="utf-8"))

    goal.turn_count = 2
    store.save(goal, event_type="turn_terminal")
    portable.write_text(json.dumps(payload), encoding="utf-8")

    recovered = store.load_current()
    assert recovered is not None
    assert recovered.turn_count == 2
    assert recovered.revision == 2


def test_goal_store_concurrent_stale_writers_allow_exactly_one_commit(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from gptty.goal_store import GoalConflictError

    state_path = tmp_path / "gptty_state.json"
    store = GoalStore(state_path)
    original = GoalState(goal_id="goal-race", status="active")
    store.save(original, event_type="goal_created")
    left = store.load("goal-race")
    right = store.load("goal-race")
    assert left is not None and right is not None
    left.turn_count = 10
    right.turn_count = 20

    def commit(goal: GoalState) -> tuple[str, int]:
        local = GoalStore(state_path)
        try:
            local.save(goal, event_type="racing_writer")
            return ("ok", goal.turn_count)
        except GoalConflictError:
            return ("conflict", goal.turn_count)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(commit, (left, right)))

    assert sorted(status for status, _ in results) == ["conflict", "ok"]
    winner = next(turns for status, turns in results if status == "ok")
    recovered = store.load("goal-race")
    assert recovered is not None
    assert recovered.turn_count == winner
    assert recovered.revision == 2


def test_goal_store_survives_process_death_after_authoritative_commit(tmp_path) -> None:
    import os
    import subprocess
    import sys

    state_path = tmp_path / "gptty_state.json"
    store = GoalStore(state_path)
    goal = GoalState(goal_id="goal-crash", status="active", turn_count=1)
    store.save(goal, event_type="goal_created")

    script = r'''
import os
import sys
from gptty.goal_store import GoalStore

class CrashAfterCommitStore(GoalStore):
    def _write_portable_projection(self, goal):
        os._exit(86)

state_path = sys.argv[1]
store = CrashAfterCommitStore(state_path)
goal = store.load("goal-crash")
goal.turn_count = 2
store.save(goal, event_type="turn_terminal", event_payload={"body": "committed"})
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run(
        [sys.executable, "-c", script, str(state_path)], env=env, check=False
    )
    assert child.returncode == 86

    recovered = store.load_current()
    assert recovered is not None
    assert recovered.turn_count == 2
    assert recovered.revision == 2
    assert store.events(recovered)[-1]["type"] == "turn_terminal"


def test_goal_store_discards_uncommitted_sqlite_write_after_process_death(tmp_path) -> None:
    import subprocess
    import sys

    state_path = tmp_path / "gptty_state.json"
    store = GoalStore(state_path)
    goal = GoalState(goal_id="goal-uncommitted", status="active", turn_count=1)
    store.save(goal, event_type="goal_created")

    script = r'''
import json
from pathlib import Path
import os
import sqlite3
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
db_path = state_path.parent / "goals" / "goal-state.sqlite3"
db = sqlite3.connect(db_path)
db.execute("PRAGMA journal_mode=WAL")
db.execute("BEGIN IMMEDIATE")
row = db.execute("SELECT state_json FROM goals WHERE goal_id='goal-uncommitted'").fetchone()
state = json.loads(row[0])
state["turn_count"] = 999
db.execute(
    "UPDATE goals SET revision=revision+1, state_json=? WHERE goal_id='goal-uncommitted'",
    (json.dumps(state),),
)
os._exit(87)
'''
    child = subprocess.run([sys.executable, "-c", script, str(state_path)], check=False)
    assert child.returncode == 87

    recovered = store.load("goal-uncommitted")
    assert recovered is not None
    assert recovered.turn_count == 1
    assert recovered.revision == 1


def test_goal_store_observed_events_are_idempotent_and_available_for_recovery(tmp_path) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(goal_id="goal-observed", status="active")
    store.save(goal, event_type="goal_created")
    payload = {
        "operation_id": "goal-observed:g1:t1",
        "tool_name": "bash",
        "label": "write file",
        "text": "echo done > marker.txt",
    }

    assert store.record_observed_event(
        goal, "tool_call_observed", payload, event_key="event-1"
    ) is True
    assert store.record_observed_event(
        goal, "tool_call_observed", payload, event_key="event-1"
    ) is False
    assert store.record_observed_event(
        goal,
        "tool_result_observed",
        {**payload, "text": "exit 0"},
        event_key="event-2",
    ) is True

    events = store.events(goal)
    assert [event["type"] for event in events] == [
        "goal_created",
        "tool_call_observed",
        "tool_result_observed",
    ]
    context = "\n".join(store.recovery_context(goal))
    assert "tool_call_observed" in context
    assert "echo done > marker.txt" in context
    assert "tool_result_observed" in context


def test_goal_store_projection_failure_does_not_undo_committed_authoritative_state(tmp_path, monkeypatch) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(goal_id="goal-projection", status="active", turn_count=1)
    store.save(goal, event_type="goal_created")

    def fail_projection(_goal):
        raise OSError("portable disk unavailable")

    monkeypatch.setattr(store, "_write_portable_projection", fail_projection)
    goal.turn_count = 2
    store.save(goal, event_type="turn_terminal")

    assert store.last_projection_error is not None
    recovered = store.load_current()
    assert recovered is not None
    assert recovered.turn_count == 2
    assert recovered.revision == 2
    assert store.events(goal)[-1]["type"] == "turn_terminal"


def test_recovery_context_keeps_old_user_steering_after_many_technical_events(tmp_path) -> None:
    store = GoalStore(tmp_path / "gptty_state.json")
    goal = GoalState(goal_id="goal-steering-retention", status="active")
    store.save(
        goal,
        event_type="goal_created",
        event_payload={"context_snapshot": ["user: original objective"]},
    )
    goal.turn_count = 1
    store.save(
        goal,
        event_type="user_steering",
        event_payload={"text": "NEVER change the public API shape."},
    )
    for index in range(45):
        store.record_observed_event(
            goal,
            "tool_result_observed",
            {
                "operation_id": "op-many",
                "label": f"technical-{index}",
                "text": "ok",
            },
            event_key=f"tech-{index}",
        )

    context = "\n".join(store.recovery_context(goal, max_events=8))
    assert "NEVER change the public API shape." in context
    assert "technical-44" in context


def test_goal_store_recovers_committed_state_from_wal_after_hard_process_exit(tmp_path) -> None:
    """A committed WAL record must survive even if the writer dies before checkpoint/projection."""
    import subprocess
    import sys

    state_path = tmp_path / "gptty_state.json"
    store = GoalStore(state_path)
    goal = GoalState(goal_id="goal-wal-recovery", status="active", turn_count=1)
    store.save(goal, event_type="goal_created")

    script = r'''
import json
import os
import sqlite3
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
db_path = state_path.parent / "goals" / "goal-state.sqlite3"
db = sqlite3.connect(db_path)
db.execute("PRAGMA journal_mode=WAL")
db.execute("PRAGMA synchronous=FULL")
db.execute("PRAGMA wal_autocheckpoint=0")
db.execute("BEGIN IMMEDIATE")
row = db.execute(
    "SELECT revision, generation, state_json FROM goals WHERE goal_id=?",
    ("goal-wal-recovery",),
).fetchone()
state = json.loads(row[2])
state["turn_count"] = 2
state["revision"] = row[0] + 1
payload = json.dumps(state, ensure_ascii=False, sort_keys=True)
db.execute(
    "UPDATE goals SET revision=?, state_json=?, updated_at=datetime('now') WHERE goal_id=?",
    (row[0] + 1, payload, "goal-wal-recovery"),
)
seq = db.execute(
    "SELECT COALESCE(MAX(seq),0)+1 FROM goal_events WHERE goal_id=?",
    ("goal-wal-recovery",),
).fetchone()[0]
db.execute(
    "INSERT INTO goal_events(goal_id,seq,generation,event_type,payload_json,created_at) VALUES(?,?,?,?,?,datetime('now'))",
    ("goal-wal-recovery", seq, row[1], "turn_terminal", json.dumps({"body":"wal committed"})),
)
db.commit()
os._exit(88)
'''
    child = subprocess.run([sys.executable, "-c", script, str(state_path)], check=False)
    assert child.returncode == 88

    recovered = store.load_current()
    assert recovered is not None
    assert recovered.turn_count == 2
    assert recovered.revision == 2
    with store._connect() as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert store.events(recovered)[-1]["payload"]["body"] == "wal committed"


def test_goal_store_machine_evidence_survives_crash_before_goal_state_advances(tmp_path) -> None:
    """Machine evidence is independently durable from the model checkpoint/state transition."""
    import os
    import subprocess
    import sys

    state_path = tmp_path / "gptty_state.json"
    store = GoalStore(state_path)
    goal = GoalState(
        goal_id="goal-evidence-crash",
        status="active",
        active_operation_id="goal-evidence-crash:g1:t1",
        active_operation_turn=1,
    )
    store.save(goal, event_type="operation_started")

    script = r'''
import os
import sys
from gptty.goal_store import GoalStore

store = GoalStore(sys.argv[1])
goal = store.load("goal-evidence-crash")
store.record_observed_event(
    goal,
    "tool_call_observed",
    {"operation_id": goal.active_operation_id, "label": "external write", "text": "write once"},
    event_key="crash-tool-call",
)
os._exit(89)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run(
        [sys.executable, "-c", script, str(state_path)], env=env, check=False
    )
    assert child.returncode == 89

    recovered = store.load_current()
    assert recovered is not None
    assert recovered.active_operation_id == "goal-evidence-crash:g1:t1"
    evidence = store.operation_evidence(recovered, recovered.active_operation_id)
    assert evidence == {"tool_calls": 1, "tool_results": 0, "unresolved_tool_calls": 1}
    context = "\n".join(store.recovery_context(recovered))
    assert "external write" in context
    assert "write once" in context
    with store._connect() as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_operation_evidence_does_not_let_unrelated_tool_result_resolve_write(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    operation_id = "goal-pairing:g1:t1"
    goal = GoalState(
        goal_id="goal-pairing",
        status="active",
        active_operation_id=operation_id,
        active_operation_turn=1,
    )
    store.save(goal, event_type="operation_started")
    store.record_observed_event(
        goal,
        "tool_call_observed",
        {"operation_id": operation_id, "tool_name": "write_api", "text": "create record"},
        event_key="call-write",
    )
    store.record_observed_event(
        goal,
        "tool_result_observed",
        {"operation_id": operation_id, "tool_name": "read_api", "text": "ok"},
        event_key="result-read",
    )

    assert store.operation_evidence(goal, operation_id) == {
        "tool_calls": 1,
        "tool_results": 1,
        "unresolved_tool_calls": 1,
    }

    store.record_observed_event(
        goal,
        "tool_result_observed",
        {"operation_id": operation_id, "tool_name": "write_api", "text": "created id=7"},
        event_key="result-write",
    )
    assert store.operation_evidence(goal, operation_id)["unresolved_tool_calls"] == 0


def test_authoritative_clear_cannot_be_undone_by_stale_portable_current_pointer(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    goal = GoalState(goal_id="goal-cleared", status="complete")
    store.save(goal, event_type="goal_created")
    stale_pointer = store.current_path().read_text(encoding="utf-8")

    # Simulate the crash window: authoritative transaction committed, but the old
    # human-readable pointer was never removed.
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM current_goal WHERE slot=1")
        db.commit()
    store.current_path().write_text(stale_pointer, encoding="utf-8")

    assert store.load_current() is None
    assert store.load("goal-cleared") is not None  # retained backup/history remains readable


def test_observed_event_refreshes_portable_journal_without_becoming_authoritative(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    goal = GoalState(goal_id="goal-portable-events", status="active")
    store.save(goal, event_type="goal_created")
    portable = store.goal_dir(goal) / "events.jsonl"
    assert "tool_call_observed" not in portable.read_text(encoding="utf-8")

    store.record_observed_event(
        goal,
        "tool_call_observed",
        {"operation_id": "op-1", "tool_name": "write_api", "text": "write once"},
        event_key="portable-event-1",
    )

    assert "tool_call_observed" in portable.read_text(encoding="utf-8")
    assert store.load_current() is not None


def test_goal_store_recovers_unique_committed_conversation_for_open_operation(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    operation_id = "goal-route:g2:t4"
    goal = GoalState(
        goal_id="goal-route",
        generation=2,
        status="active",
        active_operation_id=operation_id,
        active_operation_turn=4,
    )
    store.save(goal, event_type="operation_resumed")
    store.record_observed_event(
        goal,
        "conversation_write_committed",
        {"operation_id": operation_id, "conversation_ref": "conv-fresh-12345678"},
        event_key="route-1",
    )

    assert (
        store.operation_committed_conversation(goal, operation_id)
        == "conv-fresh-12345678"
    )
    assert "conv-fresh-12345678" in "\n".join(store.recovery_context(goal))


def test_conflicting_committed_conversation_identities_fail_closed(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    operation_id = "goal-route-conflict:g2:t4"
    goal = GoalState(
        goal_id="goal-route-conflict",
        generation=2,
        status="active",
        active_operation_id=operation_id,
        active_operation_turn=4,
    )
    store.save(goal, event_type="operation_resumed")
    for index, ref in enumerate(("conv-a-12345678", "conv-b-12345678"), start=1):
        store.record_observed_event(
            goal,
            "conversation_write_committed",
            {"operation_id": operation_id, "conversation_ref": ref},
            event_key=f"route-{index}",
        )

    assert store.operation_committed_conversation(goal, operation_id) is None
