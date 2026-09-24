from __future__ import annotations

import json
from pathlib import Path

from gptty.goal_store import GoalStore
from gptty.state import GoalCheckpoint, GoalState


def test_goal_store_writes_portable_json_checkpoint_and_multi_goal_index(tmp_path) -> None:
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
    assert payload["schema"] == 3
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
    assert not store.current_path().exists()
    index = json.loads(store.index_path().read_text(encoding="utf-8"))
    assert index["schema"] == 3
    assert index["goals"][0]["goal_id"] == "goal-123"
    assert store.load_for_conversation("conv-2") == goal
    assert set(store.bindings_for_goal(goal)) == {"conv-1", "conv-2"}


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

    recovered = store.load("goal-authority")
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

    recovered = store.load("goal-crash")
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
    recovered = store.load("goal-projection")
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

    recovered = store.load("goal-wal-recovery")
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

    recovered = store.load("goal-evidence-crash")
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


def test_legacy_current_pointer_never_overrides_conversation_routing(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    old = GoalState(
        goal_id="goal-old-pointer",
        conversation_ref="conv-old",
        conversations=["conv-old"],
        status="paused",
    )
    new = GoalState(
        goal_id="goal-routed",
        conversation_ref="conv-new",
        conversations=["conv-new"],
        status="active",
    )
    store.save(old, event_type="goal_created")
    store.save(new, event_type="goal_created")
    store.current_path().write_text("goal-old-pointer\n", encoding="utf-8")

    assert store.load_for_conversation("conv-new").goal_id == "goal-routed"
    assert store.load_for_conversation("conv-old").goal_id == "goal-old-pointer"
    assert {goal.goal_id for goal in store.list_goals()} == {
        "goal-old-pointer",
        "goal-routed",
    }


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
    assert store.load("goal-portable-events") is not None


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


def test_multiple_goals_route_independently_by_conversation(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    left = GoalState(
        goal_id="goal-left",
        conversation_ref="conv-left",
        conversations=["conv-left"],
        status="active",
        objective="left objective",
        runner_pid=111,
    )
    right = GoalState(
        goal_id="goal-right",
        conversation_ref="conv-right",
        conversations=["conv-right"],
        status="active",
        objective="right objective",
        runner_pid=222,
    )

    store.save(left, event_type="goal_created")
    store.save(right, event_type="goal_created")

    assert store.load_for_conversation("conv-left").goal_id == "goal-left"
    assert store.load_for_conversation("conv-right").goal_id == "goal-right"
    assert {goal.goal_id for goal in store.list_goals(statuses={"active"})} == {
        "goal-left",
        "goal-right",
    }
    assert store.bindings_for_goal(left) == ["conv-left"]
    assert store.bindings_for_goal(right) == ["conv-right"]


def test_unfinished_goal_binding_cannot_be_stolen_by_another_goal(tmp_path) -> None:
    import pytest
    from gptty.goal_store import GoalConflictError

    store = GoalStore(tmp_path / "state.json")
    first = GoalState(
        goal_id="goal-first",
        conversation_ref="conv-shared",
        conversations=["conv-shared"],
        status="paused",
    )
    second = GoalState(
        goal_id="goal-second",
        conversation_ref="conv-shared",
        conversations=["conv-shared"],
        status="active",
    )
    store.save(first, event_type="goal_created")

    with pytest.raises(GoalConflictError):
        store.save(second, event_type="goal_created")

    assert store.load_for_conversation("conv-shared").goal_id == "goal-first"
    assert store.load("goal-second") is None


def test_terminal_goal_conversation_can_be_rebound_to_new_goal(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    old = GoalState(
        goal_id="goal-old",
        conversation_ref="conv-reuse",
        conversations=["conv-reuse"],
        status="complete",
    )
    new = GoalState(
        goal_id="goal-new",
        conversation_ref="conv-reuse",
        conversations=["conv-reuse"],
        status="active",
    )
    store.save(old, event_type="goal_created")
    store.save(new, event_type="goal_created")

    assert store.load_for_conversation("conv-reuse").goal_id == "goal-new"
    assert any(event["type"] == "conversation_released" for event in store.events(old))


def test_committed_fresh_conversation_binds_goal_before_terminal_state_save(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    operation_id = "goal-fresh:g2:t4"
    goal = GoalState(
        goal_id="goal-fresh",
        generation=2,
        status="active",
        active_operation_id=operation_id,
        active_operation_turn=4,
    )
    store.save(goal, event_type="operation_resumed")

    store.record_observed_event(
        goal,
        "conversation_write_committed",
        {
            "operation_id": operation_id,
            "conversation_ref": "conv-fresh",
            "submission_id": "submit-1",
        },
        event_key="fresh-route",
    )

    routed = store.load_for_conversation("conv-fresh")
    assert routed is not None
    assert routed.goal_id == "goal-fresh"
    # Routing is durable immediately, while the Goal state checkpoint deliberately
    # remains unchanged until startup/terminal reconciliation advances it.
    assert routed.conversation_ref is None
    authoritative = store.load("goal-fresh")
    assert authoritative is not None
    assert authoritative.conversation_ref is None
    assert store.goal_id_for_conversation("conv-fresh") == "goal-fresh"


def test_multi_goal_index_is_not_a_singleton_pointer(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    for idx in range(2):
        store.save(
            GoalState(
                goal_id=f"goal-{idx}",
                conversation_ref=f"conv-{idx}",
                conversations=[f"conv-{idx}"],
                status="paused",
                objective=f"objective {idx}",
            ),
            event_type="goal_created",
        )
    index = json.loads(store.index_path().read_text(encoding="utf-8"))
    assert {item["goal_id"] for item in index["goals"]} == {"goal-0", "goal-1"}
    assert not store.current_path().exists()



def test_v2_singleton_store_migrates_all_conversation_bindings_to_v3(tmp_path) -> None:
    store = GoalStore(tmp_path / "state.json")
    legacy = GoalState(
        goal_id="goal-v2-migrate",
        generation=2,
        conversation_ref="conv-new",
        conversations=["conv-old", "conv-new"],
        status="paused",
        objective="migrate safely",
    )
    store.save(legacy, event_type="goal_created")

    # Simulate an on-disk v2 database: Goal JSON/history exist, singleton current
    # exists, but there is no v3 conversation routing and user_version predates v3.
    with store._connect() as db:
        db.execute("DELETE FROM goal_conversations")
        db.execute(
            "INSERT OR REPLACE INTO current_goal(slot, goal_id) VALUES (1, ?)",
            (legacy.goal_id,),
        )
        db.execute("PRAGMA user_version=2")
        db.commit()

    migrated = GoalStore(tmp_path / "state.json")
    assert migrated.load_for_conversation("conv-old").goal_id == "goal-v2-migrate"
    assert migrated.load_for_conversation("conv-new").goal_id == "goal-v2-migrate"
    assert set(migrated.bindings_for_goal("goal-v2-migrate")) == {
        "conv-old",
        "conv-new",
    }
    with migrated._connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
