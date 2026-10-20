import sqlite3

import pytest

from agentrt.checkpoint import Checkpoint, InMemoryCheckpointer
from agentrt.state import AgentState
from agentrt.types import Message, Usage


def state(text: str = "hi", step: int = 0) -> AgentState:
    return AgentState(
        messages=[Message(role="user", content=text)],
        usage=Usage(input_tokens=10, output_tokens=2),
        step=step,
    )


def test_a_checkpoint_round_trips():
    cp = InMemoryCheckpointer()
    cp.create_run("r1")
    cp.put_checkpoint(Checkpoint(run_id="r1", step=0, next_node="model", state=state("remember")))

    restored = cp.latest_checkpoint("r1", AgentState)
    assert restored.step == 0
    assert restored.next_node == "model"
    assert restored.state.messages[0].content == "remember"
    assert restored.state.usage.total == 12


def test_latest_returns_the_highest_step():
    cp = InMemoryCheckpointer()
    cp.create_run("r1")
    for step in range(5):
        cp.put_checkpoint(Checkpoint(run_id="r1", step=step, next_node="model", state=state(step=step)))

    assert cp.latest_checkpoint("r1", AgentState).step == 4


def test_an_older_checkpoint_is_still_readable():
    cp = InMemoryCheckpointer()
    cp.create_run("r1")
    for step in range(5):
        cp.put_checkpoint(
            Checkpoint(run_id="r1", step=step, next_node="model", state=state(f"step {step}", step))
        )

    old = cp.checkpoint_at("r1", 2, AgentState)
    assert old.state.messages[0].content == "step 2", "history must not be rewritten"


def test_runs_are_isolated_from_each_other():
    cp = InMemoryCheckpointer()
    cp.create_run("r1")
    cp.create_run("r2")
    cp.put_checkpoint(Checkpoint(run_id="r1", step=0, next_node="a", state=state("one")))
    cp.put_checkpoint(Checkpoint(run_id="r2", step=0, next_node="b", state=state("two")))

    assert cp.latest_checkpoint("r1", AgentState).state.messages[0].content == "one"
    assert cp.latest_checkpoint("r2", AgentState).state.messages[0].content == "two"


def test_no_checkpoint_for_an_unknown_run():
    assert InMemoryCheckpointer().latest_checkpoint("nope", AgentState) is None


def test_history_is_ordered_by_step():
    cp = InMemoryCheckpointer()
    cp.create_run("r1")
    for step in [3, 0, 2, 1]:
        cp.put_checkpoint(Checkpoint(run_id="r1", step=step, next_node="model", state=state()))

    assert [h["step"] for h in cp.history("r1")] == [0, 1, 2, 3]


def test_run_status_moves():
    cp = InMemoryCheckpointer()
    cp.create_run("r1")
    assert cp.run_status("r1") == "running"
    cp.set_run_status("r1", "completed")
    assert cp.run_status("r1") == "completed"
    assert [r["run_id"] for r in cp.list_runs(status="completed")] == ["r1"]


def test_survives_a_real_file(tmp_path):
    from agentrt.checkpoint import SqliteCheckpointer

    path = tmp_path / "run.db"
    first = SqliteCheckpointer(path)
    first.create_run("r1")
    first.put_checkpoint(Checkpoint(run_id="r1", step=0, next_node="model", state=state("persisted")))
    first.close()

    second = SqliteCheckpointer(path)
    assert second.latest_checkpoint("r1", AgentState).state.messages[0].content == "persisted"
    
    
def test_writing_the_same_step_twice_is_rejected():
    cp = InMemoryCheckpointer()
    cp.create_run("r1")
    cp.put_checkpoint(Checkpoint(run_id="r1", step=5, next_node="model", state=state("first")))

    with pytest.raises(sqlite3.IntegrityError):
        cp.put_checkpoint(Checkpoint(run_id="r1", step=5, next_node="tools", state=state("second")))

    assert cp.checkpoint_at("r1", 5, AgentState).state.messages[0].content == "first"


def test_a_checkpoint_for_an_unknown_run_is_rejected():
    cp = InMemoryCheckpointer()
    with pytest.raises(sqlite3.IntegrityError):
        cp.put_checkpoint(Checkpoint(run_id="ghost", step=0, next_node="model", state=state()))


def test_setting_the_status_of_an_unknown_run_is_rejected():
    with pytest.raises(KeyError, match="no such run"):
        InMemoryCheckpointer().set_run_status("ghost", "completed")