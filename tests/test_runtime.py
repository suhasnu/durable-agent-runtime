import pytest
from pydantic import BaseModel

from agentrt.checkpoint import InMemoryCheckpointer
from agentrt.errors import UncertainSideEffect
from agentrt.graph import END, StateGraph
from agentrt.providers import ScriptedProvider
from agentrt.runtime import Runtime
from agentrt.state import AgentState, Messages
from agentrt.types import Message
from agentrt.providers import calls, says


def test_a_state_type_without_a_step_field_still_runs():
    """The runtime maintains `step` when the state declares one, and does not
    require it. A graph with its own state type is not forced to carry it."""

    class Minimal(BaseModel):
        messages: Messages = []

    g = StateGraph(Minimal)
    g.add_node("once", lambda state, ctx: {})
    g.set_entry("once")
    g.add_edge("once", END)

    result = Runtime(graph=g.compile(), provider=ScriptedProvider([])).run()
    assert result.steps == 1


def test_the_context_step_matches_the_state_step():
    seen = {}

    def record(state, ctx):
        seen["ctx"] = ctx.step
        return {}

    g = StateGraph(AgentState)
    g.add_node("once", record)
    g.set_entry("once")
    g.add_edge("once", END)

    result = Runtime(graph=g.compile(), provider=ScriptedProvider([])).run()
    assert seen["ctx"] == result.state.step == 1


def test_nodes_receive_the_context_they_were_given():
    seen = {}

    def record(state, ctx):
        seen["system"] = ctx.system
        return {}

    g = StateGraph(AgentState)
    g.add_node("once", record)
    g.set_entry("once")
    g.add_edge("once", END)

    Runtime(graph=g.compile(), provider=ScriptedProvider([]), system="be brief").run()
    assert seen["system"] == "be brief"
    

def test_every_step_is_checkpointed():
    from agentrt.agent import build_agent
    from agentrt.checkpoint import InMemoryCheckpointer
    from agentrt.providers import calls, says
    from agentrt.tools import ToolRegistry, tool

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    cp = InMemoryCheckpointer()
    registry = ToolRegistry([ping])
    rt = Runtime(
        graph=build_agent().compile(),
        provider=ScriptedProvider([calls("ping", {}), says("done")]),
        tools=registry,
        checkpointer=cp,
    )
    result = rt.run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")

    steps = [h["step"] for h in cp.history("r1")]
    assert steps == [0, 1, 2, 3], "one row per step, including step 0"
    assert result.status == "completed"


def test_step_zero_holds_the_initial_state():
    cp = InMemoryCheckpointer()
    g = StateGraph(AgentState)
    g.add_node("once", lambda state, ctx: {})
    g.set_entry("once")
    g.add_edge("once", END)

    Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp).run(
        AgentState(messages=[Message(role="user", content="start")]), run_id="r1"
    )

    first = cp.checkpoint_at("r1", 0, AgentState)
    assert first.next_node == "once"
    assert first.state.messages[0].content == "start"


def test_the_checkpoint_records_where_to_go_next_not_where_it_came_from():
    cp = InMemoryCheckpointer()
    g = StateGraph(AgentState)
    g.add_node("a", lambda state, ctx: {})
    g.add_node("b", lambda state, ctx: {})
    g.set_entry("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp).run(run_id="r1")

    assert cp.checkpoint_at("r1", 1, AgentState).next_node == "b"
    assert cp.checkpoint_at("r1", 2, AgentState).next_node == END


def test_a_failed_run_keeps_its_state_and_its_checkpoints():
    cp = InMemoryCheckpointer()

    def boom(state, ctx):
        raise ValueError("node failed")

    g = StateGraph(AgentState)
    g.add_node("fine", lambda state, ctx: {"messages": Message(role="assistant", content="ok")})
    g.add_node("boom", boom)
    g.set_entry("fine")
    g.add_edge("fine", "boom")
    g.add_edge("boom", END)

    result = Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp).run(
        run_id="r1"
    )

    assert result.status == "failed"
    assert isinstance(result.error, ValueError)
    assert result.state.messages[-1].content == "ok", "work before the failure survives"
    assert cp.run_status("r1") == "failed"
    assert [h["step"] for h in cp.history("r1")] == [0, 1]
    

def test_a_raising_router_still_leaves_the_run_in_a_terminal_status():
    cp = InMemoryCheckpointer()

    def bad_router(state):
        raise ValueError("router exploded")

    g = StateGraph(AgentState)
    g.add_node("a", lambda state, ctx: {})
    g.set_entry("a")
    g.add_conditional_edge("a", bad_router, {END})

    result = Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp).run(
        run_id="r1"
    )

    assert result.status == "failed"
    assert isinstance(result.error, ValueError)
    assert cp.run_status("r1") == "failed", "a run must never be left marked running"


def test_resume_continues_from_the_last_checkpoint():
    from agentrt.agent import build_agent
    from agentrt.providers import calls, says
    from agentrt.tools import ToolRegistry, tool

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    cp = InMemoryCheckpointer()
    registry = ToolRegistry([ping])
    graph = build_agent().compile()

    # Die after the first model call: the script runs out, which raises.
    first = Runtime(
        graph=graph,
        provider=ScriptedProvider([calls("ping", {})]),
        tools=registry,
        checkpointer=cp,
    )
    crashed = first.run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")
    assert crashed.status == "failed"

    # A fresh Runtime, nothing carried over in memory.
    second = Runtime(
        graph=graph,
        provider=ScriptedProvider([says("done")]),
        tools=registry,
        checkpointer=cp,
    )
    result = second.resume("r1")

    assert result.status == "completed"
    assert result.last_message.content == "done"
    assert cp.run_status("r1") == "completed"


def test_resume_takes_the_step_from_the_checkpoint_not_from_memory():
    cp = InMemoryCheckpointer()
    g = StateGraph(AgentState)
    g.add_node("a", lambda state, ctx: {})
    g.add_node("b", lambda state, ctx: {})
    g.set_entry("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    rt = Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp)
    rt.run(run_id="r1")

    # The run finished at step 2. Resuming writes no new checkpoints, because
    # the last one already points at END.
    resumed = rt.resume("r1")
    assert resumed.steps == 2
    assert [h["step"] for h in cp.history("r1")] == [0, 1, 2]


def test_resuming_a_finished_run_is_a_no_op():
    cp = InMemoryCheckpointer()
    g = StateGraph(AgentState)
    g.add_node("once", lambda state, ctx: {"messages": Message(role="assistant", content="hi")})
    g.set_entry("once")
    g.add_edge("once", END)

    rt = Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp)
    rt.run(run_id="r1")
    resumed = rt.resume("r1")

    assert resumed.status == "completed"
    assert len(resumed.state.messages) == 1, "the node must not run a second time"
    assert resumed.history == []


def test_resuming_an_unknown_run_is_an_error():
    g = StateGraph(AgentState)
    g.add_node("once", lambda state, ctx: {})
    g.set_entry("once")
    g.add_edge("once", END)

    rt = Runtime(graph=g.compile(), provider=ScriptedProvider([]))
    with pytest.raises(KeyError, match="no checkpoint"):
        rt.resume("ghost")


def test_a_run_survives_a_new_process(tmp_path):
    """The whole claim, in one test: a run resumed from a different Runtime,
    a different provider and a reopened database reaches the same place."""
    from agentrt.agent import build_agent
    from agentrt.checkpoint import SqliteCheckpointer
    from agentrt.providers import calls, says
    from agentrt.tools import ToolRegistry, tool

    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    db = tmp_path / "run.db"
    graph = build_agent().compile()
    registry = ToolRegistry([ping])

    store = SqliteCheckpointer(db)
    Runtime(
        graph=graph,
        provider=ScriptedProvider([calls("ping", {})]),
        tools=registry,
        checkpointer=store,
    ).run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")
    store.close()

    reopened = SqliteCheckpointer(db)
    result = Runtime(
        graph=graph,
        provider=ScriptedProvider([says("finished in another process")]),
        tools=registry,
        checkpointer=reopened,
    ).resume("r1")

    assert result.status == "completed"
    assert result.last_message.content == "finished in another process"
    assert result.state.messages[0].content == "go", "the original question survived"
    

class ProcessKilled(BaseException):
    """Not an Exception, so nothing in the runtime catches it.

    This is how SIGKILL looks from the executor's point of view: control
    leaves mid-tool and no completion record is ever written.
    """


def test_a_tool_does_not_run_twice_when_a_crash_lands_inside_tools_node():
    """The failure the whole project is named for.

    tools_node executes the tool, then the runtime writes the checkpoint. A
    crash in between leaves the last checkpoint pointing at tools_node, so
    resume runs it again and the side effect happens twice.
    """
    from agentrt.agent import build_agent
    from agentrt.providers import says
    from agentrt.tools import ToolRegistry, tool

    charges: list[float] = []
    crash = {"armed": True}

    @tool(side_effecting=True)
    def charge_card(amount: float) -> str:
        """Charge the customer's card."""
        charges.append(amount)
        if crash["armed"]:
            crash["armed"] = False
            raise ProcessKilled
        return f"charged {amount}"

    cp = InMemoryCheckpointer()
    registry = ToolRegistry([charge_card])
    graph = build_agent().compile()

    with pytest.raises(ProcessKilled):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([calls("charge_card", {"amount": 49.0})]),
            tools=registry,
            checkpointer=cp,
        ).run(AgentState(messages=[Message(role="user", content="charge 49")]), run_id="r1")

    assert charges == [49.0], "the charge happened before the crash"

    
    resumed = Runtime(
        graph=graph,
        provider=ScriptedProvider([says("done")]),
        tools=registry,
        checkpointer=cp,
    ).resume("r1")

    assert resumed.status == "failed"
    assert isinstance(resumed.error, UncertainSideEffect)
    assert "may have executed" in str(resumed.error)
    assert charges == [49.0], "the card must be charged exactly once"
    
def test_a_completed_invocation_is_replayed_not_re_executed():
    """Resume never repeats work it already paid for, side effects or not."""