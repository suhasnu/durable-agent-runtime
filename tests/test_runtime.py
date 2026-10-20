import pytest
from pydantic import BaseModel

from agentrt.checkpoint import InMemoryCheckpointer
from agentrt.graph import END, StateGraph
from agentrt.providers import ScriptedProvider
from agentrt.runtime import Runtime
from agentrt.state import AgentState, Messages
from agentrt.types import Message


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