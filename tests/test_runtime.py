import pytest
from pydantic import BaseModel

from agentrt.graph import END, StateGraph
from agentrt.providers import ScriptedProvider
from agentrt.runtime import Runtime
from agentrt.state import AgentState, Messages


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