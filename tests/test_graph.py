import pytest

from agentrt.graph import END, StateGraph
from agentrt.state import AgentState


def noop(state, ctx):
    return {}


def build():
    g = StateGraph(AgentState)
    g.add_node("a", noop)
    g.set_entry("a")
    return g


def test_a_valid_graph_compiles():
    g = build()
    g.add_edge("a", END)
    assert g.compile().entry == "a"


def test_compile_rejects_unknown_edge_target():
    g = build()
    g.add_edge("a", "nowhere")
    with pytest.raises(ValueError, match="unknown target"):
        g.compile()


def test_compile_rejects_dangling_node():
    g = build()
    g.add_node("b", noop)
    g.add_edge("a", END)
    with pytest.raises(ValueError, match="no outgoing edge"):
        g.compile()


def test_compile_rejects_unreachable_node():
    g = build()
    g.add_node("orphan", noop)
    g.add_edge("a", END)
    g.add_edge("orphan", END)
    with pytest.raises(ValueError, match="unreachable"):
        g.compile()


def test_compile_requires_an_entry():
    g = StateGraph(AgentState)
    g.add_node("a", noop)
    g.add_edge("a", END)
    with pytest.raises(ValueError, match="no entry node"):
        g.compile()


def test_duplicate_node_names_are_rejected():
    g = build()
    with pytest.raises(ValueError, match="duplicate"):
        g.add_node("a", noop)


def test_a_node_cannot_have_both_edge_kinds():
    g = build()
    g.add_edge("a", END)
    with pytest.raises(ValueError, match="already has a static edge"):
        g.add_conditional_edge("a", lambda s: END, {END})


def test_a_conditional_edge_picks_by_state():
    g = build()
    g.add_node("b", noop)
    g.add_conditional_edge("a", lambda s: "b" if s.step > 0 else END, {"b", END})
    g.add_edge("b", END)
    compiled = g.compile()

    assert compiled.next_node("a", AgentState(step=0)) == END
    assert compiled.next_node("a", AgentState(step=1)) == "b"


def test_router_returning_an_undeclared_target_is_caught():
    g = build()
    g.add_conditional_edge("a", lambda s: "surprise", {END})
    compiled = g.compile()
    with pytest.raises(ValueError, match="not in declared targets"):
        compiled.next_node("a", AgentState())


def test_mermaid_export_mentions_every_node():
    g = build()
    g.add_edge("a", END)
    assert "a" in g.compile().to_mermaid()
    
    
def test_compiling_takes_a_snapshot_of_the_builder():
    g = build()
    g.add_edge("a", END)
    compiled = g.compile()

    g.add_edge("a", "ghost")
    assert compiled.next_node("a", AgentState()) == END, "a compiled graph must not change"


def test_a_static_cycle_with_no_exit_is_rejected():
    g = StateGraph(AgentState)
    g.add_node("a", noop)
    g.add_node("b", noop)
    g.set_entry("a")
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    with pytest.raises(ValueError, match="can never reach END"):
        g.compile()


def test_a_cycle_with_a_conditional_exit_is_allowed():
    g = StateGraph(AgentState)
    g.add_node("a", noop)
    g.add_node("b", noop)
    g.set_entry("a")
    g.add_conditional_edge("a", lambda s: "b" if s.step < 3 else END, {"b", END})
    g.add_edge("b", "a")
    g.compile()


def test_a_conditional_edge_with_no_targets_is_rejected():
    g = build()
    with pytest.raises(ValueError, match="no targets"):
        g.add_conditional_edge("a", lambda s: END, set())


def test_a_string_of_targets_is_rejected():
    g = build()
    with pytest.raises(TypeError, match="not a string"):
        g.add_conditional_edge("a", lambda s: END, "abc")