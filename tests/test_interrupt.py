import pytest

from agentrt.checkpoint import InMemoryCheckpointer
from agentrt.errors import FatalError, Interrupt, RetryableError
from agentrt.graph import END, StateGraph
from agentrt.providers import ScriptedProvider
from agentrt.runtime import Runtime
from agentrt.state import AgentState
from agentrt.types import Message


def approval_graph(seen: list | None = None) -> StateGraph:
    """prepare -> approve -> act. `approve` asks on first entry, answers on resume."""

    def approve(state, ctx):
        if seen is not None:
            seen.append((ctx.step, ctx.resume_value))
        if ctx.resume_value is None:
            raise Interrupt({"question": "ship it?"})
        return {"messages": Message(role="user", content=f"answer: {ctx.resume_value}")}

    g = StateGraph(AgentState)
    g.add_node(
        "prepare", lambda state, ctx: {"messages": Message(role="assistant", content="ready")}
    )
    g.add_node("approve", approve)
    g.add_node("act", lambda state, ctx: {"messages": Message(role="assistant", content="done")})
    g.set_entry("prepare")
    g.add_edge("prepare", "approve")
    g.add_edge("approve", "act")
    g.add_edge("act", END)
    return g


def runtime(cp, seen=None, **kwargs):
    return Runtime(
        graph=approval_graph(seen).compile(),
        provider=ScriptedProvider([]),
        checkpointer=cp,
        **kwargs,
    )


def test_an_interrupt_pauses_the_run_and_carries_the_payload():
    cp = InMemoryCheckpointer()
    result = runtime(cp).run(run_id="r1")

    assert result.status == "paused"
    assert result.interrupt == {"question": "ship it?"}
    assert result.error is None, "a pause is not a failure"
    assert cp.run_status("r1") == "paused"


def test_an_interrupt_is_control_flow_not_a_classified_error():
    assert not isinstance(Interrupt("x"), (RetryableError, FatalError))


def test_a_paused_result_holds_the_state_before_the_interrupting_node():
    result = runtime(InMemoryCheckpointer()).run(run_id="r1")

    assert result.status == "paused"
    assert [m.content for m in result.state.messages] == ["ready"]
    assert result.steps == 1, "the interrupting node did not complete a step"


def test_pausing_writes_no_checkpoint():
    cp = InMemoryCheckpointer()
    assert runtime(cp).run(run_id="r1").status == "paused"

    history = cp.history("r1")
    assert [h["step"] for h in history] == [0, 1]
    assert history[-1]["next_node"] == "approve", "the last row already points at the asker"


def test_an_interrupt_is_not_retried():
    calls = []

    def ask(state, ctx):
        calls.append(1)
        raise Interrupt("once")

    g = StateGraph(AgentState)
    g.add_node("ask", ask)
    g.set_entry("ask")
    g.add_edge("ask", END)

    result = Runtime(graph=g.compile(), provider=ScriptedProvider([])).run()

    assert result.status == "paused"
    assert len(calls) == 1


def test_resume_reenters_the_same_node_with_the_value():
    cp = InMemoryCheckpointer()
    seen: list = []
    runtime(cp, seen).run(run_id="r1")

    result = runtime(cp, seen).resume("r1", "yes")

    assert result.status == "completed"
    assert seen == [(2, None), (2, "yes")], "same node, same step, asked then answered"
    assert [m.content for m in result.state.messages] == ["ready", "answer: yes", "done"]
    assert cp.run_status("r1") == "completed"


def test_resume_after_a_pause_continues_the_step_numbering():
    """No checkpoint was written on pause, so step 2 is still free."""
    cp = InMemoryCheckpointer()
    runtime(cp).run(run_id="r1")
    result = runtime(cp).resume("r1", "yes")

    assert result.steps == 3
    assert [h["step"] for h in cp.history("r1")] == [0, 1, 2, 3]
    assert cp.checkpoint_at("r1", 2, AgentState).next_node == "act"


def test_the_resume_value_reaches_only_the_reentered_node():
    seen = []

    def ask(state, ctx):
        if ctx.resume_value is None:
            raise Interrupt("?")
        return {}

    def after(state, ctx):
        seen.append(ctx.resume_value)
        return {}

    g = StateGraph(AgentState)
    g.add_node("ask", ask)
    g.add_node("after", after)
    g.set_entry("ask")
    g.add_edge("ask", "after")
    g.add_edge("after", END)

    cp = InMemoryCheckpointer()
    rt = Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp)
    rt.run(run_id="r1")
    rt.resume("r1", "answer")

    assert seen == [None], "a later node must not mistake the answer for its own"


def test_a_node_can_interrupt_again_after_being_answered():
    def two_questions(state, ctx):
        if ctx.resume_value is None:
            raise Interrupt("first")
        if ctx.resume_value == "one":
            raise Interrupt("second")
        return {"messages": Message(role="user", content=ctx.resume_value)}

    g = StateGraph(AgentState)
    g.add_node("ask", two_questions)
    g.set_entry("ask")
    g.add_edge("ask", END)

    cp = InMemoryCheckpointer()
    rt = Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp)

    assert rt.run(run_id="r1").interrupt == "first"
    second = rt.resume("r1", "one")
    assert second.status == "paused"
    assert second.interrupt == "second"
    assert [h["step"] for h in cp.history("r1")] == [0]

    done = rt.resume("r1", "two")
    assert done.status == "completed"
    assert done.last_message.content == "two"


def test_resume_with_a_value_refuses_a_run_that_is_not_paused():
    """A value fed to a crashed run's next node would answer a question nobody asked."""

    def boom(state, ctx):
        raise ValueError("crashed")

    g = StateGraph(AgentState)
    g.add_node("boom", boom)
    g.set_entry("boom")
    g.add_edge("boom", END)

    cp = InMemoryCheckpointer()
    rt = Runtime(graph=g.compile(), provider=ScriptedProvider([]), checkpointer=cp)
    rt.run(run_id="r1")

    with pytest.raises(ValueError, match="paused"):
        rt.resume("r1", "unsolicited")
    assert cp.run_status("r1") == "failed", "refusing must not touch the status"
