import pytest
from pydantic import BaseModel

from agentrt.agent import build_agent
from agentrt.checkpoint import InMemoryCheckpointer, SqliteCheckpointer
from agentrt.errors import UncertainSideEffect
from agentrt.graph import END, StateGraph
from agentrt.providers import ScriptedProvider, calls, says
from agentrt.runtime import Runtime
from agentrt.state import AgentState, Messages
from agentrt.tools import ToolRegistry, tool
from agentrt.types import Message


def test_a_custom_state_type_runs():
    """graph can bring its own state model"""

    class Minimal(BaseModel):
        messages: Messages = []

    g = StateGraph(Minimal)
    g.add_node("once", lambda state, ctx: {})
    g.set_entry("once")
    g.add_edge("once", END)

    result = Runtime(graph=g.compile(), provider=ScriptedProvider([])).run()
    assert result.steps == 1


def test_the_context_step_numbers_the_running_node():
    seen: list[int] = []

    def record(state, ctx):
        seen.append(ctx.step)
        return {}

    g = StateGraph(AgentState)
    g.add_node("a", record)
    g.add_node("b", record)
    g.set_entry("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    result = Runtime(graph=g.compile(), provider=ScriptedProvider([])).run()

    assert seen == [1, 2], "a node sees its own step number, counting from one"
    assert result.steps == 2


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

    
    with pytest.raises(UncertainSideEffect, match="may have executed"):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([says("done")]),
            tools=registry,
            checkpointer=cp,
        ).resume("r1")

    assert charges == [49.0], "the card must be charged exactly once"
    
class CrashOnStep(InMemoryCheckpointer):
    """Dies when writing one particular checkpoint.

    The window between a tool completing and its checkpoint landing cannot be
    reached from the tool itself, so the store is where the crash goes.
    """

    def __init__(self, step: int):
        super().__init__()
        self._crash_at = step

    def put_checkpoint(self, cp):
        if cp.step == self._crash_at:
            self._crash_at = None
            raise ProcessKilled
        super().put_checkpoint(cp)


def test_a_completed_invocation_is_replayed_not_re_executed():
    """Resume never repeats work it already paid for."""

    ran: list[str] = []

    @tool
    def lookup(key: str) -> str:
        """Look something up."""
        ran.append(key)
        return f"value of {key}"

    cp = CrashOnStep(step=2)
    registry = ToolRegistry([lookup])
    graph = build_agent().compile()

    with pytest.raises(ProcessKilled):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([calls("lookup", {"key": "x"})]),
            tools=registry,
            checkpointer=cp,
        ).run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")

    assert ran == ["x"]
    assert cp.get_invocation("r1", "call_1").status == "completed"

    resumed = Runtime(
        graph=graph,
        provider=ScriptedProvider([says("the answer is value of x")]),
        tools=registry,
        checkpointer=cp,
    ).resume("r1")

    assert resumed.status == "completed"
    assert ran == ["x"], "the tool must not run a second time"
    tool_message = next(m for m in resumed.state.messages if m.role == "tool")
    assert tool_message.tool_results[0].content == "value of x", "the logged result reached the model"
    
def test_a_reused_call_id_with_different_arguments_is_refused():
    """Replaying by call id alone would silently answer the wrong question."""

    @tool
    def lookup(key: str) -> str:
        """Look something up."""
        return f"value of {key}"

    cp = InMemoryCheckpointer()
    result = Runtime(
        graph=build_agent().compile(),
        provider=ScriptedProvider([
            calls("lookup", {"key": "a"}, call_id="call_1"),
            calls("lookup", {"key": "b"}, call_id="call_1"),
            says("done"),
        ]),
        tools=ToolRegistry([lookup]),
        checkpointer=cp,
    ).run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")

    assert result.status == "failed"
    assert "different arguments" in str(result.error)
    
def test_a_read_only_tool_is_simply_re_run_after_a_crash():
    """No side effect means nothing to be uncertain about."""
    ran: list[str] = []

    @tool
    def lookup(key: str) -> str:
        """Look something up."""
        ran.append(key)
        if len(ran) == 1:
            raise ProcessKilled
        return f"value of {key}"

    cp = InMemoryCheckpointer()
    registry = ToolRegistry([lookup])
    graph = build_agent().compile()

    with pytest.raises(ProcessKilled):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([calls("lookup", {"key": "x"})]),
            tools=registry,
            checkpointer=cp,
        ).run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")

    resumed = Runtime(
        graph=graph,
        provider=ScriptedProvider([says("done")]),
        tools=registry,
        checkpointer=cp,
    ).resume("r1")

    assert resumed.status == "completed"
    assert ran == ["x", "x"], "re-running a read-only tool is free"


def test_an_idempotent_tool_is_re_run_after_a_crash():
    """It has side effects, but its author declared them safe to repeat."""
    flags: list[str] = []

    @tool(side_effecting=True, idempotent=True)
    def set_flag(name: str) -> str:
        """Set a flag. Setting it twice is the same as setting it once."""
        flags.append(name)
        if len(flags) == 1:
            raise ProcessKilled
        return f"{name}=true"

    cp = InMemoryCheckpointer()
    registry = ToolRegistry([set_flag])
    graph = build_agent().compile()

    with pytest.raises(ProcessKilled):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([calls("set_flag", {"name": "ready"})]),
            tools=registry,
            checkpointer=cp,
        ).run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")

    resumed = Runtime(
        graph=graph,
        provider=ScriptedProvider([says("done")]),
        tools=registry,
        checkpointer=cp,
    ).resume("r1")

    assert resumed.status == "completed"
    assert flags == ["ready", "ready"]


def test_reconciliation_discards_rather_than_completing_the_record():
    """The trap: a reclaimed record completed with an empty result would be
    replayed, and the model would receive "" instead of a tool result."""
    ran: list[str] = []

    @tool
    def lookup(key: str) -> str:
        """Look something up."""
        ran.append(key)
        if len(ran) == 1:
            raise ProcessKilled
        return f"value of {key}"

    cp = InMemoryCheckpointer()
    registry = ToolRegistry([lookup])
    graph = build_agent().compile()

    with pytest.raises(ProcessKilled):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([calls("lookup", {"key": "x"})]),
            tools=registry,
            checkpointer=cp,
        ).run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")

    assert len(cp.dangling_invocations("r1")) == 1

    resumed = Runtime(
        graph=graph,
        provider=ScriptedProvider([says("done")]),
        tools=registry,
        checkpointer=cp,
    ).resume("r1")

    tool_message = next(m for m in resumed.state.messages if m.role == "tool")
    assert tool_message.tool_results[0].content == "value of x", (
        "a reclaimed record must be deleted, not completed with an empty result"
    )
    
def test_a_dangling_call_to_an_unregistered_tool_is_refused():
    """No definition means no flags, so there is no basis for calling it safe."""
    cp = InMemoryCheckpointer()
    graph = build_agent().compile()

    @tool(side_effecting=True)
    def charge_card(amount: float) -> str:
        """Charge the customer's card."""
        raise ProcessKilled

    with pytest.raises(ProcessKilled):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([calls("charge_card", {"amount": 49.0})]),
            tools=ToolRegistry([charge_card]),
            checkpointer=cp,
        ).run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")

    # Resume with the tool missing from the registry.
    with pytest.raises(UncertainSideEffect):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([says("done")]),
            tools=ToolRegistry([]),
            checkpointer=cp,
        ).resume("r1")


def test_a_refused_resume_marks_the_run_as_needing_resolution():
    cp = InMemoryCheckpointer()
    graph = build_agent().compile()

    @tool(side_effecting=True)
    def charge_card(amount: float) -> str:
        """Charge the customer's card."""
        raise ProcessKilled

    registry = ToolRegistry([charge_card])

    with pytest.raises(ProcessKilled):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([calls("charge_card", {"amount": 49.0})]),
            tools=registry,
            checkpointer=cp,
        ).run(AgentState(messages=[Message(role="user", content="go")]), run_id="r1")

    with pytest.raises(UncertainSideEffect):
        Runtime(
            graph=graph,
            provider=ScriptedProvider([says("done")]),
            tools=registry,
            checkpointer=cp,
        ).resume("r1")

    assert cp.run_status("r1") == "needs_resolution"
    assert [r["run_id"] for r in cp.list_runs(status="needs_resolution")] == ["r1"]


def test_an_operator_can_resolve_a_dangling_call_and_the_run_continues():
    """The path out of a refusal: a human says what actually happened."""
    charges: list[float] = []

    @tool(side_effecting=True)
    def charge_card(amount: float) -> str:
        """Charge the customer's card."""
        charges.append(amount)
        raise ProcessKilled

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

    cp.resolve_invocation("r1", "call_1", "charged 49.0 (confirmed in the payment log)")

    resumed = Runtime(
        graph=graph,
        provider=ScriptedProvider([says("order complete")]),
        tools=registry,
        checkpointer=cp,
    ).resume("r1")

    assert resumed.status == "completed"
    assert charges == [49.0], "the tool was not run a second time"
    tool_message = next(m for m in resumed.state.messages if m.role == "tool")
    assert "confirmed in the payment log" in tool_message.tool_results[0].content