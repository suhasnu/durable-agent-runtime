import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langgraph.checkpoint.sqlite")

from agentrt.agent import build_agent  # noqa: E402
from agentrt.errors import BudgetExceeded, TruncatedResponse  # noqa: E402
from agentrt.providers import ScriptedProvider, calls, says, truncated  # noqa: E402
from agentrt.runtime import Budget, Runtime  # noqa: E402
from agentrt.state import AgentState  # noqa: E402
from agentrt.tools import ToolRegistry, tool  # noqa: E402
from agentrt.types import Message, ToolCall  # noqa: E402


def _load_port():
    path = Path(__file__).resolve().parent.parent / "examples" / "langgraph_port.py"
    spec = importlib.util.spec_from_file_location("langgraph_port", path)
    module = importlib.util.module_from_spec(spec)
    # LangGraph resolves the state's string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


port = _load_port()


@tool
def ping() -> str:
    """Ping."""
    return "pong"


def script():
    return [calls("ping", {}), says("done")]


def test_the_port_produces_the_same_conversation_as_agentrt():
    ours = Runtime(
        graph=build_agent().compile(),
        provider=ScriptedProvider(script()),
        tools=ToolRegistry([ping]),
    ).run(AgentState(messages=[Message(role="user", content="go")]))

    agent = port.build_langgraph_agent(
        ScriptedProvider(script()), ToolRegistry([ping]), port.sqlite_saver()
    )
    theirs = agent.invoke(port.initial_state("go"), port.thread("r1"))

    assert theirs["messages"] == ours.state.messages
    assert theirs["usage"] == ours.state.usage


def test_state_reloads_from_disk_as_agentrt_types(tmp_path):
    db = tmp_path / "lg.db"
    agent = port.build_langgraph_agent(
        ScriptedProvider(script()), ToolRegistry([ping]), port.sqlite_saver(db)
    )
    agent.invoke(port.initial_state("go"), port.thread("r1"))

    # A fresh saver on the same file, as a new process would open it.
    fresh = port.build_langgraph_agent(
        ScriptedProvider([]), ToolRegistry([ping]), port.sqlite_saver(db)
    )
    messages = fresh.get_state(port.thread("r1")).values["messages"]

    assert all(isinstance(m, Message) for m in messages)
    assert isinstance(messages[1].tool_calls[0], ToolCall)
    assert messages[-1].content == "done"


def test_every_step_is_checkpointed():
    agent = port.build_langgraph_agent(
        ScriptedProvider(script()), ToolRegistry([ping]), port.sqlite_saver()
    )
    agent.invoke(port.initial_state("go"), port.thread("r1"))

    steps = [s.metadata["step"] for s in agent.get_state_history(port.thread("r1"))]
    # -1 is LangGraph's input checkpoint, 0 the state after the input is applied.
    assert sorted(steps) == [-1, 0, 1, 2, 3]


def test_a_truncated_response_fails_the_run():
    agent = port.build_langgraph_agent(
        ScriptedProvider([truncated(tool_call=("ping", {}))]),
        ToolRegistry([ping]),
        port.sqlite_saver(),
    )
    with pytest.raises(TruncatedResponse):
        agent.invoke(port.initial_state("go"), port.thread("r1"))


def test_the_budget_is_checked_before_the_call():
    provider = ScriptedProvider(script())
    agent = port.build_langgraph_agent(
        provider,
        ToolRegistry([ping]),
        port.sqlite_saver(),
        budget=Budget(max_model_calls=1),
    )
    with pytest.raises(BudgetExceeded):
        agent.invoke(port.initial_state("go"), port.thread("r1"))
    assert len(provider.requests) == 1
