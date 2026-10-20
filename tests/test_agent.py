import pytest

from agentrt.agent import build_agent
from agentrt.providers import ScriptedProvider, calls, says, calls_many
from agentrt.runtime import Runtime
from agentrt.state import AgentState
from agentrt.tools import ToolRegistry, tool
from agentrt.types import Message
from agentrt.graph import END, StateGraph
from agentrt.checkpoint import InMemoryCheckpointer



CALLED: list[str] = []


@tool
def lookup(key: str) -> str:
    """Look something up."""
    CALLED.append(key)
    return f"value of {key}"


@tool
def explode() -> str:
    """Always fails."""
    raise ValueError("boom")


@pytest.fixture(autouse=True)
def clear():
    CALLED.clear()


def run(script, user_message="go", max_steps=20):
    registry = ToolRegistry([lookup, explode])
    runtime = Runtime(
        graph=build_agent().compile(),
        provider=ScriptedProvider(script),
        tools=registry,
        max_steps=max_steps,
    )
    return runtime.run(AgentState(messages=[Message(role="user", content=user_message)]))


def test_a_plain_answer_ends_the_run():
    result = run([says("hello")])
    assert result.last_message.content == "hello"
    assert result.state.usage.total == 15


def test_a_tool_call_is_executed_and_fed_back():
    result = run([calls("lookup", {"key": "x"}), says("the answer is value of x")])

    assert CALLED == ["x"]
    assert result.state.messages[2].role == "tool"
    assert result.state.messages[2].tool_results[0].content == "value of x"
    assert "value of x" in result.last_message.content


def test_an_unknown_tool_comes_back_as_an_error_result_not_an_exception():
    result = run([calls("nonexistent", {}), says("I will try something else")])
    error = result.state.messages[2].tool_results[0]
    assert error.is_error
    assert "lookup" in error.content


def test_bad_arguments_come_back_as_an_error_result():
    result = run([calls("lookup", {}), says("sorry")])
    assert result.state.messages[2].tool_results[0].is_error


def test_a_raising_tool_does_not_kill_the_run():
    result = run([calls("explode", {}), says("that failed")])
    assert result.state.messages[2].tool_results[0].is_error
    assert "boom" in result.state.messages[2].tool_results[0].content


def test_usage_accumulates_across_steps():
    result = run([calls("lookup", {"key": "x"}), says("done")])
    assert result.state.usage.total == 30


def test_the_step_limit_stops_a_loop():
    script = [calls("lookup", {"key": str(i)}, call_id=f"c{i}") for i in range(10)]
    result = run(script, max_steps=3)
    assert result.status == "failed"
    assert "did not finish" in str(result.error)
    assert len(result.state.messages) > 1, "partial work is not thrown away"


def test_the_history_records_which_nodes_ran():
    result = run([calls("lookup", {"key": "x"}), says("done")])
    assert result.history == ["model", "tools", "model"]
    
def test_several_tool_calls_in_one_turn_all_run():
    result = run([
        calls_many([("lookup", {"key": "a"}), ("lookup", {"key": "b"})]),
        says("done"),
    ])

    assert CALLED == ["a", "b"]
    results = result.state.messages[2].tool_results
    assert len(results) == 2
    assert {r.call_id for r in results} == {"call_1", "call_2"}


def test_one_failing_call_does_not_stop_the_others():
    result = run([
        calls_many([("lookup", {"key": "a"}), ("explode", {}), ("lookup", {"key": "b"})]),
        says("partly done"),
    ])

    assert CALLED == ["a", "b"]
    results = result.state.messages[2].tool_results
    assert [r.is_error for r in results] == [False, True, False]