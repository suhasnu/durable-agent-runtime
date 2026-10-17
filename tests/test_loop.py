import pytest

from agentrt.loop import run_agent
from agentrt.providers import ScriptedProvider, calls, says
from agentrt.tools import ToolRegistry, tool

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


def registry():
    return ToolRegistry([lookup, explode])


def test_a_plain_answer_ends_the_run():
    messages, usage = run_agent(ScriptedProvider([says("hello")]), registry(), "hi")
    assert messages[-1].content == "hello"
    assert usage.total == 15


def test_a_tool_call_is_executed_and_fed_back():
    provider = ScriptedProvider([calls("lookup", {"key": "x"}), says("the answer is value of x")])
    messages, _ = run_agent(provider, registry(), "look up x")

    assert CALLED == ["x"]
    assert messages[2].role == "tool"
    assert messages[2].tool_results[0].content == "value of x"
    assert "value of x" in messages[-1].content


def test_an_unknown_tool_comes_back_as_an_error_result_not_an_exception():
    provider = ScriptedProvider([calls("nonexistent", {}), says("I will try something else")])
    messages, _ = run_agent(provider, registry(), "go")

    result = messages[2].tool_results[0]
    assert result.is_error
    assert "lookup" in result.content


def test_bad_arguments_come_back_as_an_error_result():
    provider = ScriptedProvider([calls("lookup", {}), says("sorry")])
    messages, _ = run_agent(provider, registry(), "go")
    assert messages[2].tool_results[0].is_error


def test_a_raising_tool_does_not_kill_the_run():
    provider = ScriptedProvider([calls("explode", {}), says("that failed")])
    messages, _ = run_agent(provider, registry(), "go")

    assert messages[2].tool_results[0].is_error
    assert "boom" in messages[2].tool_results[0].content


def test_usage_accumulates_across_steps():
    provider = ScriptedProvider([calls("lookup", {"key": "x"}), says("done")])
    _, usage = run_agent(provider, registry(), "go")
    assert usage.total == 30


def test_the_step_limit_stops_a_loop():
    script = [calls("lookup", {"key": str(i)}, call_id=f"c{i}") for i in range(10)]
    with pytest.raises(RuntimeError, match="did not finish"):
        run_agent(ScriptedProvider(script), registry(), "go", max_steps=3)