from agentrt.agent import build_agent
from agentrt.checkpoint import InMemoryCheckpointer
from agentrt.errors import BudgetExceeded, FatalError
from agentrt.providers import ScriptedProvider, calls, says
from agentrt.runtime import Budget, Runtime
from agentrt.state import AgentState
from agentrt.tools import ToolRegistry, tool
from agentrt.types import Message


@tool
def lookup(key: str) -> str:
    """Look something up."""
    return f"value of {key}"


def runtime(provider, budget, max_tokens=50, **kwargs):
    return Runtime(
        graph=build_agent().compile(),
        provider=provider,
        tools=ToolRegistry([lookup]),
        max_tokens=max_tokens,
        budget=budget,
        **kwargs,
    )


def start():
    return AgentState(messages=[Message(role="user", content="go")])


def test_a_call_that_could_overrun_the_token_budget_is_refused():
    """60 spent plus a 50 token reservation is 110, over a budget of 100."""
    provider = ScriptedProvider(
        [
            calls("lookup", {"key": "a"}, input_tokens=40, output_tokens=20),
            says("done"),
        ]
    )
    result = runtime(provider, Budget(max_total_tokens=100)).run(start())

    assert result.status == "failed"
    assert isinstance(result.error, BudgetExceeded)
    assert isinstance(result.error, FatalError), "a budget does not refill on retry"
    assert len(provider.requests) == 1, "refused before the call, not totalled after"


def test_a_call_that_exactly_fits_the_reservation_is_allowed():
    """50 spent plus a 50 token reservation is exactly the budget of 100."""
    provider = ScriptedProvider(
        [
            calls("lookup", {"key": "a"}, input_tokens=30, output_tokens=20),
            says("done", input_tokens=30, output_tokens=20),
        ]
    )
    result = runtime(provider, Budget(max_total_tokens=100)).run(start())

    assert result.status == "completed"
    assert len(provider.requests) == 2


def test_the_first_call_is_refused_when_max_tokens_alone_exceeds_the_budget():
    provider = ScriptedProvider([says("done")])
    result = runtime(provider, Budget(max_total_tokens=100), max_tokens=200).run(start())

    assert isinstance(result.error, BudgetExceeded)
    assert provider.requests == [], "nothing is spent on a call that cannot fit"


def test_the_model_call_budget_stops_before_the_extra_call():
    provider = ScriptedProvider(
        [
            calls("lookup", {"key": "a"}, call_id="call_1"),
            calls("lookup", {"key": "b"}, call_id="call_2"),
            says("done"),
        ]
    )
    result = runtime(provider, Budget(max_model_calls=2)).run(start())

    assert result.status == "failed"
    assert isinstance(result.error, BudgetExceeded)
    assert len(provider.requests) == 2


def test_a_run_within_both_budgets_completes():
    provider = ScriptedProvider([calls("lookup", {"key": "a"}), says("done")])
    result = runtime(provider, Budget(max_total_tokens=1000, max_model_calls=5)).run(start())

    assert result.status == "completed"
    assert result.state.messages[-1].content == "done"
    assert result.state.usage.calls == 2, "the count rides on the checkpointed usage"


def test_the_model_call_budget_counts_calls_made_before_a_resume():
    """A count kept only in memory would start again at zero in a new process."""
    cp = InMemoryCheckpointer()
    first = ScriptedProvider([calls("lookup", {"key": "a"}), FatalError("provider down")])
    failed = runtime(first, Budget(), checkpointer=cp).run(start(), run_id="r1")
    assert failed.state.usage.calls == 1, "a call that raised left no usage to count"

    second = ScriptedProvider([says("done")])
    result = runtime(second, Budget(max_model_calls=1), checkpointer=cp).resume("r1")

    assert isinstance(result.error, BudgetExceeded)
    assert second.requests == [], "the one allowed call was already made"
