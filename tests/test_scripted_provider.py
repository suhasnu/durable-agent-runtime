import pytest

from agentrt.providers import ScriptedProvider, calls, says
from agentrt.types import Message


def test_responses_come_back_in_order():
    provider = ScriptedProvider([says("one"), says("two")])
    assert provider.complete([]).message.content == "one"
    assert provider.complete([]).message.content == "two"


def test_requests_are_recorded():
    provider = ScriptedProvider([says("ok")])
    provider.complete([Message(role="user", content="hi")], system="be nice")
    assert provider.calls[0]["system"] == "be nice"


def test_running_past_the_script_fails_loudly():
    provider = ScriptedProvider([says("only one")])
    provider.complete([])
    with pytest.raises(AssertionError, match="exhausted"):
        provider.complete([])


def test_a_scripted_exception_is_raised():
    provider = ScriptedProvider([RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="boom"):
        provider.complete([])


def test_calls_helper_builds_a_tool_call():
    completion = calls("search", {"q": "x"})
    assert completion.stop_reason == "tool_use"
    assert completion.message.tool_calls[0].name == "search"