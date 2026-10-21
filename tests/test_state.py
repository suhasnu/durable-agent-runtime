import pytest
from pydantic import ValidationError

from agentrt.state import AgentState, apply_updates
from agentrt.types import Message, Usage


def test_messages_append_rather_than_replace():
    state = AgentState(messages=[Message(role="user", content="hi")])
    out = apply_updates(state, {"messages": Message(role="assistant", content="hello")})
    assert [m.content for m in out.messages] == ["hi", "hello"]


def test_a_list_of_messages_appends_too():
    state = AgentState(messages=[Message(role="user", content="hi")])
    out = apply_updates(state, {"messages": [
        Message(role="assistant", content="a"),
        Message(role="tool", content="b"),
    ]})
    assert len(out.messages) == 3


def test_usage_accumulates():
    state = AgentState(usage=Usage(input_tokens=10, output_tokens=2))
    out = apply_updates(state, {"usage": Usage(input_tokens=5, output_tokens=1)})
    assert out.usage.input_tokens == 15
    assert out.usage.output_tokens == 3


def test_updates_do_not_share_objects_with_the_old_state():
    s0 = AgentState(messages=[Message(role="user", content="hi")])
    s1 = apply_updates(s0, {"messages": Message(role="assistant", content="hello")})

    assert s1.messages[0] is not s0.messages[0]
    s1.messages[0].content = "MUTATED"
    assert s0.messages[0].content == "hi", "a checkpoint taken earlier must stay valid"


def test_a_node_cannot_mutate_state_through_the_object_it_returned():
    returned = Message(role="assistant", content="hello")
    s1 = apply_updates(AgentState(), {"messages": returned})
    returned.content = "MUTATED"
    assert s1.messages[0].content == "hello"


def test_a_badly_typed_update_is_rejected_at_the_merge():
    with pytest.raises(ValidationError):
        apply_updates(AgentState(), {"messages": [{"role": "not a role"}]})


def test_an_empty_update_still_returns_a_copy():
    state = AgentState(messages=[Message(role="user", content="hi")])
    out = apply_updates(state, {})
    assert out is not state
    assert out == state


def test_unknown_fields_are_rejected():
    with pytest.raises(KeyError, match="unknown state fields"):
        apply_updates(AgentState(), {"nonexistent": 1})


def test_state_round_trips_through_json():
    state = AgentState(
        messages=[Message(role="user", content="hi")],
        usage=Usage(input_tokens=10, output_tokens=2),
    )
    assert AgentState.model_validate_json(state.model_dump_json()) == state
    
def test_scalar_fields_replace():
    class Labelled(AgentState):
        label: str = ""

    assert apply_updates(Labelled(label="a"), {"label": "b"}).label == "b"