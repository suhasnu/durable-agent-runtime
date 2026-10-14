from agentrt.types import Message, ToolCall, Usage


def test_usage_adds():
    total = Usage(input_tokens=10, output_tokens=2) + Usage(input_tokens=5, output_tokens=1)
    assert total.input_tokens == 15
    assert total.total == 18


def test_a_message_round_trips_through_json():
    original = Message(
        role="assistant",
        tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})],
    )
    restored = Message.model_validate_json(original.model_dump_json())
    assert restored == original