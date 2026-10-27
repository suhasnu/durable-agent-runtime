from agentrt.providers.anthropic import _from_wire, _to_wire
from agentrt.types import Message, ToolCall, ToolResult


def test_tool_results_ride_on_a_user_turn():
    wire = _to_wire(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", tool_calls=[ToolCall(id="t1", name="f", arguments={"a": 1})]),
            Message(role="tool", tool_results=[ToolResult(call_id="t1", name="f", content="ok")]),
        ]
    )
    assert wire[2]["role"] == "user"
    assert wire[2]["content"][0]["type"] == "tool_result"
    assert wire[2]["content"][0]["tool_use_id"] == "t1"


def test_consecutive_tool_results_merge_into_one_turn():
    wire = _to_wire(
        [
            Message(role="tool", tool_results=[ToolResult(call_id="t1", name="f", content="a")]),
            Message(role="tool", tool_results=[ToolResult(call_id="t2", name="g", content="b")]),
        ]
    )
    assert len(wire) == 1
    assert len(wire[0]["content"]) == 2


def test_an_assistant_turn_keeps_text_alongside_its_tool_calls():
    wire = _to_wire(
        [
            Message(
                role="assistant",
                content="let me check",
                tool_calls=[ToolCall(id="t1", name="f", arguments={})],
            )
        ]
    )
    assert [b["type"] for b in wire[0]["content"]] == ["text", "tool_use"]


def test_error_results_carry_the_flag():
    wire = _to_wire(
        [
            Message(
                role="tool",
                tool_results=[ToolResult(call_id="t1", name="f", content="boom", is_error=True)],
            )
        ]
    )
    assert wire[0]["content"][0]["is_error"] is True


def test_system_messages_are_not_sent_as_turns():
    wire = _to_wire([Message(role="system", content="be nice"), Message(role="user", content="hi")])
    assert len(wire) == 1


def test_a_tool_use_response_becomes_tool_calls():
    completion = _from_wire(
        {
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "get", "input": {"k": "v"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 7},
            "model": "claude-haiku-5-5",
        }
    )
    assert completion.message.content == "checking"
    assert completion.message.tool_calls[0].arguments == {"k": "v"}
    assert completion.usage.total == 19


def test_a_plain_text_response_has_no_tool_calls():
    completion = _from_wire(
        {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}
    )
    assert completion.message.tool_calls == []
