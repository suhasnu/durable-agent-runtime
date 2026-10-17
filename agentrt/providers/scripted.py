"""A provider that returns a fixed script of completions.

Every test in this repository runs against this. A suite that needs an API key
and a network is not a test suite, it is a monitoring system, and it gets
skipped in CI within a month.

This is also the starting point for replay: a replay provider is this class
with the script loaded from a recorded trace instead of passed in.
"""

from __future__ import annotations

from typing import Any

from ..types import Completion, Message, ToolCall, Usage


class ScriptedProvider:
    name = "scripted"

    def __init__(self, script: list[Completion | Exception], model: str = "scripted-1"):
        self.model = model
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self._index = 0

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 2048,
    ) -> Completion:
        self.requests.append({"messages": list(messages), "tools": tools, "system": system})
        if self._index >= len(self.script):
            raise AssertionError(
                f"scripted provider exhausted after {self._index} calls; the agent "
                f"made more provider calls than the test expected"
            )
        item = self.script[self._index]
        self._index += 1
        if isinstance(item, Exception):
            raise item
        # Copy rather than assign, so the script itself is never mutated.
        return item.model_copy(update={"model": self.model})


def says(text: str, *, input_tokens: int = 10, output_tokens: int = 5) -> Completion:
    return Completion(
        message=Message(role="assistant", content=text),
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def calls(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call_1",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> Completion:
    """Build a completion that makes one tool call.

    Every call defaults to call_id="call_1". A script with more than one tool
    calling turn must pass call_id explicitly to each, or tool results can be
    matched to the wrong call.
    """
    return Completion(
        message=Message(
            role="assistant",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        ),
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="tool_use",
    )