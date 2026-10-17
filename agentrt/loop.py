"""A tool-calling agent as a plain loop.

This is the obvious implementation and it works. It is replaced at commit 9,
for reasons that are easier to feel than to be told: see docs/why-a-graph.md.
"""

from __future__ import annotations

from .providers.base import Provider
from .tools import InvalidToolArguments, ToolNotFound, ToolRegistry
from .types import Message, ToolResult, Usage


def run_agent(
    provider: Provider,
    registry: ToolRegistry,
    user_message: str,
    system: str | None = None,
    max_steps: int = 20,
) -> tuple[list[Message], Usage]:
    messages: list[Message] = [Message(role="user", content=user_message)]
    usage = Usage()

    for _ in range(max_steps):
        completion = provider.complete(
            messages=messages,
            tools=registry.specs() or None,
            system=system,
        )
        messages.append(completion.message)
        usage = usage + completion.usage

        if not completion.message.tool_calls:
            return messages, usage

        results = []
        for call in completion.message.tool_calls:
            try:
                content = registry.get(call.name).call(call.arguments)
                is_error = False
            except (ToolNotFound, InvalidToolArguments) as exc:
                content, is_error = str(exc), True
            except Exception as exc:  # noqa: BLE001 - tool authors raise anything
                content, is_error = f"{type(exc).__name__}: {exc}", True
            results.append(
                ToolResult(call_id=call.id, name=call.name, content=content, is_error=is_error)
            )

        messages.append(Message(role="tool", tool_results=results))

    raise RuntimeError(f"agent did not finish within {max_steps} steps")