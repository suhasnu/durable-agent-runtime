"""The same two-node agent, built on LangGraph instead of agentrt's runtime.

Model and tools nodes, a conditional edge, a checkpoint per step. The provider,
the tools and the wire types are agentrt's, so this runs offline against a
ScriptedProvider and the only thing swapped out is the executor and its store.

What is deliberately not ported is the write-ahead log. The tools node calls
each tool directly, the way a LangGraph tool node does, so that a crash shows
what LangGraph itself does about side effects. See
examples/crash_and_resume_langgraph.py.

    pip install -e ".[langgraph]"
    python examples/langgraph_port.py
"""

from __future__ import annotations

import operator
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agentrt.errors import TruncatedResponse
from agentrt.providers import ScriptedProvider, calls, says
from agentrt.providers.base import Provider
from agentrt.runtime import Budget
from agentrt.tools import InvalidToolArguments, ToolNotFound, ToolRegistry, tool
from agentrt.types import Message, ToolResult, Usage


class LGState(TypedDict):
    # operator.add on lists is LangGraph's append reducer, and on Usage it is
    # Usage.__add__, so both fields merge the way agentrt's reducers do.
    messages: Annotated[list[Message], operator.add]
    usage: Annotated[Usage, operator.add]


def build_langgraph_agent(
    provider: Provider,
    tools: ToolRegistry,
    checkpointer: SqliteSaver | None = None,
    *,
    system: str | None = None,
    max_tokens: int = 2048,
    budget: Budget | None = None,
):
    """Compile the agent. Provider and tools are closed over, since a LangGraph
    node receives only the state."""
    budget = budget or Budget()

    def model_node(state: LGState) -> dict[str, Any]:
        budget.check(state["usage"], max_tokens)
        completion = provider.complete(
            messages=state["messages"],
            tools=tools.specs() or None,
            system=system,
            max_tokens=max_tokens,
        )
        if completion.stop_reason == "max_tokens":
            raise TruncatedResponse(max_tokens)
        return {
            "messages": [completion.message],
            "usage": completion.usage.model_copy(update={"calls": 1}),
        }

    def tools_node(state: LGState) -> dict[str, Any]:
        results = []
        for call in state["messages"][-1].tool_calls:
            try:
                content, is_error = tools.get(call.name).call(call.arguments), False
            except (ToolNotFound, InvalidToolArguments) as exc:
                content, is_error = str(exc), True
            except Exception as exc:  # noqa: BLE001 - tool authors raise anything
                content, is_error = f"{type(exc).__name__}: {exc}", True
            results.append(
                ToolResult(call_id=call.id, name=call.name, content=content, is_error=is_error)
            )
        return {"messages": [Message(role="tool", tool_results=results)]}

    def route_after_model(state: LGState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(LGState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model, ["tools", END])
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=checkpointer)


def sqlite_saver(path: str | Path = ":memory:") -> SqliteSaver:
    """A SqliteSaver that is allowed to load agentrt's state types.

    LangGraph's default serializer loads any class named in a checkpoint and
    warns that this will be blocked in a later version. Naming the types here
    keeps the port working once it is. ToolCall and ToolResult are not listed
    because they travel inside Message and pydantic rebuilds them.
    """
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("agentrt.types", "Message"), ("agentrt.types", "Usage")]
    )
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False), serde=serde)


def initial_state(prompt: str) -> LGState:
    return {"messages": [Message(role="user", content=prompt)], "usage": Usage()}


def thread(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


@tool
def lookup(key: str) -> str:
    """Look something up."""
    return f"value of {key}"


def main() -> None:
    saver = sqlite_saver()
    agent = build_langgraph_agent(
        ScriptedProvider([calls("lookup", {"key": "a"}), says("a is 'value of a'")]),
        ToolRegistry([lookup]),
        saver,
    )
    final = agent.invoke(initial_state("what is a?"), thread("demo"))

    for message in final["messages"]:
        text = (
            message.content
            or [c.name for c in message.tool_calls]
            or [r.content for r in message.tool_results]
        )
        print(f"  {message.role:<9} {text}")
    print(f"\n  usage: {final['usage']}")
    steps = [s.metadata["step"] for s in agent.get_state_history(thread("demo"))]
    print(f"  checkpoints (steps, newest first): {steps}")


if __name__ == "__main__":
    main()
