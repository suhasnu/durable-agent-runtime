"""A tool-calling agent, assembled from the pieces.

Two nodes: ask the model, run what it asked for. Note how little is here.
Everything hard lives in the runtime and applies to any graph.
"""

from __future__ import annotations

from typing import Any

from .graph import END, StateGraph
from .runtime import RunContext, execute_tool_calls
from .state import AgentState
from .types import Message


def model_node(state: AgentState, ctx: RunContext) -> dict[str, Any]:
    completion = ctx.provider.complete(
        messages=state.messages,
        tools=ctx.tools.specs() or None,
        system=ctx.system,
    )
    return {"messages": completion.message, "usage": completion.usage}


def tools_node(state: AgentState, ctx: RunContext) -> dict[str, Any]:
    last = state.messages[-1]
    if not last.tool_calls:
        return {}
    results = execute_tool_calls(ctx, last.tool_calls)
    return {"messages": Message(role="tool", tool_results=results)}


def route_after_model(state: AgentState) -> str:
    return "tools" if state.messages[-1].tool_calls else END


def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.set_entry("model")
    graph.add_conditional_edge("model", route_after_model, {"tools", END})
    graph.add_edge("tools", "model")
    return graph