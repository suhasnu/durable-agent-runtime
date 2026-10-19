"""The executor.

Drives a compiled graph: run a node, apply its updates, work out what's next,
repeat. Persistence arrives at commit 10; this is the shape it hangs off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from .graph import END, CompiledGraph
from .providers.base import Provider
from .state import apply_updates
from .tools import ToolRegistry
from .types import Message


@dataclass
class RunContext:
    """What a node is allowed to reach. Passed in, never imported."""

    provider: Provider
    tools: ToolRegistry
    system: str | None = None
    step: int = 0


@dataclass
class RunResult:
    state: BaseModel
    steps: int
    history: list[str] = field(default_factory=list)

    @property
    def last_message(self) -> Message | None:
        messages = getattr(self.state, "messages", [])
        return messages[-1] if messages else None


class Runtime:
    """Drives a compiled graph.

    `max_steps` counts node executions, not model turns. One tool-calling
    round is three steps (model, tools, model), so the ceiling is roughly
    half what the same number meant in the old loop.
    """
    def __init__(
        self,
        graph: CompiledGraph,
        provider: Provider,
        tools: ToolRegistry | None = None,
        system: str | None = None,
        max_steps: int = 20,
    ):
        self.graph = graph
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.system = system
        self.max_steps = max_steps

    def run(self, initial: BaseModel | None = None) -> RunResult:
        state = initial if initial is not None else self.graph.state_type()
        node = self.graph.entry
        history: list[str] = []
        ctx = RunContext(provider=self.provider, tools=self.tools, system=self.system)

        step = 0
        while node != END:
            if step >= self.max_steps:
                raise RuntimeError(f"agent did not finish within {self.max_steps} steps")

            ctx.step = step + 1
            history.append(node)

            updates = self.graph.node(node)(state, ctx) or {}
            step += 1
            if "step" in type(state).model_fields:
                # Optional bookkeeping: maintained when the state declares it,
                # so a graph with its own state type is not forced to carry it.
                updates = {**updates, "step": step}

            # One merge per node, not two. apply_updates deep copies and
            # validates, so a second call doubles the cost of every step.
            state = apply_updates(state, updates)
            node = self.graph.next_node(node, state)

        return RunResult(state=state, steps=step, history=history)