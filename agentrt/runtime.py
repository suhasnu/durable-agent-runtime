"""The executor.

Drives a compiled graph: run a node, apply its updates, work out what's next,
checkpoint, repeat. Every run leaves a checkpoint per transition and ends in a
terminal status, completed or failed.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from .checkpoint import Checkpoint, InMemoryCheckpointer, SqliteCheckpointer
from .graph import END, CompiledGraph
from .providers.base import Provider
from .state import apply_updates
from .tools import ToolRegistry
from .types import Message, new_id


@dataclass
class RunContext:
    """What a node is allowed to reach. Passed in, never imported."""

    provider: Provider
    tools: ToolRegistry
    system: str | None = None
    step: int = 0


RunStatus = Literal["completed", "failed"]


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    state: BaseModel
    steps: int
    error: Exception | None = None
    history: list[str] = field(default_factory=list)

    @property
    def last_message(self) -> Message | None:
        messages = getattr(self.state, "messages", [])
        return messages[-1] if messages else None


class Runtime:
    """Drives a compiled graph, checkpointing after every transition.

    `max_steps` counts node executions, not model turns. One tool-calling
    round is three steps (model, tools, model).
    """

    def __init__(
        self,
        graph: CompiledGraph,
        provider: Provider,
        tools: ToolRegistry | None = None,
        checkpointer: SqliteCheckpointer | None = None,
        system: str | None = None,
        max_steps: int = 20,
    ):
        self.graph = graph
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.system = system
        self.max_steps = max_steps

    def run(self, initial: BaseModel | None = None, run_id: str | None = None) -> RunResult:
        state = initial if initial is not None else self.graph.state_type()
        run_id = run_id or new_id("run")
        self.checkpointer.create_run(run_id, {"state_type": self.graph.state_type.__name__})

        # Step 0 is the state before anything executed. Without it a crash
        # inside the first node leaves nothing to resume from.
        self.checkpointer.put_checkpoint(
            Checkpoint(run_id=run_id, step=0, next_node=self.graph.entry, state=state)
        )
        return self._drive(run_id, state, self.graph.entry, step=0)

    def _drive(self, run_id: str, state: BaseModel, node: str, step: int) -> RunResult:
        history: list[str] = []
        ctx = RunContext(provider=self.provider, tools=self.tools, system=self.system)

        while node != END:
            if step >= self.max_steps:
                error = RuntimeError(f"agent did not finish within {self.max_steps} steps")
                return self._finish(run_id, "failed", state, step, history, error)

            ctx.step = step + 1
            history.append(node)

            # The whole transition, not just the node. A router or a checkpoint
            # write that raises would otherwise escape and leave the run marked
            # "running" forever, which is indistinguishable from a paused run.
            try:
                updates = self.graph.node(node)(state, ctx) or {}

                step += 1
                if "step" in type(state).model_fields:
                    # Optional bookkeeping: maintained when the state declares
                    # it, so a graph with its own state type is not forced to
                    # carry it.
                    updates = {**updates, "step": step}

                # One merge per node, not two. apply_updates deep copies and
                # validates, so a second call doubles the cost of every step.
                state = apply_updates(state, updates)

                next_node = self.graph.next_node(node, state)

                # After the transition is decided, so the saved next_node is
                # where execution actually continues. Saving the node that just
                # ran would re-run it on resume.
                self.checkpointer.put_checkpoint(
                    Checkpoint(run_id=run_id, step=step, next_node=next_node, state=state)
                )
            except Exception as exc:  # noqa: BLE001 - classification is the caller's job
                return self._finish(run_id, "failed", state, step, history, exc)

            node = next_node
            
        return self._finish(run_id, "completed", state, step, history)

    def _finish(
        self,
        run_id: str,
        status: RunStatus,
        state: BaseModel,
        step: int,
        history: list[str],
        error: Exception | None = None,
    ) -> RunResult:
        # If the store is what broke, recording the status will break too, and
        # the caller needs the error that caused the failure, not this one.
        with contextlib.suppress(Exception):
            self.checkpointer.set_run_status(run_id, status)
        return RunResult(
            run_id=run_id, status=status, state=state, steps=step, error=error, history=history
        )