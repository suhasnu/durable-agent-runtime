"""The executor.

Drives a compiled graph: run a node, apply its updates, work out what's next,
checkpoint, repeat. Every run leaves a checkpoint per transition and ends in a
terminal status, completed or failed.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from .checkpoint import Checkpoint, InMemoryCheckpointer, SqliteCheckpointer, hash_args
from .errors import FatalError, RetryableError, UncertainSideEffect
from .graph import END, CompiledGraph
from .providers.base import Provider
from .state import apply_updates
from .tools import InvalidToolArguments, ToolNotFound, ToolRegistry
from .types import Message, ToolCall, ToolResult, new_id


@dataclass
class RetryPolicy:
    """How a node retries after a transient failure.

    Retrying lives here rather than in the provider so that attempts are
    visible to the run, and so the two layers cannot multiply.
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    backoff: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")


@dataclass
class RunContext:
    """What a node is allowed to reach. Passed in, never imported."""

    provider: Provider
    tools: ToolRegistry
    run_id: str
    checkpointer: SqliteCheckpointer
    system: str | None = None
    step: int = 0
    retries: int = 0


RunStatus = Literal["completed", "failed"]


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    state: BaseModel
    steps: int
    error: Exception | None = None
    retries: int = 0
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
        retry: RetryPolicy | None = None,
    ):
        self.graph = graph
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.checkpointer = checkpointer or InMemoryCheckpointer()
        self.system = system
        self.max_steps = max_steps
        self.retry = retry or RetryPolicy()

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

    def _run_node_with_retry(self, node: str, state: BaseModel, ctx: RunContext) -> dict[str, Any]:
        """Run a node, retrying transient failures.

        Retrying a node is safe even when it has side effects: the write-ahead
        log replays completed tool calls from the previous attempt instead of
        executing them again.
        """
        delay = self.retry.initial_delay
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                return self.graph.node(node)(state, ctx) or {}
            except RetryableError as exc:
                if attempt == self.retry.max_attempts:
                    raise
                ctx.retries += 1
                wait = exc.retry_after if exc.retry_after is not None else delay
                time.sleep(min(wait, self.retry.max_delay))
                delay = min(delay * self.retry.backoff, self.retry.max_delay)
        raise AssertionError("unreachable")

    def resume(self, run_id: str) -> RunResult:
        """Continue a run from its last checkpoint.

        Everything comes from the checkpoint row: the state, the step, and the
        node to run next. Nothing is taken from this Runtime object, which is
        what lets a fresh process pick up a run the previous one died in.
        """
        checkpoint = self.checkpointer.latest_checkpoint(run_id, self.graph.state_type)
        if checkpoint is None:
            raise KeyError(f"no checkpoint for run {run_id!r}")

        if checkpoint.next_node == END:
            # Already finished. Flipping the status to running and back would
            # make a finished run briefly look like one being driven.
            return RunResult(
                run_id=run_id,
                status="completed",
                state=checkpoint.state,
                steps=checkpoint.step,
            )

        self._reconcile_dangling(run_id)

        self.checkpointer.set_run_status(run_id, "running")
        return self._drive(run_id, checkpoint.state, checkpoint.next_node, step=checkpoint.step)

    def _reconcile_dangling(self, run_id: str) -> None:
        """Decide what to do about tools that started and never reported back.

        A `started` record with no `completed` record means the process died
        between committing the intent and committing the outcome, so the effect
        may or may not have happened and nothing in the database can say which.

        Three cases, and only three:
          not side-effecting  -> re-running is free, discard the record
          idempotent          -> the author promised twice equals once, discard
          otherwise           -> refuse; a human decides via resolve_invocation
        """
        for record in self.checkpointer.dangling_invocations(run_id):
            try:
                tool = self.tools.get(record.tool_name)
            except ToolNotFound:
                # Without the tool's definition there is no way to read its
                # flags, so there is no basis for calling it safe to retry.
                self.checkpointer.set_run_status(run_id, "needs_resolution")
                raise UncertainSideEffect(record.tool_name, record.call_id) from None

            if not tool.side_effecting or tool.idempotent:
                self.checkpointer.discard_invocation(run_id, record.call_id)
                continue

            self.checkpointer.set_run_status(run_id, "needs_resolution")
            raise UncertainSideEffect(record.tool_name, record.call_id)

    def _drive(self, run_id: str, state: BaseModel, node: str, step: int) -> RunResult:
        history: list[str] = []
        ctx = RunContext(
            provider=self.provider,
            tools=self.tools,
            run_id=run_id,
            checkpointer=self.checkpointer,
            system=self.system,
        )

        while node != END:
            if step >= self.max_steps:
                error = RuntimeError(f"agent did not finish within {self.max_steps} steps")
                return self._finish(run_id, "failed", state, step, history, error, ctx.retries)

            ctx.step = step + 1
            history.append(node)

            # The whole transition, not just the node. A router or a checkpoint
            # write that raises would otherwise escape and leave the run marked
            # "running" forever, which is indistinguishable from a paused run.
            try:
                updates = self._run_node_with_retry(node, state, ctx)
                step += 1
                state = apply_updates(state, updates)

                next_node = self.graph.next_node(node, state)

                # After the transition is decided, so the saved next_node is
                # where execution actually continues. Saving the node that just
                # ran would re-run it on resume.
                self.checkpointer.put_checkpoint(
                    Checkpoint(run_id=run_id, step=step, next_node=next_node, state=state)
                )
            except Exception as exc:  # noqa: BLE001 - classification is the caller's job
                return self._finish(run_id, "failed", state, step, history, exc, ctx.retries)

            node = next_node

        return self._finish(run_id, "completed", state, step, history, retries=ctx.retries)

    def _finish(
        self,
        run_id: str,
        status: RunStatus,
        state: BaseModel,
        step: int,
        history: list[str],
        error: Exception | None = None,
        retries: int = 0,
    ) -> RunResult:
        # If the store is what broke, recording the status will break too, and
        # the caller needs the error that caused the failure, not this one.
        with contextlib.suppress(Exception):
            self.checkpointer.set_run_status(run_id, status)
        return RunResult(
            run_id=run_id,
            status=status,
            state=state,
            steps=step,
            error=error,
            history=history,
            retries=retries,
        )


def execute_tool_calls(ctx: RunContext, calls: list[ToolCall]) -> list[ToolResult]:
    """Run tool calls with write-ahead logging.

    The ordering is the point:

      1. look up a prior completed record   -> makes resume idempotent for free
      2. write `started` and COMMIT         -> survives a crash in step 3
      3. execute the tool                   -> the dangerous part
      4. write `completed` and COMMIT       -> closes the window

    Doing 2 after 3 is the obvious implementation and it is wrong: the crash
    window it leaves is exactly the window where the effect happens.
    """
    results: list[ToolResult] = []

    for call in calls:
        prior = ctx.checkpointer.get_invocation(ctx.run_id, call.id)

        if prior is not None and prior.status == "completed" and prior.result is not None:
            if prior.args_hash != hash_args(call.name, call.arguments):
                # Same call id, different arguments. Replaying the stored result
                # would answer a question nobody asked, silently.
                raise FatalError(
                    f"call id {call.id!r} was already used for {prior.tool_name!r} with "
                    f"different arguments; its logged result does not belong to this call"
                )
            results.append(
                ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=prior.result,
                    is_error=prior.is_error,
                )
            )
            continue

        if prior is not None:
            # Unreachable in the resume path: _reconcile_dangling has already
            # discarded or refused every started record. A safety net, so a bug
            # in reconciliation surfaces here rather than as a repeated effect.
            raise UncertainSideEffect(prior.tool_name, prior.call_id)

        try:
            tool = ctx.tools.get(call.name)
        except ToolNotFound as exc:
            results.append(
                ToolResult(call_id=call.id, name=call.name, content=str(exc), is_error=True)
            )
            continue

        ctx.checkpointer.begin_invocation(ctx.run_id, call.id, call.name, call.arguments)

        try:
            content = tool.call(call.arguments)
            is_error = False
        except InvalidToolArguments as exc:
            content, is_error = str(exc), True
        except Exception as exc:  # noqa: BLE001 - tool authors raise anything
            content, is_error = f"{type(exc).__name__}: {exc}", True

        ctx.checkpointer.complete_invocation(ctx.run_id, call.id, content, is_error)
        results.append(
            ToolResult(call_id=call.id, name=call.name, content=content, is_error=is_error)
        )

    return results
