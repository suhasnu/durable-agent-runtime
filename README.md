# durable-agent-runtime

[![ci](https://github.com/suhasnu/durable-agent-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/suhasnu/durable-agent-runtime/actions/workflows/ci.yml)

An agent runtime with durable execution, written from scratch. No LangGraph, no
CrewAI. The dependencies are pydantic and the standard library's SQLite.

## The problem

An agent is a loop: ask the model, run the tool it asked for, feed the result
back, repeat. The loop is twenty lines. What makes it hard is that the process
can die at any point in it.

Say the agent calls a tool that charges a customer's card. The process is killed
after the charge goes through but before the result is written down. On restart,
nothing in the database says the charge happened, so a naive runtime calls the
tool again and the customer is charged twice.

Checkpointing after each step does not fix this. It moves the window, it does
not close it. The window is between the effect happening and the record of it
being saved, and no amount of saving afterwards can cover a crash that lands
inside it.

## The approach

Record the intent before acting, not the outcome after.

Before a tool runs, the runtime commits a `started` row to SQLite. After it
returns, it commits a `completed` row with the result. On resume, every call is
in exactly one of three states:

| what the log holds | what it means | what the runtime does |
| --- | --- | --- |
| no record | the tool never ran | run it |
| `completed` | it ran and the result is saved | replay the saved result, never call again |
| `started`, no `completed` | it may or may not have run | cannot be decided from the log |

The third row is the honest one, and it is the reason the design exists. The
runtime does not guess. It resolves the call only when the tool itself says it
can be resolved, because the tool was declared non-side-effecting or idempotent.
Otherwise it refuses to resume and raises `UncertainSideEffect`, leaving the run
for a human to settle.

The full argument, including why the ordering of each write is what it is, is in
[docs/durability.md](docs/durability.md).

## See it crash

The demo kills a process mid-run and resumes it in a new one. It uses a scripted
model provider, so it needs no API key and no network.

```
$ python examples/crash_and_resume.py

================================================================
1. run, and die mid-charge
================================================================
process killed mid-tool
charges in ledger: 1

================================================================
2. what the database knows
================================================================
  charge_card  call=call_1  status=started

  a started record with no completed record: the outcome is unknown

================================================================
3. a naive resume would charge again, so this one refuses
================================================================
  refused: tool 'charge_card' (call call_1) may have executed before the crash; outcome unknown. Mark the tool idempotent, or resolve this run manually.

  run status: needs_resolution
charges in ledger: 1  (still one)

================================================================
4. an operator confirms what happened, then the run continues
================================================================
  status: completed
  final:  Order complete. The card was charged once.

charges in ledger: 1  (still one)

================================================================
5. the same crash, but the tool is declared idempotent
================================================================
process killed mid-tool again

  resumed with no human involved: completed
  set_order_status ran 2 times, which its author declared safe

================================================================
the point
================================================================
Two crashes, two different answers. The charge happened exactly once
and needed a human, because nothing could tell whether it had gone
through. The status update simply ran again, because its author
declared that safe. The runtime never guessed either way.
```

## Quick start

```
pip install -e ".[dev]"
pytest
```

The test suite runs against a scripted provider rather than a live API. A suite
that needs a key and a network is not a test suite, it is a monitoring system,
and it gets skipped within a month. All 116 tests run offline.

For a real call against the Anthropic API, set `ANTHROPIC_API_KEY` and run
`python examples/live.py`.

## What is in it

- **Tools** declared with a decorator. The JSON schema the model sees is derived
  from the Python signature, so the schema cannot drift from the function.
- **Typed state** with reducers, so a node returns what it changed rather than
  the whole state, and the merge rule lives on the field.
- **A state graph** validated at compile time: unknown edge targets, unreachable
  nodes and nodes that can never reach the end are errors before anything runs.
- **A checkpoint store** that is insert-only, so the history of a run is a
  record you can walk, not just the last known position.
- **An error taxonomy** in the type system. Every failure answers one question,
  retry or stop, and `RetryableError`, `FatalError` and `UncertainSideEffect`
  are how it answers.
- **Retries at the runtime, not the provider.** The provider classifies a
  failure, the runtime decides what to do about it, so two retry layers can
  never multiply.
- **Truncation treated as a failure.** A response cut off at the token ceiling
  is not a finished answer, and a tool call cut off halfway can still parse, so
  the run fails rather than acting on it.
- **Budgets checked before a call, not after.** The check reserves what the next
  call may cost, because by the time a call returns, the money is spent.

## Not done yet

Human-in-the-loop interrupts, so a run can pause for approval and resume later.
Then the same agent ported to LangGraph, kept alongside this one, with a written
comparison of what the framework gives you and what it costs.

Known gaps and deferred decisions, each with the condition that would make it
worth fixing, are in [docs/roadmap.md](docs/roadmap.md).

## Why build it

Because using a framework and understanding one are different things, and the
difference shows up the first time something fails in a way the documentation
does not cover.