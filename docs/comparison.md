# agentrt and LangGraph, compared

The same two-node agent was built twice: once on agentrt, once on LangGraph in
`examples/langgraph_port.py`. Both use the same tools, wire types and scripted
provider, so the only difference is the executor and its store. Everything
below was observed by running code, against langgraph 1.2.19,
langgraph-checkpoint 4.3.1 and langgraph-checkpoint-sqlite 3.1.3 on Python
3.12.10. Where a claim rests on something not exercised here, it says so.

The crash results are recorded in full in
[langgraph-crash.md](langgraph-crash.md).

## What LangGraph gave for free

**The executor.** The port is 149 lines, a demo included. On the agentrt side
the same ground is covered by `graph.py`, `runtime.py` and `checkpoint.py`, 855
lines. That is not a fair comparison line for line, because agentrt's figure
includes the write-ahead log and reconciliation, which LangGraph does not have.
But graph validation, routing, the step loop and checkpointing came with the
import.

**A checkpoint store with history.** `SqliteSaver` created its own schema,
keyed runs by `thread_id`, and `get_state_history` returned every step. It
switched the database to WAL mode on its own and left `synchronous` at FULL,
SQLite's default, which ends up where agentrt's hand-set pragmas do.

**A choice of durability.** `durability="sync"`, `"async"` or `"exit"` trades
crash safety for less waiting on writes. agentrt has one setting. The modes
behaved differently under a crash, which is covered under disagreements below.

**Interrupts.** `interrupt(payload)` paused the run and `Command(resume=value)`
continued it, with an id on each interrupt that agentrt does not have. The
design turned out to be the same as agentrt's, see below.

**More than this port used.** The installed package exports `Send`,
`RetryPolicy`, `CachePolicy` and several stream modes on `invoke`. None of them
were exercised here, so this page makes no claim about how they behave.

## What it cost

**35 packages.** A clean environment with `.[dev]` held 14 packages. Adding
`.[langgraph]` made it 49, including `httpx`, `requests`, `websockets`,
`langsmith` and `langchain-core`, for an agent that never opens a socket.
agentrt depends on pydantic.

**A store that trusts its rows.** By default LangGraph rebuilds whatever class
a checkpoint names, and only warns: "Deserializing unregistered type
agentrt.types.Message from checkpoint. This will be blocked in a future
version." The port passes `allowed_msgpack_modules` to make the list explicit.
agentrt cannot do this, because the caller passes `state_type` and the row
never names a class.

**State you cannot read with sqlite3.** LangGraph stores checkpoints as
msgpack `BLOB`s. agentrt stores JSON in a `TEXT` column, which any SQLite
client can query. This mattered in practice: working out what the crash had
left on disk meant going through LangGraph's own API, `get_state`, and in async
mode that API was wrong (next section).

**No run status.** The store has two tables, `checkpoints` and `writes`. A node
that raises propagates out of `invoke` as the original exception, and the
error is recorded only on the pending task. Whether a thread finished, failed,
crashed or is paused has to be worked out from the checkpoints and the pending
tasks. agentrt keeps a `runs` row with a status, and `run()` returns a
`RunResult` rather than raising.

**Smaller frictions.**
- Nodes receive only the state, so the port closes over the provider and the
  tools. agentrt passes a `RunContext`.
- LangGraph resolves the state's type hints through `sys.modules`. A module
  loaded from a file path failed with `NameError: Annotated` until it was
  registered there.
- Every run writes two checkpoints before the first node, step -1 for the raw
  input and step 0 once it is applied. agentrt writes one.
- `recursion_limit` raises `GraphRecursionError` when hit. A limit of 5
  allowed 3 model calls. agentrt's `max_steps` returns a failed
  `RunResult`, and `Budget.max_model_calls` counts model calls directly.

## What it does not do

**Record a side effect before it happens.** This is the result that matters.
With a process killed after `charge_card` landed, resuming from the same file
charged the card again in every run of every mode that could resume:

| durability | charges after resume | model called again |
| --- | --- | --- |
| `sync` | 2, five runs out of five | never |
| `async`, the default | 2, five runs out of five | in two runs out of five |
| `exit` | the run could not be resumed, `EmptyInputError`. Starting over from the input charged again | |
| agentrt | 1, and the resume refused with `needs_resolution` | |

Nothing in LangGraph's store recorded that the tool started. After the crash
the pending `tools` task showed `error=None`, which is exactly how it looks if
the tool had never run. With nothing to tell the two apart, LangGraph runs the
node again.

**Tell a caller which tools are safe to repeat.** There are no flags for
side-effecting or idempotent tools, no refusal, and no operator step like
`resolve_invocation`. Making a tool safe to run twice is left to its author.

**Report pending work honestly after an async crash.** In probe runs,
`get_state` after an async crash reported step -1 or step 0 with `next=()`,
which reads as nothing left to do. `invoke(None)` then resumed the run and
charged the card.

## Where the designs genuinely disagree

**When the record is written.** LangGraph records what a step returned, after
it returns. agentrt records what a tool is about to do, before it does it, and
what it did once it is done. Checkpointing after each step moves the crash
window, it does not close it, and the crash test shows the window: it is the
whole tools node.

**Whether durability is a dial.** LangGraph treats durability as a performance
setting and defaults to the faster one. The cost of `async` showed up as a
repeated model call in two runs out of five, a call already paid for, and
`exit` lost the run entirely. agentrt has no dial: every checkpoint and every
invocation record commits with `synchronous=FULL` before execution moves on.
Whether the dial is worth having depends on whether a repeated model call
costs more than the waiting. agentrt never measured the waiting, so this page
does not either.

**Who owns a crash in the middle of a side effect.** LangGraph leaves it to
the tool author. agentrt makes it the runtime's job, and only lets a tool be
repeated when the tool says that is safe. LangGraph's approach is simpler, and
it is correct whenever tools are idempotent. agentrt's approach costs a table
and a reconciliation step, and it is the only one of the two that charged the
card once.

**How a failure reaches the caller.** LangGraph raises. agentrt returns a
result and leaves a terminal status in the database, so a crashed process and
a failed run can be told apart later by reading one row.

**What a misplaced resume does.** `Command(resume="again")` on a thread that
had already finished returned the finished state and did nothing. agentrt's
`resume(run_id, value)` raises `ValueError` when the run is not paused,
because a value with no question waiting for it is almost always a bug in the
caller.

## Where they agree

Interrupts. Both re-enter the interrupting node from the top on resume, so one
node does both the asking and the answering. In the LangGraph probe the node
ran twice, once to ask and once to answer, and pausing wrote no checkpoint of
its own: the history was steps -1, 0 and 1 while paused, and step 2 appeared
only after the resume. That is commit 17's design. It means the roadmap entry
"A node's work before an interrupt runs again on resume" applies to LangGraph
too, and so does its advice to put the interrupt check at the top of the node.

## Summary

LangGraph is the more complete executor. It has more built in, a store that
needed no setup, and an interrupt design that matched this project's. It does not
attempt the problem agentrt exists for. The crash test is a fair statement of
the difference: same agent, same crash, two charges against one.
