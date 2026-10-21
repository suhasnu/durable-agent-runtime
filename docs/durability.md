# Durability

This runtime makes one claim: **a run can be killed at any instant and resumed
without losing work or repeating a side effect.**

Everything below is what that costs.


## Why state had to stop being local variables

The first version of this agent was a `while` loop. It worked, and it could not
be resumed, because the conversation lived in a local variable and the position
in the run was the interpreter's program counter. When the process died, there
was nothing to write down. [docs/why-a-graph.md](why-a-graph.md) covers that
change; this document picks up after it.

## Where the checkpoint goes

Each step of a run does this, in this order:

run the node -> apply its updates -> decide the next node -> CHECKPOINT -> continue

The placement is the whole thing. Checkpointing before the node runs would mean
a resume runs that node again, which for a tool means a second email. Writing it
after the node but before the next node is chosen would save what happened
without saving what to do about it.

Step 0 is written before anything executes, holding the initial state with
`next_node` set to the entry node. Without it, a crash inside the very first
node leaves no row at all and the run is unrecoverable from its first instant.

Rows are inserted, never updated. Step 5's row keeps describing step 5 forever,
which is why `put_checkpoint` uses a plain `INSERT` and a duplicate step raises
rather than overwriting. That is also what makes `checkpoint_at(run_id, step)`
useful for more than crash recovery.

## The part that is actually hard

Checkpointing solves resumption. It is also what causes the double charge.

Suppose a tool charges a card and the process dies before the next checkpoint
commits. Resume loads the last good checkpoint, which is from *before* the
charge, and runs the tool again. The customer is charged twice.

No amount of checkpointing fixes this. Nothing was lost. The world changed and
the database did not find out.

## Recording intent instead of outcome

The fix is to write down what is *about to* happen, and commit it, before doing
it. `execute_tool_calls` does this for every tool call:

1. look for an existing record for this `call_id`
2. `INSERT status='started'` and commit
3. execute the tool          <- the crash window
4. `UPDATE status='completed'` with the result, and commit

Step 2 before step 3 is the entire design. Writing the record after the tool
would leave no trace of a call that died mid-flight, which is exactly the call a
resume needs to know about.

A crash now leaves one of three states on disk:

| What is on disk | What it means | What resume does |
| --- | --- | --- |
| no record | the tool never started | run it |
| `started`, no `completed` | **unknown** | see below |
| `completed` with a result | it finished | replay the stored result |

The third row is a free bonus. Because a completed record is replayed rather
than re-executed, a resumed run never repeats work it already paid for, even for
tools with no side effects at all. A crash between step 4 and the checkpoint is
therefore harmless: the result is already on disk and the resume reads it back.

Replaying compares `args_hash` before trusting a stored result. Two calls
sharing an id with different arguments would otherwise return an answer to a
question nobody asked, silently.

## The row that cannot be decided

A `started` record with no `completed` record is genuinely undecidable. The
process may have died before the API call, during it, or after it returned but
before the commit. Nothing in the database distinguishes these, and nothing can,
because the information lives in a system this runtime does not control.

So it does not guess. `Runtime._reconcile_dangling` runs at the start of every
resume and has exactly three branches:

- **the tool has no side effects**, so re-running it is free: the record is
  discarded and the call is treated as never attempted
- **the tool is declared `idempotent`**, so its author has stated that running
  it twice is the same as running it once: the record is discarded too
- **anything else**: the run is marked `needs_resolution` and the resume
  refuses with `UncertainSideEffect`

A fourth case behaves like the third: if the tool is no longer in the registry,
its flags cannot be read, so there is no basis for calling it safe.

The runtime cannot work out which case a tool belongs to. Only the person who
wrote it can, which is why `side_effecting` and `idempotent` are declared on the
tool rather than inferred.

The way out of a refusal is a human. `resolve_invocation(run_id, call_id,
result)` records what actually happened, after checking the payment log or
whatever the equivalent is, and the next resume replays that result and carries
on. Runs waiting for this are listed by `list_runs(status="needs_resolution")`.

## Two details that are easy to get wrong

**`synchronous=FULL`.** SQLite's default lets a committed transaction sit in the
operating system's buffers for a while. That is fine for a cache and wrong here,
because the entire argument above is "the record reached disk before the tool
ran", and a record in a buffer has not reached disk. The cost is write
throughput, which a few hundred writes per run can afford.

**Discard, do not complete.** Reclaiming a record means deleting it. The first
version of `_reconcile_dangling` marked reclaimed records `completed` with an
empty result, which looks equivalent and is not: the next pass finds a completed
record, replays it, and hands the model an empty string where a tool result
belongs. The tool never runs and nothing reports an error.
`test_reconciliation_discards_rather_than_completing_the_record` is the test
that catches it.

## What this does not solve

- **Tool granularity.** A tool that makes three API calls internally and dies
  after the second is one invocation here. Splitting that up is the tool's
  problem, not the runtime's.
- **Concurrent resumes.** Nothing stops two processes driving the same run. Both
  would execute the same node and the slower one would overwrite the winner's
  status. This needs a lease on the run, plus a way to recover a run whose
  driver was killed.
- **Provider idempotency keys.** The runtime tracks `call_id` locally but does
  not pass one to the provider's own idempotency mechanism. That is what would
  make the `started` case *decidable* rather than merely detectable, and it is
  the single change that would shrink the hardest branch above.
- **`idempotent` is a promise, not a check.** The runtime takes the tool
  author's word for it. A tool wrongly marked idempotent will be re-run after a
  crash, and nothing here will notice.
- **Stored state has no schema version.** Adding a required field to a state
  model makes every checkpoint written before that change unreadable.

## The tests that hold this up

- `test_a_tool_does_not_run_twice_when_a_crash_lands_inside_tools_node`, the
  failure this whole design exists for
- `test_a_completed_invocation_is_replayed_not_re_executed`, the free bonus
- `test_reconciliation_discards_rather_than_completing_the_record`, the trap
- `test_an_operator_can_resolve_a_dangling_call_and_the_run_continues`, the way
  out of a refusal
- `test_a_run_survives_a_new_process`, a reopened database, a fresh `Runtime`,
  and the run carries on

All of them run against a scripted provider, with no network and no API key.
`examples/crash_and_resume.py` shows the same thing as a narrative.