# Roadmap

Known gaps that were deferred on purpose. Each one is small today and becomes
a real problem once a specific feature lands.

## Scripted provider stores shallow copies

`ScriptedProvider.complete` records `list(messages)`, which copies the list but
not the `Message` objects inside it. If the agent loop mutates a message after
sending it, the recorded request changes too, and assertions on the request log
check the mutated version instead of what was actually sent. Script items are
also returned by reference, so a test that reuses a `says(...)` object, or a
runtime that edits a completion, leaks changes back into the script.

Fix when: the loop starts mutating message history, or tests start asserting on
the request log in detail. Options are `model_copy(deep=True)` on record and
return, or frozen pydantic models.

## Parallel tool calls in `calls()`

`calls()` builds a completion with exactly one `ToolCall`. Providers can return
several tool calls in one turn, and the loop will need to execute them and
return all results before the next completion.

Fix when: the agent loop handles more than one tool call per turn. Likely shape
is a helper that takes several `(name, arguments, call_id)` entries.

## mypy for Protocol conformance

Nothing checks that `ScriptedProvider` satisfies `Provider`. Because `Provider`
is a structural `Protocol`, a signature drift in either one fails silently until
a real call breaks.

Fix when: a second provider exists, or before CI is set up. Add mypy (or
pyright) to dev dependencies and CI, plus a static assignment such as
`_: Provider = ScriptedProvider([])` so the check has something to verify.

## The agent graph never checks `stop_reason`

`route_after_model` sends any completion without tool calls to `END`. A
completion that stopped because it hit the token limit is treated as if the
model chose to stop, so a truncated answer reads as a complete one. A
truncation in the middle of a tool call can also leave its arguments
incomplete, and `tools_node` would then run the call with them.

Fix when: commit 15, since fixing it properly needs `max_tokens` plumbed
through the provider call (next entry). The router should branch on
`stop_reason` and treat `max_tokens` as its own outcome, not as success.

## `model_node` never passes `max_tokens`

`model_node` calls `ctx.provider.complete` without `max_tokens`, so every run
uses the provider default of 2048. Callers have no way to raise the limit for
long answers or lower it to cap cost.

Fix when: alongside the `stop_reason` check, since the two interact. Carry
`max_tokens` on `Runtime` and `RunContext`, and have `model_node` pass it
through.

## Tool results on the final step are discarded

If the step budget runs out right after `tools_node`, the tools have already
executed, side effects included, but the runtime raises the step limit error
before `model_node` runs again. The results never reach the model. The work
happened, but nothing downstream knows it did.

This is the side effect problem that commits 13 and 14 address.

## Hitting `max_steps` discards the run

When the budget runs out, `Runtime.run` raises `RuntimeError` instead of
returning a `RunResult`. The state built up so far is lost with it: messages,
tool results and the token count, so a caller cannot even report what the
failed run cost.

Fix when: commit 16, when budgets return a status on the `RunResult` instead
of raising.

## `max_steps` counts node executions, not model turns

The old loop counted one step per model turn. `Runtime` counts every node it
runs, and one tool calling round is `model` then `tools`, so the same
`max_steps` allows about half as many model calls as before.

Fix when: a run hits the limit sooner than its author expected, or budgets
move to commit 16 and can count model turns, tokens or nodes separately.

## SSL certificate failures are classified as retryable

`AnthropicProvider._post` maps every `urllib.error.URLError` to
`RetryableError`. A failed certificate check arrives as a `URLError` wrapping
`ssl.SSLCertVerificationError`, so it is retried even though no retry can fix
it. Telling it apart from a real connection failure means inspecting
`exc.reason`, which in practice comes down to string matching, so it was
deferred.

Fix when: a misconfigured environment wastes a run on pointless retries, or
the error classification gets its own tests for `URLError` causes.

## Retry moves from the provider to the runtime

Decision: the provider classifies, the runtime retries. Today
`AnthropicProvider._post_with_retry` retries internally, which means retries
the runtime cannot see or record, and which would multiply with any retry the
runtime adds on top.

Fix when: commit 15. In the same commit, the provider's `max_attempts` is set
to 1 so the two retry layers can never stack.

## `apply_updates` copies the whole state on every step

`apply_updates` deep copies each merged value and rebuilds the state through
`model_validate`, so every step costs O(n) in the length of the message list.
That is acceptable for now because checkpointing already serialises the whole
state on every step, so the copy does not change the overall cost.

Fix when: runs get long enough for per step copying to show up. The
mitigation is structural sharing, for example immutable messages that new
states can reference instead of copy.

## The append reducer treats a tuple as a single item

`_append` spreads an update only when it is a `list`. A tuple of messages is
appended as one item instead of being spread, so it then fails validation, or
worse, passes it.

Fix when: a node returns a tuple, or the reducers get tests for update shapes.
Spread any non string sequence, or reject tuples explicitly.

## `to_mermaid` breaks on some node names

`CompiledGraph.to_mermaid` uses each node name as its Mermaid id. A node named
after a Mermaid reserved word, such as `end`, or one whose name contains spaces
or punctuation, produces a flowchart that will not render.

Fix when: someone names a node that way, or the diagram is rendered somewhere
automatically. Emit generated ids such as `n0`, `n1` and put the node name in
the label.

## Each checkpointer statement is its own transaction

`SqliteCheckpointer` opens its connection with `isolation_level=None`, so every
statement commits on its own. That is fine while each write is a single
statement. Commit 13 needs a checkpoint write and a run status update to
commit together or not at all, and today a crash between the two leaves them
disagreeing.

Fix when: commit 13. Add an explicit transaction helper, for example a context
manager that issues `BEGIN` and then `COMMIT` or `ROLLBACK`, and route both
writes through it.

## `Checkpoint.step` and `state.step` can disagree

The step number lives in two places: the `step` column on the checkpoint row,
and the `step` field on the state it stores. They are maintained separately and
nothing asserts they agree, so a resume could read one number from the row and
another from the state.

Fix when: the runtime starts writing checkpoints, since that is where the two
are first set together. Either assert they match in `put_checkpoint`, or derive
one from the other.

## Checkpoints carry no schema version

A checkpoint stores the state as JSON and is read back with whatever
`state_type` the caller passes. Adding a required field to a state model makes
every older checkpoint fail validation, so runs saved before the change can no
longer be resumed or inspected.

Fix when: a state model changes after real runs have been saved. Store a schema
version on each checkpoint and migrate old rows on read, or give every new
field a default.

## One checkpointer connection is shared across threads unguarded

`check_same_thread=False` lets one connection be used from several threads, but
nothing serialises access to it. A reader such as a trace viewer should open its
own connection rather than share the runtime's, which WAL mode already
supports.

Fix when: a second thread touches the checkpointer, for example a live trace
viewer. Open a connection per reader, or guard the shared one with a lock.

## The interrupt path must not write a checkpoint

When a run is interrupted before a node, the checkpoint from the previous
transition already records that node as `next_node`, so it is already the
correct resume point. Writing another checkpoint on the interrupt would reuse
that step number, and `put_checkpoint` now rejects a duplicate step with
`IntegrityError`.

Fix when: commit 17, when interrupts land. The interrupt path updates the run
status only and leaves the checkpoints alone.
