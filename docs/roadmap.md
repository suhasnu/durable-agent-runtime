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

## Agent loop never checks `stop_reason`

`run_agent` treats any completion without tool calls as a finished answer. A
completion that stopped because it hit the token limit is returned as if the
model chose to stop, so a truncated answer reads as a complete one. A
truncation in the middle of a tool call can also leave its arguments
incomplete.

Fix when: the loop is replaced, or before any real provider is wired in. The
loop should branch on `stop_reason` and treat `max_tokens` as its own outcome,
not as success.

## Agent loop never passes `max_tokens`

`run_agent` calls `provider.complete` without `max_tokens`, so every run uses
the provider default of 2048. Callers have no way to raise the limit for long
answers or lower it to cap cost.

Fix when: alongside the `stop_reason` check, since the two interact. Expose
`max_tokens` on `run_agent` and pass it through.

## Tool results on the final step are discarded

If the last allowed completion requests tools, the loop executes them, side
effects included, then raises the step limit error without ever returning the
results to the model. The work happened, but nothing downstream knows it did.

This is the side effect problem that commits 13 and 14 address.
