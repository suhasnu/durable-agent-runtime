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
