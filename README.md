# durable-agent-runtime

An agent runtime with durable execution, built from scratch to understand what
agent frameworks actually do.

A run can be killed at any instant and resumed without losing work or repeating
a side effect.

**Status:** in development. Tool registry, provider interface, typed state with
reducers, a validated state graph, and a SQLite checkpoint store. The agent runs
as a two node graph on the runtime; the checkpointer is not yet wired into it.
See [docs/why-a-graph.md](docs/why-a-graph.md).

No LangGraph, no CrewAI. Dependencies: pydantic, and SQLite from the standard library.