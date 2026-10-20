# durable-agent-runtime

An agent runtime with durable execution, built from scratch to understand what
agent frameworks actually do.

A run can be killed at any instant and resumed without losing work or repeating
a side effect.

**Status:** in development. Tool registry, provider interface, typed state with
reducers, a validated state graph, and a SQLite checkpoint store. The agent runs
as a two node graph on the runtime, which writes a checkpoint after every
transition and can resume a run from its last checkpoint, in a new process.
Side effects are not yet protected: a resumed run can repeat a tool call.
See [docs/why-a-graph.md](docs/why-a-graph.md).

No LangGraph, no CrewAI. Dependencies: pydantic, and SQLite from the standard library.