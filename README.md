# durable-agent-runtime

An agent runtime with durable execution, built from scratch to understand what
agent frameworks actually do.

A run can be killed at any instant and resumed without losing work or repeating
a side effect.

**Status:** in development. Tool registry, provider interface, typed state with
reducers, and a validated state graph. The agent now runs as a two node graph
on the runtime, and the old loop is gone, see
[docs/why-a-graph.md](docs/why-a-graph.md).

No LangGraph, no CrewAI. Dependencies: pydantic, and SQLite from the standard library.