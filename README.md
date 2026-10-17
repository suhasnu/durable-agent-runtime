# durable-agent-runtime

An agent runtime with durable execution, built from scratch to understand what
agent frameworks actually do.

A run can be killed at any instant and resumed without losing work or repeating
a side effect.

**Status:** in development. Tool registry, provider interface, and a working
agent loop. The loop is about to be replaced by a state machine, see
[docs/why-a-graph.md](docs/why-a-graph.md).

No LangGraph, no CrewAI. Dependencies: pydantic, and SQLite from the standard library.