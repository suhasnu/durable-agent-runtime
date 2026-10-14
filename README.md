# durable-agent-runtime

An agent runtime with durable execution, built from scratch to understand what
agent frameworks actually do.

A run can be killed at any instant and resumed without losing work or repeating
a side effect.

**Status:** in development.

No LangGraph, no CrewAI. Dependencies: pydantic, and SQLite from the standard library.