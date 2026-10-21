"""Agent state.

Commit 5 kept state in local variables inside a while loop. That works until
you want to inspect, pause, persist or resume a run, at which point you find
the state lives on the Python stack and the stack is gone. Everything durable
in this runtime follows from making state an explicit object instead.

Nodes do not mutate state. They return a dict of updates, and the runtime
applies each one through that field's reducer. The reducer is what lets two
nodes both contribute to `messages` without one clobbering the other.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Annotated, Any, get_args, get_origin

from pydantic import BaseModel, Field

from .types import Message, Usage


class Reducer:
    """Field metadata saying how an update combines with the current value."""

    def __init__(self, fn: Callable[[Any, Any], Any], name: str):
        self.fn = fn
        self.name = name

    def __repr__(self) -> str:
        return f"Reducer({self.name})"


def _append(current: list[Any], update: Any) -> list[Any]:
    if update is None:
        return current
    items = update if isinstance(update, list) else [update]
    return [*current, *items]


def _add(current: Any, update: Any) -> Any:
    return current if update is None else current + update


append = Reducer(_append, "append")
add = Reducer(_add, "add")
replace = Reducer(lambda _current, update: update, "replace")

Messages = Annotated[list[Message], append]
Tokens = Annotated[Usage, add]


class AgentState(BaseModel):
    """Default state for a tool-calling agent. Subclass it to add fields."""

    messages: Messages = Field(default_factory=list)
    usage: Tokens = Field(default_factory=Usage)


def reducer_for(model: type[BaseModel], field: str) -> Reducer:
    """Find the reducer declared on a field, defaulting to replace."""
    annotation = model.model_fields[field].rebuild_annotation()
    if get_origin(annotation) is Annotated:
        for meta in get_args(annotation)[1:]:
            if isinstance(meta, Reducer):
                return meta
    return replace


def apply_updates(state: BaseModel, updates: dict[str, Any]) -> BaseModel:
    """Return a NEW state with `updates` folded in through each field's reducer.

    A copy, not a mutation, so a checkpoint written before this call still
    describes the state it was taken from.
    """
    if not updates:
        return type(state).model_validate(state.model_dump())

    unknown = set(updates) - set(type(state).model_fields)
    if unknown:
        raise KeyError(f"node returned unknown state fields: {sorted(unknown)}")

    merged = {}
    for key, update in updates.items():
        value = reducer_for(type(state), key).fn(getattr(state, key), update)
        # Copy explicitly. Pydantic passes model instances through validation
        # untouched rather than rebuilding them, so model_validate alone leaves
        # the new state sharing objects with the old one.
        merged[key] = deepcopy(value)

    # Rebuild through validation rather than model_copy(update=...). That method
    # deep copies the existing fields but drops the update values in untouched,
    # so the new state would share objects with the old one and a checkpoint
    # could change after it was taken. It also skips validation, so a node
    # returning the wrong type would fail far from the node that caused it.
    return type(state).model_validate({**state.model_dump(), **merged})