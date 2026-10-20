"""The state machine.

A node is a function from state to a dict of updates. An edge says which node
runs next, either statically or by inspecting the state. Compiling validates
the wiring up front, because a typo in an edge target should fail at import
time rather than forty steps into a run that has already cost money.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

START = "__start__"
END = "__end__"

# A node receives the state and the run context (provider, tools, checkpointer).
# Passing the context explicitly rather than through a global is what makes a
# node callable from a test without standing up a runtime.
Node = Callable[[BaseModel, Any], dict[str, Any]]
Router = Callable[[BaseModel], str]


@dataclass(frozen=True)
class _ConditionalEdge:
    router: Router
    targets: frozenset[str]


@dataclass
class StateGraph:
    state_type: type[BaseModel]
    nodes: dict[str, Node] = field(default_factory=dict)
    static_edges: dict[str, str] = field(default_factory=dict)
    conditional_edges: dict[str, _ConditionalEdge] = field(default_factory=dict)
    entry: str | None = None

    def add_node(self, name: str, fn: Node) -> StateGraph:
        if name in (START, END):
            raise ValueError(f"{name!r} is reserved")
        if name in self.nodes:
            raise ValueError(f"duplicate node {name!r}")
        self.nodes[name] = fn
        return self

    def set_entry(self, name: str) -> StateGraph:
        self.entry = name
        return self

    def add_edge(self, source: str, target: str) -> StateGraph:
        if source in self.conditional_edges:
            raise ValueError(f"node {source!r} already has a conditional edge")
        self.static_edges[source] = target
        return self

    def add_conditional_edge(
        self, source: str, router: Router, targets: Iterable[str]
    ) -> StateGraph:
        if source in self.static_edges:
            raise ValueError(f"node {source!r} already has a static edge")
        if isinstance(targets, str):
            # A bare string would silently become a set of characters.
            raise TypeError("targets must be a collection of node names, not a string")
        frozen = frozenset(targets)
        if not frozen:
            raise ValueError(f"conditional edge from {source!r} has no targets")
        self.conditional_edges[source] = _ConditionalEdge(router, frozen)
        return self

    def compile(self) -> CompiledGraph:
        if self.entry is None:
            raise ValueError("no entry node set")
        if self.entry not in self.nodes:
            raise ValueError(f"entry node {self.entry!r} is not defined")

        valid = set(self.nodes) | {END}
        for source, target in self.static_edges.items():
            self._check_source(source)
            if target not in valid:
                raise ValueError(f"edge {source!r} -> {target!r}: unknown target")
        for source, edge in self.conditional_edges.items():
            self._check_source(source)
            unknown = edge.targets - valid
            if unknown:
                raise ValueError(
                    f"conditional edge from {source!r}: unknown targets {sorted(unknown)}"
                )

        dangling = set(self.nodes) - set(self.static_edges) - set(self.conditional_edges)
        if dangling:
            raise ValueError(f"nodes with no outgoing edge: {sorted(dangling)}")

        self._check_reachable()
        self._check_can_reach_end()
        return CompiledGraph(self)

    def _check_source(self, source: str) -> None:
        if source not in self.nodes:
            raise ValueError(f"edge from unknown node {source!r}")

    def _check_reachable(self) -> None:
        seen: set[str] = set()
        frontier = [self.entry]
        while frontier:
            current = frontier.pop()
            if current in seen or current == END:
                continue
            seen.add(current)
            if current in self.static_edges:
                frontier.append(self.static_edges[current])
            elif current in self.conditional_edges:
                frontier.extend(self.conditional_edges[current].targets)
        unreachable = set(self.nodes) - seen
        if unreachable:
            raise ValueError(f"unreachable nodes: {sorted(unreachable)}")
          
    def _check_can_reach_end(self) -> None:
        """Every node must have some path to END.

        Forward reachability says a node can be entered. This says it can be
        left. A static cycle with no exit passes the first check and is an
        infinite loop that only the step budget stops.
        """
        reverse: dict[str, set[str]] = {}
        for source, target in self.static_edges.items():
            reverse.setdefault(target, set()).add(source)
        for source, edge in self.conditional_edges.items():
            for target in edge.targets:
                reverse.setdefault(target, set()).add(source)

        can_reach: set[str] = set()
        frontier = [END]
        while frontier:
            current = frontier.pop()
            for predecessor in reverse.get(current, ()):
                if predecessor not in can_reach:
                    can_reach.add(predecessor)
                    frontier.append(predecessor)

        stuck = set(self.nodes) - can_reach
        if stuck:
            raise ValueError(f"nodes that can never reach END: {sorted(stuck)}")


class CompiledGraph:
    """An immutable, validated graph. Executing it is the runtime's job.

    The builder's contents are copied at compile time. Sharing them would let
    a later `add_edge` reach a compiled graph without passing validation.
    """

    def __init__(self, graph: StateGraph):
        self.state_type = graph.state_type
        self.entry = graph.entry
        self._nodes = dict(graph.nodes)
        self._static = dict(graph.static_edges)
        self._conditional = dict(graph.conditional_edges)

    def node(self, name: str) -> Node:
        return self._nodes[name]

    def next_node(self, source: str, state: BaseModel) -> str:
        if source in self._static:
            return self._static[source]
        edge = self._conditional[source]
        target = edge.router(state)
        if target not in edge.targets:
            raise ValueError(
                f"router for {source!r} returned {target!r}, "
                f"not in declared targets {sorted(edge.targets)}"
            )
        return target

    def to_mermaid(self) -> str:
        lines = ["flowchart TD", f"    {START}([start]) --> {self.entry}"]
        for source, target in self._static.items():
            lines.append(f"    {source} --> {_label(target)}")
        for source, edge in self._conditional.items():
            for target in sorted(edge.targets):
                lines.append(f"    {source} -.-> {_label(target)}")
        return "\n".join(lines)
      
      
def _label(node: str) -> str:
    return f"{END}([end])" if node == END else node