"""Tool registry.

A tool is a Python function. Its JSON schema is derived from its signature,
never written by hand: a hand-written schema drifts from the function it
describes and the drift is silent.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model


class InvalidToolArguments(Exception):
    def __init__(self, tool_name: str, detail: str):
        super().__init__(f"invalid arguments for {tool_name}: {detail}")
        self.tool_name = tool_name
        self.detail = detail


class ToolNotFound(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    input_schema: dict[str, Any]
    side_effecting: bool
    idempotent: bool
    _model: type

    def validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        try:
            # dict() is shallow, so nested model arguments keep their types.
            return dict(self._model(**raw))
        except ValidationError as exc:
            raise InvalidToolArguments(self.name, _compact_errors(exc)) from exc

    def call(self, arguments: dict[str, Any]) -> str:
        result = self.fn(**self.validate(arguments))
        return result if isinstance(result, str) else json.dumps(result, default=str)


def _compact_errors(exc: ValidationError) -> str:
    """Model-readable validation failure, one line.

    Pydantic's default repr is multi-line and cites internal URLs. This goes
    back to the model as a tool result, so it needs the field and the problem
    and nothing else.
    """
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _schema_from_signature(fn: Callable[..., Any], name: str) -> tuple[dict[str, Any], type]:
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"tool {name!r}: *args and **kwargs cannot be described to a model")
        # pydantic would drop these as private attributes or clash with its own.
        if pname.startswith(("_", "model_")) or hasattr(BaseModel, pname):
            raise TypeError(f"tool {name!r}: parameter name {pname!r} is reserved by pydantic")

        annotation = hints.get(pname)
        if annotation is None:
            raise TypeError(f"tool {name!r}: parameter {pname!r} needs a type annotation")

        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, default)

    model = create_model(f"{name}_Args", **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)

    return schema, model


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    side_effecting: bool = False,
    idempotent: bool = False,
) -> Any:
    def wrap(f: Callable[..., Any]) -> Tool:
        tool_name = name or f.__name__
        doc = inspect.getdoc(f)
        if not doc:
            raise TypeError(f"tool {tool_name!r} needs a docstring: it becomes the description")
        schema, model = _schema_from_signature(f, tool_name)
        return Tool(
            name=tool_name,
            description=doc.strip().split("\n\n")[0],
            fn=f,
            input_schema=schema,
            side_effecting=side_effecting,
            idempotent=idempotent,
            _model=model,
        )

    return wrap if fn is None else wrap(fn)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name {t.name!r}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotFound(f"unknown tool {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]