import pytest
from pydantic import BaseModel

from agentrt.tools import InvalidToolArguments, ToolNotFound, ToolRegistry, tool


@tool
def add(a: int, b: int = 0) -> int:
    """Add two integers."""
    return a + b


@tool(side_effecting=True)
def send(to: str, body: str) -> str:
    """Send a message."""
    return f"sent to {to}"


def test_schema_comes_from_the_signature():
    schema = add.input_schema
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["required"] == ["a"]
    assert "b" not in schema.get("required", [])


def test_description_comes_from_the_docstring():
    assert add.description == "Add two integers."


def test_missing_docstring_is_rejected_at_definition_time():
    with pytest.raises(TypeError, match="docstring"):

        @tool
        def nameless(x: int) -> int:
            return x


def test_missing_annotation_is_rejected_at_definition_time():
    with pytest.raises(TypeError, match="type annotation"):

        @tool
        def untyped(x):
            """No annotation."""
            return x


def test_bad_arguments_raise_with_a_model_readable_message():
    with pytest.raises(InvalidToolArguments) as exc:
        add.validate({"a": "not a number"})
    assert "a:" in exc.value.detail
    assert "\n" not in exc.value.detail


def test_coercion_follows_pydantic():
    assert add.validate({"a": "3", "b": 4}) == {"a": 3, "b": 4}


def test_non_string_return_is_serialised():
    @tool
    def facts() -> dict:
        """Return a mapping."""
        return {"x": 1}

    assert facts.call({}) == '{"x": 1}'


def test_registry_rejects_duplicates():
    registry = ToolRegistry([add])
    with pytest.raises(ValueError, match="duplicate"):
        registry.add(add)


def test_unknown_tool_lists_what_exists():
    registry = ToolRegistry([add, send])
    with pytest.raises(ToolNotFound, match="add, send"):
        registry.get("subtract")


def test_side_effect_flags_default_off():
    assert add.side_effecting is False
    assert send.side_effecting is True
    assert send.idempotent is False


class Point(BaseModel):
    x: int


def test_model_parameters_arrive_as_models():
    @tool
    def where(p: Point) -> str:
        """Report what the function receives."""
        return type(p).__name__

    assert where.call({"p": {"x": 1}}) == "Point"


def test_underscore_parameter_is_rejected_at_definition_time():
    # pydantic treats leading underscores as private attributes, so the
    # parameter would silently vanish from the schema.
    with pytest.raises(TypeError, match="_hidden"):

        @tool
        def hidden(_hidden: int) -> int:
            """Underscore parameter."""
            return _hidden


def _clashes_with_model_config(model_config: int) -> int:
    """Clashes with pydantic configuration."""
    return model_config


def _shadows_schema(schema: int) -> int:
    """Shadows a BaseModel method."""
    return schema


@pytest.mark.parametrize(
    ("fn", "pname"),
    [(_clashes_with_model_config, "model_config"), (_shadows_schema, "schema")],
)
def test_pydantic_reserved_names_are_rejected_at_definition_time(fn, pname):
    with pytest.raises(TypeError, match=pname):
        tool(fn)
