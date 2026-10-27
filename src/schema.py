"""Input schemas and strict generated-call validation."""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParameterSpec(BaseModel):
    """A parameter type declared in functions_definition.json."""

    model_config = ConfigDict(extra="allow")
    type: str

    @field_validator("type")
    @classmethod
    def valid_type(cls, value: str) -> str:
        """Accept the supported scalar argument types."""
        if value not in ("string", "number", "integer", "boolean"):
            raise ValueError(f"Unsupported argument type: {value}")
        return value


class FunctionSpec(BaseModel):
    """Public function description supplied to the model."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str
    parameters: Dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptSpec(BaseModel):
    """One user request."""

    model_config = ConfigDict(extra="forbid")
    prompt: str


def validate_call(call: Dict[str, Any], functions: List[FunctionSpec]) -> None:
    """Reject output that fails the exact project schema."""
    if set(call) != {"prompt", "name", "parameters"}:
        raise ValueError(
            "Output must contain exactly prompt, name, parameters"
            )
    if not isinstance(call["prompt"], str):
        raise ValueError("prompt must be a string")
    function = next((item for item in functions
                     if item.name == call["name"]), None)
    if function is None:
        raise ValueError(f"Unknown function: {call['name']!r}")
    values = call["parameters"]
    if not isinstance(values, dict):
        raise ValueError("parameters must be an object")
    if set(values) != set(function.parameters):
        raise ValueError("Function parameter names do not match definition")
    for name, specification in function.parameters.items():
        value = values[name]
        kind = specification.type
        if kind == "string" and not isinstance(value, str):
            raise ValueError(f"{name} must be string")
        if kind == "boolean" and type(value) is not bool:
            raise ValueError(f"{name} must be boolean")
        if kind == "integer" and type(value) is not int:
            raise ValueError(f"{name} must be integer")
        if kind == "number" and type(value) is not float:
            raise ValueError(f"{name} must be number")
