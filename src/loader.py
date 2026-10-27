"""Safe input-file reading."""

import json
from pathlib import Path
from typing import Any, List, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .schema import FunctionSpec, PromptSpec

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_models(path: Path, model_type: Type[ModelT]) -> List[ModelT]:
    """Load a JSON list and validate every entry with Pydantic."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data: Any = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read valid JSON from {path}: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")

    try:
        return [
            model_type.model_validate(entry)
            for entry in data
        ]
    except ValidationError as exc:
        raise ValueError(
            f"Invalid entries in {path}: {exc}"
        ) from exc


def load_inputs(
    functions_path: Path,
    prompts_path: Path,
) -> tuple[List[FunctionSpec], List[PromptSpec]]:
    """Load input files and reject invalid function definitions."""
    functions = load_models(functions_path, FunctionSpec)
    prompts = load_models(prompts_path, PromptSpec)

    if not functions:
        raise ValueError("At least one function must be defined")

    names = [function.name for function in functions]

    if len(set(names)) != len(names):
        raise ValueError("Duplicate function names")

    for index, item in enumerate(prompts, start=1):
        if not item.prompt.strip():
            raise ValueError(
                f"Prompt #{index} cannot be empty or whitespace-only"
            )

    return functions, prompts
