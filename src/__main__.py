
"""Command line entry point for the Call Me Maybe project."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .candidates import find_candidates
from .constrained_decoder import choose_json_value, choose_option
from .llm import SDKAdapter
from .loader import load_inputs
from .output import save_results
from .schema import FunctionSpec, validate_call
from .visualizer import (
    show_prompt,
    show_function,
    show_argument,
    show_valid,
)


def generate_call(
    model: SDKAdapter,
    prompt: str,
    functions: List[FunctionSpec],
    index: int,
    total: int,
) -> Dict[str, Any]:
    """Use the LLM to select a function and extract its arguments."""
    show_prompt(prompt, index, total)

    names = [item.name for item in functions]
    definitions = [item.model_dump() for item in functions]

    context = (
        "Choose the single most appropriate function for the request.\n"
        "Functions: "
        + json.dumps(definitions, ensure_ascii=False)
        + "\n"
        f"Request: {prompt}\n"
        "Function name: "
    )

    selected = choose_option(model, context, names)
    show_function(selected)

    function = next(
        item for item in functions
        if item.name == selected
    )

    arguments: Dict[str, Any] = {}
    parameter_names = list(function.parameters.keys())

    numeric_names = [
        name
        for name, parameter in function.parameters.items()
        if parameter.type in ("number", "integer")
    ]

    for name, parameter in function.parameters.items():
        possibilities = find_candidates(prompt, parameter.type)

        if not possibilities:
            raise ValueError(
                f"Cannot extract {name} "
                f"({parameter.type}) from {prompt!r}"
            )

        if parameter.type in ("number", "integer"):
            if len(possibilities) == len(numeric_names):
                position = numeric_names.index(name)
                possibilities = [possibilities[position]]

        argument_position = parameter_names.index(name) + 1

        context = (
            "Extract the value of one function argument.\n"
            "Select the candidate matching the CURRENT argument.\n"
            "Use the meaning of the user request.\n"
            "Do not repeat a previous argument unless appropriate.\n"
            "Return only one JSON value.\n\n"
            f"Request: {prompt}\n"
            f"Function: {selected}\n"
            f"Description: {function.description}\n"
            f"All arguments: {parameter_names}\n"
            f"Current argument: {name}\n"
            f"Position: {argument_position}/{len(parameter_names)}\n"
            f"Type: {parameter.type}\n"
            "Previously selected: "
            + json.dumps(arguments, ensure_ascii=False)
            + "\n"
            "Available values: "
            + json.dumps(possibilities, ensure_ascii=False)
            + "\n"
            "JSON value: "
        )

        value = choose_json_value(
            model,
            context,
            possibilities,
        )

        if parameter.type == "number":
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError(
                    f"Invalid numeric value for {name}: {value!r}"
                )
            value = float(value)

        if parameter.type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Invalid integer value for {name}: {value!r}"
                )

        # IMPORTANT: Keep these lines inside the loop.
        # Every generated parameter must be stored.
        arguments[name] = value
        show_argument(name, value)

    call: Dict[str, Any] = {
        "prompt": prompt,
        "name": selected,
        "parameters": arguments,
    }

    validate_call(call, functions)
    show_valid()

    return call


def main() -> int:
    """Parse CLI arguments, generate calls, and save JSON output."""
    parser = argparse.ArgumentParser(
        description="LLM function calling"
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
    )

    args = parser.parse_args()

    try:
        functions, prompts = load_inputs(
            args.functions_definition,
            args.input,
        )

        calls: List[Dict[str, Any]] = []

        if prompts:
            model = SDKAdapter()
            total = len(prompts)

            for index, item in enumerate(prompts, start=1):
                result = generate_call(
                    model,
                    item.prompt,
                    functions,
                    index,
                    total,
                )
                calls.append(result)

        save_results(args.output, calls)

        print(
            f"\nSaved {len(calls)} function calls "
            f"to {args.output}"
        )
        return 0

    except (
        OSError,
        ValueError,
        RuntimeError,
        OverflowError,
        TypeError,
        IndexError,
        KeyError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
