"""Display function-calling progress in the terminal."""

import json
from typing import Any


def show_prompt(prompt: str, index: int, total: int) -> None:
    """Display the prompt currently being processed."""
    print(f"\n[{index}/{total}] {prompt}", flush=True)


def show_function(name: str) -> None:
    """Display the function selected by the model."""
    print(f"  Function : {name}", flush=True)


def show_argument(name: str, value: Any) -> None:
    """Display one generated argument."""
    formatted = json.dumps(value, ensure_ascii=False)
    print(f"  {name:<9}: {formatted}", flush=True)


def show_valid() -> None:
    """Display successful schema validation."""
    print("  Status   : VALID", flush=True)
