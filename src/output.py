"""Write results with deterministic UTF-8 JSON formatting."""

import json
from pathlib import Path
from typing import Any, Dict, List


def save_results(path: Path, calls: List[Dict[str, Any]]) -> None:
    """Create the output directory and write valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(calls, handle, ensure_ascii=False, indent=2,
                  allow_nan=False)
        handle.write("\n")
