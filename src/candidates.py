"""Collect typed possible arguments
from user requests, without selecting one."""

import re
from typing import List

_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?(?![\w.])"
)
_QUOTED = re.compile(r"""["']([^"']*)["']""")
_WORD = re.compile(r"[\w-]+", re.UNICODE)


def find_candidates(prompt: str, kind: str) -> List[object]:
    """Return distinct typed candidates extracted from a request."""
    if kind in ("number", "integer"):
        result: List[object] = []

        for match in _NUMBER.finditer(prompt):
            value = float(match.group())

            if kind == "integer":
                if value.is_integer():
                    result.append(int(value))
            else:
                result.append(value)

        return list(dict.fromkeys(result))

    if kind == "boolean":
        return [True, False]

    if kind == "string":
        quoted = [
            match.group(1)
            for match in _QUOTED.finditer(prompt)
        ]

        if quoted:
            return list(dict.fromkeys(quoted))

        words = _WORD.findall(prompt)

        if not words:
            return []

        if len(words) == 2:
            return [words[1]]

        candidates: List[str] = list(words)

        for size in range(2, min(len(words), 5) + 1):
            for start in range(len(words) - size + 1):
                phrase = " ".join(words[start:start + size])

                if phrase != prompt.strip():
                    candidates.append(phrase)

        return list(dict.fromkeys(candidates))

    raise ValueError(f"Unsupported type: {kind}")
