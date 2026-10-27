"""Token-constrained selection from a finite collection of JSON-safe choices.

A small model can get trapped generating an unbounded JSON number.  Instead,
create finite, typed candidates from the request and let the LLM choose among
those candidates using *masked next-token logits*.  This is genuine constrained
decoding: a chosen token must belong to a candidate tokenization prefix.
"""

import json
from typing import List, Protocol


class TokenModel(Protocol):
    """Public tokenizer and model methods needed by the decoder."""

    def encode(self, text: str) -> List[int]:
        """Encode a string into token IDs."""
        ...

    def logits(self, ids: List[int]) -> List[float]:
        """Get next-token logits."""
        ...


def choose_option(model: TokenModel, context: str,
                  options: List[str], max_steps: int = 128) -> str:
    """Choose exactly one option, masking tokens outside candidate prefixes.

    Each option is represented by its own token IDs. If tokenizations share a
    prefix, the model determines which candidate to keep at each step. This
    function refuses cases where a completed candidate is a token-prefix of
    another, rather than silently making an arbitrary choice.
    """
    options = list(dict.fromkeys(options))
    if not options:
        raise ValueError("Cannot select from an empty candidate set")
    if len(options) == 1:
        return options[0]
    if any(not option for option in options):
        raise ValueError("Empty candidate")
    sequences = {option: model.encode(option) for option in options}
    if any(not seq for seq in sequences.values()):
        raise ValueError("Tokenizer produced an empty candidate")
    generated: List[int] = []
    context_ids = model.encode(context)
    for _ in range(max_steps):
        remaining = [option for option, seq in sequences.items()
                     if seq[:len(generated)] == generated]
        if not remaining:
            raise ValueError("No valid candidate prefix")
        finished = [option for option in remaining
                    if len(sequences[option]) == len(generated)]
        if len(remaining) == 1 and finished:
            return finished[0]
        if finished:
            raise ValueError("Ambiguous candidate token-prefix")
        allowed = {sequences[option][len(generated)] for option in remaining}
        scores = model.logits(context_ids + generated)
        if not allowed or max(allowed) >= len(scores):
            raise ValueError("Model vocabulary does not match tokenizer IDs")
        selected = max(sorted(allowed), key=lambda token: scores[token])
        generated.append(selected)
    raise ValueError("Candidate selection exceeded token budget")


def choose_json_value(model: TokenModel, context: str,
                      candidates: List[object]) -> object:
    """Select one typed value using LLM logits, with finite JSON candidates.

    The LLM chooses the candidate, not a Python rule. Finite candidates ensure
    that every possible completed output is syntactically correct JSON.
    """
    if not candidates:
        raise ValueError("No candidate values found in request")
    encoded = {json.dumps(value, ensure_ascii=False, allow_nan=False): value
               for value in candidates}
    key = choose_option(model, context, list(encoded))
    return encoded[key]
