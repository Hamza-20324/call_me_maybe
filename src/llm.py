"""Adapter using only the published SDK interfaces."""

from typing import Any, List


class SDKAdapter:
    """Wrap the subject-provided Small_LLM_Model public API."""

    def __init__(self) -> None:
        """Initialize the official LLM SDK."""
        try:
            from llm_sdk import (  # type: ignore
                Small_LLM_Model,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Missing provided llm_sdk/. "
                "Copy the official SDK next to src/."
            ) from exc

        try:
            self.model: Any = Small_LLM_Model()
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize Small_LLM_Model; "
                "check SDK dependencies and model installation."
            ) from exc

    def encode(self, text: str) -> List[int]:
        """Encode text through the public tokenizer API."""
        encoded: Any = self.model.encode(text)

        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()

        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]

        return [int(item) for item in encoded]

    def decode(self, ids: List[int]) -> str:
        """Decode tokens through the public SDK method."""
        return str(self.model.decode(ids))

    def logits(self, ids: List[int]) -> List[float]:
        """Retrieve next-token scores from the SDK."""
        raw: Any = self.model.get_logits_from_input_ids(ids)

        if hasattr(raw, "tolist"):
            raw = raw.tolist()

        if raw and isinstance(raw[0], list):
            raw = raw[-1]

        return [float(value) for value in raw]
