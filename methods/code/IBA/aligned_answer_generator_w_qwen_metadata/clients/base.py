"""Common interface for legacy EchoSight generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..types import GeneratorInput, GeneratorResponse


class GeneratorClient(ABC):
    """Minimal surface matching the needs of the Qwen-aligned workflow."""

    @abstractmethod
    def answer(
        self,
        request: GeneratorInput,
        *,
        generation_kwargs: Dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        """Produce an answer for the provided metadata bundle."""


def _normalize_response(output: Any) -> GeneratorResponse:
    if isinstance(output, GeneratorResponse):
        return output
    text = str(output) if output is not None else ""
    return GeneratorResponse(answer=text, raw_response=text)
