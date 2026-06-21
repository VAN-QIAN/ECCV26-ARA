"""Thin wrapper around :class:`model.answer_generator.MistralAnswerGenerator`."""

from __future__ import annotations

from typing import Any, Dict, Optional

from model.answer_generator import MistralAnswerGenerator

from .base import GeneratorClient, _normalize_response
from ..types import GeneratorInput, GeneratorResponse


class MistralGeneratorClient(GeneratorClient):
    """Expose the legacy Mistral generator through the generic client API."""

    def __init__(
        self,
        *,
        device: str,
        model_path: str,
        use_embedding_model: bool = False,
    ) -> None:
        self._generator = MistralAnswerGenerator(
            device=device,
            model_path=model_path,
            use_embedding_model=use_embedding_model,
        )

    def answer(
        self,
        request: GeneratorInput,
        *,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> GeneratorResponse:
        _ = generation_kwargs  # generation knobs are not supported by the legacy class
        output = self._generator.llm_answering(
            question=request.question,
            entry=request.entry,
            entry_section=request.entry_section,
            dataset_name=request.dataset_name,
        )
        return _normalize_response(output)
