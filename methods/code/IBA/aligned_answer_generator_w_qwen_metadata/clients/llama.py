"""Wrapper for the Meta LLaMA-3 answer generator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from model.answer_generator import LLaMA3AnswerGenerator

from .base import GeneratorClient, _normalize_response
from ..types import GeneratorInput, GeneratorResponse


class LLaMA3GeneratorClient(GeneratorClient):
    def __init__(self, *, device: str, model_path: Optional[str] = None) -> None:
        path = model_path or "meta-llama/Meta-Llama-3-8B-Instruct"
        self._generator = LLaMA3AnswerGenerator(device=device, model_path=path)

    def answer(
        self,
        request: GeneratorInput,
        *,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> GeneratorResponse:
        _ = generation_kwargs
        output = self._generator.llm_answering(
            question=request.question,
            entry=request.entry,
            entry_section=request.entry_section,
            dataset_name=request.dataset_name,
        )
        return _normalize_response(output)
