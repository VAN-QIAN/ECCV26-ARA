"""Adapters for reusing EchoSight answer generators with Qwen metadata."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Optional

from model.retriever import WikipediaKnowledgeBaseEntry

from .factory import create_client, resolve_settings
from .metadata_adapter import adapt_metadata_row
from .types import GeneratorInput, GeneratorResponse, GeneratorSettings
from .clients import GeneratorClient


class AlignedAnswerGenerator:
    """High-level helper that converts metadata rows and queries a legacy generator."""

    def __init__(
        self,
        client: GeneratorClient,
        *,
        kb_by_url: Optional[Mapping[str, WikipediaKnowledgeBaseEntry]] = None,
    ) -> None:
        self._client = client
        self._kb_by_url = kb_by_url or {}

    @property
    def client(self) -> GeneratorClient:
        return self._client

    def answer_metadata(
        self,
        row: MutableMapping[str, object],
        *,
        generation_kwargs: Optional[dict] = None,
    ) -> GeneratorResponse:
        request = adapt_metadata_row(row, self._kb_by_url)
        return self._client.answer(request, generation_kwargs=generation_kwargs)


def build_aligned_generator(
    name: str,
    *,
    kb_by_url: Optional[Mapping[str, WikipediaKnowledgeBaseEntry]] = None,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    vllm_base_url: Optional[str] = None,
    vllm_api_key: Optional[str] = None,
    vllm_timeout: Optional[float] = None,
    vllm_model_name: Optional[str] = None,
) -> AlignedAnswerGenerator:
    settings = resolve_settings(
        name,
        device=device,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        vllm_base_url=vllm_base_url,
        vllm_api_key=vllm_api_key,
        vllm_timeout=vllm_timeout,
        vllm_model_name=vllm_model_name,
    )
    client = create_client(settings)
    return AlignedAnswerGenerator(client, kb_by_url=kb_by_url)


__all__ = [
    "AlignedAnswerGenerator",
    "GeneratorInput",
    "GeneratorResponse",
    "GeneratorSettings",
    "build_aligned_generator",
    "resolve_settings",
]
