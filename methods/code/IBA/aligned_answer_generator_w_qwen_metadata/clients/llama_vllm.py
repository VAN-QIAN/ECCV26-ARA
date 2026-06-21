"""OpenAI-compatible vLLM host client for LLaMA-3 style answer generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from model.answer_generator import (
    _adjust_prompt_length,
    _build_dataset_prompt_messages,
    reconstruct_wiki_article,
)

from .base import GeneratorClient
from ..types import GeneratorInput, GeneratorResponse


class LLaMA3VLLMHostGeneratorClient(GeneratorClient):
    """Run LLaMA answer generation against a remote vLLM OpenAI endpoint."""

    def __init__(
        self,
        *,
        device: str,
        model_path: Optional[str] = None,
        base_url: Optional[str],
        api_key: str = "EMPTY",
        request_timeout: float = 120.0,
        model_name: Optional[str] = None,
    ) -> None:
        del device  # Kept for API compatibility with local HF clients.
        normalized_base = (base_url or "").strip()
        if not normalized_base:
            raise ValueError("vLLM base_url is required for LLaMA vLLM host client.")
        if normalized_base.endswith("/"):
            normalized_base = normalized_base[:-1]
        if not normalized_base.endswith("/v1"):
            normalized_base = f"{normalized_base}/v1"

        self._client = OpenAI(
            base_url=normalized_base,
            api_key=api_key or "EMPTY",
            timeout=float(request_timeout),
        )
        self._model_name = (
            (model_name or "").strip()
            or (model_path or "").strip()
            or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )

    @staticmethod
    def _extract_text_from_response(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        chunks.append(str(item.get("text", "")))
                    elif item.get("text") is not None:
                        chunks.append(str(item.get("text")))
                else:
                    chunks.append(str(item))
            return "".join(chunks)
        return str(content)

    @staticmethod
    def _context_from_request(request: GeneratorInput) -> Optional[str]:
        if request.entry_section:
            return request.entry_section
        if request.entry is not None:
            context = reconstruct_wiki_article(request.entry)
            return _adjust_prompt_length(context, 4096)
        if request.context_text:
            return request.context_text
        return None

    def answer(
        self,
        request: GeneratorInput,
        *,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> GeneratorResponse:
        generation_kwargs = generation_kwargs or {}
        context_text = self._context_from_request(request)
        messages = _build_dataset_prompt_messages(
            question=request.question,
            context_text=context_text,
            dataset_name=request.dataset_name,
            require_reasoning=False,
        )
        dataset = (request.dataset_name or "").strip().lower()
        if dataset != "infoseek" and (not messages or messages[0].get("role") != "system"):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "You are a helpful assistant for answering encyclopedic questions.",
                },
            )

        max_new_tokens = int(generation_kwargs.get("max_new_tokens") or 128)
        temperature_value = generation_kwargs.get("temperature")
        temperature = 0.6 if temperature_value is None else max(0.0, float(temperature_value))

        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        top_p_value = generation_kwargs.get("top_p")
        if top_p_value is not None:
            payload["top_p"] = float(top_p_value)

        response = self._client.chat.completions.create(**payload)
        if not response.choices:
            return GeneratorResponse(answer="", raw_response="")
        content = response.choices[0].message.content
        text = self._extract_text_from_response(content).strip()
        return GeneratorResponse(answer=text, raw_response=text)
