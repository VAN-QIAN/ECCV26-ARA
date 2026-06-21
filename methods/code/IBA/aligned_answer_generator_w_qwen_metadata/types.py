"""Typed primitives for running EchoSight generators with Qwen metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from model.retriever import WikipediaKnowledgeBaseEntry


@dataclass
class GeneratorInput:
    """Normalized view over a single Qwen metadata row."""

    question: str
    dataset_name: Optional[str]
    entry: Optional[WikipediaKnowledgeBaseEntry]
    entry_section: Optional[str]
    context_mode: Optional[str]
    context_text: Optional[str]
    reranked_sections: List[str]
    image_path: Optional[str]
    raw: Dict[str, Any]


@dataclass
class GeneratorResponse:
    """Lightweight container mirroring EchoSight generator outputs."""

    answer: str
    raw_response: str


@dataclass
class GeneratorSettings:
    """Configuration bundle controlling generator instantiation."""

    name: str
    device: str
    model_path: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    vllm_base_url: Optional[str] = None
    vllm_api_key: str = "EMPTY"
    vllm_timeout: float = 120.0
    vllm_model_name: Optional[str] = None
