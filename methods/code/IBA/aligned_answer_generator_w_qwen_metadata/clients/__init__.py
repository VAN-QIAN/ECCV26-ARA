"""Client registry for aligned EchoSight answer generators."""

from __future__ import annotations

from typing import Dict, Type

from .base import GeneratorClient
from .llama import LLaMA3GeneratorClient
from .llama_vllm import LLaMA3VLLMHostGeneratorClient
from .mistral import MistralGeneratorClient


CLIENTS: Dict[str, Type[GeneratorClient]] = {
    "mistral": MistralGeneratorClient,
    "llama-3": LLaMA3GeneratorClient,
    "llama-3.1": LLaMA3GeneratorClient,
    "llama-3.1-8b": LLaMA3GeneratorClient,
}

__all__ = [
    "CLIENTS",
    "GeneratorClient",
    "MistralGeneratorClient",
    "LLaMA3GeneratorClient",
    "LLaMA3VLLMHostGeneratorClient",
]
