"""Static configuration for aligned answer generator clients."""

from __future__ import annotations

from typing import Dict

from .types import GeneratorSettings


GENERATOR_ALIASES: Dict[str, str] = {
    "mistral": "mistral",
    "mistral-7b": "mistral",
    "mistral-7b-instruct": "mistral",
    "mistral-7b-instruct-v0.2": "mistral",
    "llama": "llama-3",
    "llama3": "llama-3",
    "llama-3": "llama-3",
    "llama-3-8b": "llama-3",
    "llama-3.1": "llama-3.1-8b",
    "llama-3.1-8b": "llama-3.1-8b",
    "llama-3.1-8b-instruct": "llama-3.1-8b",
}


DEFAULT_SETTINGS: Dict[str, GeneratorSettings] = {
    "mistral": GeneratorSettings(
        name="mistral",
        device="cuda:0",
        model_path="mistralai/Mistral-7B-Instruct-v0.1",
    ),
    "llama-3": GeneratorSettings(
        name="llama-3",
        device="cuda:0",
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "llama-3.1-8b": GeneratorSettings(
        name="llama-3.1-8b",
        device="cuda:0",
        model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    ),
}


def canonical_name(name: str) -> str:
    key = name.lower().strip()
    return GENERATOR_ALIASES.get(key, key)
