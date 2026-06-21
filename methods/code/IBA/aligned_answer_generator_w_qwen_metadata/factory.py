"""Factory helpers for constructing aligned answer generator clients."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

from .config import DEFAULT_SETTINGS, canonical_name
from .types import GeneratorSettings
from .clients import CLIENTS, GeneratorClient
from .clients.llama_vllm import LLaMA3VLLMHostGeneratorClient


def resolve_settings(
    name: str,
    *,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    vllm_base_url: Optional[str] = None,
    vllm_api_key: Optional[str] = None,
    vllm_timeout: Optional[float] = None,
    vllm_model_name: Optional[str] = None,
) -> GeneratorSettings:
    canonical = canonical_name(name)
    base = DEFAULT_SETTINGS.get(canonical)
    if base is None:
        if not device:
            raise ValueError(
                f"Generator '{name}' is unknown. Provide explicit device/model_path overrides."
            )
        base = GeneratorSettings(name=canonical, device=device, model_path=model_path)
    settings = replace(base)
    if device is not None:
        settings.device = device
    if model_path is not None:
        settings.model_path = model_path
    if max_new_tokens is not None:
        settings.max_new_tokens = max_new_tokens
    if temperature is not None:
        settings.temperature = temperature
    if vllm_base_url is not None:
        settings.vllm_base_url = vllm_base_url
    if vllm_api_key is not None:
        settings.vllm_api_key = vllm_api_key
    if vllm_timeout is not None:
        settings.vllm_timeout = float(vllm_timeout)
    if vllm_model_name is not None:
        settings.vllm_model_name = vllm_model_name
    return settings


def create_client(settings: GeneratorSettings) -> GeneratorClient:
    canonical = canonical_name(settings.name)
    if settings.vllm_base_url and canonical in {"llama-3", "llama-3.1", "llama-3.1-8b"}:
        init_kwargs: Dict[str, Any] = {
            "device": settings.device,
            "model_path": settings.model_path,
            "base_url": settings.vllm_base_url,
            "api_key": settings.vllm_api_key,
            "request_timeout": settings.vllm_timeout,
            "model_name": settings.vllm_model_name,
        }
        return LLaMA3VLLMHostGeneratorClient(**init_kwargs)
    client_cls = CLIENTS.get(canonical)
    if client_cls is None:
        raise ValueError(f"Unsupported generator '{settings.name}'")
    init_kwargs: Dict[str, Any] = {
        "device": settings.device,
    }
    if settings.model_path is not None:
        init_kwargs["model_path"] = settings.model_path
    return client_cls(**init_kwargs)


__all__ = ["resolve_settings", "create_client"]
