"""Utilities for constructing dataset-aware answer prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

_INFOSSEEK_SYSTEM_PROMPT = (
    "You always answer the question the user asks. Do not answer anything else."
)

_INFOSSEEK_ONE_SHOT = (
    "Context: The sounthern side of the Alps is next to Lake Como.\n\n"
    "Question: Which body of water is this mountain located in or next to? "
    "Just answer the questions , no explanations needed. Short answer is: Lake Como"
)

_INFOSSEEK_USER_TEMPLATE = (
    "Context: {context}\n"
    "Question: {question} Just answer the questions , no explanations needed. "
    "Short answer is:"
)

_DEFAULT_INSTRUCTION_PREFIX = (
    "You are a knowledgeable assistant. Examine the image carefully before using any contextual evidence."
)

_DEFAULT_CONTEXT_GUIDANCE = (
    "Use ONLY the evidence text if it is helpful."
)

_DEFAULT_NO_CONTEXT_GUIDANCE = (
    "If the image alone is insufficient to answer, reply 'Not sure'."
)

_DEFAULT_REASONING_GUIDANCE = (
    "Briefly explain your reasoning before giving the final answer."
)


@dataclass(frozen=True)
class PromptParts:
    """Structured chat prompt components."""

    system: Optional[str]
    user: str


def _normalize_context_text(context: Optional[str]) -> str:
    text = (context or "").strip()
    return text if text else "None"


def _build_augmented_image_note(prompt_metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(prompt_metadata, Mapping):
        return None
    mode = str(prompt_metadata.get("augmented_mode") or "").strip().lower()
    if mode not in {"method1", "method2"}:
        return None
    target_side = str(prompt_metadata.get("augmented_target_side") or "").strip().lower()
    query_variant = str(prompt_metadata.get("augmented_query_variant") or "").strip().lower()
    parts = ["The input image is a composite image containing a target object and a distractor object."]
    if target_side:
        parts.append(f"Answer about the target object on the {target_side} side only.")
    else:
        parts.append("Answer only about the target object referred to by the question.")
    if mode == "method2" and query_variant == "without_position":
        if target_side:
            parts.append(
                f"The question may omit positional words; it still refers to the {target_side} object."
            )
        else:
            parts.append("The question may omit positional words; do not answer about the distractor object.")
    else:
        parts.append("Ignore the distractor object.")
    return " ".join(parts)


def build_prompt_parts(
    *,
    dataset_name: Optional[str],
    question: str,
    context_text: Optional[str],
    require_reasoning: bool,
    prompt_metadata: Optional[Mapping[str, Any]] = None,
) -> PromptParts:
    """Return dataset-specific prompt pieces for answer generation."""
    dataset = (dataset_name or "").strip().lower()
    normalized_context = _normalize_context_text(context_text)
    stripped_question = question.strip()
    augmented_image_note = _build_augmented_image_note(prompt_metadata)

    if dataset.startswith("infoseek"):
        user_question_block = _INFOSSEEK_USER_TEMPLATE.format(
            context=normalized_context,
            question=stripped_question,
        )
        if augmented_image_note:
            user_question_block = (
                f"Image note: {augmented_image_note}\n" + user_question_block
            )
        user_prompt = "\n\n".join(
            (
                _INFOSSEEK_ONE_SHOT,
                user_question_block,
            )
        )
        return PromptParts(system=_INFOSSEEK_SYSTEM_PROMPT, user=user_prompt)

    instructions = [_DEFAULT_INSTRUCTION_PREFIX]
    if context_text:
        instructions.append(_DEFAULT_CONTEXT_GUIDANCE)
    else:
        instructions.append(_DEFAULT_NO_CONTEXT_GUIDANCE)
    if require_reasoning:
        instructions.append(_DEFAULT_REASONING_GUIDANCE)
    if augmented_image_note:
        instructions.append(augmented_image_note)

    instruction_block = "\n".join(instructions)
    if context_text:
        user_prompt = (
            f"{instruction_block}\n"
            "---- Evidence ----\n"
            f"{context_text}\n"
            "------------------\n"
            f"Question: {stripped_question}\n"
            "Answer:"
        )
    else:
        user_prompt = (
            f"{instruction_block}\n"
            f"Question: {stripped_question}\n"
            "Answer:"
        )
    return PromptParts(system=None, user=user_prompt)


__all__ = ["PromptParts", "build_prompt_parts"]
