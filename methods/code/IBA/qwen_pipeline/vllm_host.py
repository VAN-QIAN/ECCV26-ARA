"""OpenAI-compatible vLLM host adapter for Qwen-2.5-VL."""

from __future__ import annotations

import base64
import collections
import logging
import sys
from importlib import util as importlib_util
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from openai import OpenAI
from PIL import Image

_qwen_module = sys.modules.get("model.qwen_vl")
if _qwen_module is None:
    _QWEN_SPEC = importlib_util.spec_from_file_location(
        "model.qwen_vl",
        Path(__file__).resolve().parent.parent / "model" / "Qwen-vl.py",
    )
    if _QWEN_SPEC is None or _QWEN_SPEC.loader is None:
        raise ImportError("Unable to load model/Qwen-vl.py")
    _qwen_module = importlib_util.module_from_spec(_QWEN_SPEC)
    sys.modules[_QWEN_SPEC.name] = _qwen_module
    _QWEN_SPEC.loader.exec_module(_qwen_module)
QwenVLModel = _qwen_module.QwenVLModel
AnswerResult = _qwen_module.AnswerResult
CandidateScore = _qwen_module.CandidateScore
IdentificationResult = _qwen_module.IdentificationResult
IdentificationScores = _qwen_module.IdentificationScores

LOGGER = logging.getLogger(__name__)


class VLLMHostQwenVLModel(QwenVLModel):
    """QwenVLModel-compatible wrapper backed by a remote vLLM OpenAI endpoint."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        request_timeout: float = 120.0,
        max_context_tokens: int = 1024,
        max_image_side: Optional[int] = 2048,
        max_image_area: Optional[int] = 4_194_304,
        image_format: str = "JPEG",
        image_quality: int = 90,
        enable_image_cache: bool = True,
        image_cache_size: int = 2048,
    ) -> None:
        base_url = (api_base or "http://127.0.0.1:8000/v1").strip()
        if not base_url:
            base_url = "http://127.0.0.1:8000/v1"
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self.api_base = base_url
        self.api_key = api_key or "EMPTY"
        self.request_timeout = float(request_timeout)
        self.image_format = str(image_format or "JPEG").upper()
        self.image_quality = max(1, min(100, int(image_quality)))
        self.enable_image_cache = bool(enable_image_cache)
        self.image_cache_size = max(0, int(image_cache_size))
        self._image_data_url_cache: "collections.OrderedDict[str, str]" = collections.OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=self.request_timeout,
        )
        super().__init__(
            model_name=model_name,
            device="cpu",
            max_context_tokens=max_context_tokens,
            max_image_side=max_image_side,
            max_image_area=max_image_area,
        )

    def _load_model(self) -> None:
        # Keep QwenVLModel's parsing/prompt helpers while delegating generation to vLLM host.
        self.processor = None
        self.model = None
        LOGGER.info(
            "Using remote vLLM host for model %s via %s",
            self.model_name,
            self.api_base,
        )

    @staticmethod
    def _normalize_path_key(path: str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def _get_cached_data_url(self, path: str) -> Optional[str]:
        if not self.enable_image_cache or self.image_cache_size <= 0:
            return None
        key = self._normalize_path_key(path)
        value = self._image_data_url_cache.get(key)
        if value is None:
            self._cache_misses += 1
            return None
        self._cache_hits += 1
        self._image_data_url_cache.move_to_end(key)
        return value

    def _set_cached_data_url(self, path: str, data_url: str) -> None:
        if not self.enable_image_cache or self.image_cache_size <= 0:
            return
        key = self._normalize_path_key(path)
        self._image_data_url_cache[key] = data_url
        self._image_data_url_cache.move_to_end(key)
        while len(self._image_data_url_cache) > self.image_cache_size:
            self._image_data_url_cache.popitem(last=False)

    def _image_path_to_data_url(self, image_path: str) -> str:
        cached = self._get_cached_data_url(image_path)
        if cached is not None:
            return cached
        image = self._load_image(image_path)
        encoded = self._image_to_data_url(image)
        self._set_cached_data_url(image_path, encoded)
        return encoded

    def _is_deepseek_backend(self) -> bool:
        model_name = str(getattr(self, "model_name", "") or "").strip().lower()
        api_base = str(getattr(self, "api_base", "") or "").strip().lower()
        return "deepseek" in model_name or "deepseek" in api_base

    def _strict_system_prompt(self, *, expects_json: bool) -> Optional[str]:
        if not self._is_deepseek_backend():
            return None
        if expects_json:
            return "You are a strict auditor. Return JSON only."
        return "You are a strict visual entity recognizer. Follow the required answer format exactly."

    def _format_identification_prompt(
        self,
        candidate_titles: Sequence[str],
        question: Optional[str],
        instruction: Optional[str],
        top_k: int,
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
    ) -> Tuple[List[str], str]:
        labels = [chr(ord("A") + idx) for idx in range(len(candidate_titles))]
        option_lines: List[str] = []
        for idx, (label, title) in enumerate(zip(labels, candidate_titles)):
            line = f"{label}. {title}"
            if candidate_similarities and idx < len(candidate_similarities):
                similarity = candidate_similarities[idx]
                if similarity is not None:
                    try:
                        line += f" (image similarity: {float(similarity):.3f})"
                    except (TypeError, ValueError):
                        pass
            option_lines.append(line)
        base_instruction = (
            "You are an expert visual entity recognizer. Look at the image and here are some potentially relevant options."
        )
        if instruction:
            base_instruction += f"\n{instruction.strip()}"
        question_text = question or "Which option best matches the image?"
        if top_k <= 1:
            guidance = (
                "Reply with 'Answer: <label>' where <label> is one of the option letters. "
                "Do not include explanations."
            )
        else:
            guidance = (
                f"Reply with 'Answer: <label1>, <label2>, ...' listing the top {top_k} option letters "
                "from most to least likely based on the image. Do not include explanations."
            )
        if self._is_deepseek_backend():
            if top_k <= 1:
                guidance = "Output exactly one line: Answer: <label>"
            else:
                guidance = (
                    f"Output exactly one line: Answer: <label1>, <label2>, ... with exactly {top_k} labels."
                )
        options_block = "\n".join(option_lines)
        prompt = (
            f"{base_instruction}\n"
            f"Question: {question_text}\n"
            f"Options:\n{options_block}\n"
            f"{guidance}"
        )
        return labels, prompt

    def _image_to_data_url(self, image: Image.Image) -> str:
        fmt = self.image_format if self.image_format in {"JPEG", "PNG", "WEBP"} else "JPEG"
        mime = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }[fmt]
        image_to_save = image
        if fmt == "JPEG" and image_to_save.mode != "RGB":
            image_to_save = image_to_save.convert("RGB")
        payload = BytesIO()
        save_kwargs: Dict[str, Any] = {}
        if fmt == "JPEG":
            save_kwargs["quality"] = self.image_quality
        image_to_save.save(payload, format=fmt, **save_kwargs)
        encoded = base64.b64encode(payload.getvalue()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _normalize_content_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        item_type = item.get("type")
        if item_type == "text":
            return {"type": "text", "text": str(item.get("text", ""))}

        if item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and "url" in image_url:
                return {"type": "image_url", "image_url": {"url": str(image_url["url"])}}
            if isinstance(image_url, str) and image_url:
                return {"type": "image_url", "image_url": {"url": image_url}}
            return None

        if item_type != "image":
            return None

        image_obj = item.get("image")
        if isinstance(image_obj, Image.Image):
            return {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_obj)}}
        if isinstance(image_obj, str) and image_obj:
            return {"type": "image_url", "image_url": {"url": self._image_path_to_data_url(image_obj)}}
        return None

    def _to_openai_messages(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str):
                serialized.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                serialized.append({"role": role, "content": str(content)})
                continue
            packed_content: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict):
                    normalized = self._normalize_content_item(item)
                    if normalized is not None:
                        packed_content.append(normalized)
            serialized.append({"role": role, "content": packed_content})
        return serialized

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
                        continue
                    text_value = item.get("text")
                    if text_value is not None:
                        chunks.append(str(text_value))
                else:
                    chunks.append(str(item))
            return "".join(chunks)
        return str(content)

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        del images  # Images are serialized from the message payload directly.
        response = self._chat_completions_create(
            messages=self._to_openai_messages(messages),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if not response.choices:
            return ""
        content = response.choices[0].message.content
        return self._extract_text_from_response(content).strip()

    @staticmethod
    def _is_unsupported_token_param_error(exc: Exception, param_name: str) -> bool:
        message = str(exc).lower()
        if not message:
            return False
        if param_name.lower() not in message:
            return False
        unsupported_hints = (
            "unsupported parameter",
            "not supported",
            "unknown parameter",
            "invalid_request_error",
        )
        return any(hint in message for hint in unsupported_hints)

    def _prefers_max_completion_tokens(self) -> bool:
        model_name = str(self.model_name or "").strip().lower()
        # GPT-5 family requires max_completion_tokens on chat.completions.
        return model_name.startswith("gpt-5")

    def _chat_completions_create(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ):
        token_budget = max(1, int(max_new_tokens))
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": max(0.0, float(temperature)),
        }
        primary = "max_completion_tokens" if self._prefers_max_completion_tokens() else "max_tokens"
        secondary = "max_tokens" if primary == "max_completion_tokens" else "max_completion_tokens"
        try:
            return self.client.chat.completions.create(
                **request_kwargs,
                **{primary: token_budget},
            )
        except Exception as exc:
            if not self._is_unsupported_token_param_error(exc, primary):
                raise
            LOGGER.warning(
                "chat.completions rejected %s for model %s; retrying with %s",
                primary,
                self.model_name,
                secondary,
            )
            return self.client.chat.completions.create(
                **request_kwargs,
                **{secondary: token_budget},
            )

    def _build_image_text_messages(
        self,
        image_path: str,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._image_path_to_data_url(image_path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return messages

    def _parse_probability_payload(
        self,
        raw: str,
        candidate_titles: Sequence[str],
    ) -> IdentificationScores:
        parsed = self._extract_first_json_object(raw)
        if not parsed:
            raise ValueError("Unable to parse probability JSON from Qwen response.")

        limit = len(candidate_titles)
        candidate_probs_raw = parsed.get("candidate_probabilities")
        values: List[Optional[float]] = []
        if isinstance(candidate_probs_raw, list):
            for item in candidate_probs_raw:
                if isinstance(item, dict):
                    value = item.get("probability") or item.get("score")
                else:
                    value = item
                values.append(self._coerce_probability(value))
        if not values:
            candidates_block = parsed.get("candidates")
            if isinstance(candidates_block, list):
                for entry in candidates_block:
                    value = None
                    if isinstance(entry, dict):
                        value = (
                            entry.get("probability")
                            or entry.get("score")
                            or entry.get("likelihood")
                        )
                    values.append(self._coerce_probability(value))
        if not values:
            raise ValueError("Probability list missing in Qwen response.")

        if len(values) < limit:
            values.extend([None] * (limit - len(values)))
        candidate_probs: List[float] = []
        for idx in range(limit):
            prob = values[idx] if idx < len(values) else None
            if prob is None:
                candidate_probs.append(0.0)
            else:
                candidate_probs.append(max(0.0, float(prob)))
        none_raw = (
            parsed.get("none_probability")
            if "none_probability" in parsed
            else parsed.get("none_of_the_above_probability", parsed.get("other_probability"))
        )
        none_prob = self._coerce_probability(none_raw)
        if none_prob is None:
            none_prob = max(0.0, 1.0 - sum(candidate_probs))
        total = sum(candidate_probs) + none_prob
        if total > 0:
            candidate_probs = [prob / total for prob in candidate_probs]
            none_prob = none_prob / total
        candidates = [
            CandidateScore(index=idx, title=title, probability=candidate_probs[idx])
            for idx, title in enumerate(candidate_titles)
        ]
        return IdentificationScores(
            candidates=candidates,
            none_probability=max(0.0, min(1.0, none_prob)),
            raw_response=raw,
        )

    def score_candidates(
        self,
        image_path: str,
        candidate_titles: Sequence[str],
        question: Optional[str] = None,
        instruction: Optional[str] = None,
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> IdentificationScores:
        if not candidate_titles:
            raise ValueError("candidate_titles must not be empty.")
        prompt = self._format_scoring_prompt(
            candidate_titles,
            question,
            instruction,
            candidate_similarities=candidate_similarities,
        )
        messages = self._build_image_text_messages(
            image_path=image_path,
            prompt=prompt,
            system_prompt=self._strict_system_prompt(expects_json=True),
        )
        raw = self._generate(messages, images=None, max_new_tokens=max_new_tokens, temperature=temperature)
        return self._parse_probability_payload(raw=raw, candidate_titles=candidate_titles)

    def identify(
        self,
        image_path: str,
        candidate_titles: Sequence[str],
        question: Optional[str] = None,
        instruction: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 1,
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
    ) -> IdentificationResult:
        if not candidate_titles:
            raise ValueError("candidate_titles must not be empty.")
        top_k = max(1, min(top_k, len(candidate_titles)))
        labels, prompt = self._format_identification_prompt(
            candidate_titles,
            question,
            instruction,
            top_k=top_k,
            candidate_similarities=candidate_similarities,
        )
        messages = self._build_image_text_messages(
            image_path=image_path,
            prompt=prompt,
            system_prompt=self._strict_system_prompt(expects_json=False),
        )
        raw = self._generate(messages, images=None, max_new_tokens=max_new_tokens, temperature=temperature)
        selection, idx, matched_by, ranked_indices, ranked_labels = self._parse_identification(
            raw,
            labels,
            candidate_titles,
            top_k=top_k,
        )
        if ranked_indices and idx < 0:
            idx = ranked_indices[0]
            selection = candidate_titles[idx]
            matched_by = matched_by or "ranked_labels"
        if idx < 0 and candidate_titles:
            idx = 0
            selection = candidate_titles[0]
        return IdentificationResult(
            raw_response=raw,
            selected_option=selection,
            selected_index=idx,
            matched_by=matched_by,
            ranked_labels=ranked_labels,
            ranked_indices=ranked_indices,
        )

    def _format_joint_identification_prompt(
        self,
        candidate_titles: Sequence[str],
        question: Optional[str],
        instruction: Optional[str],
        top_k: int,
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
    ) -> Tuple[List[str], str]:
        labels = [chr(ord("A") + idx) for idx in range(len(candidate_titles))]
        option_lines: List[str] = []
        for idx, (label, title) in enumerate(zip(labels, candidate_titles)):
            line = f"{label}. {title}"
            if candidate_similarities and idx < len(candidate_similarities):
                similarity = candidate_similarities[idx]
                if similarity is not None:
                    try:
                        line += f" (retrieval similarity: {float(similarity):.3f})"
                    except (TypeError, ValueError):
                        pass
            option_lines.append(line)
        question_text = question or "Which candidate best matches this image?"
        guidance = (
            "Return ONLY a JSON object with keys:\n"
            "1) ranked_labels: array of option labels ordered from most likely to least likely "
            f"(length should be {top_k})\n"
            "2) candidate_probabilities: array of probabilities for ALL candidates in the same order as options\n"
            "3) none_probability: probability that none of the candidates is correct\n"
            "All probabilities must be decimals, non-negative, and sum to 1 within 0.01."
        )
        if self._is_deepseek_backend():
            guidance = (
                "Return JSON only (no markdown fences) with keys:\n"
                "ranked_labels: array of option labels ordered from most likely to least likely "
                f"(exactly {top_k} labels)\n"
                "candidate_probabilities: array of probabilities for all candidates in option order\n"
                "none_probability: probability that none of the candidates is correct\n"
                "All probabilities must be numeric, non-negative, and sum to 1 within 0.01."
            )
        base_instruction = "You are an expert visual entity recognizer."
        if instruction:
            base_instruction += f"\n{instruction.strip()}"
        prompt = (
            f"{base_instruction}\n"
            f"Question: {question_text}\n"
            f"Options:\n{chr(10).join(option_lines)}\n"
            f"{guidance}"
        )
        return labels, prompt

    def identify_and_score(
        self,
        image_path: str,
        candidate_titles: Sequence[str],
        question: Optional[str] = None,
        instruction: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 1,
        candidate_similarities: Optional[Sequence[Optional[float]]] = None,
    ) -> Tuple[IdentificationResult, IdentificationScores]:
        if not candidate_titles:
            raise ValueError("candidate_titles must not be empty.")
        top_k = max(1, min(top_k, len(candidate_titles)))
        labels, prompt = self._format_joint_identification_prompt(
            candidate_titles=candidate_titles,
            question=question,
            instruction=instruction,
            top_k=top_k,
            candidate_similarities=candidate_similarities,
        )
        messages = self._build_image_text_messages(
            image_path=image_path,
            prompt=prompt,
            system_prompt=self._strict_system_prompt(expects_json=True),
        )
        raw = self._generate(messages, images=None, max_new_tokens=max_new_tokens, temperature=temperature)

        parsed = self._extract_first_json_object(raw)
        ranked_labels: List[str] = []
        if isinstance(parsed, dict):
            ranked_labels_raw = parsed.get("ranked_labels")
            if isinstance(ranked_labels_raw, list):
                for item in ranked_labels_raw:
                    label = str(item).strip().upper()
                    if label in labels and label not in ranked_labels:
                        ranked_labels.append(label)
            ranked_indices_raw = parsed.get("ranked_indices")
            if not ranked_labels and isinstance(ranked_indices_raw, list):
                for item in ranked_indices_raw:
                    try:
                        idx = int(item)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(labels):
                        label = labels[idx]
                        if label not in ranked_labels:
                            ranked_labels.append(label)
        if not ranked_labels:
            ranked_labels = self._extract_ranked_labels(raw, labels)
        ranked_labels = ranked_labels[:top_k]
        ranked_indices = [labels.index(label) for label in ranked_labels]

        selection, selected_index, matched_by, parsed_ranked_indices, parsed_ranked_labels = self._parse_identification(
            raw,
            labels,
            candidate_titles,
            top_k=top_k,
        )
        if ranked_indices:
            selected_index = ranked_indices[0]
            selection = candidate_titles[selected_index]
            matched_by = "joint_json"
        elif parsed_ranked_indices:
            ranked_indices = parsed_ranked_indices[:top_k]
            ranked_labels = parsed_ranked_labels[:top_k]

        if selected_index < 0 and candidate_titles:
            selected_index = 0
            selection = candidate_titles[0]
            if not ranked_indices:
                ranked_indices = [0]
                ranked_labels = ["A"]
            matched_by = matched_by or "fallback_first_option"

        identification = IdentificationResult(
            raw_response=raw,
            selected_option=selection,
            selected_index=selected_index,
            matched_by=matched_by,
            ranked_labels=ranked_labels,
            ranked_indices=ranked_indices,
        )
        scores = self._parse_probability_payload(raw=raw, candidate_titles=candidate_titles)
        return identification, scores

    def answer_question(
        self,
        image_path: Optional[str],
        question: str,
        context: Optional[Union[str, Sequence[str]]] = None,
        require_reasoning: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        return_full: bool = False,
        dataset_name: Optional[str] = None,
        prompt_metadata: Optional[Dict[str, Any]] = None,
    ) -> Union[str, AnswerResult]:
        context_text = self._prepare_context(context)
        prompt_parts = self._build_prompt_parts(
            question=question,
            context_text=context_text,
            require_reasoning=require_reasoning,
            dataset_name=dataset_name,
            prompt_metadata=prompt_metadata,
        )
        messages: List[Dict[str, Any]] = []
        if prompt_parts.system:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt_parts.system}],
                }
            )
        user_content: List[Dict[str, Any]] = []
        if image_path:
            user_content.append(
                {"type": "image_url", "image_url": {"url": self._image_path_to_data_url(image_path)}}
            )
        user_content.append({"type": "text", "text": prompt_parts.user})
        messages.append({"role": "user", "content": user_content})

        raw = self._generate(
            messages,
            images=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        answer = self._extract_answer(raw)
        result = AnswerResult(raw_response=raw, answer=answer)
        return result if return_full else result.answer

    def image_cache_stats(self) -> Dict[str, int]:
        return {
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "size": len(self._image_data_url_cache),
            "capacity": int(self.image_cache_size),
        }


__all__ = ["VLLMHostQwenVLModel"]
