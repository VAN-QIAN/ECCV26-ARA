"""InfoSeek answer-reward style scoring utilities.

This module is a lightweight, dependency-free adaptation of the scoring logic
used by the ReflectiVA InfoSeek `answer_reward_utils.py` reference
and `score_infoseek_methods_with_answer_reward.py`.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional, Tuple


SCORING_INFO = {
    "source": "answer_reward_utils_reference",
    "string_time_rule": "normalized_reference_substring_in_prediction",
    "numerical_rule": "metric_numerical_range(tolerance=0.1)",
}

_PUNCTUATION = string.punctuation + "‘’´`_"
_DIGIT_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "entailment": "yes",
    "true": "yes",
    "contradiction": "no",
    "false": "no",
}
_CONTRACTIONS = {
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "isnt": "isn't",
    "shouldnt": "shouldn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
    "youve": "you've",
    "theyre": "they're",
    "ive": "i've",
    "im": "i'm",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def standardize_digits_and_contractions(text: str) -> str:
    output: List[str] = []
    for token in text.split():
        token = _DIGIT_MAP.get(token, token)
        token = _CONTRACTIONS.get(token, token)
        output.append(token)
    return " ".join(output)


def preprocess_answer(answer: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(the answer is|a|an|the)\b", " ", text)

    def replace_punctuation(text: str) -> str:
        to_replace = set(_PUNCTUATION)
        return "".join("" if ch in to_replace else ch for ch in text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    answer = answer.lower().replace("\n", " ").replace("\t", " ").strip()
    if answer.startswith("<extra_id_0> "):
        answer = answer.replace("<extra_id_0> ", "", 1)
    answer = replace_punctuation(answer)
    answer = remove_articles(answer)
    answer = standardize_digits_and_contractions(answer)
    return white_space_fix(answer)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punctuation(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def _find_all(text: str, char: str):
    idx = text.find(char)
    while idx != -1:
        yield idx
        idx = text.find(char, idx + 1)


def clean_str_range(text: str) -> str:
    idx_list = list(_find_all(text, "-"))
    idx_replace = [idx for idx in idx_list if idx >= 1 and text[idx - 1].isdigit()]
    return "".join(" - " if idx in idx_replace else ch for idx, ch in enumerate(text))


def process_numerical_answer(string_number: str):
    string_number = clean_str_range(string_number)
    matches = re.findall(
        r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
        string_number,
    )
    matches = [match.replace(",", "").strip(".") for match in matches]
    values: List[float] = []
    for number in matches:
        if number.count(".") > 1:
            number = number.split(".")[0]
        values.append(float(number))

    if len(values) > 2:
        values = values[:2]
    if len(values) == 2:
        return [values[0], values[1]] if values[0] <= values[1] else values[0]
    if len(values) == 1:
        return values[0]
    return [0, 0]


def safe_division(x: float, y: float) -> float:
    return x / y if y != 0 else 0.0


def range_intersection_over_union(x_list: List[float], y_list: List[float]) -> float:
    min_1, max_1 = min(x_list), max(x_list)
    min_2, max_2 = min(y_list), max(y_list)
    overlap = max(0.0, min(max_1, max_2) - max(min_1, min_2))
    len_x = (max_1 - min_1) + 1e-12
    len_y = (max_2 - min_2) + 1e-12
    return safe_division(overlap, len_x + len_y - overlap)


def metric_numerical_range(pred: Any, answer: Any, tolerance: float = 0.1) -> int:
    answer = list(answer) if isinstance(answer, tuple) else answer
    pred = list(pred) if isinstance(pred, tuple) else pred

    if not isinstance(answer, list):
        answer = [answer * (1 - tolerance), answer * (1 + tolerance)]

    if not isinstance(pred, list):
        return int(answer[0] <= pred <= answer[1])

    if answer[0] <= pred[0] <= answer[1] and answer[0] <= pred[1] <= answer[1]:
        return 1
    return int(range_intersection_over_union(pred, answer) >= 0.5 - 1e-12)


def _answer_references(answer_eval: Any) -> List[str]:
    values = answer_eval if isinstance(answer_eval, list) else [answer_eval]
    return [_safe_text(ref) for ref in values if _safe_text(ref)]


def score_method_prediction(
    example: Dict[str, Any],
    prediction_entry: Optional[Dict[str, Any]],
) -> Tuple[int, Optional[Any]]:
    """Score one method prediction using answer-reward style matching."""
    if not prediction_entry:
        return 0, None

    prediction_text = _safe_text(prediction_entry.get("prediction"))
    if not prediction_text:
        return 0, None

    question_type = _safe_text(example.get("question_type")).lower()
    references = _answer_references(example.get("answer_eval", []))
    if not references:
        return 0, None

    if question_type == "numerical":
        candidate_text = standardize_digits_and_contractions(
            prediction_text.lower().replace("\n", " ").replace("\t", " ").strip()
        )
        candidate_value = process_numerical_answer(candidate_text)
        if candidate_value == [0, 0]:
            return 0, candidate_value

        best = 0
        for reference in references:
            ref_value = process_numerical_answer(
                standardize_digits_and_contractions(
                    reference.lower().replace("\n", " ").replace("\t", " ").strip()
                )
            )
            if ref_value == [0, 0]:
                continue
            best = max(best, metric_numerical_range(candidate_value, ref_value))
            if best == 1:
                break
        return int(best), candidate_value

    processed_prediction = normalize_answer(preprocess_answer(prediction_text))
    for reference in references:
        processed_reference = normalize_answer(preprocess_answer(reference))
        if processed_reference and processed_reference in processed_prediction:
            return 1, processed_prediction
    return 0, processed_prediction
