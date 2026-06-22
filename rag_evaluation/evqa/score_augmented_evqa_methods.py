#!/usr/bin/env python3
"""Unify EVQA prediction files from multiple methods and run EVQA scoring.

Scoring logic follows the EchoSight `evqa_eval.py` reference:
- references = answer.split("|")
- score each example with evqa_utils.evaluate_example(...)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_FILE = str(REPO_ROOT / "data/ground_truth/evqa_fixed.csv")
DEFAULT_EVQA_EVAL_ROOT = str(REPO_ROOT / "rag_evaluation/evqa_eval")
DEFAULT_IBA_ANCHOR_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/IBA_anchor.jsonl")
)
DEFAULT_IBA_AUGMENTED1_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/IBA_augmented_method1.jsonl")
)
DEFAULT_IBA_AUGMENTED2_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/IBA_augmented_method2.jsonl")
)
DEFAULT_ECHOSIGHT_ANCHOR_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/EchoSight_anchor.jsonl")
)
DEFAULT_ECHOSIGHT_AUGMENTED1_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/EchoSight_augmented_method1.jsonl")
)
DEFAULT_ECHOSIGHT_AUGMENTED2_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/EchoSight_augmented_method2.jsonl")
)
DEFAULT_WIKIPRF_ANCHOR_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/Wiki_PRF_anchor.jsonl")
)
DEFAULT_WIKIPRF_AUGMENTED1_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/Wiki_PRF_augmented_method1.jsonl")
)
DEFAULT_WIKIPRF_AUGMENTED2_PATH = (
    str(REPO_ROOT / "outputs/raw_methods/evqa/augmented/Wiki_PRF_augmented_method2.jsonl")
)
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "results/evaluation/evqa/augmented")
VALID_QUESTION_TYPES = {"automatic", "templated", "multi_answer", "infoseek"}


@dataclass
class MethodConfig:
    name: str
    source_path: str
    loader: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EVQA outputs from multiple methods to unified "
            "{data_id,prediction} and score with EVQA evaluate_example."
        )
    )
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--evqa-eval-root", default=DEFAULT_EVQA_EVAL_ROOT)
    parser.add_argument("--iba-anchor-path", default=DEFAULT_IBA_ANCHOR_PATH)
    parser.add_argument("--iba-augmented1-path", default=DEFAULT_IBA_AUGMENTED1_PATH)
    parser.add_argument("--iba-augmented2-path", default=DEFAULT_IBA_AUGMENTED2_PATH)
    parser.add_argument("--echosight-anchor-path", default=DEFAULT_ECHOSIGHT_ANCHOR_PATH)
    parser.add_argument("--echosight-augmented1-path", default=DEFAULT_ECHOSIGHT_AUGMENTED1_PATH)
    parser.add_argument("--echosight-augmented2-path", default=DEFAULT_ECHOSIGHT_AUGMENTED2_PATH)
    parser.add_argument("--wikiprf-anchor-path", default=DEFAULT_WIKIPRF_ANCHOR_PATH)
    parser.add_argument("--wikiprf-augmented1-path", default=DEFAULT_WIKIPRF_AUGMENTED1_PATH)
    parser.add_argument("--wikiprf-augmented2-path", default=DEFAULT_WIKIPRF_AUGMENTED2_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--disable-ouriba-log-fallback",
        action="store_true",
        help="Disable parsing IBA log files when JSONL has no prediction field.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug option: evaluate only first N GT examples (0 means all).",
    )
    parser.add_argument(
        "--allow-exact-match-fallback",
        action="store_true",
        help=(
            "Use a dependency-light exact-match fallback if EVQA/BEM scoring "
            "cannot be imported. Intended for smoke tests only."
        ),
    )
    return parser.parse_args()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^0-9a-zA-Z_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "method"


def parse_boxed_answer(text: str) -> Optional[str]:
    marker = "\\boxed"
    marker_idx = text.rfind(marker)
    if marker_idx < 0:
        return None

    brace_start = text.find("{", marker_idx)
    if brace_start < 0:
        return None

    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : idx].strip()
    return None


def extract_final_answer_text(raw_prediction: Any) -> str:
    text = _safe_text(raw_prediction)
    if not text:
        return ""

    boxed = parse_boxed_answer(text)
    if boxed:
        return boxed

    answer_tag = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if answer_tag:
        candidate = _safe_text(answer_tag.group(1))
        boxed_from_tag = parse_boxed_answer(candidate)
        return boxed_from_tag if boxed_from_tag else candidate

    final_match = re.search(r"final answer\s*[:：]\s*(.+)$", text, flags=re.I | re.S)
    if final_match:
        candidate = _safe_text(final_match.group(1))
        boxed_from_final = parse_boxed_answer(candidate)
        return boxed_from_final if boxed_from_final else candidate

    return text


def split_reference_answers(answer_text: str) -> List[str]:
    return [ans.strip() for ans in _safe_text(answer_text).split("|") if ans.strip()]


_FALLBACK_DIGIT_MAP = {
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
}


def _fallback_preprocess_answer(answer: Any) -> str:
    text = _safe_text(answer).lower().replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\b(the answer is|a|an|the)\b", " ", text)
    text = "".join("" if char in string.punctuation + "'`_" else char for char in text)
    tokens = [_FALLBACK_DIGIT_MAP.get(token, token) for token in text.split()]
    return " ".join(tokens)


def exact_match_evaluate_example(
    question: str,
    reference_list: List[str],
    candidate: str,
    question_type: str,
) -> float:
    del question
    processed_candidate = _fallback_preprocess_answer(candidate)
    if _safe_text(question_type) == "multi_answer":
        candidate_parts = [
            _fallback_preprocess_answer(part)
            for part in candidate.replace(" and ", ",").replace(" & ", ",").split(",")
        ]
        candidate_set = {part for part in candidate_parts if part}
        for reference in reference_list:
            reference_parts = [
                _fallback_preprocess_answer(part)
                for part in _safe_text(reference).split("&&")
            ]
            reference_set = {part for part in reference_parts if part}
            if reference_set:
                iou = len(reference_set & candidate_set) / len(reference_set | candidate_set)
                if iou >= 0.5:
                    return 1.0
        return 0.0

    for reference in reference_list:
        if _fallback_preprocess_answer(reference) == processed_candidate:
            return 1.0
    return 0.0


def strip_augmented_evqa_suffix(data_id: str) -> str:
    text = _safe_text(data_id)
    if not text:
        return ""

    # Augmented EVQA predictions often append variant suffixes, e.g.
    # E-VQA_1234__anchor / __method1 / __method1__with_position / __method2__without_position.
    match = re.match(r"^(E-VQA_\d+)(?:__.*)?$", text)
    return match.group(1) if match else text


def align_prediction_data_ids_to_gt(
    raw_predictions: List[Dict[str, Any]],
    gt_ids: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gt_id_set = set(gt_ids)
    aligned_predictions: List[Dict[str, Any]] = []

    exact_match_count = 0
    suffix_stripped_match_count = 0
    unmatched_count = 0

    for pred in raw_predictions:
        record = dict(pred)
        source_data_id = _safe_text(record.get("data_id"))
        record["source_data_id"] = source_data_id

        if not source_data_id:
            record["data_id_alignment"] = "missing_data_id"
            aligned_predictions.append(record)
            continue

        if source_data_id in gt_id_set:
            exact_match_count += 1
            record["data_id"] = source_data_id
            record["data_id_alignment"] = "exact_match"
            aligned_predictions.append(record)
            continue

        stripped_data_id = strip_augmented_evqa_suffix(source_data_id)
        if stripped_data_id and stripped_data_id in gt_id_set:
            suffix_stripped_match_count += 1
            record["data_id"] = stripped_data_id
            record["data_id_alignment"] = "suffix_stripped_to_match_gt"
        else:
            unmatched_count += 1
            record["data_id"] = source_data_id
            record["data_id_alignment"] = "unmatched_kept_as_is"
        aligned_predictions.append(record)

    metadata = {
        "total_predictions": len(aligned_predictions),
        "exact_match_count": exact_match_count,
        "suffix_stripped_match_count": suffix_stripped_match_count,
        "unmatched_count": unmatched_count,
    }
    return aligned_predictions, metadata


def load_evqa_ground_truth(test_file: str) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"EVQA test file not found: {path}")

    qid2example: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []
    duplicate_ids = 0
    skipped_invalid_qtype = 0
    missing_data_id = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"data_id", "question", "question_type", "answer"}
        if reader.fieldnames is None:
            raise ValueError(f"CSV is empty or malformed: {path}")
        missing_cols = required - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"EVQA test file missing columns {sorted(missing_cols)}; "
                f"available columns: {reader.fieldnames}"
            )

        for row in reader:
            data_id = _safe_text(row.get("data_id"))
            if not data_id:
                missing_data_id += 1
                continue
            qtype = _safe_text(row.get("question_type")).lower()
            if qtype not in VALID_QUESTION_TYPES:
                skipped_invalid_qtype += 1
                continue
            question = _safe_text(row.get("question"))
            answer = _safe_text(row.get("answer"))

            if data_id in qid2example:
                duplicate_ids += 1
            else:
                ordered_ids.append(data_id)
            qid2example[data_id] = {
                "data_id": data_id,
                "question": question,
                "question_type": qtype,
                "answer": answer,
            }

    metadata = {
        "test_file": str(path),
        "loaded_count": len(ordered_ids),
        "duplicate_ids": duplicate_ids,
        "missing_data_id": missing_data_id,
        "skipped_invalid_qtype": skipped_invalid_qtype,
    }
    return qid2example, ordered_ids, metadata


def load_jsonl_predictions(
    source_path: str,
    *,
    prediction_keys: Iterable[str] = ("prediction",),
    postprocess: Optional[Callable[[Any], str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    key_list = list(prediction_keys)
    predictions: List[Dict[str, Any]] = []
    dropped_missing_id = 0
    dropped_missing_prediction = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue

            data_id = _safe_text(record.get("data_id"))
            if not data_id:
                dropped_missing_id += 1
                continue

            prediction_raw = None
            for key in key_list:
                if key in record and record.get(key) is not None:
                    prediction_raw = record.get(key)
                    break
            if prediction_raw is None:
                dropped_missing_prediction += 1
                continue

            prediction = (
                postprocess(prediction_raw)
                if postprocess is not None
                else _safe_text(prediction_raw)
            )
            predictions.append({"data_id": data_id, "prediction": prediction})

    metadata = {
        "loader": "jsonl",
        "source_path": str(path),
        "loaded_count": len(predictions),
        "dropped_missing_id": dropped_missing_id,
        "dropped_missing_prediction": dropped_missing_prediction,
        "prediction_keys": key_list,
    }
    return predictions, metadata


def _load_json_predictions_from_file(
    path: Path,
    *,
    prediction_keys: Iterable[str],
    postprocess: Optional[Callable[[Any], str]] = None,
) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    key_list = list(prediction_keys)
    predictions: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        data_id = _safe_text(item.get("data_id"))
        if not data_id:
            continue
        raw_prediction = None
        for key in key_list:
            if key in item and item.get(key) is not None:
                raw_prediction = item.get(key)
                break
        if raw_prediction is None:
            continue
        prediction = (
            postprocess(raw_prediction)
            if postprocess is not None
            else _safe_text(raw_prediction)
        )
        predictions.append({"data_id": data_id, "prediction": prediction})
    return predictions


def load_reflectiva_predictions(source_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"ReflectiVA source not found: {path}")

    prediction_keys = ("prediction", "answers", "answer")
    predictions: List[Dict[str, Any]] = []
    scanned_files = 0

    if path.is_file():
        predictions.extend(
            _load_json_predictions_from_file(path, prediction_keys=prediction_keys)
        )
        scanned_files = 1
    else:
        json_files = sorted(path.glob("*.json"))
        scanned_files = len(json_files)
        for file_path in json_files:
            predictions.extend(
                _load_json_predictions_from_file(
                    file_path, prediction_keys=prediction_keys
                )
            )

    metadata = {
        "loader": "reflectiva_json",
        "source_path": str(path),
        "loaded_count": len(predictions),
        "scanned_files": scanned_files,
        "prediction_keys": list(prediction_keys),
    }
    return predictions, metadata


def parse_ouriba_log_answers(log_path: Path) -> Dict[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    start_pattern = re.compile(
        r"^\d{4}-\d{2}-\d{2} [0-9:,.\- ]+ - INFO - Answer (E-VQA_\d+) \|",
        flags=re.M,
    )
    starts = list(start_pattern.finditer(text))

    parsed: Dict[str, str] = {}
    marker = '| answer="'
    for idx, match in enumerate(starts):
        data_id = match.group(1)
        start = match.start()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        block = text[start:end]
        answer_pos = block.find(marker)
        if answer_pos < 0:
            continue
        answer = block[answer_pos + len(marker) :].strip()
        if answer.endswith('"'):
            answer = answer[:-1].rstrip()
        if not answer:
            continue
        parsed[data_id] = answer
    return parsed


def load_ouriba_predictions(
    source_path: str,
    *,
    enable_log_fallback: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"IBA file not found: {path}")

    key_candidates = ("prediction", "answer", "final_answer", "raw_response", "answers")
    base_predictions, metadata = load_jsonl_predictions(
        source_path,
        prediction_keys=key_candidates,
    )
    predictions = list(base_predictions)
    existing_ids = {
        _safe_text(entry.get("data_id"))
        for entry in base_predictions
        if _safe_text(entry.get("data_id"))
    }

    log_predictions_added = 0
    used_logs: List[str] = []
    if enable_log_fallback:
        log_candidates = [
            path.with_name("evqa_answer.log"),
            path.with_name("evqa_answer.nohup.out"),
        ]
        for log_path in log_candidates:
            if not log_path.exists():
                continue
            parsed = parse_ouriba_log_answers(log_path)
            used_logs.append(str(log_path))
            for data_id, prediction in parsed.items():
                if data_id not in existing_ids:
                    predictions.append({"data_id": data_id, "prediction": prediction})
                    existing_ids.add(data_id)
                    log_predictions_added += 1

    metadata["loader"] = "ouriba_jsonl_plus_log_fallback"
    metadata["log_fallback_enabled"] = bool(enable_log_fallback)
    metadata["log_files_used"] = used_logs
    metadata["log_predictions_added"] = log_predictions_added
    metadata["loaded_count_after_fallback"] = len(predictions)
    if not base_predictions and enable_log_fallback and log_predictions_added > 0:
        metadata["warning"] = (
            "No prediction field found in IBA JSONL; used answers parsed from logs."
        )
    return predictions, metadata


def deduplicate_predictions(
    predictions: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    pred_map: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    for pred in predictions:
        data_id = _safe_text(pred.get("data_id"))
        if not data_id:
            continue
        if data_id in pred_map:
            duplicates += 1
        pred_map[data_id] = {"data_id": data_id, "prediction": _safe_text(pred.get("prediction"))}
    return pred_map, duplicates


def build_raw_prediction_metadata(
    raw_predictions: List[Dict[str, Any]],
    gt_ids: List[str],
    *,
    eval_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    gt_id_set = set(gt_ids)
    eval_id_set = set(eval_ids) if eval_ids is not None else gt_id_set

    occurrence_count_by_id: Dict[str, int] = {}
    last_index_by_id: Dict[str, int] = {}
    for idx, pred in enumerate(raw_predictions):
        data_id = _safe_text(pred.get("data_id"))
        if not data_id:
            continue
        occurrence_count_by_id[data_id] = occurrence_count_by_id.get(data_id, 0) + 1
        last_index_by_id[data_id] = idx

    seen_count_by_id: Dict[str, int] = {}
    metadata_records: List[Dict[str, Any]] = []
    for idx, pred in enumerate(raw_predictions):
        data_id = _safe_text(pred.get("data_id"))
        source_data_id = _safe_text(pred.get("source_data_id"))
        data_id_alignment = _safe_text(pred.get("data_id_alignment"))
        prediction = _safe_text(pred.get("prediction"))

        occurrence_idx = 0
        occurrence_count = 0
        is_duplicate = False
        kept_after_dedup = False
        in_gt = False
        in_eval_subset = False

        if data_id:
            seen_count_by_id[data_id] = seen_count_by_id.get(data_id, 0) + 1
            occurrence_idx = seen_count_by_id[data_id]
            occurrence_count = occurrence_count_by_id.get(data_id, 0)
            is_duplicate = occurrence_count > 1
            kept_after_dedup = idx == last_index_by_id.get(data_id, idx)
            in_gt = data_id in gt_id_set
            in_eval_subset = data_id in eval_id_set

        metadata_records.append(
            {
                "raw_index": idx,
                "source_data_id": source_data_id,
                "data_id": data_id,
                "data_id_alignment": data_id_alignment,
                "data_id_changed_for_alignment": bool(source_data_id and source_data_id != data_id),
                "prediction": prediction,
                "has_prediction": bool(prediction),
                "occurrence_index_for_data_id": occurrence_idx,
                "occurrence_count_for_data_id": occurrence_count,
                "is_duplicate_data_id": is_duplicate,
                "kept_after_dedup": kept_after_dedup,
                "in_ground_truth": in_gt,
                "in_eval_subset": in_eval_subset,
            }
        )

    return metadata_records


def write_jsonl(path: Path, entries: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")


def resolve_evaluate_example(evqa_eval_root: str) -> Callable[..., float]:
    root = Path(evqa_eval_root)
    if not root.exists():
        raise FileNotFoundError(f"evqa_eval root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)

    try:
        from evqa_utils import evaluate_example  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import evqa_utils.evaluate_example. "
            f"Make sure dependencies are installed and --evqa-eval-root is correct. Error: {exc}"
        ) from exc
    return evaluate_example


def score_predictions(
    method_name: str,
    pred_map: Dict[str, Dict[str, Any]],
    qid2example: Dict[str, Dict[str, Any]],
    ordered_ids: List[str],
    *,
    evaluate_example_fn: Callable[..., float],
    max_samples: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    evaluate_ids = ordered_ids[:max_samples] if max_samples and max_samples > 0 else ordered_ids

    total_gt = len(evaluate_ids)
    evaluated = 0
    score_sum = 0.0
    missing_ids: List[str] = []
    per_prediction_metadata: List[Dict[str, Any]] = []

    per_type: Dict[str, Dict[str, float]] = {}
    for data_id in evaluate_ids:
        example = qid2example[data_id]
        qtype = _safe_text(example.get("question_type")).lower()
        if qtype not in per_type:
            per_type[qtype] = {"total": 0.0, "evaluated": 0.0, "score_sum": 0.0}
        per_type[qtype]["total"] += 1.0

        prediction_entry = pred_map.get(data_id)
        candidate = _safe_text(prediction_entry.get("prediction")) if prediction_entry else ""
        references = split_reference_answers(_safe_text(example.get("answer")))
        meta_entry: Dict[str, Any] = {
            "data_id": data_id,
            "question_type": qtype,
            "question": _safe_text(example.get("question")),
            "references": references,
            "prediction": candidate,
            "has_prediction": bool(candidate),
            "was_evaluated": False,
            "eval_status": "",
            "score": None,
            "score_gt_zero": None,
        }
        if not candidate:
            missing_ids.append(data_id)
            meta_entry["eval_status"] = "missing_prediction"
            per_prediction_metadata.append(meta_entry)
            continue

        if not references:
            missing_ids.append(data_id)
            meta_entry["eval_status"] = "missing_reference"
            per_prediction_metadata.append(meta_entry)
            continue

        score = float(
            evaluate_example_fn(
                example.get("question", ""),
                reference_list=references,
                candidate=candidate,
                question_type=qtype,
            )
        )
        score_sum += score
        evaluated += 1
        per_type[qtype]["evaluated"] += 1.0
        per_type[qtype]["score_sum"] += score
        meta_entry["was_evaluated"] = True
        meta_entry["eval_status"] = "scored"
        meta_entry["score"] = score
        meta_entry["score_gt_zero"] = score > 0.0
        per_prediction_metadata.append(meta_entry)

    by_type_output: Dict[str, Dict[str, Any]] = {}
    for qtype, stats in per_type.items():
        total = int(stats["total"])
        evaluated_q = int(stats["evaluated"])
        score_sum_q = float(stats["score_sum"])
        by_type_output[qtype] = {
            "score_sum": score_sum_q,
            "total": total,
            "evaluated": evaluated_q,
            "coverage": (evaluated_q / total) if total else 0.0,
            "average_on_evaluated": (score_sum_q / evaluated_q) if evaluated_q else 0.0,
            "average_over_total": (score_sum_q / total) if total else 0.0,
        }

    summary = {
        "method": method_name,
        "total_gt": total_gt,
        "evaluated": evaluated,
        "coverage": (evaluated / total_gt) if total_gt else 0.0,
        "score_sum": score_sum,
        "average_on_evaluated": (score_sum / evaluated) if evaluated else 0.0,
        "average_over_total": (score_sum / total_gt) if total_gt else 0.0,
        "missing_count": len(missing_ids),
        "missing_ids_sample": missing_ids[:20],
        "per_type": by_type_output,
    }
    return summary, per_prediction_metadata


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qid2example, ordered_ids, gt_meta = load_evqa_ground_truth(args.test_file)
    print(f"Loaded EVQA GT: {len(ordered_ids)} rows from {args.test_file}")

    try:
        evaluate_example_fn = resolve_evaluate_example(args.evqa_eval_root)
        print(f"Loaded scoring function from: {args.evqa_eval_root}")
    except RuntimeError as exc:
        if not args.allow_exact_match_fallback:
            raise
        print(
            "Warning: EVQA/BEM scoring is unavailable; using exact-match "
            f"fallback for this run only. Original error: {exc}"
        )
        evaluate_example_fn = exact_match_evaluate_example

    methods: List[MethodConfig] = [
        MethodConfig(
            name="IBA-anchor",
            source_path=args.iba_anchor_path,
            loader=lambda p: load_ouriba_predictions(
                p,
                enable_log_fallback=not args.disable_ouriba_log_fallback,
            ),
        ),
        MethodConfig(
            name="IBA-augmented_method1",
            source_path=args.iba_augmented1_path,
            loader=lambda p: load_ouriba_predictions(
                p,
                enable_log_fallback=not args.disable_ouriba_log_fallback,
            ),
        ),
        MethodConfig(
            name="IBA-augmented_method2",
            source_path=args.iba_augmented2_path,
            loader=lambda p: load_ouriba_predictions(
                p,
                enable_log_fallback=not args.disable_ouriba_log_fallback,
            ),
        ),
        MethodConfig(
            name="EchoSight-anchor",
            source_path=args.echosight_anchor_path,
            loader=lambda p: load_jsonl_predictions(p, prediction_keys=("prediction",)),
        ),
        MethodConfig(
            name="EchoSight-augmented_method1",
            source_path=args.echosight_augmented1_path,
            loader=lambda p: load_jsonl_predictions(p, prediction_keys=("prediction",)),
        ),
        MethodConfig(
            name="EchoSight-augmented_method2",
            source_path=args.echosight_augmented2_path,
            loader=lambda p: load_jsonl_predictions(p, prediction_keys=("prediction",)),
        ),
        MethodConfig(
            name="Wiki-PRF-anchor",
            source_path=args.wikiprf_anchor_path,
            loader=lambda p: load_jsonl_predictions(p, prediction_keys=("prediction",)),
        ),
        MethodConfig(
            name="Wiki-PRF-augmented_method1",
            source_path=args.wikiprf_augmented1_path,
            loader=lambda p: load_jsonl_predictions(p, prediction_keys=("prediction",)),
        ),
        MethodConfig(
            name="Wiki-PRF-augmented_method2",
            source_path=args.wikiprf_augmented2_path,
            loader=lambda p: load_jsonl_predictions(p, prediction_keys=("prediction",)),
        ),
    ]

    summary: Dict[str, Any] = {
        "test_file": args.test_file,
        "evqa_eval_root": args.evqa_eval_root,
        "gt_metadata": gt_meta,
        "max_samples": int(args.max_samples),
        "methods": {},
    }
    eval_ids = ordered_ids[: int(args.max_samples)] if int(args.max_samples) > 0 else ordered_ids

    for method in methods:
        print(f"\nProcessing method: {method.name}")
        raw_predictions, loader_meta = method.loader(method.source_path)
        raw_predictions, data_id_alignment_meta = align_prediction_data_ids_to_gt(
            raw_predictions,
            ordered_ids,
        )
        loader_meta["data_id_alignment"] = data_id_alignment_meta
        raw_metadata_records = build_raw_prediction_metadata(
            raw_predictions,
            ordered_ids,
            eval_ids=eval_ids,
        )
        pred_map, duplicate_count = deduplicate_predictions(raw_predictions)

        aligned_entries = [
            {"data_id": data_id, "prediction": pred_map[data_id]["prediction"]}
            for data_id in ordered_ids
            if data_id in pred_map and _safe_text(pred_map[data_id]["prediction"])
        ]
        unified_path = output_dir / f"{_slugify(method.name)}_unified.jsonl"
        write_jsonl(unified_path, aligned_entries)
        raw_metadata_path = output_dir / f"{_slugify(method.name)}_raw_records_metadata.jsonl"
        write_jsonl(raw_metadata_path, raw_metadata_records)

        score_result, per_prediction_metadata = score_predictions(
            method.name,
            pred_map,
            qid2example,
            ordered_ids,
            evaluate_example_fn=evaluate_example_fn,
            max_samples=int(args.max_samples),
        )
        scored_metadata_path = output_dir / f"{_slugify(method.name)}_scored_metadata.jsonl"
        write_jsonl(scored_metadata_path, per_prediction_metadata)
        score_result["loader"] = loader_meta
        score_result["duplicate_id_count"] = duplicate_count
        score_result["unified_path"] = str(unified_path)
        score_result["raw_metadata_path"] = str(raw_metadata_path)
        score_result["raw_metadata_count"] = len(raw_metadata_records)
        score_result["scored_metadata_path"] = str(scored_metadata_path)
        score_result["scored_metadata_count"] = len(per_prediction_metadata)
        summary["methods"][method.name] = score_result

        print(
            f"{method.name}: "
            f"avg_eval={score_result['average_on_evaluated']:.4f}, "
            f"avg_all={score_result['average_over_total']:.4f}, "
            f"coverage={score_result['coverage']:.4f}, "
            f"evaluated={score_result['evaluated']}/{score_result['total_gt']}"
        )
        warning = loader_meta.get("warning")
        if warning:
            print(f"Warning ({method.name}): {warning}")
        alignment_meta = loader_meta.get("data_id_alignment", {})
        if alignment_meta.get("suffix_stripped_match_count", 0):
            print(
                f"ID alignment ({method.name}): "
                f"suffix_stripped={alignment_meta['suffix_stripped_match_count']}, "
                f"exact={alignment_meta.get('exact_match_count', 0)}, "
                f"unmatched={alignment_meta.get('unmatched_count', 0)}"
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
