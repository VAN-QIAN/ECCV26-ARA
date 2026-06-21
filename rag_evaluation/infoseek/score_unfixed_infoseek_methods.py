#!/usr/bin/env python3
"""Convert multi-method InfoSeek predictions to a unified format and score them.

The scoring logic reuses `score_method_prediction` from
`compute_score_enhanced_string_with_bem.py`.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# try:
from compute_score_enhanced_string_with_bem import score_method_prediction
# except ImportError:
# from rag_evaluation.infoseek.compute_score_enhanced_string_with_bem import (  # type: ignore
#     score_method_prediction,
# )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT_CSV = str(REPO_ROOT / "data/ground_truth/infoseek_unfixed_subset.csv")
DEFAULT_OURIBA_PATH = str(REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/OurIBA.jsonl")
DEFAULT_OURIBA_QWEN_PATH = str(REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/OurIBA_Qwen.jsonl")
DEFAULT_ECHOSIGHT_PATH = str(REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/EchoSight.jsonl")
DEFAULT_REFLECTIVA_DIR = str(REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/ReflectiVA")
DEFAULT_WIKIPRF_PATH = str(REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/Wiki_PRF.jsonl")
DEFAULT_COMEM_PATH = str(REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/CoMEM.jsonl")
DEFAULT_PARAMETER_LLAVA_PATH = str(REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/Parameter_LLaVA.jsonl")
DEFAULT_PARAMETER_QWEN_PATH = str(REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/Parameter_Qwen.jsonl")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "results/evaluation/infoseek/unfixed")


@dataclass
class MethodConfig:
    name: str
    source_path: str
    loader: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert predictions from multiple methods to unified "
            "{data_id, prediction} format and score with InfoSeek logic."
        )
    )
    parser.add_argument("--ground-truth-csv", default=DEFAULT_GT_CSV)
    parser.add_argument("--ouriba-path", default=DEFAULT_OURIBA_PATH)
    parser.add_argument("--ouriba-qwen-path", default=DEFAULT_OURIBA_QWEN_PATH)
    parser.add_argument("--echosight-path", default=DEFAULT_ECHOSIGHT_PATH)
    parser.add_argument("--reflectiva-dir", default=DEFAULT_REFLECTIVA_DIR)
    parser.add_argument("--wikiprf-path", default=DEFAULT_WIKIPRF_PATH)
    parser.add_argument("--comem-path", default=DEFAULT_COMEM_PATH)
    parser.add_argument("--parameter-llava-path", default=DEFAULT_PARAMETER_LLAVA_PATH)
    parser.add_argument("--parameter-qwen-path", default=DEFAULT_PARAMETER_QWEN_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--anchor-method",
        default="OurIBA",
        help=(
            "Anchor method used as the first dimension for combination buckets. "
            "All methods are still considered (2^N buckets)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Debug option: evaluate only first N GT examples (0 means all).",
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


def _coerce_numeric_if_possible(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    text = _safe_text(value).replace(",", "")
    if not text:
        return text
    if re.fullmatch(r"[+-]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return value
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)", text):
        try:
            return float(text)
        except ValueError:
            return value
    return value


def _parse_answer_field(answer: Any, question_type: str) -> List[Any]:
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        return []

    values: List[Any]
    if isinstance(answer, list):
        values = answer
    else:
        text = _safe_text(answer)
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    values = parsed
                else:
                    values = [text]
            except Exception:
                values = [text]
        elif "|" in text:
            values = [item.strip() for item in text.split("|") if item.strip()]
        else:
            values = [text]

    qtype = _safe_text(question_type).lower()
    if qtype == "numerical":
        return [_coerce_numeric_if_possible(v) for v in values]
    return [_safe_text(v) for v in values]


def load_ground_truth(gt_csv: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    path = Path(gt_csv)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"data_id", "question", "question_type", "answer"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV missing columns: {sorted(missing)}")

    qid2example: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []
    for row in df.itertuples(index=False):
        data_id = _safe_text(getattr(row, "data_id", ""))
        if not data_id:
            continue
        question = _safe_text(getattr(row, "question", ""))
        qtype = _safe_text(getattr(row, "question_type", "String")) or "String"
        answer_raw = getattr(row, "answer", "")
        answer_eval = _parse_answer_field(answer_raw, qtype)

        qid2example[data_id] = {
            "data_id": data_id,
            "question": question,
            "question_type": qtype,
            "answer_eval": answer_eval,
            "data_split": "fixed_infoseek",
        }
        ordered_ids.append(data_id)

    return qid2example, ordered_ids


def read_first_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    first_non_empty = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                first_non_empty = line.strip()
                break

    if not first_non_empty:
        raise ValueError(f"No JSON object found in file: {path}")

    try:
        obj = json.loads(first_non_empty)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text.lstrip())
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at beginning of file: {path}")
    return obj


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


def load_jsonl_predictions(
    source_path: str,
    *,
    prediction_key: str = "prediction",
    postprocess: Optional[Callable[[Any], str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    predictions: List[Dict[str, Any]] = []
    dropped_missing_id = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data_id = _safe_text(record.get("data_id"))
            if not data_id:
                dropped_missing_id += 1
                continue
            prediction_raw = record.get(prediction_key)
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
    }
    return predictions, metadata


def load_reflectiva_predictions(
    source_dir: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    directory = Path(source_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"ReflectiVA source is not a directory: {directory}")

    json_files = sorted(directory.glob("*.json"))
    predictions: List[Dict[str, Any]] = []
    malformed_files: List[str] = []

    for file_path in json_files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            malformed_files.append(str(file_path))
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            data_id = _safe_text(item.get("data_id"))
            if not data_id:
                continue
            prediction = _safe_text(item.get("prediction"))
            predictions.append({"data_id": data_id, "prediction": prediction})

    metadata = {
        "loader": "reflectiva_dir",
        "source_path": str(directory),
        "json_file_count": len(json_files),
        "loaded_count": len(predictions),
        "malformed_files": malformed_files,
    }
    return predictions, metadata


def load_wikiprf_predictions(
    source_path: str,
    gt_ids: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(source_path)

    # Newer Wiki-PRF generation detail output is JSONL with inline data_id.
    if path.suffix.lower() == ".jsonl":
        predictions, metadata = load_jsonl_predictions(
            source_path,
            prediction_key="prediction",
        )
        metadata["loader"] = "wikiprf_generation_details_jsonl"
        metadata["id_source"] = "inline_data_id"
        return predictions, metadata

    obj = read_first_json_object(path)

    results = obj.get("results")
    if not isinstance(results, list):
        raise ValueError(f'Wiki-PRF file has no list "results": {path}')

    data_ids = obj.get("data_id")
    id_source = "inline_data_id"

    if not isinstance(data_ids, list) or len(data_ids) != len(results):
        raw_outputs_path = Path(str(path).replace(".json", "_raw_outputs.jsonl"))
        id_source = "gt_order_fallback"
        if raw_outputs_path.exists():
            raw_ids: List[str] = []
            raw_predictions: List[str] = []
            with raw_outputs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    rid = _safe_text(record.get("data_id"))
                    if not rid:
                        continue
                    raw_ids.append(rid)
                    raw_predictions.append(_safe_text(record.get("prediction")))
            if raw_ids and len(raw_ids) == len(results):
                data_ids = raw_ids
                results = raw_predictions
                id_source = f"raw_outputs:{raw_outputs_path.name}"

    if not isinstance(data_ids, list) or len(data_ids) != len(results):
        data_ids = gt_ids[: len(results)]
        id_source = "gt_order_fallback"

    predictions = []
    for data_id, prediction in zip(data_ids, results):
        data_id_text = _safe_text(data_id)
        if not data_id_text:
            continue
        predictions.append(
            {"data_id": data_id_text, "prediction": _safe_text(prediction)}
        )

    metadata = {
        "loader": "wikiprf_json",
        "source_path": str(path),
        "loaded_count": len(predictions),
        "id_source": id_source,
        "raw_result_count": len(results),
    }
    if id_source == "gt_order_fallback":
        metadata["warning"] = (
            "Wiki-PRF output does not contain data_id. "
            "IDs are aligned by GT row order."
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
) -> List[Dict[str, Any]]:
    gt_id_set = set(gt_ids)

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
        prediction = _safe_text(pred.get("prediction"))

        occurrence_idx = 0
        occurrence_count = 0
        is_duplicate = False
        kept_after_dedup = False
        in_gt = False

        if data_id:
            seen_count_by_id[data_id] = seen_count_by_id.get(data_id, 0) + 1
            occurrence_idx = seen_count_by_id[data_id]
            occurrence_count = occurrence_count_by_id.get(data_id, 0)
            is_duplicate = occurrence_count > 1
            kept_after_dedup = idx == last_index_by_id.get(data_id, idx)
            in_gt = data_id in gt_id_set

        metadata_records.append(
            {
                "raw_index": idx,
                "data_id": data_id,
                "prediction": prediction,
                "has_prediction": bool(prediction),
                "occurrence_index_for_data_id": occurrence_idx,
                "occurrence_count_for_data_id": occurrence_count,
                "is_duplicate_data_id": is_duplicate,
                "kept_after_dedup": kept_after_dedup,
                "in_ground_truth": in_gt,
            }
        )

    return metadata_records


def write_jsonl(path: Path, entries: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")


def score_predictions(
    method_name: str,
    pred_map: Dict[str, Dict[str, Any]],
    qid2example: Dict[str, Dict[str, Any]],
    gt_ids: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    type_total: Dict[str, int] = {}
    type_correct: Dict[str, int] = {}

    total = len(gt_ids)
    matched = 0
    non_empty = 0
    correct = 0
    scored_records: List[Dict[str, Any]] = []
    record_map: Dict[str, Dict[str, Any]] = {}

    for data_id in gt_ids:
        example = qid2example[data_id]
        qtype = _safe_text(example.get("question_type", "String")) or "String"
        type_total[qtype] = type_total.get(qtype, 0) + 1

        pred_entry = pred_map.get(data_id)
        prediction_raw = pred_entry.get("prediction") if pred_entry else None
        prediction_text = _safe_text(prediction_raw) if pred_entry else ""
        available = pred_entry is not None
        non_empty_prediction = bool(prediction_text)

        if pred_entry is not None:
            matched += 1
            if non_empty_prediction:
                non_empty += 1

        score, processed_prediction = score_method_prediction(example, pred_entry)
        correct += int(score)
        type_correct[qtype] = type_correct.get(qtype, 0) + int(score)

        record = {
            "data_id": data_id,
            "question_type": qtype,
            "question": example.get("question"),
            "answer_eval": example.get("answer_eval"),
            "available": available,
            "prediction": prediction_text if available else None,
            "non_empty_prediction": non_empty_prediction,
            "processed_prediction": processed_prediction,
            "score": int(score),
            "correct": bool(score),
        }
        scored_records.append(record)
        record_map[data_id] = record

    per_type = {}
    for qtype, count in type_total.items():
        hit = type_correct.get(qtype, 0)
        per_type[qtype] = {
            "correct": hit,
            "total": count,
            "accuracy": (hit / count) if count else 0.0,
        }

    extra_prediction_ids = [pid for pid in pred_map.keys() if pid not in qid2example]
    summary = {
        "method": method_name,
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total) if total else 0.0,
        },
        "coverage": {
            "matched_gt_ids": matched,
            "total_gt_ids": total,
            "matched_ratio": (matched / total) if total else 0.0,
            "non_empty_predictions": non_empty,
        },
        "per_type": per_type,
        "extra_prediction_count": len(extra_prediction_ids),
        "extra_prediction_ids_sample": extra_prediction_ids[:20],
    }
    return summary, scored_records, record_map


def build_comparison_records(
    gt_ids: List[str],
    qid2example: Dict[str, Dict[str, Any]],
    method_names: List[str],
    method_record_maps: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for data_id in gt_ids:
        example = qid2example[data_id]
        methods_view: Dict[str, Any] = {}
        for method_name in method_names:
            method_record = method_record_maps[method_name].get(data_id)
            if method_record is None:
                methods_view[method_name] = {
                    "available": False,
                    "non_empty_prediction": False,
                    "prediction": None,
                    "processed_prediction": None,
                    "score": 0,
                    "correct": False,
                }
                continue

            methods_view[method_name] = {
                "available": bool(method_record.get("available")),
                "non_empty_prediction": bool(method_record.get("non_empty_prediction")),
                "prediction": method_record.get("prediction"),
                "processed_prediction": method_record.get("processed_prediction"),
                "score": int(method_record.get("score", 0)),
                "correct": bool(method_record.get("correct")),
            }

        records.append(
            {
                "data_id": data_id,
                "question_type": _safe_text(example.get("question_type")),
                "question": example.get("question"),
                "answer_eval": example.get("answer_eval"),
                "methods": methods_view,
            }
        )
    return records


def combination_key(method_order: List[str], methods_view: Dict[str, Any]) -> str:
    parts: List[str] = []
    for method_name in method_order:
        method_info = methods_view.get(method_name, {})
        tag = "T" if bool(method_info.get("correct")) else "F"
        parts.append(f"{_slugify(method_name)}_{tag}")
    return "__".join(parts)


def all_combination_keys(method_order: List[str]) -> List[str]:
    keys: List[str] = []
    for flags in itertools.product([False, True], repeat=len(method_order)):
        parts: List[str] = []
        for method_name, is_correct in zip(method_order, flags):
            parts.append(f"{_slugify(method_name)}_{'T' if is_correct else 'F'}")
        keys.append("__".join(parts))
    return keys


def write_question_type_splits(
    comparison_records: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, str]:
    split_dir = output_dir / "comparison_by_question_type"
    split_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in comparison_records:
        qtype = _safe_text(record.get("question_type")) or "Unknown"
        grouped.setdefault(qtype, []).append(record)

    output_paths: Dict[str, str] = {}
    for qtype, records in grouped.items():
        path = split_dir / f"{_slugify(qtype)}.jsonl"
        write_jsonl(path, records)
        output_paths[qtype] = str(path)
    return output_paths


def write_combination_buckets(
    comparison_records: List[Dict[str, Any]],
    method_order: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    bucket_dir = output_dir / "combination_buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    all_keys = all_combination_keys(method_order)
    bucket_records: Dict[str, List[Dict[str, Any]]] = {key: [] for key in all_keys}

    for record in comparison_records:
        key = combination_key(method_order, record.get("methods", {}))
        if key not in bucket_records:
            bucket_records[key] = []
        bucket_records[key].append(record)

    bucket_files: Dict[str, str] = {}
    bucket_counts: Dict[str, int] = {}
    for key, records in bucket_records.items():
        path = bucket_dir / f"{key}.jsonl"
        write_jsonl(path, records)
        bucket_files[key] = str(path)
        bucket_counts[key] = len(records)

    return {
        "anchor_order": method_order,
        "bucket_dir": str(bucket_dir),
        "bucket_counts": bucket_counts,
        "bucket_files": bucket_files,
        "num_buckets": len(bucket_records),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qid2example, all_gt_ids = load_ground_truth(args.ground_truth_csv)
    gt_ids = (
        all_gt_ids[: int(args.max_samples)]
        if int(args.max_samples) > 0
        else all_gt_ids
    )
    print(
        f"Loaded GT: {len(gt_ids)} rows from {args.ground_truth_csv}"
        f" (source rows: {len(all_gt_ids)})"
    )

    methods: List[MethodConfig] = [
        MethodConfig(
            name="OurIBA",
            source_path=args.ouriba_path,
            loader=lambda p: load_jsonl_predictions(p),
        ),
        MethodConfig(
            name="OurIBA-Qwen",
            source_path=args.ouriba_qwen_path,
            loader=lambda p: load_jsonl_predictions(p),
        ),
        MethodConfig(
            name="EchoSight",
            source_path=args.echosight_path,
            loader=lambda p: load_jsonl_predictions(p),
        ),
        MethodConfig(
            name="ReflectiVA",
            source_path=args.reflectiva_dir,
            loader=lambda p: load_reflectiva_predictions(p),
        ),
        MethodConfig(
            name="Wiki-PRF",
            source_path=args.wikiprf_path,
            loader=lambda p: load_wikiprf_predictions(p, gt_ids),
        ),
        MethodConfig(
            name="CoMEM",
            source_path=args.comem_path,
            loader=lambda p: load_jsonl_predictions(
                p,
                postprocess=extract_final_answer_text,
            ),
        ),
        MethodConfig(
            name="Parameter-LLaVA",
            source_path=args.parameter_llava_path,
            loader=lambda p: load_jsonl_predictions(p),
        ),
        MethodConfig(
            name="Parameter-Qwen",
            source_path=args.parameter_qwen_path,
            loader=lambda p: load_jsonl_predictions(p),
        ),
    ]

    summary: Dict[str, Any] = {
        "ground_truth_csv": args.ground_truth_csv,
        "max_samples": int(args.max_samples),
        "gt_count": len(gt_ids),
        "methods": {},
    }
    method_names: List[str] = []
    method_record_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    method_record_files: Dict[str, str] = {}
    method_raw_record_files: Dict[str, str] = {}

    per_method_record_dir = output_dir / "method_records"
    per_method_record_dir.mkdir(parents=True, exist_ok=True)
    per_method_raw_record_dir = output_dir / "raw_method_records"
    per_method_raw_record_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        print(f"\nProcessing method: {method.name}")
        raw_predictions, loader_meta = method.loader(method.source_path)
        raw_metadata_records = build_raw_prediction_metadata(raw_predictions, gt_ids)
        pred_map, duplicate_count = deduplicate_predictions(raw_predictions)

        aligned_entries = [
            {"data_id": data_id, "prediction": pred_map[data_id]["prediction"]}
            for data_id in gt_ids
            if data_id in pred_map
        ]
        unified_path = output_dir / f"{_slugify(method.name)}_unified.jsonl"
        write_jsonl(unified_path, aligned_entries)
        raw_metadata_path = (
            per_method_raw_record_dir / f"{_slugify(method.name)}_raw_records_metadata.jsonl"
        )
        write_jsonl(raw_metadata_path, raw_metadata_records)

        score_result, scored_records, record_map = score_predictions(
            method.name,
            pred_map,
            qid2example,
            gt_ids,
        )
        method_record_path = per_method_record_dir / f"{_slugify(method.name)}_records.jsonl"
        write_jsonl(method_record_path, scored_records)

        score_result["loader"] = loader_meta
        score_result["duplicate_id_count"] = duplicate_count
        score_result["unified_path"] = str(unified_path)
        score_result["raw_metadata_path"] = str(raw_metadata_path)
        score_result["raw_metadata_count"] = len(raw_metadata_records)
        score_result["scored_metadata_path"] = str(method_record_path)
        score_result["scored_metadata_count"] = len(scored_records)
        score_result["record_path"] = str(method_record_path)
        summary["methods"][method.name] = score_result
        method_names.append(method.name)
        method_record_maps[method.name] = record_map
        method_record_files[method.name] = str(method_record_path)
        method_raw_record_files[method.name] = str(raw_metadata_path)

        acc = score_result["overall"]["accuracy"]
        cov = score_result["coverage"]["matched_ratio"]
        print(
            f"{method.name}: accuracy={acc:.4f}, "
            f"coverage={cov:.4f}, unified={unified_path}"
        )
        warning = loader_meta.get("warning")
        if warning:
            print(f"Warning ({method.name}): {warning}")

    if args.anchor_method not in method_names:
        raise ValueError(
            f'Anchor method "{args.anchor_method}" is not in method list: {method_names}'
        )
    anchor_order = [args.anchor_method] + [
        m for m in method_names if m != args.anchor_method
    ]

    comparison_records = build_comparison_records(
        gt_ids,
        qid2example,
        method_names,
        method_record_maps,
    )
    comparison_path = output_dir / "comparison_records.jsonl"
    write_jsonl(comparison_path, comparison_records)

    qtype_split_paths = write_question_type_splits(comparison_records, output_dir)
    combo_meta = write_combination_buckets(comparison_records, anchor_order, output_dir)

    summary["record_outputs"] = {
        "method_record_dir": str(per_method_record_dir),
        "method_record_files": method_record_files,
        "raw_method_record_dir": str(per_method_raw_record_dir),
        "raw_method_record_files": method_raw_record_files,
        "comparison_path": str(comparison_path),
        "question_type_split_paths": qtype_split_paths,
        "combination_buckets": combo_meta,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
