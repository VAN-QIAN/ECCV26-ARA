"""I/O helpers for the Qwen pipeline."""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from utils import get_test_question, load_csv_data

from .types import RetrievalRecord


_AUGMENTED_METHOD2_QUERY_VARIANTS = ("with_position", "without_position", "legacy")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _load_csv_rows_raw(test_file: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(test_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def _looks_like_augmented_infoseek_csv(fieldnames: Sequence[str]) -> bool:
    header = set(fieldnames)
    required = {
        "anchor_data_id",
        "anchor_question",
        "anchor_image_path",
        "method1_composite_image_path",
        "method2_composite_image_path",
    }
    return required.issubset(header)


def _resolve_augmented_image_path(
    image_path: str,
    *,
    test_file_path: str,
    path_root: Optional[str] = None,
) -> Optional[str]:
    path = _normalize_text(image_path)
    if not path:
        return None
    if os.path.isabs(path):
        return path

    candidates: List[str] = []
    if path_root:
        candidates.append(os.path.normpath(os.path.join(path_root, path)))

    # Relative to cwd (legacy behavior in reference script)
    candidates.append(os.path.abspath(path))

    # Relative to the CSV directory and its parents (helps with nested results folders)
    csv_path = Path(test_file_path).resolve()
    candidate_bases: List[Path] = [csv_path.parent]
    candidate_bases.extend(list(csv_path.parents[:4]))
    for base in candidate_bases:
        candidates.append(str((base / path).resolve()))

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate

    # Return the first non-empty candidate for debugging/logging even if missing.
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def _pick_augmented_query(
    row: Dict[str, str],
    *,
    augmented_mode: str,
    augmented_query_variant: str,
) -> str:
    if augmented_mode == "anchor":
        return _normalize_text(row.get("anchor_question")) or _normalize_text(row.get("question"))
    if augmented_mode == "method1":
        return _normalize_text(row.get("method1_query")) or _normalize_text(row.get("anchor_question"))
    if augmented_query_variant == "with_position":
        return _normalize_text(row.get("method2_query_with_position")) or _normalize_text(row.get("method2_query"))
    if augmented_query_variant == "without_position":
        return _normalize_text(row.get("method2_query_without_position")) or _normalize_text(row.get("method2_query"))
    return _normalize_text(row.get("method2_query")) or _normalize_text(row.get("anchor_question"))


def _pick_augmented_ground_truth(
    row: Dict[str, str],
    *,
    augmented_mode: str,
    augmented_ground_truth_target: str,
) -> str:
    if augmented_mode == "anchor":
        return _normalize_text(row.get("anchor_wikipedia_url")) or _normalize_text(row.get("wikipedia_url"))
    if augmented_ground_truth_target == "distractor":
        return _normalize_text(row.get("distractor_wikipedia_url")) or _normalize_text(
            row.get(f"{augmented_mode}_pair_wikipedia_url")
        )
    if augmented_ground_truth_target == "pair":
        return _normalize_text(row.get(f"{augmented_mode}_pair_wikipedia_url"))
    return _normalize_text(row.get("anchor_wikipedia_url")) or _normalize_text(row.get("wikipedia_url"))


def _build_augmented_candidate_ids(
    *,
    anchor_data_id: str,
    augmented_mode: str,
    augmented_query_variant: str,
) -> List[str]:
    candidates: List[str] = []
    if not anchor_data_id:
        return candidates
    if augmented_mode == "anchor":
        candidates.extend([f"{anchor_data_id}__anchor", anchor_data_id])
    else:
        # Reference script saved composite ids as:
        #   {anchor_data_id}__{method}__{query_variant}
        candidate_variants = [augmented_query_variant]
        candidate_variants.extend(
            variant for variant in _AUGMENTED_METHOD2_QUERY_VARIANTS if variant != augmented_query_variant
        )
        for variant in candidate_variants:
            candidates.append(f"{anchor_data_id}__{augmented_mode}__{variant}")
        candidates.append(f"{anchor_data_id}__{augmented_mode}")
        candidates.append(anchor_data_id)
    seen = set()
    return [cid for cid in candidates if cid and not (cid in seen or seen.add(cid))]


def _build_augmented_example(
    idx: int,
    row: Dict[str, str],
    *,
    test_file: str,
    augmented_mode: str,
    augmented_query_variant: str,
    augmented_ground_truth_target: str,
    augmented_image_path_root: Optional[str],
) -> Dict[str, Any]:
    mode = augmented_mode.lower()
    if mode not in {"anchor", "method1", "method2"}:
        raise ValueError(f"Unsupported augmented_mode={augmented_mode!r}")
    query_variant = (augmented_query_variant or "with_position").strip().lower()
    if query_variant not in _AUGMENTED_METHOD2_QUERY_VARIANTS:
        query_variant = "with_position"

    anchor_data_id = _normalize_text(row.get("anchor_data_id"))
    target_side = _normalize_text(row.get("target_side")).lower()
    question = _pick_augmented_query(row, augmented_mode=mode, augmented_query_variant=query_variant)
    ground_truth_url = _pick_augmented_ground_truth(
        row,
        augmented_mode=mode,
        augmented_ground_truth_target=(augmented_ground_truth_target or "anchor").strip().lower(),
    )
    answer = (
        _normalize_text(row.get(f"{mode}_expected_answer"))
        if mode in {"method1", "method2"}
        else ""
    ) or _normalize_text(row.get("anchor_answer")) or _normalize_text(row.get("answer"))
    question_type = (
        _normalize_text(row.get("anchor_question_type"))
        or _normalize_text(row.get("question_type"))
        or "String"
    )

    if mode == "anchor":
        image_field = "anchor_image_path"
        image_path_value = row.get("anchor_image_path", "")
        image_id = _normalize_text(row.get("anchor_image_id"))
        data_id = f"{anchor_data_id}__anchor" if anchor_data_id else f"composite_anchor_{idx:06d}"
    else:
        image_field = f"{mode}_composite_image_path"
        image_path_value = row.get(image_field, "")
        image_id = _normalize_text(row.get("anchor_image_id")) or _normalize_text(row.get("anchor_data_id"))
        if anchor_data_id:
            data_id = f"{anchor_data_id}__{mode}__{query_variant}"
        else:
            data_id = f"composite_{mode}_{idx:06d}"

    resolved_image_path = _resolve_augmented_image_path(
        image_path_value or "",
        test_file_path=test_file,
        path_root=augmented_image_path_root,
    )
    candidate_ids = _build_augmented_candidate_ids(
        anchor_data_id=anchor_data_id,
        augmented_mode=mode,
        augmented_query_variant=query_variant,
    )
    if data_id and data_id not in candidate_ids:
        candidate_ids = [data_id] + candidate_ids

    example: Dict[str, Any] = {
        "data_id": data_id,
        "question": question,
        "wikipedia_url": ground_truth_url,
        "ground_truth_url": ground_truth_url,
        "gold_wikipedia_url": ground_truth_url,
        "answer": answer,
        "question_type": question_type,
        "dataset_name": "infoseek",
        "dataset_image_ids": image_id or _normalize_text(Path(resolved_image_path).name if resolved_image_path else ""),
        "image_path_override": resolved_image_path,
        "_candidate_data_ids": candidate_ids,
        "augmented_mode": mode,
        "augmented_query_variant": query_variant,
        "augmented_ground_truth_target": (augmented_ground_truth_target or "anchor").strip().lower(),
        "augmented_is_composite": mode in {"method1", "method2"},
        "augmented_target_side": target_side,
        "augmented_anchor_data_id": anchor_data_id,
        "augmented_anchor_question": _normalize_text(row.get("anchor_question")),
        "augmented_anchor_answer": _normalize_text(row.get("anchor_answer")),
        "augmented_anchor_wikipedia_url": _normalize_text(row.get("anchor_wikipedia_url")),
        "augmented_distractor_wikipedia_url": _normalize_text(row.get("distractor_wikipedia_url")),
        "augmented_image_field": image_field,
        "augmented_image_path_raw": _normalize_text(image_path_value),
        "augmented_method1_pair_wikipedia_url": _normalize_text(row.get("method1_pair_wikipedia_url")),
        "augmented_method2_pair_wikipedia_url": _normalize_text(row.get("method2_pair_wikipedia_url")),
    }
    return example


def iter_examples(
    test_file: str,
    *,
    augmented_csv_mode: str = "off",
    augmented_query_variant: str = "with_position",
    augmented_ground_truth_target: str = "anchor",
    augmented_image_path_root: Optional[str] = None,
) -> Iterator[Tuple[int, Dict[str, str]]]:
    mode = (augmented_csv_mode or "off").strip().lower()
    if mode in {"anchor", "method1", "method2"}:
        rows, fieldnames = _load_csv_rows_raw(test_file)
        if not _looks_like_augmented_infoseek_csv(fieldnames):
            raise ValueError(
                "augmented_csv_mode is enabled, but the CSV header does not match the augmented InfoSeek schema. "
                f"Path={test_file}"
            )
        for idx, row in enumerate(rows):
            yield idx, _build_augmented_example(
                idx,
                row,
                test_file=test_file,
                augmented_mode=mode,
                augmented_query_variant=augmented_query_variant,
                augmented_ground_truth_target=augmented_ground_truth_target,
                augmented_image_path_root=augmented_image_path_root,
            )
        return

    rows, header = load_csv_data(test_file)
    for idx in range(len(rows)):
        yield idx, get_test_question(idx, rows, header)


def _normalize_retrieval_blob(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize retrieval payload from either raw retrieval rows or metadata rows."""
    if not isinstance(row, dict):
        return None

    retrieval_meta = row.get("retrieval_meta")
    if isinstance(retrieval_meta, dict):
        blob = dict(retrieval_meta)
        if "retrieved_entries" not in blob and row.get("retrieved_entries") is not None:
            blob["retrieved_entries"] = row.get("retrieved_entries")
        if "initial_retrieved_entries" not in blob and row.get("candidate_urls") is not None:
            blob["initial_retrieved_entries"] = row.get("candidate_urls")
        if "reranked_sections" not in blob:
            # Metadata rows may store sections under qwen_reranked_sections.
            blob["reranked_sections"] = row.get("qwen_reranked_sections") or row.get("reranked_sections") or []
        return blob

    has_retrieval_fields = any(
        key in row
        for key in (
            "retrieved_entries",
            "initial_retrieved_entries",
            "retrieval_similarities",
            "reranked_sections",
        )
    )
    if has_retrieval_fields:
        return row
    return None


def _index_jsonl_records(path: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    invalid_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(row, dict):
                invalid_lines += 1
                continue
            data_id = row.get("data_id") or row.get("original_data_id")
            if not data_id:
                invalid_lines += 1
                continue
            blob = _normalize_retrieval_blob(row)
            if blob is None:
                invalid_lines += 1
                continue
            indexed[str(data_id)] = blob

    if indexed:
        return indexed

    raise ValueError(
        "No retrieval records found in JSONL file. "
        f"Path={path}, invalid_or_non_retrieval_lines={invalid_lines}. "
        "Expected rows with data_id plus retrieved_entries/initial_retrieved_entries "
        "or metadata rows containing retrieval_meta."
    )


def load_retrieval_results(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    stripped = raw.lstrip()
    if not stripped:
        raise ValueError(f"Retrieval results file is empty: {path}")

    # Fast path: standard JSON object keyed by data_id.
    if stripped[0] in "{[":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Typical when a JSONL file is passed to json.loads (e.g., "Extra data").
            if "Extra data" in str(exc):
                return _index_jsonl_records(path)
            raise

        if isinstance(parsed, dict):
            # Canonical shape: {"data_id": {...}, ...}
            if parsed and all(isinstance(value, dict) for value in parsed.values()):
                return parsed
            # Single retrieval row dict.
            blob = _normalize_retrieval_blob(parsed)
            key = parsed.get("data_id") or parsed.get("original_data_id")
            if key and blob is not None:
                return {str(key): blob}
            raise ValueError(
                f"Unsupported retrieval JSON structure in {path}. "
                "Expected object keyed by data_id or a retrieval row with data_id."
            )

        if isinstance(parsed, list):
            indexed: Dict[str, Dict[str, Any]] = {}
            for row in parsed:
                if not isinstance(row, dict):
                    continue
                data_id = row.get("data_id") or row.get("original_data_id")
                blob = _normalize_retrieval_blob(row)
                if data_id and blob is not None:
                    indexed[str(data_id)] = blob
            if indexed:
                return indexed
            raise ValueError(
                f"Unsupported retrieval JSON list structure in {path}. "
                "Expected list rows with data_id and retrieval fields."
            )

        raise ValueError(f"Unsupported retrieval file format in {path}.")

    # Non-JSON prefix -> treat as JSONL.
    return _index_jsonl_records(path)


def build_retrieval_record(
    example: Dict[str, str],
    data_id: str,
    blob: Dict,
    image_path: Optional[str],
    candidate_ids: Sequence[str],
) -> RetrievalRecord:
    dataset = example.get("dataset_name") or "unknown"
    image_id = example.get("dataset_image_ids", "")
    candidate_urls: List[str] = blob.get("initial_retrieved_entries") or blob.get("retrieved_entries", [])
    reranked_urls: List[str] = blob.get("retrieved_entries", []) or candidate_urls
    reranked_sections: List[str] = blob.get("reranked_sections", [])
    augmented_meta: Optional[Dict[str, Any]] = None
    if any(str(key).startswith("augmented_") for key in example.keys()):
        augmented_meta = {
            "mode": example.get("augmented_mode"),
            "query_variant": example.get("augmented_query_variant"),
            "ground_truth_target": example.get("augmented_ground_truth_target"),
            "is_composite": example.get("augmented_is_composite"),
            "target_side": example.get("augmented_target_side"),
            "anchor_data_id": example.get("augmented_anchor_data_id"),
            "anchor_question": example.get("augmented_anchor_question"),
            "anchor_answer": example.get("augmented_anchor_answer"),
            "anchor_wikipedia_url": example.get("augmented_anchor_wikipedia_url"),
            "distractor_wikipedia_url": example.get("augmented_distractor_wikipedia_url"),
            "image_field": example.get("augmented_image_field"),
            "image_path_raw": example.get("augmented_image_path_raw"),
            "method1_pair_wikipedia_url": example.get("augmented_method1_pair_wikipedia_url"),
            "method2_pair_wikipedia_url": example.get("augmented_method2_pair_wikipedia_url"),
        }
    return RetrievalRecord(
        data_id=data_id,
        question=example["question"],
        image_path=image_path,
        dataset_name=dataset,
        candidate_urls=candidate_urls,
        reranked_urls=reranked_urls,
        reranked_sections=reranked_sections,
        meta={
            "retrieval_similarities": blob.get("retrieval_similarities", []),
            "initial_retrieved_entries": blob.get("initial_retrieved_entries", []),
            "retrieved_entries": blob.get("retrieved_entries", []),
            "image_id": image_id,
            "original_data_id": example.get("data_id"),
            "candidate_data_ids": list(candidate_ids),
            "dataset_name": dataset,
            **({"augmented": augmented_meta} if augmented_meta else {}),
        },
    )


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def load_metadata(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
