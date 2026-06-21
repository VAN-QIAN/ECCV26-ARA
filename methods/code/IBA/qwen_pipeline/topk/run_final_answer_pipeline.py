"""Generate final answers from existing metadata with CSV question overrides."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from utils import seed_everything

from .pipeline import TopKPipelineConfig, TopKQwenPipeline


DATASET_PRESETS: Dict[str, Dict[str, str]] = {
    "evqa": {
        "csv_path": "/data2/QianMa/FixKBVQA/EVQA_results_final_check/evqa_final_check_Feb12.csv",
        "metadata_path": "/data/qianMa/EchoSight/ECCV_results/evqa_our_metadata_with_bge_Oct23_1_0.5_0.5.jsonl",
        "knowledge_base": "/data/qianMa/EchoSight/KB_EVQA/encyclopedic_kb_wiki.json",
    },
    "infoseek": {
        "csv_path": "/data2/QianMa/ECCV/Wiki-PRF/test/infoseek_final_recheck_Feb7.csv",
        "metadata_path": (
            "/data/qianMa/EchoSight/ECCV_results/"
            "infoseek_qwen_top3_identified_rescored_with_retrieval_similarity_with_bge_Oct15_0.5_0.5_1.jsonl"
        ),
        "knowledge_base": "/data/qianMa/EchoSight/wiki_100_dict_v4.json",
    },
}

MINIMAL_METADATA_KEYS: Sequence[str] = (
    "data_id",
    "image_path",
    "dataset_name",
    "context_mode",
    "context_text",
    "context_source_url",
    "context_section_title",
    "selected_url",
    "selected_title",
    "selected_index",
    "context_source_rank",
    "context_section_score",
    "context_source_probability",
    "fallback_reason",
)


@dataclass
class CSVQuestionMap:
    question_by_id: Dict[str, str]
    ordered_ids: List[str]
    rows_total: int
    rows_with_data_id: int
    rows_missing_question: int
    conflicting_duplicates: int


@dataclass
class AlignmentStats:
    metadata_rows_total: int = 0
    metadata_rows_written: int = 0
    metadata_rows_invalid_json: int = 0
    metadata_rows_missing_data_id: int = 0
    metadata_rows_non_csv_skipped: int = 0
    metadata_rows_duplicate_data_id_skipped: int = 0
    csv_ids_matched: int = 0
    missing_csv_ids: List[str] = field(default_factory=list)


def _coalesce_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    string_value = value.strip()
    return string_value or None


def _parse_fallback_columns(raw_value: str) -> List[str]:
    values = [column.strip() for column in raw_value.split(",")]
    return [column for column in values if column]


def _safe_data_id(raw_value: object) -> Optional[str]:
    if raw_value is None:
        return None
    string_value = str(raw_value).strip()
    return string_value or None


def _extract_csv_question(
    row: Dict[str, str],
    primary_column: str,
    fallback_columns: Iterable[str],
) -> str:
    candidates = [primary_column] + list(fallback_columns)
    for column in candidates:
        if column not in row:
            continue
        text = (row.get(column) or "").strip()
        if text:
            return text
    return ""


def load_csv_question_map(
    csv_path: str,
    *,
    data_id_column: str,
    question_column: str,
    question_fallback_columns: Sequence[str],
) -> CSVQuestionMap:
    question_by_id: Dict[str, str] = {}
    ordered_ids: List[str] = []
    seen_order: Set[str] = set()
    rows_total = 0
    rows_with_data_id = 0
    rows_missing_question = 0
    conflicting_duplicates = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears empty or malformed: {csv_path}")
        missing_columns = [column for column in [data_id_column] if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                f"CSV missing required columns {missing_columns}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            rows_total += 1
            data_id = _safe_data_id(row.get(data_id_column))
            if data_id is None:
                continue
            rows_with_data_id += 1
            if data_id not in seen_order:
                seen_order.add(data_id)
                ordered_ids.append(data_id)

            question = _extract_csv_question(
                row=row,
                primary_column=question_column,
                fallback_columns=question_fallback_columns,
            )
            if not question:
                rows_missing_question += 1
                continue
            previous = question_by_id.get(data_id)
            if previous is not None and previous != question:
                conflicting_duplicates += 1
            question_by_id[data_id] = question

    return CSVQuestionMap(
        question_by_id=question_by_id,
        ordered_ids=ordered_ids,
        rows_total=rows_total,
        rows_with_data_id=rows_with_data_id,
        rows_missing_question=rows_missing_question,
        conflicting_duplicates=conflicting_duplicates,
    )


def _extract_dataset_name(row: Dict[str, object]) -> Optional[str]:
    dataset_name = row.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name.strip():
        return dataset_name.strip()
    retrieval_meta = row.get("retrieval_meta")
    if isinstance(retrieval_meta, dict):
        nested_name = retrieval_meta.get("dataset_name")
        if isinstance(nested_name, str) and nested_name.strip():
            return nested_name.strip()
    return None


def _to_section_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _build_aligned_row(
    row: Dict[str, object],
    *,
    question: str,
    keep_full_metadata: bool,
    include_reranked_sections: bool,
) -> Dict[str, object]:
    if keep_full_metadata:
        output: Dict[str, object] = dict(row)
    else:
        output = {key: row[key] for key in MINIMAL_METADATA_KEYS if key in row}

    data_id = _safe_data_id(row.get("data_id")) or _safe_data_id(row.get("original_data_id"))
    if data_id is not None:
        output["data_id"] = data_id
    output["question"] = question

    dataset_name = _extract_dataset_name(row)
    if dataset_name is not None:
        output["dataset_name"] = dataset_name

    if include_reranked_sections:
        sections_value = row.get("reranked_sections")
        if sections_value is None:
            sections_value = row.get("qwen_reranked_sections")
        output["reranked_sections"] = _to_section_list(sections_value)

    return output


def align_metadata_questions(
    *,
    metadata_path: str,
    output_metadata_path: str,
    csv_map: CSVQuestionMap,
    include_non_csv_ids: bool,
    keep_full_metadata: bool,
    deduplicate_by_data_id: bool,
    include_reranked_sections: bool,
    log_every: int,
    max_rows: Optional[int],
) -> AlignmentStats:
    stats = AlignmentStats()
    csv_questions = csv_map.question_by_id
    matched_ids: Set[str] = set()
    written_ids: Set[str] = set()

    output_path = Path(output_metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, raw_line in enumerate(fin, start=1):
            if max_rows is not None and stats.metadata_rows_total >= max_rows:
                break
            line = raw_line.strip()
            if not line:
                continue
            stats.metadata_rows_total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats.metadata_rows_invalid_json += 1
                continue
            if not isinstance(row, dict):
                stats.metadata_rows_invalid_json += 1
                continue

            data_id = _safe_data_id(row.get("data_id")) or _safe_data_id(row.get("original_data_id"))
            if data_id is None:
                stats.metadata_rows_missing_data_id += 1
                continue

            if deduplicate_by_data_id and data_id in written_ids:
                stats.metadata_rows_duplicate_data_id_skipped += 1
                continue

            question_override = csv_questions.get(data_id)
            if question_override is None:
                if not include_non_csv_ids:
                    stats.metadata_rows_non_csv_skipped += 1
                    continue
                fallback_question = row.get("question")
                question = str(fallback_question).strip() if fallback_question is not None else ""
            else:
                question = question_override
                matched_ids.add(data_id)

            aligned = _build_aligned_row(
                row,
                question=question,
                keep_full_metadata=keep_full_metadata,
                include_reranked_sections=include_reranked_sections,
            )
            json.dump(aligned, fout, ensure_ascii=False)
            fout.write("\n")

            stats.metadata_rows_written += 1
            written_ids.add(data_id)

            if (
                not include_non_csv_ids
                and deduplicate_by_data_id
                and len(matched_ids) >= len(csv_questions)
            ):
                break

            if log_every > 0 and line_no % log_every == 0:
                print(
                    f"[align] scanned={line_no} written={stats.metadata_rows_written} "
                    f"matched_csv_ids={len(matched_ids)}"
                )

    stats.csv_ids_matched = len(matched_ids)
    stats.missing_csv_ids = [
        data_id
        for data_id in csv_map.ordered_ids
        if data_id in csv_questions and data_id not in matched_ids
    ]
    return stats


def _build_answer_config(args: argparse.Namespace) -> TopKPipelineConfig:
    return TopKPipelineConfig(
        test_file="",
        retrieval_results="",
        knowledge_base=args.knowledge_base,
        qwen_model_name=args.qwen_model_name,
        qwen_device=args.qwen_device,
        qwen_backend=args.qwen_backend,
        qwen_vllm_base_url=_coalesce_optional_path(args.qwen_vllm_base_url),
        qwen_vllm_api_key=args.qwen_vllm_api_key,
        qwen_vllm_timeout=args.qwen_vllm_timeout,
        qwen_vllm_model_name=_coalesce_optional_path(args.qwen_vllm_model_name),
        qwen_vllm_enable_image_cache=not args.disable_qwen_vllm_image_cache,
        qwen_vllm_image_cache_size=max(0, int(args.qwen_vllm_image_cache_size)),
        identification_call_mode="two_step",
        answer_temperature=args.answer_temperature,
        answer_max_new_tokens=args.answer_max_new_tokens,
        require_reasoning=args.require_reasoning,
        answer_backend=args.answer_backend,
        answer_backend_device=_coalesce_optional_path(args.answer_backend_device),
        answer_backend_model_path=_coalesce_optional_path(args.answer_backend_model_path),
        answer_backend_vllm_base_url=_coalesce_optional_path(args.answer_backend_vllm_base_url),
        answer_backend_vllm_api_key=args.answer_backend_vllm_api_key,
        answer_backend_vllm_timeout=args.answer_backend_vllm_timeout,
        answer_backend_vllm_model_name=_coalesce_optional_path(args.answer_backend_vllm_model_name),
        inat_mapping_path=_coalesce_optional_path(args.inat_mapping_path),
        answer_rerank_sections=args.answer_rerank_sections,
        evqa_landmark_root=_coalesce_optional_path(args.evqa_landmark_root),
        answer_timing_summary_path=_coalesce_optional_path(args.answer_timing_summary_path),
        log_file=_coalesce_optional_path(args.log_file),
        log_level=args.log_level,
        section_reranker_backend=args.section_reranker_backend,
        section_reranker=_coalesce_optional_path(args.section_reranker),
        section_score_weight=args.section_score_weight,
        retrieval_similarity_weight=args.retrieval_similarity_weight,
        identification_probability_weight=args.identification_probability_weight,
        identification_score_mode=args.identification_score_mode,
        section_score_source=args.section_score_source,
    )


def _default_aligned_metadata_path(output_path: str) -> str:
    out = Path(output_path)
    return str(out.with_name(f"{out.stem}.aligned_metadata.jsonl"))


def _apply_dataset_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.dataset != "custom":
        preset = DATASET_PRESETS[args.dataset]
        if args.csv_path is None:
            args.csv_path = preset["csv_path"]
        if args.metadata_path is None:
            args.metadata_path = preset["metadata_path"]
        if args.knowledge_base is None:
            args.knowledge_base = preset["knowledge_base"]

    missing = [
        name
        for name in ("csv_path", "metadata_path", "knowledge_base")
        if not getattr(args, name, None)
    ]
    if missing:
        parser.error(
            f"Missing required paths: {missing}. Provide them explicitly or set --dataset evqa/infoseek."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align metadata by data_id with updated CSV questions, then generate final answers."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["custom", "evqa", "infoseek"],
        default="custom",
        help="Use built-in csv/metadata/knowledge-base paths for EVQA or InfoSeek.",
    )
    parser.add_argument("--csv_path", type=str, default=None, help="CSV containing updated questions.")
    parser.add_argument("--metadata_path", type=str, default=None, help="Existing metadata JSONL to reuse contexts.")
    parser.add_argument("--knowledge_base", type=str, default=None, help="KB JSON path for answer pipeline init.")
    parser.add_argument("--output_path", type=str, required=True, help="Final answers JSONL output path.")
    parser.add_argument(
        "--aligned_metadata_path",
        type=str,
        default=None,
        help="Optional aligned metadata output path (default: alongside output_path).",
    )
    parser.add_argument("--data_id_column", type=str, default="data_id")
    parser.add_argument("--question_column", type=str, default="question")
    parser.add_argument(
        "--question_fallback_columns",
        type=str,
        default="question_before_review,question_original",
        help="Comma-separated fallback columns used when question_column is empty.",
    )
    parser.add_argument(
        "--include_non_csv_ids",
        action="store_true",
        help="Keep metadata rows whose data_id is absent in CSV (uses metadata question).",
    )
    parser.add_argument(
        "--allow_missing_metadata_ids",
        action="store_true",
        help="Do not fail when some CSV data_id values are not found in metadata.",
    )
    parser.add_argument(
        "--keep_full_metadata",
        action="store_true",
        help="Keep every original metadata field in aligned output (larger files).",
    )
    parser.add_argument(
        "--no_deduplicate_metadata",
        action="store_true",
        help="Do not deduplicate metadata rows by data_id during alignment.",
    )
    parser.add_argument("--log_every", type=int, default=5000)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional debug cap for scanned metadata rows.",
    )

    # Answer generation options (aligned with run_top3_identification_pipeline_vllm.py).
    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--qwen_device", type=str, default="cuda:0")
    parser.add_argument("--qwen_backend", choices=["hf", "vllm_host"], default="vllm_host")
    parser.add_argument("--qwen_vllm_base_url", type=str, default="http://127.0.0.1:8001/v1")
    parser.add_argument("--qwen_vllm_api_key", type=str, default="EMPTY")
    parser.add_argument("--qwen_vllm_timeout", type=float, default=120.0)
    parser.add_argument("--qwen_vllm_model_name", type=str, default=None)
    parser.add_argument(
        "--qwen_vllm_image_cache_size",
        type=int,
        default=2048,
        help="LRU size for image_path->data_url cache on vLLM client.",
    )
    parser.add_argument(
        "--disable_qwen_vllm_image_cache",
        action="store_true",
        help="Disable image_path->data_url cache for vLLM client.",
    )
    parser.add_argument("--answer_temperature", type=float, default=0.0)
    parser.add_argument("--answer_max_new_tokens", type=int, default=512)
    parser.add_argument("--answer_backend", type=str, default="qwen")
    parser.add_argument("--answer_backend_device", type=str, default=None)
    parser.add_argument("--answer_backend_model_path", type=str, default=None)
    parser.add_argument("--answer_backend_vllm_base_url", type=str, default=None)
    parser.add_argument("--answer_backend_vllm_api_key", type=str, default="EMPTY")
    parser.add_argument("--answer_backend_vllm_timeout", type=float, default=120.0)
    parser.add_argument("--answer_backend_vllm_model_name", type=str, default=None)
    parser.add_argument("--require_reasoning", action="store_true")
    parser.add_argument("--use_image", action="store_true")
    parser.add_argument("--section_reranker_backend", choices=["auto", "bge"], default="auto")
    parser.add_argument("--section_reranker", type=str, default=None)
    parser.add_argument("--answer_rerank_sections", action="store_true")
    parser.add_argument("--inat_mapping_path", type=str, default="/data/qianMa/EchoSight/images/val_id2name.json")
    parser.add_argument("--evqa_landmark_root", type=str, default="/data/qianMa/EchoSight/E-VQA/landmark")
    parser.add_argument(
        "--answer_timing_summary_path",
        type=str,
        default=None,
        help="Optional JSON path to write answer-stage runtime summary.",
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--section_score_weight", type=float, default=1.0)
    parser.add_argument("--retrieval_similarity_weight", type=float, default=0.0)
    parser.add_argument("--identification_probability_weight", type=float, default=0.0)
    parser.add_argument("--identification_score_mode", choices=["multiply", "add"], default="multiply")
    parser.add_argument("--section_score_source", choices=["blended", "raw"], default="blended")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _apply_dataset_defaults(args, parser)

    seed_everything(42)
    print("#" * 50)
    print("Seeded everything with 42")
    print("#" * 50)

    aligned_metadata_path = args.aligned_metadata_path or _default_aligned_metadata_path(args.output_path)
    fallback_columns = _parse_fallback_columns(args.question_fallback_columns)

    csv_map = load_csv_question_map(
        args.csv_path,
        data_id_column=args.data_id_column,
        question_column=args.question_column,
        question_fallback_columns=fallback_columns,
    )
    print(
        "[csv] "
        f"rows_total={csv_map.rows_total} "
        f"rows_with_data_id={csv_map.rows_with_data_id} "
        f"unique_data_ids={len(csv_map.ordered_ids)} "
        f"mapped_questions={len(csv_map.question_by_id)} "
        f"missing_question_rows={csv_map.rows_missing_question} "
        f"conflicting_duplicates={csv_map.conflicting_duplicates}"
    )

    stats = align_metadata_questions(
        metadata_path=args.metadata_path,
        output_metadata_path=aligned_metadata_path,
        csv_map=csv_map,
        include_non_csv_ids=args.include_non_csv_ids,
        keep_full_metadata=args.keep_full_metadata,
        deduplicate_by_data_id=not args.no_deduplicate_metadata,
        include_reranked_sections=args.answer_rerank_sections,
        log_every=max(0, int(args.log_every)),
        max_rows=args.max_rows,
    )
    print(
        "[align] "
        f"scanned={stats.metadata_rows_total} "
        f"written={stats.metadata_rows_written} "
        f"invalid_json={stats.metadata_rows_invalid_json} "
        f"missing_data_id={stats.metadata_rows_missing_data_id} "
        f"non_csv_skipped={stats.metadata_rows_non_csv_skipped} "
        f"duplicate_skipped={stats.metadata_rows_duplicate_data_id_skipped} "
        f"matched_csv_ids={stats.csv_ids_matched} "
        f"missing_csv_ids={len(stats.missing_csv_ids)}"
    )

    if stats.metadata_rows_written == 0:
        raise ValueError(
            "No aligned metadata rows were produced. "
            "Check data_id column names and input file consistency."
        )

    if stats.missing_csv_ids and not args.allow_missing_metadata_ids:
        preview = ", ".join(stats.missing_csv_ids[:10])
        raise ValueError(
            "Some CSV data_id values were not found in metadata. "
            f"count={len(stats.missing_csv_ids)}, sample=[{preview}] "
            "Use --allow_missing_metadata_ids to continue anyway."
        )

    config = _build_answer_config(args)
    pipeline = TopKQwenPipeline(config)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.answer_from_metadata(
        metadata_path=aligned_metadata_path,
        output_path=args.output_path,
        use_image=args.use_image,
    )

    print(f"[done] aligned_metadata={aligned_metadata_path}")
    print(f"[done] answers={args.output_path}")


if __name__ == "__main__":
    main()
