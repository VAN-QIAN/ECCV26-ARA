"""CLI entrypoint for Top-K Qwen identification with remote API backends."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .pipeline import TopKPipelineConfig, TopKQwenPipeline
from utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INAT_MAPPING_PATH = str(REPO_ROOT / "data/images/echosight_inat_val_id2name.json")
DEFAULT_EVQA_LANDMARK_ROOT = str(REPO_ROOT / "data/images/evqa_landmark_images")


def _coalesce_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    string_value = value.strip()
    return string_value or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen entity identification with top-k support and remote API backends",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Perform retrieval alignment, ask Qwen via API for top-k entities, rerank sections, and export metadata.",
    )
    prepare.add_argument("--test_file", type=str, required=True, help="CSV with evaluation questions")
    prepare.add_argument("--retrieval_results", type=str, required=True, help="JSON retrieval blob keyed by data_id")
    prepare.add_argument("--knowledge_base", type=str, required=True, help="Path to serialized KB JSON")
    prepare.add_argument("--metadata_path", type=str, required=True, help="Destination JSONL for metadata output")
    prepare.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    prepare.add_argument("--qwen_device", type=str, default="cuda:0")
    prepare.add_argument("--qwen_backend", choices=["hf", "vllm_host", "openai_api"], default="vllm_host")
    prepare.add_argument("--qwen_vllm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    prepare.add_argument("--qwen_vllm_api_key", type=str, default="EMPTY")
    prepare.add_argument("--qwen_vllm_timeout", type=float, default=120.0)
    prepare.add_argument("--qwen_vllm_model_name", type=str, default=None)
    prepare.add_argument("--qwen_openai_base_url", type=str, default="https://api.openai.com/v1")
    prepare.add_argument("--qwen_openai_api_key", type=str, default=None)
    prepare.add_argument("--qwen_openai_timeout", type=float, default=120.0)
    prepare.add_argument("--qwen_openai_model_name", type=str, default=None)
    prepare.add_argument(
        "--qwen_vllm_image_cache_size",
        type=int,
        default=2048,
        help="LRU size for image_path->data_url cache on vLLM client.",
    )
    prepare.add_argument(
        "--disable_qwen_vllm_image_cache",
        action="store_true",
        help="Disable image_path->data_url cache for vLLM client.",
    )
    prepare.add_argument("--identification_top_k", type=int, default=20, help="Number of retrieval candidates sent to Qwen")
    prepare.add_argument("--identification_select_top", type=int, default=3, help="Number of ranked options to request from Qwen")
    prepare.add_argument(
        "--top1_identification_only",
        action="store_true",
        help="Run only top1 identification and skip section rerank (article context from selected entity).",
    )
    prepare.add_argument(
        "--identification_call_mode",
        choices=["two_step", "one_step"],
        default="two_step",
        help="two_step: identify then score; one_step: identify+score in a single vLLM call.",
    )
    prepare.add_argument(
        "--identification_score_top_k",
        type=int,
        default=3,
        help="How many ranked options should receive probability estimates",
    )
    prepare.add_argument("--identification_temperature", type=float, default=0.0)
    prepare.add_argument("--identification_max_new_tokens", type=int, default=256)
    prepare.add_argument(
        "--identification_include_similarity",
        action="store_true",
        help="Include initial retrieval image similarities in the identification prompt.",
    )
    prepare.add_argument("--entity_top_k", type=int, default=3, help="How many entities to expand into sections")
    prepare.add_argument("--context_mode", choices=["article", "section"], default="section")
    prepare.add_argument(
        "--section_pool_size",
        type=int,
        default=0,
        help="Deprecated; kept for compatibility. Non-positive values use all sections.",
    )
    prepare.add_argument("--section_reranker_backend", choices=["auto", "bge"], default="bge")
    prepare.add_argument("--section_reranker", type=str, default=None, help="HF reranker checkpoint when using 'bge'")
    prepare.add_argument(
        "--section_score_weight",
        type=float,
        default=1.0,
        help="Weight assigned to the reranker section score when combining final section scores.",
    )
    prepare.add_argument(
        "--retrieval_similarity_weight",
        type=float,
        default=0.0,
        help="Weight assigned to the initial retrieval similarity when combining final section scores.",
    )
    prepare.add_argument(
        "--identification_probability_weight",
        type=float,
        default=0.0,
        help="Weight assigned to the identification probability when combining final section scores.",
    )
    prepare.add_argument(
        "--identification_score_mode",
        choices=["multiply", "add"],
        default="multiply",
        help="How to incorporate identification probability with the section score before weighting.",
    )
    prepare.add_argument(
        "--section_score_source",
        choices=["blended", "raw"],
        default="blended",
        help="Whether to use blended or raw reranker scores when fusing section scores.",
    )
    prepare.add_argument(
        "--section_score_normalization",
        choices=["none", "minmax"],
        default="none",
        help="Optional normalization applied to selected section scores before fusion.",
    )
    prepare.add_argument("--use_reranked_sections_first", action="store_true", help="Keep reranker ordering ahead of KB reconstruction")
    prepare.add_argument("--answer_backend", type=str, default="qwen")
    prepare.add_argument("--answer_backend_device", type=str, default=None)
    prepare.add_argument("--answer_backend_model_path", type=str, default=None)
    prepare.add_argument("--answer_backend_vllm_base_url", type=str, default=None)
    prepare.add_argument("--answer_backend_vllm_api_key", type=str, default="EMPTY")
    prepare.add_argument("--answer_backend_vllm_timeout", type=float, default=120.0)
    prepare.add_argument("--answer_backend_vllm_model_name", type=str, default=None)
    prepare.add_argument("--answer_temperature", type=float, default=0.0)
    prepare.add_argument("--answer_max_new_tokens", type=int, default=512)
    prepare.add_argument("--require_reasoning", action="store_true")
    prepare.add_argument("--answer_rerank_sections", action="store_true")
    prepare.add_argument("--inat_mapping_path", type=str, default=DEFAULT_INAT_MAPPING_PATH)
    prepare.add_argument("--evqa_landmark_root", type=str, default=DEFAULT_EVQA_LANDMARK_ROOT)
    prepare.add_argument(
        "--prepare_timing_summary_path",
        type=str,
        default=None,
        help="Optional JSON path to write prepare-stage runtime summary.",
    )
    prepare.add_argument("--log_file", type=str, default=None)
    prepare.add_argument("--log_level", type=str, default="INFO")

    answer = subparsers.add_parser(
        "answer",
        help="Consume prepared metadata and run the answer generator over the selected sections",
    )
    answer.add_argument("--metadata_path", type=str, required=True, help="Metadata JSONL produced by 'prepare'")
    answer.add_argument("--output_path", type=str, required=True, help="Where to write answers JSONL")
    answer.add_argument("--knowledge_base", type=str, required=True, help="KB JSON used during preparation")
    answer.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    answer.add_argument("--qwen_device", type=str, default="cuda:0")
    answer.add_argument("--qwen_backend", choices=["hf", "vllm_host", "openai_api"], default="vllm_host")
    answer.add_argument("--qwen_vllm_base_url", type=str, default="http://127.0.0.1:8000/v1")
    answer.add_argument("--qwen_vllm_api_key", type=str, default="EMPTY")
    answer.add_argument("--qwen_vllm_timeout", type=float, default=120.0)
    answer.add_argument("--qwen_vllm_model_name", type=str, default=None)
    answer.add_argument("--qwen_openai_base_url", type=str, default="https://api.openai.com/v1")
    answer.add_argument("--qwen_openai_api_key", type=str, default=None)
    answer.add_argument("--qwen_openai_timeout", type=float, default=120.0)
    answer.add_argument("--qwen_openai_model_name", type=str, default=None)
    answer.add_argument(
        "--qwen_vllm_image_cache_size",
        type=int,
        default=2048,
        help="LRU size for image_path->data_url cache on vLLM client.",
    )
    answer.add_argument(
        "--disable_qwen_vllm_image_cache",
        action="store_true",
        help="Disable image_path->data_url cache for vLLM client.",
    )
    answer.add_argument(
        "--identification_call_mode",
        choices=["two_step", "one_step"],
        default="two_step",
        help="Kept for config consistency; affects prepare identification only.",
    )
    answer.add_argument("--answer_temperature", type=float, default=0.0)
    answer.add_argument("--answer_max_new_tokens", type=int, default=512)
    answer.add_argument("--answer_backend", type=str, default="qwen")
    answer.add_argument("--answer_backend_device", type=str, default=None)
    answer.add_argument("--answer_backend_model_path", type=str, default=None)
    answer.add_argument("--answer_backend_vllm_base_url", type=str, default=None)
    answer.add_argument("--answer_backend_vllm_api_key", type=str, default="EMPTY")
    answer.add_argument("--answer_backend_vllm_timeout", type=float, default=120.0)
    answer.add_argument("--answer_backend_vllm_model_name", type=str, default=None)
    answer.add_argument("--require_reasoning", action="store_true")
    answer.add_argument("--use_image", action="store_true")
    answer.add_argument("--section_reranker_backend", choices=["auto", "bge"], default="auto")
    answer.add_argument("--section_reranker", type=str, default=None)
    answer.add_argument("--answer_rerank_sections", action="store_true")
    answer.add_argument("--inat_mapping_path", type=str, default=DEFAULT_INAT_MAPPING_PATH)
    answer.add_argument("--evqa_landmark_root", type=str, default=DEFAULT_EVQA_LANDMARK_ROOT)
    answer.add_argument(
        "--answer_timing_summary_path",
        type=str,
        default=None,
        help="Optional JSON path to write answer-stage runtime summary.",
    )
    answer.add_argument("--log_file", type=str, default=None)
    answer.add_argument("--log_level", type=str, default="INFO")
    answer.add_argument("--section_score_weight", type=float, default=1.0)
    answer.add_argument("--retrieval_similarity_weight", type=float, default=0.0)
    answer.add_argument("--identification_probability_weight", type=float, default=0.0)
    answer.add_argument("--identification_score_mode", choices=["multiply", "add"], default="multiply")
    answer.add_argument("--section_score_source", choices=["blended", "raw"], default="blended")
    answer.add_argument("--section_score_normalization", choices=["none", "minmax"], default="none")

    return parser


def _build_config_from_args(args: argparse.Namespace, phase: str) -> TopKPipelineConfig:
    # Shared values between prepare and answer phases.
    top1_identification_only = bool(getattr(args, "top1_identification_only", False))
    effective_identification_select_top = (
        1 if top1_identification_only else getattr(args, "identification_select_top", 1)
    )
    effective_entity_top_k = 1 if top1_identification_only else getattr(args, "entity_top_k", 3)
    effective_context_mode = "article" if top1_identification_only else getattr(args, "context_mode", "section")
    base_kwargs = dict(
        qwen_model_name=args.qwen_model_name,
        qwen_device=args.qwen_device,
        qwen_backend=getattr(args, "qwen_backend", "hf"),
        qwen_vllm_base_url=_coalesce_optional_path(getattr(args, "qwen_vllm_base_url", None)),
        qwen_vllm_api_key=getattr(args, "qwen_vllm_api_key", "EMPTY"),
        qwen_vllm_timeout=getattr(args, "qwen_vllm_timeout", 120.0),
        qwen_vllm_model_name=_coalesce_optional_path(getattr(args, "qwen_vllm_model_name", None)),
        qwen_openai_base_url=_coalesce_optional_path(getattr(args, "qwen_openai_base_url", None)),
        qwen_openai_api_key=_coalesce_optional_path(getattr(args, "qwen_openai_api_key", None)),
        qwen_openai_timeout=getattr(args, "qwen_openai_timeout", 120.0),
        qwen_openai_model_name=_coalesce_optional_path(getattr(args, "qwen_openai_model_name", None)),
        qwen_vllm_enable_image_cache=not getattr(args, "disable_qwen_vllm_image_cache", False),
        qwen_vllm_image_cache_size=max(0, int(getattr(args, "qwen_vllm_image_cache_size", 2048))),
        identification_top_k=getattr(args, "identification_top_k", 5),
        identification_select_top=effective_identification_select_top,
        identification_call_mode=getattr(args, "identification_call_mode", "two_step"),
        identification_temperature=getattr(args, "identification_temperature", 0.0),
        identification_max_new_tokens=getattr(args, "identification_max_new_tokens", 256),
        identification_include_similarity=getattr(args, "identification_include_similarity", False),
        context_mode=effective_context_mode,
        section_pool_size=getattr(args, "section_pool_size", 0),
        use_reranked_sections_first=getattr(args, "use_reranked_sections_first", False),
        answer_temperature=args.answer_temperature,
        answer_max_new_tokens=args.answer_max_new_tokens,
        require_reasoning=args.require_reasoning,
        answer_backend=args.answer_backend,
        answer_backend_device=_coalesce_optional_path(args.answer_backend_device),
        answer_backend_model_path=_coalesce_optional_path(args.answer_backend_model_path),
        answer_backend_vllm_base_url=_coalesce_optional_path(
            getattr(args, "answer_backend_vllm_base_url", None)
        ),
        answer_backend_vllm_api_key=getattr(args, "answer_backend_vllm_api_key", "EMPTY"),
        answer_backend_vllm_timeout=getattr(args, "answer_backend_vllm_timeout", 120.0),
        answer_backend_vllm_model_name=_coalesce_optional_path(
            getattr(args, "answer_backend_vllm_model_name", None)
        ),
        inat_mapping_path=_coalesce_optional_path(args.inat_mapping_path),
        answer_rerank_sections=getattr(args, "answer_rerank_sections", False),
        evqa_landmark_root=_coalesce_optional_path(args.evqa_landmark_root),
        prepare_timing_summary_path=_coalesce_optional_path(
            getattr(args, "prepare_timing_summary_path", None)
        ),
        answer_timing_summary_path=_coalesce_optional_path(
            getattr(args, "answer_timing_summary_path", None)
        ),
        log_file=_coalesce_optional_path(args.log_file),
        log_level=args.log_level,
        section_reranker_backend=getattr(args, "section_reranker_backend", "auto"),
        section_reranker=_coalesce_optional_path(getattr(args, "section_reranker", None)),
        entity_top_k=effective_entity_top_k,
        top1_identification_only=top1_identification_only,
        section_score_weight=getattr(args, "section_score_weight", 1.0),
        retrieval_similarity_weight=getattr(args, "retrieval_similarity_weight", 0.0),
        identification_probability_weight=getattr(args, "identification_probability_weight", 0.0),
        identification_score_mode=getattr(args, "identification_score_mode", "multiply"),
        section_score_source=getattr(args, "section_score_source", "blended"),
        section_score_normalization=getattr(args, "section_score_normalization", "none"),
    )
    if hasattr(args, "identification_score_top_k"):
        base_kwargs["identification_score_top_k"] = (
            0 if top1_identification_only else getattr(args, "identification_score_top_k")
        )

    if phase == "prepare":
        return TopKPipelineConfig(
            test_file=args.test_file,
            retrieval_results=args.retrieval_results,
            knowledge_base=args.knowledge_base,
            **base_kwargs,
        )
    # answer phase - metadata already contains retrieval outputs.
    return TopKPipelineConfig(
        test_file="",
        retrieval_results="",
        knowledge_base=args.knowledge_base,
        **base_kwargs,
    )


def run_prepare(args: argparse.Namespace) -> None:
    config = _build_config_from_args(args, phase="prepare")
    pipeline = TopKQwenPipeline(config)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.prepare_metadata(args.metadata_path)


def run_answer(args: argparse.Namespace) -> None:
    config = _build_config_from_args(args, phase="answer")
    pipeline = TopKQwenPipeline(config)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.answer_from_metadata(
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        use_image=args.use_image,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seed_everything(42)
    print("#"*50)
    print(f"Seeded everything with 42")
    print("#"*50)
    if args.command == "prepare":
        run_prepare(args)
    else:
        run_answer(args)


if __name__ == "__main__":
    main()
