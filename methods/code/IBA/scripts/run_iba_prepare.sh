#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_iba_prepare.sh --dataset evqa [--dry-run]
  bash scripts/run_iba_prepare.sh --dataset infoseek [--dry-run]

Environment overrides:
  TEST_FILE RETRIEVAL_RESULTS KNOWLEDGE_BASE METADATA_PATH
  QWEN_BACKEND QWEN_VLLM_BASE_URL QWEN_VLLM_MODEL_NAME SECTION_RERANKER
  IDENTIFICATION_TOP_K IDENTIFICATION_SELECT_TOP ENTITY_TOP_K CUDA_VISIBLE_DEVICES
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMERA_READY_ROOT="$(cd "${METHOD_ROOT}/../../.." && pwd)"

DATASET="evqa"
DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

OUTPUT_DIR="${OUTPUT_DIR:-${CAMERA_READY_ROOT}/outputs/generated_methods/IBA}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
QWEN_BACKEND="${QWEN_BACKEND:-vllm_host}"
QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
QWEN_VLLM_API_KEY="${QWEN_VLLM_API_KEY:-EMPTY}"
QWEN_VLLM_MODEL_NAME="${QWEN_VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
IDENTIFICATION_TOP_K="${IDENTIFICATION_TOP_K:-20}"
IDENTIFICATION_SELECT_TOP="${IDENTIFICATION_SELECT_TOP:-3}"
IDENTIFICATION_SCORE_TOP_K="${IDENTIFICATION_SCORE_TOP_K:-3}"
ENTITY_TOP_K="${ENTITY_TOP_K:-3}"
SECTION_RERANKER="${SECTION_RERANKER:-BAAI/bge-reranker-v2-m3}"
SECTION_SCORE_WEIGHT="${SECTION_SCORE_WEIGHT:-1.0}"
RETRIEVAL_SIMILARITY_WEIGHT="${RETRIEVAL_SIMILARITY_WEIGHT:-0.5}"
IDENTIFICATION_PROBABILITY_WEIGHT="${IDENTIFICATION_PROBABILITY_WEIGHT:-0.5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$DATASET" in
  evqa)
    TEST_FILE="${TEST_FILE:-${CAMERA_READY_ROOT}/data/ground_truth/evqa_fixed.csv}"
    RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS:-${CAMERA_READY_ROOT}/data/retrieval/echosight_reranker_evqa_k20.jsonl}"
    KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-${CAMERA_READY_ROOT}/data/kb/evqa_encyclopedic_kb_wiki.json}"
    METADATA_PATH="${METADATA_PATH:-${OUTPUT_DIR}/evqa_metadata.jsonl}"
    ;;
  infoseek)
    TEST_FILE="${TEST_FILE:-${CAMERA_READY_ROOT}/data/ground_truth/infoseek_fixed.csv}"
    RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS:-${CAMERA_READY_ROOT}/data/retrieval/echosight_reranker_infoseek_k20.jsonl}"
    KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-${CAMERA_READY_ROOT}/data/kb/infoseek_wiki_100_dict_v4.json}"
    METADATA_PATH="${METADATA_PATH:-${OUTPUT_DIR}/infoseek_metadata.jsonl}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}. Use evqa or infoseek." >&2
    exit 1
    ;;
esac

TIMING_PATH="${TIMING_PATH:-${METADATA_PATH%.jsonl}_timing.json}"
LOG_FILE="${LOG_FILE:-${METADATA_PATH%.jsonl}.log}"

CMD=(
  python -m qwen_pipeline.topk.run_top3_identification_pipeline_vllm
  prepare
  --test_file "$TEST_FILE"
  --retrieval_results "$RETRIEVAL_RESULTS"
  --knowledge_base "$KNOWLEDGE_BASE"
  --metadata_path "$METADATA_PATH"
  --qwen_model_name "$QWEN_MODEL_NAME"
  --qwen_backend "$QWEN_BACKEND"
  --qwen_vllm_base_url "$QWEN_VLLM_BASE_URL"
  --qwen_vllm_api_key "$QWEN_VLLM_API_KEY"
  --qwen_vllm_model_name "$QWEN_VLLM_MODEL_NAME"
  --identification_top_k "$IDENTIFICATION_TOP_K"
  --identification_select_top "$IDENTIFICATION_SELECT_TOP"
  --identification_score_top_k "$IDENTIFICATION_SCORE_TOP_K"
  --identification_include_similarity
  --entity_top_k "$ENTITY_TOP_K"
  --context_mode section
  --section_reranker_backend bge
  --section_reranker "$SECTION_RERANKER"
  --section_score_weight "$SECTION_SCORE_WEIGHT"
  --retrieval_similarity_weight "$RETRIEVAL_SIMILARITY_WEIGHT"
  --identification_probability_weight "$IDENTIFICATION_PROBABILITY_WEIGHT"
  --prepare_timing_summary_path "$TIMING_PATH"
  --log_file "$LOG_FILE"
)
CMD+=("${EXTRA_ARGS[@]}")

print_command() {
  printf 'cd %q\n' "$METHOD_ROOT"
  printf 'export PYTHONPATH=%q\n' "${METHOD_ROOT}:${PYTHONPATH:-}"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_command
  exit 0
fi

mkdir -p "$(dirname "$METADATA_PATH")"
for required in "$TEST_FILE" "$RETRIEVAL_RESULTS" "$KNOWLEDGE_BASE"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

cd "$METHOD_ROOT"
export PYTHONPATH="${METHOD_ROOT}:${PYTHONPATH:-}"
"${CMD[@]}"
printf '\nIBA metadata output: %s\n' "$METADATA_PATH"
