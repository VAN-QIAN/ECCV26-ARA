#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_iba_answer.sh --dataset evqa [--dry-run]
  bash scripts/run_iba_answer.sh --dataset infoseek [--dry-run]

Environment overrides:
  CSV_PATH METADATA_PATH KNOWLEDGE_BASE OUTPUT_PATH ALIGNED_METADATA_PATH
  ANSWER_BACKEND ANSWER_BACKEND_VLLM_BASE_URL ANSWER_BACKEND_VLLM_MODEL_NAME
  QWEN_BACKEND QWEN_VLLM_BASE_URL QWEN_VLLM_MODEL_NAME CUDA_VISIBLE_DEVICES
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
ANSWER_BACKEND="${ANSWER_BACKEND:-qwen}"
ANSWER_BACKEND_VLLM_BASE_URL="${ANSWER_BACKEND_VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
ANSWER_BACKEND_VLLM_API_KEY="${ANSWER_BACKEND_VLLM_API_KEY:-EMPTY}"
ANSWER_BACKEND_VLLM_MODEL_NAME="${ANSWER_BACKEND_VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
QWEN_BACKEND="${QWEN_BACKEND:-vllm_host}"
QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
QWEN_VLLM_API_KEY="${QWEN_VLLM_API_KEY:-EMPTY}"
QWEN_VLLM_MODEL_NAME="${QWEN_VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
SECTION_RERANKER="${SECTION_RERANKER:-BAAI/bge-reranker-v2-m3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$DATASET" in
  evqa)
    CSV_PATH="${CSV_PATH:-${CAMERA_READY_ROOT}/data/ground_truth/evqa_fixed.csv}"
    KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-${CAMERA_READY_ROOT}/data/kb/evqa_encyclopedic_kb_wiki.json}"
    GENERATED_METADATA="${OUTPUT_DIR}/evqa_metadata.jsonl"
    OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/evqa_answers.jsonl}"
    ALIGNED_METADATA_PATH="${ALIGNED_METADATA_PATH:-${OUTPUT_DIR}/evqa_aligned_metadata.jsonl}"
    ;;
  infoseek)
    CSV_PATH="${CSV_PATH:-${CAMERA_READY_ROOT}/data/ground_truth/infoseek_fixed.csv}"
    KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-${CAMERA_READY_ROOT}/data/kb/infoseek_wiki_100_dict_v4.json}"
    GENERATED_METADATA="${OUTPUT_DIR}/infoseek_metadata.jsonl"
    OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/infoseek_answers.jsonl}"
    ALIGNED_METADATA_PATH="${ALIGNED_METADATA_PATH:-${OUTPUT_DIR}/infoseek_aligned_metadata.jsonl}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}. Use evqa or infoseek." >&2
    exit 1
    ;;
esac

if [[ -z "${METADATA_PATH:-}" ]]; then
  METADATA_PATH="$GENERATED_METADATA"
fi

CMD=(
  python -m qwen_pipeline.topk.run_final_answer_pipeline
  --dataset custom
  --csv_path "$CSV_PATH"
  --metadata_path "$METADATA_PATH"
  --knowledge_base "$KNOWLEDGE_BASE"
  --output_path "$OUTPUT_PATH"
  --aligned_metadata_path "$ALIGNED_METADATA_PATH"
  --answer_backend "$ANSWER_BACKEND"
  --answer_backend_vllm_base_url "$ANSWER_BACKEND_VLLM_BASE_URL"
  --answer_backend_vllm_api_key "$ANSWER_BACKEND_VLLM_API_KEY"
  --answer_backend_vllm_model_name "$ANSWER_BACKEND_VLLM_MODEL_NAME"
  --qwen_backend "$QWEN_BACKEND"
  --qwen_vllm_base_url "$QWEN_VLLM_BASE_URL"
  --qwen_vllm_api_key "$QWEN_VLLM_API_KEY"
  --qwen_vllm_model_name "$QWEN_VLLM_MODEL_NAME"
  --section_reranker_backend bge
  --section_reranker "$SECTION_RERANKER"
  --answer_rerank_sections
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

mkdir -p "$(dirname "$OUTPUT_PATH")"
for required in "$CSV_PATH" "$METADATA_PATH" "$KNOWLEDGE_BASE"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

cd "$METHOD_ROOT"
export PYTHONPATH="${METHOD_ROOT}:${PYTHONPATH:-}"
"${CMD[@]}"
printf '\nIBA answer output: %s\n' "$OUTPUT_PATH"
