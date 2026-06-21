#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_echosight_answer.sh --dataset evqa [--dry-run]
  bash scripts/run_echosight_answer.sh --dataset infoseek [--dry-run]

Environment overrides:
  TEST_FILE RETRIEVAL_RESULTS ANSWER_GENERATOR LLM_CHECKPOINT OUTPUT_PATH
  CUDA_VISIBLE_DEVICES
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

OUTPUT_DIR="${OUTPUT_DIR:-${CAMERA_READY_ROOT}/outputs/generated_methods/EchoSight}"
ANSWER_GENERATOR="${ANSWER_GENERATOR:-llama3}"
LLM_CHECKPOINT="${LLM_CHECKPOINT:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "$DATASET" in
  evqa)
    TEST_FILE="${TEST_FILE:-${CAMERA_READY_ROOT}/data/ground_truth/evqa_fixed_final_check_Feb12.csv}"
    RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS:-${OUTPUT_DIR}/evqa_reranker_k20.json}"
    OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/evqa_answers.jsonl}"
    DATASET_ARGS=()
    ;;
  infoseek)
    TEST_FILE="${TEST_FILE:-${CAMERA_READY_ROOT}/data/ground_truth/infoseek_fixed_final_recheck_Feb7.csv}"
    RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS:-${OUTPUT_DIR}/infoseek_reranker_k20.json}"
    OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/infoseek_answers.jsonl}"
    DATASET_ARGS=(--dataset_name infoseek)
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}. Use evqa or infoseek." >&2
    exit 1
    ;;
esac

CMD=(
  python -m test.test_answer_generator
  --test_file "$TEST_FILE"
  --retrieval_results "$RETRIEVAL_RESULTS"
  --answer_generator "$ANSWER_GENERATOR"
  --llm_checkpoint "$LLM_CHECKPOINT"
  --output_file "$OUTPUT_PATH"
  "${DATASET_ARGS[@]}"
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
for required in "$TEST_FILE" "$RETRIEVAL_RESULTS"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

cd "$METHOD_ROOT"
export PYTHONPATH="${METHOD_ROOT}:${PYTHONPATH:-}"
"${CMD[@]}"
printf '\nEchoSight answer output: %s\n' "$OUTPUT_PATH"

