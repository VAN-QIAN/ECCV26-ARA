#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/methods/_common.sh"

SAMPLE_SIZE="${SAMPLE_SIZE:-1}"
DATASET_NAME="infoseek_fixed"
STEPS="${STEPS:-600}"
RECALL_TOP_K="${RECALL_TOP_K:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/generated_methods/Wiki-PRF/infoseek_fixed}"
if [[ -z "${OUTPUT_TEMPLATE:-}" ]]; then
  OUTPUT_TEMPLATE="${OUTPUT_DIR}/sample_results_{DATASET}_step{STEPS}_topk${RECALL_TOP_K}.json"
fi
EVAL_DIR="${EVAL_DIR:-${ROOT}/results/evaluation/infoseek/fixed_from_generated_wikiprf_sample}"

MODEL_GPU_ID="${MODEL_GPU_ID:-0}"
FILTER_GPU_ID="${FILTER_GPU_ID:-1}"
RETRIEVER_GPU_ID="${RETRIEVER_GPU_ID:-1}"
MASTER_PORT="${MASTER_PORT:-13201}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

resolved_output="${OUTPUT_TEMPLATE//\{DATASET\}/${DATASET_NAME}}"
resolved_output="${resolved_output//\{STEPS\}/${STEPS}}"
GENERATION_DETAILS="${GENERATION_DETAILS:-${resolved_output}_0_generation_details.jsonl}"

mkdir -p "${OUTPUT_DIR}"
if [[ "${FORCE:-0}" == "1" ]]; then
  rm -f "${resolved_output}_0.json" \
    "${resolved_output}_0_generation_details.jsonl" \
    "${resolved_output}_0_raw_outputs.jsonl" \
    "${resolved_output}_0_recall_details.jsonl"
fi

if [[ ! -s "${GENERATION_DETAILS}" ]]; then
  bash "${ROOT}/scripts/methods/run_wikiprf.sh" \
    --dataset infoseek \
    --split fixed \
    --max_samples "${SAMPLE_SIZE}" \
    --output_path "${OUTPUT_TEMPLATE}" \
    --steps "${STEPS}" \
    --recall_top_k "${RECALL_TOP_K}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --model_gpu_id "${MODEL_GPU_ID}" \
    --filter_gpu_id "${FILTER_GPU_ID}" \
    --retriever_gpu_id "${RETRIEVER_GPU_ID}" \
    --master_port "${MASTER_PORT}"
else
  printf 'Existing Wiki-PRF sample output: %s\n' "${GENERATION_DETAILS}"
  printf 'Set FORCE=1 to rerun Wiki-PRF generation.\n'
fi

if [[ ! -s "${GENERATION_DETAILS}" ]]; then
  echo "missing Wiki-PRF generation details: ${GENERATION_DETAILS}" >&2
  exit 1
fi

activate_conda_env KBVQA_eval
python "${ROOT}/rag_evaluation/infoseek/score_fixed_infoseek_methods.py" \
  --max-samples "${SAMPLE_SIZE}" \
  --wikiprf-path "${GENERATION_DETAILS}" \
  --output-dir "${EVAL_DIR}"

printf '\nWiki-PRF generation details: %s\n' "${GENERATION_DETAILS}"
printf 'Wiki-PRF sample evaluation: %s\n' "${EVAL_DIR}/summary.json"
