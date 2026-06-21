#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/methods/_common.sh"

SAMPLE_SIZE="${SAMPLE_SIZE:-1}"
SAMPLE_CSV="${SAMPLE_CSV:-${ROOT}/data/samples/infoseek_fixed_1.csv}"
RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS:-/data/qianMa/EchoSight/ECCV_results/EchoSight_reranker_full_InfoSeek_k20_20260213_095859.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/generated_methods/IBA/infoseek/fixed}"
METADATA_PATH="${METADATA_PATH:-${OUTPUT_DIR}/sample_metadata.jsonl}"
ANSWER_PATH="${ANSWER_PATH:-${OUTPUT_DIR}/sample_answers.jsonl}"
ALIGNED_METADATA_PATH="${ALIGNED_METADATA_PATH:-${OUTPUT_DIR}/sample_aligned_metadata.jsonl}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/results/evaluation/infoseek/fixed_from_generated_iba_sample}"

QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://127.0.0.1:8001/v1}"
ANSWER_BACKEND_VLLM_BASE_URL="${ANSWER_BACKEND_VLLM_BASE_URL:-${QWEN_VLLM_BASE_URL}}"
ANSWER_BACKEND="${ANSWER_BACKEND:-qwen}"
ANSWER_BACKEND_VLLM_MODEL_NAME="${ANSWER_BACKEND_VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
IDENTIFICATION_TOP_K="${IDENTIFICATION_TOP_K:-3}"
IDENTIFICATION_SELECT_TOP="${IDENTIFICATION_SELECT_TOP:-1}"
IDENTIFICATION_SCORE_TOP_K="${IDENTIFICATION_SCORE_TOP_K:-1}"
ENTITY_TOP_K="${ENTITY_TOP_K:-1}"
IDENTIFICATION_MAX_NEW_TOKENS="${IDENTIFICATION_MAX_NEW_TOKENS:-64}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-96}"
export QWEN_VLLM_BASE_URL ANSWER_BACKEND_VLLM_BASE_URL ANSWER_BACKEND ANSWER_BACKEND_VLLM_MODEL_NAME
export IDENTIFICATION_TOP_K IDENTIFICATION_SELECT_TOP IDENTIFICATION_SCORE_TOP_K ENTITY_TOP_K
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for required in "${SAMPLE_CSV}" "${RETRIEVAL_RESULTS}"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing required path: ${required}" >&2
    exit 1
  fi
done

if command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "${QWEN_VLLM_BASE_URL%/}/models" >/dev/null; then
    echo "Qwen vLLM endpoint is not reachable: ${QWEN_VLLM_BASE_URL}" >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"
if [[ "${FORCE:-0}" == "1" ]]; then
  rm -f "${METADATA_PATH}" "${METADATA_PATH%.jsonl}_timing.json" "${METADATA_PATH%.jsonl}.log" \
    "${ANSWER_PATH}" "${ALIGNED_METADATA_PATH}"
fi

if [[ ! -s "${ANSWER_PATH}" ]]; then
  activate_conda_env echosight

  TEST_FILE="${SAMPLE_CSV}" \
  RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  METADATA_PATH="${METADATA_PATH}" \
  QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL}" \
  bash "${ROOT}/methods/code/IBA/scripts/run_iba_prepare.sh" \
    --dataset infoseek \
    --identification_max_new_tokens "${IDENTIFICATION_MAX_NEW_TOKENS}"

  CSV_PATH="${SAMPLE_CSV}" \
  METADATA_PATH="${METADATA_PATH}" \
  OUTPUT_PATH="${ANSWER_PATH}" \
  ALIGNED_METADATA_PATH="${ALIGNED_METADATA_PATH}" \
  QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL}" \
  ANSWER_BACKEND="${ANSWER_BACKEND}" \
  ANSWER_BACKEND_VLLM_BASE_URL="${ANSWER_BACKEND_VLLM_BASE_URL}" \
  ANSWER_BACKEND_VLLM_MODEL_NAME="${ANSWER_BACKEND_VLLM_MODEL_NAME}" \
  bash "${ROOT}/methods/code/IBA/scripts/run_iba_answer.sh" \
    --dataset infoseek \
    --answer_max_new_tokens "${ANSWER_MAX_NEW_TOKENS}"
else
  printf 'Existing IBA sample answer: %s\n' "${ANSWER_PATH}"
  printf 'Set FORCE=1 to rerun prepare and answer generation.\n'
fi

if [[ ! -s "${ANSWER_PATH}" ]]; then
  echo "missing IBA sample answer output: ${ANSWER_PATH}" >&2
  exit 1
fi

activate_conda_env KBVQA_eval
python "${ROOT}/rag_evaluation/infoseek/score_fixed_infoseek_methods.py" \
  --max-samples "${SAMPLE_SIZE}" \
  --ouriba-path "${ANSWER_PATH}" \
  --output-dir "${EVAL_DIR}"

printf '\nIBA sample metadata: %s\n' "${METADATA_PATH}"
printf 'IBA sample answer: %s\n' "${ANSWER_PATH}"
printf 'IBA sample evaluation: %s\n' "${EVAL_DIR}/summary.json"
