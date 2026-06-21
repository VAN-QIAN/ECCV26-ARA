#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/methods/_common.sh"

SAMPLE_SIZE="${SAMPLE_SIZE:-1}"
SIMILAR_NUM="${SIMILAR_NUM:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SAMPLE_SIZE SIMILAR_NUM CUDA_VISIBLE_DEVICES

bash "${ROOT}/scripts/methods/run_comem.sh" \
  --dataset infoseek \
  --split fixed \
  --max-samples "${SAMPLE_SIZE}"

COMEM_OUTPUT="${ROOT}/outputs/generated_methods/CoMEM/infoseek/fixed/qwen2.5_CoMEM_custom_${SIMILAR_NUM}.jsonl"
if [[ ! -s "${COMEM_OUTPUT}" ]]; then
  echo "missing CoMEM sample output: ${COMEM_OUTPUT}" >&2
  exit 1
fi

activate_conda_env KBVQA_eval
python "${ROOT}/rag_evaluation/infoseek/score_fixed_infoseek_methods.py" \
  --max-samples "${SAMPLE_SIZE}" \
  --comem-path "${COMEM_OUTPUT}" \
  --output-dir "${ROOT}/results/evaluation/infoseek/fixed_from_generated_comem_sample"

printf '\nSample output: %s\n' "${COMEM_OUTPUT}"
printf 'Sample evaluation: %s\n' "${ROOT}/results/evaluation/infoseek/fixed_from_generated_comem_sample/summary.json"
