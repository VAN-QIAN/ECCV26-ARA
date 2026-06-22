#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAX_SAMPLES="${MAX_SAMPLES:-1}"

SAMPLE_GT="${ROOT}/data/samples/infoseek_fixed_1.csv"
IBA_SAMPLE="${ROOT}/outputs/generated_methods/IBA/infoseek/fixed/sample_answers.jsonl"
COMEM_SAMPLE="${ROOT}/outputs/generated_methods/CoMEM/infoseek/fixed/qwen2.5_CoMEM_custom_1.jsonl"
WIKIPRF_SAMPLE="${ROOT}/outputs/generated_methods/Wiki-PRF/infoseek_fixed/sample_results_infoseek_fixed_step600_topk3.json_0_generation_details.jsonl"
REFLECTIVA_SAMPLE_DIR="${ROOT}/outputs/generated_methods/ReflectiVA/infoseek/fixed"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/evaluation/smoke/infoseek_fixed}"

for required in "${SAMPLE_GT}" "${IBA_SAMPLE}" "${COMEM_SAMPLE}" "${WIKIPRF_SAMPLE}" "${REFLECTIVA_SAMPLE_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    printf 'missing smoke input: %s\n' "${required}" >&2
    exit 1
  fi
done

python "${ROOT}/rag_evaluation/infoseek/score_fixed_infoseek_methods.py" \
  --ground-truth-csv "${SAMPLE_GT}" \
  --iba-path "${IBA_SAMPLE}" \
  --echosight-path "${IBA_SAMPLE}" \
  --reflectiva-dir "${REFLECTIVA_SAMPLE_DIR}" \
  --wikiprf-path "${WIKIPRF_SAMPLE}" \
  --comem-path "${COMEM_SAMPLE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}"

printf 'smoke evaluation summary: %s\n' "${OUTPUT_DIR}/summary.json"
