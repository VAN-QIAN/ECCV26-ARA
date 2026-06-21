#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_CSV="${ROOT}/data/samples/evqa_fixed_1.csv"
OUTPUT_PATH="${ROOT}/outputs/generated_methods/EchoSight/evqa/fixed/sample_reranker_k3.json"

if [[ -s "${OUTPUT_PATH}" && "${FORCE:-0}" != "1" ]]; then
  printf 'Existing EchoSight sample output: %s\n' "${OUTPUT_PATH}"
  printf 'Set FORCE=1 to rerun retrieval/reranking.\n'
  exit 0
fi

if [[ ! -s "${SAMPLE_CSV}" ]]; then
  echo "missing sample CSV: ${SAMPLE_CSV}" >&2
  exit 1
fi

conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${conda_base}/etc/profile.d/conda.sh"
set +u
conda activate echosight
set -u

TEST_FILE="${SAMPLE_CSV}" \
OUTPUT_DIR="${ROOT}/outputs/generated_methods/EchoSight/evqa/fixed" \
SAVE_RESULT_PATH="${OUTPUT_PATH}" \
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-3}" \
TOP_KS="${TOP_KS:-1,3}" \
bash "${ROOT}/methods/code/EchoSight/scripts/run_echosight_reranker.sh" --dataset evqa

printf '\nEchoSight sample output: %s\n' "${OUTPUT_PATH}"
