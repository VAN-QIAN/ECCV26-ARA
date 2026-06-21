#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/methods/run_reflectiva.sh --dataset evqa --split fixed

Options:
  --dataset evqa|infoseek
  --split fixed|unfixed
  --dry-run

Uses conda env: reflectiva
EOF
}

DATASET="evqa"
SPLIT="fixed"
DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

ROOT="$(repo_root)"
GT_CSV="$(gt_csv_for_split "${ROOT}" "${DATASET}" "${SPLIT}")"
METHOD_ROOT="${ROOT}/methods/code/ReflectiVA"
OUTPUT_DIR="${ROOT}/outputs/generated_methods/ReflectiVA/${DATASET}/${SPLIT}"

activate_conda_env reflectiva

mkdir -p "${OUTPUT_DIR}"
if [[ "${DATASET}" == "evqa" ]]; then
  DATA_PATH="${DATA_PATH:-${OUTPUT_DIR}/test_one_hop_${SPLIT}.json}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'python %q --csv %q --output %q\n' "${METHOD_ROOT}/data_evqa/adapt_data.py" "${GT_CSV}" "${DATA_PATH}"
  elif [[ ! -e "${DATA_PATH}" ]]; then
    python "${METHOD_ROOT}/data_evqa/adapt_data.py" --csv "${GT_CSV}" --output "${DATA_PATH}"
  fi
  cmd=(bash "${METHOD_ROOT}/scripts/run_reflectiva_evqa.sh")
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
  DATA_PATH="${DATA_PATH}" OUTPUT_DIR="${OUTPUT_DIR}" "${cmd[@]}" "${EXTRA_ARGS[@]}"
elif [[ "${DATASET}" == "infoseek" ]]; then
  cmd=(bash "${METHOD_ROOT}/scripts/run_reflectiva_infoseek.sh")
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
  INPUT_CSV="${GT_CSV}" QUESTION_CSV="${GT_CSV}" OUTPUT_DIR="${OUTPUT_DIR}" "${cmd[@]}" "${EXTRA_ARGS[@]}"
else
  echo "unsupported dataset: ${DATASET}" >&2
  exit 1
fi

