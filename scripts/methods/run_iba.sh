#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/methods/run_iba.sh --dataset evqa --split fixed --stage all

Options:
  --dataset evqa|infoseek
  --split fixed|unfixed
  --stage prepare|answer|all
  --dry-run

Uses conda env: echosight
EOF
}

DATASET="evqa"
SPLIT="fixed"
STAGE="all"
DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

ROOT="$(repo_root)"
GT_CSV="$(gt_csv_for_split "${ROOT}" "${DATASET}" "${SPLIT}")"
METHOD_ROOT="${ROOT}/methods/code/IBA"
OUTPUT_DIR="${ROOT}/outputs/generated_methods/IBA/${DATASET}/${SPLIT}"
METADATA_PATH="${METADATA_PATH:-${OUTPUT_DIR}/metadata.jsonl}"
ANSWER_PATH="${ANSWER_PATH:-${OUTPUT_DIR}/answers.jsonl}"
ALIGNED_METADATA_PATH="${ALIGNED_METADATA_PATH:-${OUTPUT_DIR}/aligned_metadata.jsonl}"
RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS:-${ROOT}/outputs/raw_methods/${DATASET}/${SPLIT}/EchoSight.jsonl}"

case "${STAGE}" in
  prepare|answer|all) ;;
  *) echo "unsupported stage: ${STAGE}" >&2; exit 1 ;;
esac

activate_conda_env echosight

mkdir -p "${OUTPUT_DIR}"
if [[ "${STAGE}" == "prepare" || "${STAGE}" == "all" ]]; then
  cmd=(bash "${METHOD_ROOT}/scripts/run_iba_prepare.sh" --dataset "${DATASET}")
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
  TEST_FILE="${GT_CSV}" RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS}" METADATA_PATH="${METADATA_PATH}" "${cmd[@]}" "${EXTRA_ARGS[@]}"
fi

if [[ "${STAGE}" == "answer" || "${STAGE}" == "all" ]]; then
  cmd=(bash "${METHOD_ROOT}/scripts/run_iba_answer.sh" --dataset "${DATASET}")
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
  CSV_PATH="${GT_CSV}" METADATA_PATH="${METADATA_PATH}" OUTPUT_PATH="${ANSWER_PATH}" ALIGNED_METADATA_PATH="${ALIGNED_METADATA_PATH}" "${cmd[@]}" "${EXTRA_ARGS[@]}"
fi

