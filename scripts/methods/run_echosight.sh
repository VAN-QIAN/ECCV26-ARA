#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/methods/run_echosight.sh --dataset evqa --split fixed --stage all

Options:
  --dataset evqa|infoseek
  --split fixed|unfixed
  --stage reranker|answer|all
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
METHOD_ROOT="${ROOT}/methods/code/EchoSight"
OUTPUT_DIR="${ROOT}/outputs/generated_methods/EchoSight/${DATASET}/${SPLIT}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-20}"
RERANK_PATH="${RERANK_PATH:-${OUTPUT_DIR}/reranker_k${RETRIEVAL_TOP_K}.jsonl}"
ANSWER_PATH="${ANSWER_PATH:-${OUTPUT_DIR}/answers.jsonl}"

case "${STAGE}" in
  reranker|answer|all) ;;
  *) echo "unsupported stage: ${STAGE}" >&2; exit 1 ;;
esac

activate_conda_env echosight

mkdir -p "${OUTPUT_DIR}"
if [[ "${STAGE}" == "reranker" || "${STAGE}" == "all" ]]; then
  cmd=(bash "${METHOD_ROOT}/scripts/run_echosight_reranker.sh" --dataset "${DATASET}")
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
  TEST_FILE="${GT_CSV}" OUTPUT_DIR="${OUTPUT_DIR}" SAVE_RESULT_PATH="${RERANK_PATH}" RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K}" "${cmd[@]}" "${EXTRA_ARGS[@]}"
fi

if [[ "${STAGE}" == "answer" || "${STAGE}" == "all" ]]; then
  cmd=(bash "${METHOD_ROOT}/scripts/run_echosight_answer.sh" --dataset "${DATASET}")
  [[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
  TEST_FILE="${GT_CSV}" OUTPUT_DIR="${OUTPUT_DIR}" RETRIEVAL_RESULTS="${RETRIEVAL_RESULTS:-${RERANK_PATH}}" OUTPUT_PATH="${ANSWER_PATH}" "${cmd[@]}" "${EXTRA_ARGS[@]}"
fi

