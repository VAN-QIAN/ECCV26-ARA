#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/methods/run_wikiprf.sh --dataset evqa --split fixed [--dry-run]

Options:
  --dataset evqa|infoseek
  --split fixed|unfixed
  --dry-run

Uses conda env: echosight
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
METHOD_ROOT="${ROOT}/methods/code/Wiki-PRF"
SCRIPT="${METHOD_ROOT}/scripts/run_${DATASET}_${SPLIT}.sh"
if [[ ! -x "${SCRIPT}" && ! -f "${SCRIPT}" ]]; then
  echo "unsupported dataset/split: ${DATASET}/${SPLIT}" >&2
  exit 1
fi

activate_conda_env echosight

cmd=(bash "${SCRIPT}")
[[ "${DRY_RUN}" == "1" ]] && cmd+=(--dry-run)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}" "${EXTRA_ARGS[@]}"
  printf '\n'
  exit 0
fi
"${cmd[@]}" "${EXTRA_ARGS[@]}"
