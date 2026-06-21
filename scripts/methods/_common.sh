#!/usr/bin/env bash
set -euo pipefail

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

activate_conda_env() {
  local env_name="$1"
  if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    return
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    return
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available; activate ${env_name} or set SKIP_CONDA_ACTIVATE=1" >&2
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  set +u
  conda activate "${env_name}"
  set -u
}

gt_csv_for_split() {
  local root="$1"
  local dataset="$2"
  local split="$3"
  case "${dataset}:${split}" in
    evqa:fixed) printf '%s\n' "${root}/data/ground_truth/evqa_fixed_final_check_Feb12.csv" ;;
    evqa:unfixed) printf '%s\n' "${root}/data/ground_truth/evqa_unfixed_test_with_id.csv" ;;
    infoseek:fixed) printf '%s\n' "${root}/data/ground_truth/infoseek_fixed_final_recheck_Feb7.csv" ;;
    infoseek:unfixed) printf '%s\n' "${root}/data/ground_truth/infoseek_unfixed_subset.csv" ;;
    *) echo "unsupported dataset/split: ${dataset}/${split}" >&2; exit 1 ;;
  esac
}
