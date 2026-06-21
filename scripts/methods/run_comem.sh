#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/methods/run_comem.sh --dataset evqa --split fixed [--dry-run]

Options:
  --dataset evqa|infoseek
  --split fixed|unfixed
  --max-samples N
  --dry-run

Uses conda env: CoMEM by default. Override with CONDA_ENV=...
EOF
}

DATASET="evqa"
SPLIT="fixed"
DRY_RUN=0
MAX_SAMPLES=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

ROOT="$(repo_root)"
GT_CSV="$(gt_csv_for_split "${ROOT}" "${DATASET}" "${SPLIT}")"
METHOD_ROOT="${ROOT}/methods/code/CoMEM"
OUTPUT_DIR="${ROOT}/outputs/generated_methods/CoMEM/${DATASET}/${SPLIT}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT}/data/checkpoints/CoMEM}"
SIMILAR_NUM="${SIMILAR_NUM:-10}"
MODEL_NAME="${MODEL_NAME:-qwen2.5}"

case "${DATASET}:${SPLIT}" in
  evqa:fixed)
    PY_SCRIPT="CoMEM-inference/EVQA/run_EVQA_finetunekv_clip_customized.py"
    MDS_DIR="${MDS_DIR:-${METHOD_ROOT}/CoMEM-inference/EVQA/Custom_test_full}"
    ;;
  evqa:unfixed)
    PY_SCRIPT="CoMEM-inference/EVQA/run_EVQA_finetunekv_clip_customized_unfixed.py"
    MDS_DIR="${MDS_DIR:-${METHOD_ROOT}/CoMEM-inference/EVQA/Custom_test_full_unfixed}"
    ;;
  infoseek:fixed)
    PY_SCRIPT="CoMEM-inference/infoseek/run_infoseek_finetunekv_clip_customized.py"
    MDS_DIR="${MDS_DIR:-${METHOD_ROOT}/CoMEM-inference/infoseek/Custom_test_full}"
    ;;
  infoseek:unfixed)
    PY_SCRIPT="CoMEM-inference/infoseek/run_infoseek_finetunekv_clip_customized_unfixed.py"
    MDS_DIR="${MDS_DIR:-${METHOD_ROOT}/CoMEM-inference/infoseek/Custom_test_full_unfixed}"
    ;;
  *) echo "unsupported dataset/split: ${DATASET}/${SPLIT}" >&2; exit 1 ;;
esac

activate_conda_env "${CONDA_ENV:-CoMEM}"

mkdir -p "${OUTPUT_DIR}"
cmd=(
  python "${PY_SCRIPT}"
  --model_name "${MODEL_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --similar_num "${SIMILAR_NUM}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --metadata_path "${GT_CSV}"
  --mds_dir "${MDS_DIR}"
)
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  cmd+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ "${DATASET}" == "infoseek" ]]; then
  cmd+=(--image_root "${IMAGE_ROOT:-${ROOT}/data/images/infoseek_val_images}")
fi

cmd+=("${EXTRA_ARGS[@]}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'cd %q\n' "${METHOD_ROOT}"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

cd "${METHOD_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"${cmd[@]}"
