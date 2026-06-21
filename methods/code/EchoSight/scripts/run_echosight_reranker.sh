#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_echosight_reranker.sh --dataset evqa [--dry-run]
  bash scripts/run_echosight_reranker.sh --dataset infoseek [--dry-run]

Environment overrides:
  TEST_FILE KNOWLEDGE_BASE FAISS_INDEX QFORMER_CKPT_PATH SAVE_RESULT_PATH
  RETRIEVAL_TOP_K TOP_KS RETRIEVER_VIT CUDA_VISIBLE_DEVICES
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMERA_READY_ROOT="$(cd "${METHOD_ROOT}/../../.." && pwd)"

DATASET="evqa"
DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

OUTPUT_DIR="${OUTPUT_DIR:-${CAMERA_READY_ROOT}/outputs/generated_methods/EchoSight}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-20}"
TOP_KS="${TOP_KS:-1,2,3,5,10,20}"
RETRIEVER_VIT="${RETRIEVER_VIT:-eva-clip}"
QFORMER_CKPT_PATH="${QFORMER_CKPT_PATH:-/data/qianMa/EchoSight/reranker.pth}"
# test_reranker_echo_score.py uses cuda:0 for QFormer and cuda:1 for CLIP.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

case "$DATASET" in
  evqa)
    TEST_FILE="${TEST_FILE:-${CAMERA_READY_ROOT}/data/ground_truth/evqa_fixed_final_check_Feb12.csv}"
    KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-${CAMERA_READY_ROOT}/data/kb/evqa_encyclopedic_kb_wiki.json}"
    FAISS_INDEX="${FAISS_INDEX:-${CAMERA_READY_ROOT}/data/kb/KB_EVQA/root/FAISS_INDEX/EVA-CLIP_2/evqa_index_full}"
    SAVE_RESULT_PATH="${SAVE_RESULT_PATH:-${OUTPUT_DIR}/evqa_reranker_k${RETRIEVAL_TOP_K}.json}"
    ;;
  infoseek)
    TEST_FILE="${TEST_FILE:-${CAMERA_READY_ROOT}/data/ground_truth/infoseek_fixed_final_recheck_Feb7.csv}"
    KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-${CAMERA_READY_ROOT}/data/kb/infoseek_wiki_100_dict_v4.json}"
    FAISS_INDEX="${FAISS_INDEX:-${CAMERA_READY_ROOT}/data/kb/KB_infoseek}"
    SAVE_RESULT_PATH="${SAVE_RESULT_PATH:-${OUTPUT_DIR}/infoseek_reranker_k${RETRIEVAL_TOP_K}.json}"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}. Use evqa or infoseek." >&2
    exit 1
    ;;
esac

case "${FAISS_INDEX}" in
  */) ;;
  *) FAISS_INDEX="${FAISS_INDEX}/" ;;
esac

CMD=(
  python -m test.test_reranker_echo_score
  --test_file "$TEST_FILE"
  --knowledge_base "$KNOWLEDGE_BASE"
  --faiss_index "$FAISS_INDEX"
  --retriever_vit "$RETRIEVER_VIT"
  --top_ks "$TOP_KS"
  --retrieval_top_k "$RETRIEVAL_TOP_K"
  --perform_qformer_reranker
  --qformer_ckpt_path "$QFORMER_CKPT_PATH"
  --save_result
  --save_result_path "$SAVE_RESULT_PATH"
)

if [[ -n "${RESUME_FROM:-}" ]]; then
  CMD+=(--resume_from "$RESUME_FROM")
fi
CMD+=("${EXTRA_ARGS[@]}")

print_command() {
  printf 'cd %q\n' "$METHOD_ROOT"
  printf 'export PYTHONPATH=%q\n' "${METHOD_ROOT}:${PYTHONPATH:-}"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_command
  exit 0
fi

mkdir -p "$(dirname "$SAVE_RESULT_PATH")"
for required in "$TEST_FILE" "$KNOWLEDGE_BASE" "$FAISS_INDEX" "$QFORMER_CKPT_PATH"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

cd "$METHOD_ROOT"
export PYTHONPATH="${METHOD_ROOT}:${PYTHONPATH:-}"
"${CMD[@]}"
printf '\nEchoSight reranker output: %s\n' "$SAVE_RESULT_PATH"
