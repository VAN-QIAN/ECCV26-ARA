#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run ReflectiVA on EVQA with the EchoSight KB.

Usage:
  bash scripts/run_reflectiva_evqa.sh [--dry-run] [extra python args...]

Common env overrides:
  MODEL_PATH, MODEL_NAME, DATA_PATH, IMAGE_ROOT, INDEX_PATH, KNOWLEDGE_BASE
  OUTPUT_DIR, OUTPUT_PATH, ENTITY_K, PART, TOTAL_PART, CUDA_VISIBLE_DEVICES
  USE_FLASH_ATTN=1
EOF
}

DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CAMERA_READY_ROOT="$(cd "$METHOD_ROOT/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-aimagelab/ReflectiVA}"
MODEL_NAME="${MODEL_NAME:-llava_llama_3.1}"
DATA_PATH="${DATA_PATH:-$METHOD_ROOT/data_evqa/test_one_hop_Feb14.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$CAMERA_READY_ROOT/data/images/reflectiva_evqa_inference_images}"
INDEX_PATH="${INDEX_PATH:-$CAMERA_READY_ROOT/data/kb/reflectiva_evqa_EVA_image}"
KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-$CAMERA_READY_ROOT/data/kb/evqa_encyclopedic_kb_wiki.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$CAMERA_READY_ROOT/outputs/generated_methods/ReflectiVA/evqa}"
ENTITY_K="${ENTITY_K:-5}"
PART="${PART:-0}"
TOTAL_PART="${TOTAL_PART:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INPUT_STEM="$(basename "$DATA_PATH" .json)"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/split_${PART}_${INPUT_STEM}_k${ENTITY_K}.json}"

export PYTHONPATH="$METHOD_ROOT:${PYTHONPATH:-}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export PART TOTAL_PART CUDA_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR"

cmd=(
  python rag_evaluation/encyclopedic/release_retrieval_echo_kb.py
  --model_path "$MODEL_PATH"
  --model_name "$MODEL_NAME"
  --data_path "$DATA_PATH"
  --image_root "$IMAGE_ROOT"
  --index_path "$INDEX_PATH"
  --entity_k "$ENTITY_K"
  --use_eva_to_retrieve
  --retriever_path BAAI/EVA-CLIP-8B
  --retriever_processor_path openai/clip-vit-large-patch14
  --kb_wikipedia_path "$KNOWLEDGE_BASE"
  --short_prompt
  --answers_file "$OUTPUT_PATH"
)

if [[ "${USE_FLASH_ATTN:-0}" == "1" ]]; then
  cmd+=(--use_flash_attn)
fi

cmd+=("${EXTRA_ARGS[@]}")

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'cd %q\n' "$METHOD_ROOT"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

cd "$METHOD_ROOT"
"${cmd[@]}"
