#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run ReflectiVA on InfoSeek with the EchoSight KB.

Usage:
  bash scripts/run_reflectiva_infoseek.sh [--dry-run] [extra python args...]

Common env overrides:
  MODEL_PATH, MODEL_NAME, INPUT_CSV, QUESTION_CSV, IMAGE_ROOT, INDEX_PATH
  KNOWLEDGE_BASE, OUTPUT_DIR, OUTPUT_PATH, ENTITY_K, PART, TOTAL_PART
  CUDA_VISIBLE_DEVICES, SAMPLES_PER_PART
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
INPUT_CSV="${INPUT_CSV:-$CAMERA_READY_ROOT/data/ground_truth/infoseek_fixed.csv}"
QUESTION_CSV="${QUESTION_CSV:-$INPUT_CSV}"
IMAGE_ROOT="${IMAGE_ROOT:-$CAMERA_READY_ROOT/data/images/reflectiva_infoseek_val_image}"
INDEX_PATH="${INDEX_PATH:-$CAMERA_READY_ROOT/data/kb/KB_infoseek}"
KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-$CAMERA_READY_ROOT/data/kb/infoseek_wiki_100_dict_v4.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$CAMERA_READY_ROOT/outputs/generated_methods/ReflectiVA/infoseek}"
ENTITY_K="${ENTITY_K:-5}"
PART="${PART:-0}"
TOTAL_PART="${TOTAL_PART:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INPUT_STEM="$(basename "$INPUT_CSV" .csv)"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTPUT_DIR/split_${PART}_${INPUT_STEM}_k${ENTITY_K}.json}"

export PYTHONPATH="$METHOD_ROOT:${PYTHONPATH:-}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export PART TOTAL_PART CUDA_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR"

cmd=(
  python rag_evaluation/infoseek/release_retrieval_echo_kb.py
  --model_path "$MODEL_PATH"
  --model_name "$MODEL_NAME"
  --index_path "$INDEX_PATH"
  --entity_k "$ENTITY_K"
  --use_eva_to_retrieve
  --retriever_path BAAI/EVA-CLIP-8B
  --retriever_processor_path openai/clip-vit-large-patch14
  --image_root "$IMAGE_ROOT"
  --input_csv "$INPUT_CSV"
  --question_csv "$QUESTION_CSV"
  --kb_wikipedia_path "$KNOWLEDGE_BASE"
  --short_prompt
  --answers_file "$OUTPUT_PATH"
)

if [[ -n "${SAMPLES_PER_PART:-}" ]]; then
  cmd+=(--samples_per_part "$SAMPLES_PER_PART")
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
