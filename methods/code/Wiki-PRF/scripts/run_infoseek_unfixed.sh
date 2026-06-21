#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAMERA_READY_ROOT="$(cd "${METHOD_ROOT}/../../.." && pwd)"

export PYTHONPATH="${METHOD_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${METHOD_ROOT}/run_wikiprf.py" \
  --dataset infoseek_unfixed \
  --model_path "${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}" \
  --peft_model_path "${PEFT_MODEL_PATH:-${CAMERA_READY_ROOT}/data/checkpoints/Wiki-PRF}" \
  --knowledge_base "${KNOWLEDGE_BASE:-${CAMERA_READY_ROOT}/data/kb/infoseek_wiki_100_dict_v4.json}" \
  --faiss_root "${FAISS_ROOT:-${CAMERA_READY_ROOT}/data/kb/KB_infoseek}" \
  --output_dir "${OUTPUT_DIR:-${CAMERA_READY_ROOT}/outputs/generated_methods/Wiki-PRF}" \
  --model_gpu_id "${MODEL_GPU_ID:-0}" \
  --filter_gpu_id "${FILTER_GPU_ID:-0}" \
  --retriever_gpu_id "${RETRIEVER_GPU_ID:-0}" \
  "$@"
