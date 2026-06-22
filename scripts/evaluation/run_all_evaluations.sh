#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-${ROOT}/rag_evaluation/evqa_eval/tfhub_cache}"
EVQA_EXTRA_ARGS=()
if [[ "${ALLOW_EXACT_MATCH_FALLBACK:-0}" == "1" ]]; then
  EVQA_EXTRA_ARGS+=(--allow-exact-match-fallback)
fi

python "${ROOT}/rag_evaluation/evqa/score_fixed_evqa_methods.py" \
  --max-samples "${MAX_SAMPLES}" \
  "${EVQA_EXTRA_ARGS[@]}"
python "${ROOT}/rag_evaluation/evqa/score_unfixed_evqa_methods.py" \
  --max-samples "${MAX_SAMPLES}" \
  "${EVQA_EXTRA_ARGS[@]}"
python "${ROOT}/rag_evaluation/evqa/score_augmented_evqa_methods.py" \
  --max-samples "${MAX_SAMPLES}" \
  "${EVQA_EXTRA_ARGS[@]}"
python "${ROOT}/rag_evaluation/infoseek/score_fixed_infoseek_methods.py" \
  --max-samples "${MAX_SAMPLES}"
python "${ROOT}/rag_evaluation/infoseek/score_unfixed_infoseek_methods.py" \
  --max-samples "${MAX_SAMPLES}"
python "${ROOT}/rag_evaluation/infoseek/score_augmented_infoseek_methods.py" \
  --max-samples "${MAX_SAMPLES}"
