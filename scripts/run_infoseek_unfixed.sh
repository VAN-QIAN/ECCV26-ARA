#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-${ROOT}/rag_evaluation/evqa_eval/tfhub_cache}"
python "${ROOT}/rag_evaluation/infoseek/score_unfixed_infoseek_methods.py" "$@"
