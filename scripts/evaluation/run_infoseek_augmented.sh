#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python "${ROOT}/rag_evaluation/infoseek/score_augmented_infoseek_methods.py" "$@"
