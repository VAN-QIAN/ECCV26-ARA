#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

required_paths=(
  ".gitignore"
  "README.md"
  "configs/paths.example.env"
  "data/README.md"
  "data/ground_truth/evqa_unfixed_test_with_id.csv"
  "data/ground_truth/evqa_fixed.csv"
  "data/ground_truth/infoseek_unfixed_subset.csv"
  "data/ground_truth/infoseek_fixed.csv"
  "data/augmented/evqa/evqa_challenging_queries_full_seed3185_with_images.csv"
  "data/augmented/infoseek/infoseek_challenging_queries_full_seed3185_with_images.csv"
  "data/samples/README.md"
  "data/samples/evqa_fixed_1.csv"
  "data/samples/infoseek_fixed_1.csv"
  "data/checkpoints/README.md"
  "data/checkpoints/.gitkeep"
  "data/images/README.md"
  "data/images/augmented/README.md"
  "data/images/augmented/evqa/composite_images_full_seed3185/method1"
  "data/images/augmented/evqa/composite_images_full_seed3185/method1/.gitkeep"
  "data/images/augmented/evqa/composite_images_full_seed3185/method2"
  "data/images/augmented/evqa/composite_images_full_seed3185/method2/.gitkeep"
  "data/images/augmented/infoseek/composite_images_full_seed3185/method1"
  "data/images/augmented/infoseek/composite_images_full_seed3185/method1/.gitkeep"
  "data/images/augmented/infoseek/composite_images_full_seed3185/method2"
  "data/images/augmented/infoseek/composite_images_full_seed3185/method2/.gitkeep"
  "data/kb/README.md"
  "data/kb/.gitkeep"
  "data/retrieval/README.md"
  "data/retrieval/.gitkeep"
  "data/wikiprf/README.md"
  "data/wikiprf/.gitkeep"
  "rag_evaluation/evqa_eval/evqa_utils.py"
  "rag_evaluation/evqa/score_fixed_evqa_methods.py"
  "rag_evaluation/evqa/score_unfixed_evqa_methods.py"
  "rag_evaluation/evqa/score_augmented_evqa_methods.py"
  "rag_evaluation/infoseek/score_fixed_infoseek_methods.py"
  "rag_evaluation/infoseek/score_unfixed_infoseek_methods.py"
  "rag_evaluation/infoseek/score_augmented_infoseek_methods.py"
  "rag_evaluation/infoseek/answer_reward_utils.py"
  "methods/code/EchoSight/README.md"
  "methods/code/EchoSight/scripts/run_echosight_reranker.sh"
  "methods/code/EchoSight/scripts/run_echosight_answer.sh"
  "methods/code/IBA/README.md"
  "methods/code/IBA/scripts/run_iba_prepare.sh"
  "methods/code/IBA/scripts/run_iba_answer.sh"
  "methods/code/ReflectiVA/README.md"
  "methods/code/CoMEM/README.md"
  "methods/code/Wiki-PRF/README.md"
  "outputs/README.md"
  "outputs/raw_methods/README.md"
  "outputs/raw_methods/evqa/fixed/.gitkeep"
  "outputs/raw_methods/evqa/unfixed/.gitkeep"
  "outputs/raw_methods/evqa/augmented/.gitkeep"
  "outputs/raw_methods/infoseek/fixed/.gitkeep"
  "outputs/raw_methods/infoseek/unfixed/.gitkeep"
  "outputs/raw_methods/infoseek/augmented/.gitkeep"
  "scripts/setup_local_assets.sh"
  "scripts/methods/run_echosight.sh"
  "scripts/methods/run_iba.sh"
  "scripts/methods/run_reflectiva.sh"
  "scripts/methods/run_comem.sh"
  "scripts/methods/run_wikiprf.sh"
  "scripts/check_method_image_alignment.py"
  "scripts/run_comem_infoseek_sample.sh"
  "scripts/run_echosight_evqa_reranker_sample.sh"
  "scripts/run_iba_infoseek_sample.sh"
  "scripts/run_wikiprf_infoseek_sample.sh"
  "scripts/evaluation/README.md"
  "scripts/evaluation/run_all_evaluations.sh"
  "scripts/evaluation/run_smoke.sh"
  "scripts/evaluation/run_evqa_fixed.sh"
  "scripts/evaluation/run_evqa_unfixed.sh"
  "scripts/evaluation/run_evqa_augmented.sh"
  "scripts/evaluation/run_infoseek_fixed.sh"
  "scripts/evaluation/run_infoseek_unfixed.sh"
  "scripts/evaluation/run_infoseek_augmented.sh"
)

missing=0
for path in "${required_paths[@]}"; do
  if [[ ! -e "${ROOT}/${path}" ]]; then
    printf 'missing: %s\n' "${path}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

python -m py_compile \
  "${ROOT}/scripts/check_method_image_alignment.py" \
  "${ROOT}/rag_evaluation/evqa/score_fixed_evqa_methods.py" \
  "${ROOT}/rag_evaluation/evqa/score_unfixed_evqa_methods.py" \
  "${ROOT}/rag_evaluation/evqa/score_augmented_evqa_methods.py" \
  "${ROOT}/rag_evaluation/infoseek/score_fixed_infoseek_methods.py" \
  "${ROOT}/rag_evaluation/infoseek/score_unfixed_infoseek_methods.py" \
  "${ROOT}/rag_evaluation/infoseek/score_augmented_infoseek_methods.py" \
  "${ROOT}/rag_evaluation/infoseek/answer_reward_utils.py" \
  "${ROOT}/rag_evaluation/evqa_eval/evqa_utils.py" \
  "${ROOT}/methods/code/IBA/qwen_pipeline/pipeline.py" \
  "${ROOT}/methods/code/Wiki-PRF/run_wikiprf.py"

printf 'camera-ready check passed: %s\n' "${ROOT}"
