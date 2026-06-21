#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

required_paths=(
  "README.md"
  "ECCV_2026_Qian_Fixing_KBVQA.pdf"
  "data/README.md"
  "data/ground_truth/evqa_unfixed_test_with_id.csv"
  "data/ground_truth/evqa_fixed_final_check_Feb12.csv"
  "data/ground_truth/infoseek_unfixed_subset.csv"
  "data/ground_truth/infoseek_fixed_final_recheck_Feb7.csv"
  "data/augmented/evqa/evqa_challenging_queries_full_seed3185.csv"
  "data/augmented/evqa/evqa_challenging_queries_full_seed3185_with_images.csv"
  "data/augmented/infoseek/infoseek_challenging_queries_full_seed3185.csv"
  "data/augmented/infoseek/infoseek_challenging_queries_full_seed3185_with_images.csv"
  "data/samples/README.md"
  "data/samples/evqa_fixed_1.csv"
  "data/samples/infoseek_fixed_1.csv"
  "data/kb/evqa_encyclopedic_kb_wiki.json"
  "data/kb/infoseek_wiki_100_dict_v4.json"
  "data/images/evqa_landmark_images"
  "data/images/augmented/README.md"
  "data/images/augmented/evqa/composite_images_full_seed3185/method1"
  "data/images/augmented/evqa/composite_images_full_seed3185/method2"
  "data/images/augmented/infoseek/composite_images_full_seed3185/method1"
  "data/images/augmented/infoseek/composite_images_full_seed3185/method2"
  "data/checkpoints/CoMEM"
  "data/checkpoints/Wiki-PRF"
  "rag_evaluation/evqa_eval/evqa_utils.py"
  "rag_evaluation/evqa_eval/bem_model"
  "rag_evaluation/evqa/score_fixed_evqa_methods.py"
  "rag_evaluation/evqa/score_unfixed_evqa_methods.py"
  "rag_evaluation/evqa/score_augmented_evqa_methods.py"
  "rag_evaluation/infoseek/score_fixed_infoseek_methods.py"
  "rag_evaluation/infoseek/score_unfixed_infoseek_methods.py"
  "rag_evaluation/infoseek/score_augmented_infoseek_methods.py"
  "rag_evaluation/infoseek/compute_score_enhanced_string_with_bem.py"
  "methods/code/EchoSight/README.md"
  "methods/code/EchoSight/scripts/run_echosight_reranker.sh"
  "methods/code/EchoSight/scripts/run_echosight_answer.sh"
  "methods/code/IBA/README.md"
  "methods/code/IBA/scripts/run_iba_prepare.sh"
  "methods/code/IBA/scripts/run_iba_answer.sh"
  "methods/code/ReflectiVA/README.md"
  "methods/code/CoMEM/README.md"
  "methods/code/CoMEM/CoMEM-inference/EVQA/Custom_test_full"
  "methods/code/CoMEM/CoMEM-inference/EVQA/Custom_test_full_unfixed"
  "methods/code/CoMEM/CoMEM-inference/infoseek/Custom_test_full"
  "methods/code/CoMEM/CoMEM-inference/infoseek/Custom_test_full_unfixed"
  "methods/code/Wiki-PRF/README.md"
  "outputs/README.md"
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
  "scripts/run_evqa_augmented.sh"
  "scripts/run_infoseek_augmented.sh"
  "outputs/raw_methods/evqa/fixed/OurIBA.jsonl"
  "outputs/raw_methods/evqa/fixed/EchoSight.jsonl"
  "outputs/raw_methods/evqa/fixed/ReflectiVA.json"
  "outputs/raw_methods/evqa/fixed/Wiki_PRF.jsonl"
  "outputs/raw_methods/evqa/fixed/CoMEM.jsonl"
  "outputs/raw_methods/evqa/augmented/IBA_anchor.jsonl"
  "outputs/raw_methods/evqa/augmented/IBA_augmented_method1.jsonl"
  "outputs/raw_methods/evqa/augmented/IBA_augmented_method2.jsonl"
  "outputs/raw_methods/evqa/augmented/EchoSight_anchor.jsonl"
  "outputs/raw_methods/evqa/augmented/EchoSight_augmented_method1.jsonl"
  "outputs/raw_methods/evqa/augmented/EchoSight_augmented_method2.jsonl"
  "outputs/raw_methods/evqa/augmented/Wiki_PRF_anchor.jsonl"
  "outputs/raw_methods/evqa/augmented/Wiki_PRF_augmented_method1.jsonl"
  "outputs/raw_methods/evqa/augmented/Wiki_PRF_augmented_method2.jsonl"
  "outputs/raw_methods/infoseek/fixed/OurIBA.jsonl"
  "outputs/raw_methods/infoseek/fixed/EchoSight.jsonl"
  "outputs/raw_methods/infoseek/fixed/ReflectiVA"
  "outputs/raw_methods/infoseek/fixed/Wiki_PRF.jsonl"
  "outputs/raw_methods/infoseek/fixed/CoMEM.jsonl"
  "outputs/raw_methods/infoseek/augmented/IBA_anchor.jsonl"
  "outputs/raw_methods/infoseek/augmented/IBA_augmented_method1.jsonl"
  "outputs/raw_methods/infoseek/augmented/IBA_augmented_method2.jsonl"
  "outputs/raw_methods/infoseek/augmented/EchoSight_anchor.jsonl"
  "outputs/raw_methods/infoseek/augmented/EchoSight_augmented_method1.jsonl"
  "outputs/raw_methods/infoseek/augmented/EchoSight_augmented_method2.jsonl"
  "outputs/raw_methods/infoseek/augmented/Wiki_PRF_anchor.jsonl"
  "outputs/raw_methods/infoseek/augmented/Wiki_PRF_augmented_method1.jsonl"
  "outputs/raw_methods/infoseek/augmented/Wiki_PRF_augmented_method2.jsonl"
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
  "${ROOT}/rag_evaluation/infoseek/compute_score_enhanced_string_with_bem.py" \
  "${ROOT}/rag_evaluation/evqa_eval/evqa_utils.py" \
  "${ROOT}/methods/code/IBA/qwen_pipeline/pipeline.py" \
  "${ROOT}/methods/code/Wiki-PRF/run_wikiprf.py"

printf 'camera-ready check passed: %s\n' "${ROOT}"
