#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/configs/paths.env}"
STRICT="${STRICT:-0}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  printf 'No local env file found: %s\n' "${ENV_FILE}" >&2
  printf 'Copy configs/paths.example.env to configs/paths.env and edit it first.\n' >&2
  exit 1
fi

is_placeholder() {
  [[ -z "${1:-}" || "${1}" == /path/to/* ]]
}

link_asset() {
  local target="$1"
  local link="$2"
  if is_placeholder "${target}"; then
    printf 'skip unset target: %s\n' "${link}"
    return 0
  fi
  if [[ ! -e "${target}" ]]; then
    printf 'missing target: %s -> %s\n' "${link}" "${target}" >&2
    [[ "${STRICT}" == "1" ]] && return 1
    return 0
  fi

  mkdir -p "$(dirname "${link}")"
  if [[ -L "${link}" ]]; then
    rm "${link}"
  elif [[ -e "${link}" ]]; then
    printf 'skip existing non-symlink: %s\n' "${link}" >&2
    [[ "${STRICT}" == "1" ]] && return 1
    return 0
  fi
  ln -s "${target}" "${link}"
  printf 'linked: %s -> %s\n' "${link}" "${target}"
}

ECHOSIGHT_ROOT="${ECHOSIGHT_ROOT:-}"
REFLECTIVA_ROOT="${REFLECTIVA_ROOT:-}"
COMEM_ROOT="${COMEM_ROOT:-}"
COMEM_CHECKPOINT_DIR="${COMEM_CHECKPOINT_DIR:-}"
WIKIPRF_ROOT="${WIKIPRF_ROOT:-}"
WIKIPRF_CHECKPOINT_DIR="${WIKIPRF_CHECKPOINT_DIR:-}"
WIKIPRF_INFOSEEK_UNFIXED_OUTPUT="${WIKIPRF_INFOSEEK_UNFIXED_OUTPUT:-}"
IBA_EVQA_FIXED_OUTPUT="${IBA_EVQA_FIXED_OUTPUT:-}"
IBA_EVQA_UNFIXED_OUTPUT="${IBA_EVQA_UNFIXED_OUTPUT:-}"
IBA_INFOSEEK_FIXED_OUTPUT="${IBA_INFOSEEK_FIXED_OUTPUT:-}"
IBA_INFOSEEK_UNFIXED_OUTPUT="${IBA_INFOSEEK_UNFIXED_OUTPUT:-}"
BEM_MODEL_DIR="${BEM_MODEL_DIR:-}"

link_asset "${COMEM_CHECKPOINT_DIR}" "${ROOT}/data/checkpoints/CoMEM"
link_asset "${WIKIPRF_CHECKPOINT_DIR}" "${ROOT}/data/checkpoints/Wiki-PRF"
link_asset "${ECHOSIGHT_ROOT}/reranker.pth" "${ROOT}/data/checkpoints/EchoSight/reranker.pth"

link_asset "${ECHOSIGHT_ROOT}/images" "${ROOT}/data/images/echosight_images"
link_asset "${ECHOSIGHT_ROOT}/images/val_id2name.json" "${ROOT}/data/images/echosight_inat_val_id2name.json"
link_asset "${ECHOSIGHT_ROOT}/E-VQA/landmark" "${ROOT}/data/images/evqa_landmark_images"
link_asset "${ECHOSIGHT_ROOT}/images/val" "${ROOT}/data/images/evqa_val_images"
link_asset "${ECHOSIGHT_ROOT}/InfoSeek/infoseek_val" "${ROOT}/data/images/infoseek_val_images"
link_asset "${ECHOSIGHT_ROOT}/InfoSeek/wikipedia_images_full" "${ROOT}/data/images/infoseek_wikipedia_images_full"
link_asset "${REFLECTIVA_ROOT}/evqa_inference_images" "${ROOT}/data/images/reflectiva_evqa_inference_images"
link_asset "${REFLECTIVA_ROOT}/infoseek_val_image" "${ROOT}/data/images/reflectiva_infoseek_val_image"

link_asset "${ECHOSIGHT_ROOT}/KB_EVQA" "${ROOT}/data/kb/KB_EVQA"
link_asset "${ECHOSIGHT_ROOT}/KB_infoseek" "${ROOT}/data/kb/KB_infoseek"
link_asset "${ECHOSIGHT_ROOT}/KB_EVQA/encyclopedic_kb_wiki.json" "${ROOT}/data/kb/evqa_encyclopedic_kb_wiki.json"
link_asset "${ECHOSIGHT_ROOT}/KB_infoseek/wiki_100_dict_v4.json" "${ROOT}/data/kb/infoseek_wiki_100_dict_v4.json"
link_asset "${REFLECTIVA_ROOT}/evqa_EVA_image" "${ROOT}/data/kb/reflectiva_evqa_EVA_image"

link_asset "${WIKIPRF_ROOT}/test/EVQA_with_initial_retrieval.jsonl" "${ROOT}/data/wikiprf/evqa_fixed_with_initial_retrieval.jsonl"
link_asset "${WIKIPRF_ROOT}/test/EVQA_unfixed_with_initial_retrieval.jsonl" "${ROOT}/data/wikiprf/evqa_unfixed_with_initial_retrieval.jsonl"
link_asset "${WIKIPRF_ROOT}/test/infoseek_with_initial_retrieval_recheck_Feb7.jsonl" "${ROOT}/data/wikiprf/infoseek_fixed_with_initial_retrieval.jsonl"
link_asset "${WIKIPRF_ROOT}/test/InfoSeek_unfixed_with_initial_retrieval.jsonl" "${ROOT}/data/wikiprf/infoseek_unfixed_with_initial_retrieval.jsonl"

link_asset "${ECHOSIGHT_ROOT}/ECCV_results/echo_reranker_evqa_k20_20260213_100145.jsonl" "${ROOT}/data/retrieval/echosight_reranker_evqa_k20.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_reranker_full_InfoSeek_k20_20260213_095859.jsonl" "${ROOT}/data/retrieval/echosight_reranker_infoseek_k20.jsonl"
link_asset "${ECHOSIGHT_ROOT}/InfoSeek/infoseek_val.jsonl" "${ROOT}/data/retrieval/infoseek_val.jsonl"
link_asset "${ECHOSIGHT_ROOT}/InfoSeek/infoseek_val_withkb.jsonl" "${ROOT}/data/retrieval/infoseek_val_withkb.jsonl"
link_asset "${ECHOSIGHT_ROOT}/infoseek_val_qtype.jsonl" "${ROOT}/data/retrieval/infoseek_val_qtype.jsonl"
link_asset "${REFLECTIVA_ROOT}/data_evqa/test_one_hop_Feb14.json" "${ROOT}/data/retrieval/reflectiva_evqa_test_one_hop_Feb14.json"

link_asset "${COMEM_ROOT}/CoMEM-inference/EVQA/Custom_test_full" "${ROOT}/methods/code/CoMEM/CoMEM-inference/EVQA/Custom_test_full"
link_asset "${COMEM_ROOT}/CoMEM-inference/EVQA/Custom_test_full_unfixed" "${ROOT}/methods/code/CoMEM/CoMEM-inference/EVQA/Custom_test_full_unfixed"
link_asset "${COMEM_ROOT}/CoMEM-inference/infoseek/Custom_test_full" "${ROOT}/methods/code/CoMEM/CoMEM-inference/infoseek/Custom_test_full"
link_asset "${COMEM_ROOT}/CoMEM-inference/infoseek/Custom_test_full_unfixed" "${ROOT}/methods/code/CoMEM/CoMEM-inference/infoseek/Custom_test_full_unfixed"

link_asset "${COMEM_ROOT}/CoMEM-inference/EVQA/fixed_output/qwen2.5_CoMEM_custom_10.jsonl" "${ROOT}/outputs/raw_methods/evqa/fixed/CoMEM.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_evqa_to_eval_k20.jsonl" "${ROOT}/outputs/raw_methods/evqa/fixed/EchoSight.jsonl"
link_asset "${IBA_EVQA_FIXED_OUTPUT}" "${ROOT}/outputs/raw_methods/evqa/fixed/IBA.jsonl"
link_asset "${REFLECTIVA_ROOT}/output/Reflectiva_evqa_echo_kb_eva_index_I2I/split_0_test_one_hop_Feb14_k5.json" "${ROOT}/outputs/raw_methods/evqa/fixed/ReflectiVA.json"
link_asset "${WIKIPRF_ROOT}/results_EVQA_fixed_fixed_step600_Feb23_topk3.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/evqa/fixed/Wiki_PRF.jsonl"

link_asset "${COMEM_ROOT}/CoMEM-inference/EVQA/unfixed_output/qwen2.5_CoMEM_custom_10.jsonl" "${ROOT}/outputs/raw_methods/evqa/unfixed/CoMEM.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_evqa_to_eval_k20.jsonl" "${ROOT}/outputs/raw_methods/evqa/unfixed/EchoSight.jsonl"
link_asset "${IBA_EVQA_UNFIXED_OUTPUT}" "${ROOT}/outputs/raw_methods/evqa/unfixed/IBA.jsonl"
link_asset "${REFLECTIVA_ROOT}/output/Reflectiva_evqa_echo_kb_eva_index_I2I/split_0_test_one_hop_k5_Nov11_unfixed.json" "${ROOT}/outputs/raw_methods/evqa/unfixed/ReflectiVA.json"
link_asset "${WIKIPRF_ROOT}/results_EVQA_unfixed_step600_Feb21_unfixed.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/evqa/unfixed/Wiki_PRF.jsonl"

link_asset "${COMEM_ROOT}/CoMEM-inference/infoseek/fixed_output/qwen2.5_CoMEM_custom_10.jsonl" "${ROOT}/outputs/raw_methods/infoseek/fixed/CoMEM.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_infoseek_to_eval_k20.jsonl" "${ROOT}/outputs/raw_methods/infoseek/fixed/EchoSight.jsonl"
link_asset "${IBA_INFOSEEK_FIXED_OUTPUT}" "${ROOT}/outputs/raw_methods/infoseek/fixed/IBA.jsonl"
link_asset "${REFLECTIVA_ROOT}/output/Feb14_Reflectiva_infoseek_fixed_echo_kb_unzipped_eva_index_Image2Image" "${ROOT}/outputs/raw_methods/infoseek/fixed/ReflectiVA"
link_asset "${WIKIPRF_ROOT}/results_infoseek_test_step600_Feb13_test.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/infoseek/fixed/Wiki_PRF.jsonl"

link_asset "${COMEM_ROOT}/CoMEM-inference/infoseek/unfixed_output/qwen2.5_CoMEM_custom_10.jsonl" "${ROOT}/outputs/raw_methods/infoseek/unfixed/CoMEM.jsonl"
link_asset "${ECHOSIGHT_ROOT}/infoseek_to_eval_k20_full_Oct15_llama3.jsonl" "${ROOT}/outputs/raw_methods/infoseek/unfixed/EchoSight.jsonl"
link_asset "${IBA_INFOSEEK_UNFIXED_OUTPUT}" "${ROOT}/outputs/raw_methods/infoseek/unfixed/IBA.jsonl"
link_asset "${REFLECTIVA_ROOT}/Delata_output/output/Reflectiva_infoseek_echo_kb_unzipped_eva_index_Image2Image" "${ROOT}/outputs/raw_methods/infoseek/unfixed/ReflectiVA"
link_asset "${WIKIPRF_INFOSEEK_UNFIXED_OUTPUT}" "${ROOT}/outputs/raw_methods/infoseek/unfixed/Wiki_PRF.jsonl"

link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_outputs/evqa_augmented_anchor_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/EchoSight_anchor.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_outputs/evqa_augmented_method1_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/EchoSight_augmented_method1.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_outputs/evqa_augmented_method2_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/EchoSight_augmented_method2.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/outputs/EVQA_augmented_Llama31_vllm_noimg/evqa_anchor_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/IBA_anchor.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/outputs/EVQA_augmented_Llama31_vllm_noimg/evqa_augmented1_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/IBA_augmented_method1.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/outputs/EVQA_augmented_Llama31_vllm_noimg/evqa_augmented2_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/IBA_augmented_method2.jsonl"
link_asset "${WIKIPRF_ROOT}/test/augmented_EVQA/results_anchor_EVQA_anchor_with_initial_retrieval_step600_Feb23_test_topk3.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/Wiki_PRF_anchor.jsonl"
link_asset "${WIKIPRF_ROOT}/test/augmented_EVQA/results_augmented1_EVQA_augmented1_with_initial_retrieval_step600_Feb23_test_topk3.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/Wiki_PRF_augmented_method1.jsonl"
link_asset "${WIKIPRF_ROOT}/test/augmented_EVQA/results_augmented2_EVQA_augmented2_with_initial_retrieval_step600_Feb23_test_topk3.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/evqa/augmented/Wiki_PRF_augmented_method2.jsonl"

link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_outputs/infoseek_augmented_anchor_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/EchoSight_anchor.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_outputs/infoseek_augmented_method1_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/EchoSight_augmented_method1.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/EchoSight_outputs/infoseek_augmented_method2_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/EchoSight_augmented_method2.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/outputs/InfoSeek_augmented_Llama31_vllm_noimg/infoseek_anchor_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/IBA_anchor.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/outputs/InfoSeek_augmented_Llama31_vllm_noimg/infoseek_augmented1_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/IBA_augmented_method1.jsonl"
link_asset "${ECHOSIGHT_ROOT}/ECCV_results/outputs/InfoSeek_augmented_Llama31_vllm_noimg/infoseek_augmented2_llama31_answers.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/IBA_augmented_method2.jsonl"
link_asset "${WIKIPRF_ROOT}/test/augmented_InfoSeek/results_infoseek_anchor_with_initial_retrieval_step600_Feb23_test_topk3.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/Wiki_PRF_anchor.jsonl"
link_asset "${WIKIPRF_ROOT}/test/augmented_InfoSeek/results_infoseek_aug_method1_withpos_with_initial_retrieval_step600_Feb23_test_topk3.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/Wiki_PRF_augmented_method1.jsonl"
link_asset "${WIKIPRF_ROOT}/test/augmented_InfoSeek/results_augmented2_infoseek_aug_method2_withoutpos_with_initial_retrieval_step600_Feb23_test_topk3.json_0_generation_details.jsonl" "${ROOT}/outputs/raw_methods/infoseek/augmented/Wiki_PRF_augmented_method2.jsonl"

if [[ -n "${BEM_MODEL_DIR}" && "${BEM_MODEL_DIR}" != /path/to/* ]]; then
  link_asset "${BEM_MODEL_DIR}" "${ROOT}/rag_evaluation/evqa_eval/bem_model"
fi
