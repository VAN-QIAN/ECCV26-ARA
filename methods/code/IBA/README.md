# IBA Qwen/TopK Pipeline

Root-level wrapper:

```bash
cd ECCV26_CameraReady
scripts/methods/run_iba.sh --dataset evqa --split fixed --stage all --dry-run
scripts/methods/run_iba.sh --dataset evqa --split fixed --stage all
```

This directory is the camera-ready IBA code snapshot used for the paper. It
contains the Qwen2.5-VL entity identification,
TopK entity expansion, BGE section reranking, and final answer-generation
pipeline used for the IBA outputs. The EchoSight baseline code is in
`../EchoSight`.

## Contents

- `qwen_pipeline/`: Qwen identification, TopK metadata preparation, reranking,
  vLLM/OpenAI-compatible backends, and final answer generation.
- `aligned_answer_generator_w_qwen_metadata/`: adapters for LLaMA/Mistral style
  answer generation from Qwen metadata.
- `model/Qwen-vl.py`: local Qwen2.5-VL wrapper.
- `model/answer_generator.py`, `model/retriever.py`, `utils/`: shared KB and
  answer utilities required by the pipeline.
- `scripts/run_iba_prepare.sh`: prepares TopK metadata from EchoSight retrieval
  results.
- `scripts/run_iba_answer.sh`: aligns metadata with fixed CSV questions and
  generates final answers.

Large metadata, KB, images, retrieval results, and model weights are referenced
by path. They are not copied into this directory.

## Environment

```bash
conda activate echosight
cd ECCV26_CameraReady/methods/code/IBA
export PYTHONPATH=$PWD
```

The default prepare path uses a Qwen vLLM/OpenAI-compatible server at
`http://127.0.0.1:8000/v1`. The default answer path uses a LLaMA-3.1 vLLM
server at `http://127.0.0.1:8001/v1`. Override the URLs below if your hosts are
different.

## Prepare Metadata

```bash
bash scripts/run_iba_prepare.sh --dataset evqa --dry-run
bash scripts/run_iba_prepare.sh --dataset evqa
```

Important overrides:

- `TEST_FILE`
- `RETRIEVAL_RESULTS`
- `KNOWLEDGE_BASE`
- `METADATA_PATH`
- `QWEN_BACKEND` (`vllm_host`, `hf`, `openai_api`)
- `QWEN_VLLM_BASE_URL`
- `QWEN_VLLM_MODEL_NAME`
- `SECTION_RERANKER`
- `CUDA_VISIBLE_DEVICES`

## Generate Answers

If `METADATA_PATH` is not set, the script uses metadata produced by
`run_iba_prepare.sh` under `outputs/generated_methods/IBA/`.

```bash
bash scripts/run_iba_answer.sh --dataset evqa --dry-run
bash scripts/run_iba_answer.sh --dataset evqa
```

Important overrides:

- `CSV_PATH`
- `METADATA_PATH`
- `KNOWLEDGE_BASE`
- `OUTPUT_PATH`
- `ALIGNED_METADATA_PATH`
- `ANSWER_BACKEND` (`llama-3.1-8b` or `qwen`)
- `ANSWER_BACKEND_VLLM_BASE_URL`
- `ANSWER_BACKEND_VLLM_MODEL_NAME`
- `QWEN_VLLM_BASE_URL`
- `QWEN_VLLM_MODEL_NAME`
