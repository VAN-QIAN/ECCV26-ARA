# EchoSight Baseline

Root-level wrapper:

```bash
cd ECCV26_CameraReady
scripts/methods/run_echosight.sh --dataset evqa --split fixed --stage all --dry-run
scripts/methods/run_echosight.sh --dataset evqa --split fixed --stage all
```

This directory is the camera-ready EchoSight-only code snapshot. The Qwen/TopK
IBA code has been moved to `../IBA`; this directory keeps the original
EchoSight retrieval, multimodal reranker, and answer-generation entrypoints
only.

## Contents

- `model/`, `dataset/`, `utils/`, `lavis/`: EchoSight retriever/reranker
  implementation and local LAVIS model definitions.
- `test/test_reranker_echo_score.py`: main retrieval + Q-Former reranking
  entrypoint used for EchoSight outputs.
- `test/test_answer_generator.py`: answer-generation entrypoint from saved
  reranker results.
- `scripts/run_echosight_reranker.sh`: runnable wrapper for EVQA/InfoSeek
  reranking.
- `scripts/run_echosight_answer.sh`: runnable wrapper for EVQA/InfoSeek answer
  generation from saved reranker results.

Large KB files, images, FAISS indices, and checkpoints are not copied here.
Defaults point to local symlinks created from `configs/paths.env` by
`scripts/setup_local_assets.sh`.

## Environment

```bash
conda activate echosight
cd ECCV26_CameraReady/methods/code/EchoSight
export PYTHONPATH=$PWD
```

`requirements.txt` lists the main package families. The fully pinned source
environment is intentionally not duplicated; use the prepared `echosight`
environment for reproduction.

## Run Reranking

Dry-run first to inspect the exact command:

```bash
bash scripts/run_echosight_reranker.sh --dataset evqa --dry-run
bash scripts/run_echosight_reranker.sh --dataset infoseek --dry-run
```

Run the actual reranker:

```bash
bash scripts/run_echosight_reranker.sh --dataset evqa
bash scripts/run_echosight_reranker.sh --dataset infoseek
```

Important overrides:

- `TEST_FILE`
- `KNOWLEDGE_BASE`
- `FAISS_INDEX`
- `QFORMER_CKPT_PATH`
- `RETRIEVAL_TOP_K`
- `SAVE_RESULT_PATH`
- `CUDA_VISIBLE_DEVICES`

## Run Answer Generation

By default this expects the reranker output written by
`run_echosight_reranker.sh`.

```bash
bash scripts/run_echosight_answer.sh --dataset evqa --dry-run
bash scripts/run_echosight_answer.sh --dataset evqa
```

Important overrides:

- `RETRIEVAL_RESULTS`
- `ANSWER_GENERATOR` (`llama3`, `mistral`, `gpt4`, `palm`)
- `LLM_CHECKPOINT`
- `OUTPUT_PATH`
