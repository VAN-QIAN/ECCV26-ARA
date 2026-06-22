# Wiki-PRF Camera-Ready Evaluation

Root-level wrapper:

```bash
cd ECCV26_CameraReady
scripts/methods/run_wikiprf.sh --dataset evqa --split fixed --dry-run
scripts/methods/run_wikiprf.sh --dataset evqa --split fixed
```

This folder keeps the minimal runnable Wiki-PRF inference code used for the
camera-ready KB-VQA evaluation.

## Kept Files

- `run_wikiprf.py`: unified EVQA/InfoSeek fixed/unfixed inference entry.
- `retriever.py`: EVA-CLIP + FAISS KB retriever.
- `answer_generator.py`: minimal wiki article reconstruction helpers.
- `build_initial_retrieval_evaclip.py`: optional utility to rebuild initial retrieval JSONL.
- `configs/*.yaml`: four data configs pointing to camera-ready soft links.
- `scripts/*.sh`: direct runnable wrappers for the four evaluation splits.

Large JSONL files and checkpoints are referenced by soft links:

- `data/wikiprf/*.jsonl` -> original Wiki-PRF initial retrieval JSONL files.
- `data/checkpoints/Wiki-PRF` -> merged Wiki-PRF checkpoint.
- `data/kb/KB_infoseek` and `data/kb/infoseek_wiki_100_dict_v4.json` -> EchoSight-hosted KB/FAISS.

## Environment

```bash
conda activate echosight
cd ECCV26_CameraReady
```

Install dependencies only if the environment is missing packages:

```bash
pip install -r methods/code/Wiki-PRF/requirements.txt
```

## Validate Paths

Dry-run reads the selected YAML and first sample image, but does not load model
weights:

```bash
methods/code/Wiki-PRF/scripts/run_infoseek_fixed.sh --dry-run
methods/code/Wiki-PRF/scripts/run_infoseek_unfixed.sh --dry-run
methods/code/Wiki-PRF/scripts/run_evqa_fixed.sh --dry-run
methods/code/Wiki-PRF/scripts/run_evqa_unfixed.sh --dry-run
```

## Run Inference

Default scripts use `CUDA_VISIBLE_DEVICES=0` and map planner/filter/retriever to
GPU id `0` inside that visible set.

```bash
methods/code/Wiki-PRF/scripts/run_infoseek_fixed.sh
methods/code/Wiki-PRF/scripts/run_infoseek_unfixed.sh
methods/code/Wiki-PRF/scripts/run_evqa_fixed.sh
methods/code/Wiki-PRF/scripts/run_evqa_unfixed.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
MODEL_GPU_ID=0 FILTER_GPU_ID=1 RETRIEVER_GPU_ID=2 \
methods/code/Wiki-PRF/scripts/run_infoseek_fixed.sh
```

Outputs are written under:

```text
outputs/generated_methods/Wiki-PRF/<dataset>/
```

The dynamic tool retrieval defaults to the EchoSight InfoSeek KB/FAISS, matching
the original Wiki-PRF evaluation scripts. Override `KNOWLEDGE_BASE` and
`FAISS_ROOT` if a different KB index is needed.

## Direct Python Entry

```bash
python methods/code/Wiki-PRF/run_wikiprf.py \
  --dataset infoseek_fixed \
  --dry-run
```

Supported `--dataset` values:

- `infoseek_fixed`
- `infoseek_unfixed`
- `evqa_fixed`
- `evqa_unfixed`

## Rebuild Initial Retrieval

The checked-in configs use existing initial retrieval JSONL soft links. To
regenerate them:

```bash
python methods/code/Wiki-PRF/build_initial_retrieval_evaclip.py \
  --input_jsonl <raw_input.jsonl> \
  --image_root <image_root> \
  --knowledge_base data/kb/infoseek_wiki_100_dict_v4.json \
  --faiss_root data/kb/KB_infoseek \
  --output_jsonl outputs/generated_methods/Wiki-PRF/initial_retrieval/custom.jsonl
```
