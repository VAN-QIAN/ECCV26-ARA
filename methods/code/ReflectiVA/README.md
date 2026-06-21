# ReflectiVA Camera-Ready Wrapper

Root-level wrapper:

```bash
cd /data2/QianMa/ECCV26_CameraReady
scripts/methods/run_reflectiva.sh --dataset evqa --split fixed --dry-run
scripts/methods/run_reflectiva.sh --dataset evqa --split fixed
```

This folder keeps the minimal ReflectiVA inference code needed for the camera-ready KB-VQA evaluation.

## Scope

- EVQA entry: `rag_evaluation/encyclopedic/release_retrieval_echo_kb.py`
- InfoSeek entry: `rag_evaluation/infoseek/release_retrieval_echo_kb.py`
- LLaVA/ReflectiVA model code required by inference only
- EVQA converted input: `data_evqa/test_one_hop_Feb14.json`

Training, serving, generic LLaVA evaluation scripts, old outputs, and copied datasets are intentionally removed.

## Data Layout

Large data stays outside this folder and is accessed through the camera-ready workspace soft-links:

- EVQA images: `../../../data/images/reflectiva_evqa_inference_images`
- InfoSeek images: `../../../data/images/reflectiva_infoseek_val_image`
- EVQA KB json: `../../../data/kb/evqa_encyclopedic_kb_wiki.json`
- EVQA FAISS index: `../../../data/kb/reflectiva_evqa_EVA_image`
- InfoSeek KB json: `../../../data/kb/infoseek_wiki_100_dict_v4.json`
- InfoSeek FAISS index: `../../../data/kb/KB_infoseek`

The EVQA FAISS index soft-link points to the original ReflectiVA `knn.index/knn.json` files because the existing index files are large and are not duplicated here.

## Environment

Use the existing environment:

```bash
conda activate reflectiva
```

`requirements.txt` documents the minimal Python packages used by these wrappers. The cluster environment already provides most of them.

## Run EVQA

```bash
cd /data2/QianMa/ECCV26_CameraReady/methods/code/ReflectiVA
bash scripts/run_reflectiva_evqa.sh --dry-run
bash scripts/run_reflectiva_evqa.sh
```

Useful overrides:

```bash
PART=0 TOTAL_PART=0 CUDA_VISIBLE_DEVICES=0 ENTITY_K=5 \
bash scripts/run_reflectiva_evqa.sh
```

For SLURM arrays, keep the original convention: `PART` is the current split id and `TOTAL_PART` is the max split id, so `TOTAL_PART=99` means 100 splits.

## Run InfoSeek

```bash
cd /data2/QianMa/ECCV26_CameraReady/methods/code/ReflectiVA
bash scripts/run_reflectiva_infoseek.sh --dry-run
bash scripts/run_reflectiva_infoseek.sh
```

The InfoSeek wrapper uses the CSV/image-folder pipeline from `methods/Delata_ReflectiVA`, adapted to the EchoSight KB files.

## Rebuild EVQA Input

Only needed if the EVQA ground-truth CSV changes:

```bash
python data_evqa/adapt_data.py
```

The script writes `data_evqa/test_one_hop_Feb14.json` by default.
