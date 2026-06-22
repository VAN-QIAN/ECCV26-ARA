# Fixing KB-VQA Camera-Ready Repo

This repository packages the camera-ready artifacts for:

- fixed and unfixed E-VQA / InfoSeek evaluation CSVs;
- augmented multi-entity query CSVs, with image folders kept as placeholders;
- reusable audit, repair, and augmentation scripts with prompts;
- one-command scoring scripts for all reported raw method outputs;
- runnable method wrappers for the method code snapshots kept in this repo.

Large images, KBs, checkpoints, and full raw generated outputs are not
duplicated here. The repo keeps placeholders plus a local setup script that can
materialize machine-local symlinks after cloning.

## Quick Start

```bash
cd ECCV26_CameraReady
conda activate KBVQA_eval
bash scripts/check_camera_ready.sh
```

To enable full method runs or scoring from preserved raw outputs on a local
machine, configure local assets once:

```bash
cp configs/paths.example.env configs/paths.env
$EDITOR configs/paths.env
bash scripts/setup_local_assets.sh
```

Run a lightweight scoring smoke test:

```bash
bash scripts/evaluation/run_smoke.sh
```

Run one real generated-output sample and immediately score it:

```bash
bash scripts/run_comem_infoseek_sample.sh
```

This writes a one-example CoMEM prediction to
`outputs/generated_methods/CoMEM/infoseek/fixed/` and scores it through
`rag_evaluation/infoseek/score_fixed_infoseek_methods.py`.

Run one real EchoSight retrieval/reranking sample:

```bash
bash scripts/run_echosight_evqa_reranker_sample.sh
```

This writes `outputs/generated_methods/EchoSight/evqa/fixed/sample_reranker_k3.json`.
Set `FORCE=1` to rerun after the sample output already exists.

Run one real IBA sample against the local Qwen vLLM server on port 8001, then
score the generated answer:

```bash
bash scripts/run_iba_infoseek_sample.sh
```

This writes sample metadata and answers under
`outputs/generated_methods/IBA/infoseek/fixed/`, then scores
`sample_answers.jsonl` directly.

Run one real Wiki-PRF sample, then score the generated details file:

```bash
bash scripts/run_wikiprf_infoseek_sample.sh
```

This writes Wiki-PRF outputs under
`outputs/generated_methods/Wiki-PRF/infoseek_fixed/`, then scores
`sample_results_infoseek_fixed_step600_topk3.json_0_generation_details.jsonl`
directly. The default uses `CUDA_VISIBLE_DEVICES=0,1`; override
`MODEL_GPU_ID`, `FILTER_GPU_ID`, and `RETRIEVER_GPU_ID` if needed.

After materializing `outputs/raw_methods/`, run all fixed/unfixed/augmented
scoring:

```bash
bash scripts/evaluation/run_all_evaluations.sh
```

Outputs are written under `results/evaluation/`.

## Data

See `data/README.md` for the full layout.

Main files:

- `data/ground_truth/evqa_unfixed_test_with_id.csv`
- `data/ground_truth/evqa_fixed.csv`
- `data/ground_truth/infoseek_unfixed_subset.csv`
- `data/ground_truth/infoseek_fixed.csv`
- `data/augmented/evqa/evqa_challenging_queries_full_seed3185_with_images.csv`
- `data/augmented/infoseek/infoseek_challenging_queries_full_seed3185_with_images.csv`

Augmented composite images are intentionally not stored. Placeholder folders are
under `data/images/augmented/`. The image paths in the augmented CSVs are
relative paths intended for a HuggingFace dataset repo layout.

## Audit, Repair, Augment

The reusable tools are under `fixing/`:

- `fixing/evqa/`: E-VQA audit, question repair, evidence checking, final check,
  challenging-query generation, and composite-image construction.
- `fixing/infoseek/`: InfoSeek KB extraction, question/evidence/QA repair,
  final recheck, challenging-query generation, and composite-image construction.

Entry summaries:

```bash
sed -n '1,220p' fixing/README.md
sed -n '1,220p' fixing/evqa/README.md
sed -n '1,220p' fixing/infoseek/InfoSeek_Check_Plan.md
```

Most audit and repair scripts use standard CLI arguments and support small
`--limit` runs. LLM-based scripts need an OpenAI-compatible API key/base URL.

## Method Inference Wrappers

Use these wrappers to run the method code snapshots. Each supports `--dry-run`
to print the exact command before loading models.

| Method | Environment | Wrapper |
| --- | --- | --- |
| EchoSight | `echosight` | `scripts/methods/run_echosight.sh --dataset evqa --split fixed --stage all` |
| IBA | `echosight` | `scripts/methods/run_iba.sh --dataset evqa --split fixed --stage all` |
| ReflectiVA | `reflectiva` | `scripts/methods/run_reflectiva.sh --dataset evqa --split fixed` |
| CoMEM | `CoMEM` | `scripts/methods/run_comem.sh --dataset evqa --split fixed` |
| Wiki-PRF | `echosight` | `scripts/methods/run_wikiprf.sh --dataset evqa --split fixed` |

Set `SKIP_CONDA_ACTIVATE=1` if the target environment is already active. Common
overrides such as `CUDA_VISIBLE_DEVICES`, `OUTPUT_DIR`, `CHECKPOINT_PATH`, and
model server URLs are supported by the underlying method scripts.

For IBA, the default OpenAI-compatible Qwen endpoint is
`http://127.0.0.1:8001/v1`, matching the local Qwen2.5-VL-7B-Instruct vLLM
server used for the runtime sample.

The paper evaluation covers five methods: EchoSight, IBA, CoMEM, ReflectiVA,
and Wiki-PRF.

## Scoring Scripts

Individual scoring commands:

```bash
bash scripts/evaluation/run_evqa_fixed.sh
bash scripts/evaluation/run_evqa_unfixed.sh
bash scripts/evaluation/run_evqa_augmented.sh
bash scripts/evaluation/run_infoseek_fixed.sh
bash scripts/evaluation/run_infoseek_unfixed.sh
bash scripts/evaluation/run_infoseek_augmented.sh
```

Each scorer accepts `--max-samples N` and path overrides for ground truth or
method outputs. InfoSeek scorers use the answer-reward style rules in
`rag_evaluation/infoseek/answer_reward_utils.py`. EVQA scorers also accept
`--allow-exact-match-fallback` for dependency-light validation.

Augmented scoring evaluates the anchor, intra-category/method1, and
inter-category/method2 variants for IBA, EchoSight, and Wiki-PRF. Their raw
prediction files are expected under `outputs/raw_methods/*/augmented/`; use
`scripts/setup_local_assets.sh` to create local symlinks to the preserved full
outputs.

## Paths

`configs/paths.example.env` is the publishable template. Copy it to
`configs/paths.env`, replace placeholders with machine-local roots, and run
`scripts/setup_local_assets.sh`. The local `configs/paths.env` file is ignored
by Git.

CoMEM and Wiki-PRF use their own local roots (`COMEM_ROOT`, `WIKIPRF_ROOT`) and
checkpoint directories. IBA preserved raw outputs are configured as explicit
files through `IBA_EVQA_FIXED_OUTPUT`, `IBA_EVQA_UNFIXED_OUTPUT`,
`IBA_INFOSEEK_FIXED_OUTPUT`, and `IBA_INFOSEEK_UNFIXED_OUTPUT`.

The paper PDF is not required at runtime and is not treated as a required
GitHub artifact by `scripts/check_camera_ready.sh`.
