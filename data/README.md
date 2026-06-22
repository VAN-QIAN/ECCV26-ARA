# Data Layout

Small CSV files are stored in this repository. Images, KB files, checkpoints,
retrieval dumps, and large generated outputs are kept outside the repo. Use
`../configs/paths.env` plus `../scripts/setup_local_assets.sh` to materialize
local symlinks when needed.

## Ground Truth

`data/ground_truth/` contains the aligned original and repaired evaluation CSVs:

| File | Rows | Source |
| --- | ---: | --- |
| `evqa_unfixed_test_with_id.csv` | 4,750 | EchoSight E-VQA test set |
| `evqa_fixed.csv` | 4,750 | FixKBVQA E-VQA final check |
| `infoseek_unfixed_subset.csv` | 1,924 | ECCV InfoSeek unfixed subset |
| `infoseek_fixed.csv` | 1,924 | FixKBVQA InfoSeek final recheck |

The fixed and unfixed files are aligned by `data_id`, so score changes reflect
question/answer/evidence repair instead of sample-set changes.

## Augmented Data

`data/augmented/` contains the multi-entity augmentation CSVs from the paper:

| File | Rows | Notes |
| --- | ---: | --- |
| `evqa/evqa_challenging_queries_full_seed3185_with_images.csv` | 3,871 | Augmented E-VQA queries plus image/composite-image paths |
| `infoseek/infoseek_challenging_queries_full_seed3185_with_images.csv` | 1,604 | Augmented InfoSeek queries plus image/composite-image paths |

Original source roots are configured locally through `../configs/paths.env`.

## Images

`data/images/` contains placeholders for the original image roots used by the
method code. Augmented composite images are intentionally not copied into the
repo. The placeholder directories are:

- `data/images/augmented/evqa/composite_images_full_seed3185/method1/`
- `data/images/augmented/evqa/composite_images_full_seed3185/method2/`
- `data/images/augmented/infoseek/composite_images_full_seed3185/method1/`
- `data/images/augmented/infoseek/composite_images_full_seed3185/method2/`

The `_with_images.csv` files use repository-relative image paths such as
`images/augmented/evqa/...`. When hosting images on HuggingFace, upload the
image folders with the same relative layout.

Augmented method prediction outputs are not data-source files. They are
materialized under `../outputs/raw_methods/evqa/augmented/` and
`../outputs/raw_methods/infoseek/augmented/` for scoring.
