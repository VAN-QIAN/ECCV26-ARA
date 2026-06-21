# Data Layout

Small CSV files are stored in this repository. Images, KB files, checkpoints,
and large generated outputs are kept outside the repo and referenced by path.

## Ground Truth

`data/ground_truth/` contains the aligned original and repaired evaluation CSVs:

| File | Rows | Source |
| --- | ---: | --- |
| `evqa_unfixed_test_with_id.csv` | 4,750 | `/data/qianMa/EchoSight/test_evqa_with_id.csv` |
| `evqa_fixed_final_check_Feb12.csv` | 4,750 | `/data2/QianMa/FixKBVQA/EVQA_results_final_check/evqa_final_check_Feb12.csv` |
| `infoseek_unfixed_subset.csv` | 1,924 | `/data2/QianMa/ECCV/infoseek_unfixed_subset.csv` |
| `infoseek_fixed_final_recheck_Feb7.csv` | 1,924 | `/data2/QianMa/FixKBVQA/InfoSeek-Fix/results_final_check/infoseek_final_recheck_Feb7.csv` |

The fixed and unfixed files are aligned by `data_id`, so score changes reflect
question/answer/evidence repair instead of sample-set changes.

## Augmented Data

`data/augmented/` contains the multi-entity augmentation CSVs from the paper:

| File | Rows | Notes |
| --- | ---: | --- |
| `evqa/evqa_challenging_queries_full_seed3185.csv` | 3,871 | Metadata-only augmented E-VQA queries |
| `evqa/evqa_challenging_queries_full_seed3185_with_images.csv` | 3,871 | Same rows plus image/composite-image paths |
| `infoseek/infoseek_challenging_queries_full_seed3185.csv` | 1,604 | Metadata-only augmented InfoSeek queries |
| `infoseek/infoseek_challenging_queries_full_seed3185_with_images.csv` | 1,604 | Same rows plus image/composite-image paths |

Sources:

- `/data2/QianMa/FixKBVQA/EVQA_results_final_check/`
- `/data2/QianMa/FixKBVQA/InfoSeek-Fix/results_final_check/`

## Images

`data/images/` contains symlinks to the original image roots used by the method
code. Augmented composite images are intentionally not copied into the repo.
The placeholder directories are:

- `data/images/augmented/evqa/composite_images_full_seed3185/method1/`
- `data/images/augmented/evqa/composite_images_full_seed3185/method2/`
- `data/images/augmented/infoseek/composite_images_full_seed3185/method1/`
- `data/images/augmented/infoseek/composite_images_full_seed3185/method2/`

To materialize augmented composite images locally, copy or symlink files from:

- `/data2/QianMa/FixKBVQA/EVQA_results_final_check/composite_images_full_seed3185/`
- `/data2/QianMa/FixKBVQA/InfoSeek-Fix/results_final_check/composite_images_full_seed3185/`

The `_with_images.csv` files keep the original relative composite-image paths
from FixKBVQA for traceability.

Augmented method prediction outputs are not data-source files. They are linked
under `../outputs/raw_methods/evqa/augmented/` and
`../outputs/raw_methods/infoseek/augmented/` for scoring.
