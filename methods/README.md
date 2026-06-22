# Method Code Snapshots

This directory contains the method code snapshots used by the camera-ready
package. Large data, outputs, checkpoints, logs, caches, and `.git` metadata are
kept out of the code snapshots and referenced through `../data` and
`../outputs/raw_methods`.

## Sources

- `code/EchoSight`: EchoSight retrieval/reranker/answer code snapshot.
- `code/IBA`: Qwen/TopK/BGE IBA code split from EchoSight.
- `code/ReflectiVA`: ReflectiVA inference subset.
- `code/CoMEM`: CoMEM inference subset.
- `code/Wiki-PRF`: camera-ready Wiki-PRF runner.

## One-Command Wrappers

Run these from the repository root. Add `--dry-run` first to inspect paths.

| Method | Env | Command |
| --- | --- | --- |
| EchoSight | `echosight` | `scripts/methods/run_echosight.sh --dataset evqa --split fixed --stage all` |
| IBA | `echosight` | `scripts/methods/run_iba.sh --dataset evqa --split fixed --stage all` |
| ReflectiVA | `reflectiva` | `scripts/methods/run_reflectiva.sh --dataset evqa --split fixed` |
| CoMEM | `CoMEM` | `scripts/methods/run_comem.sh --dataset evqa --split fixed` |
| Wiki-PRF | `echosight` | `scripts/methods/run_wikiprf.sh --dataset evqa --split fixed` |

All wrappers support `--dataset evqa|infoseek` and `--split fixed|unfixed`
where the underlying method supports both. Set `SKIP_CONDA_ACTIVATE=1` if the
correct environment is already active.

The evaluated raw prediction files are materialized under `../outputs/raw_methods`.
Fixed/unfixed outputs cover the five paper methods: EchoSight, IBA, CoMEM,
ReflectiVA, and Wiki-PRF. Augmented outputs include the paper's IBA, EchoSight,
and Wiki-PRF anchor/method1/method2 runs. The top-level scoring scripts
evaluate those raw files directly.
