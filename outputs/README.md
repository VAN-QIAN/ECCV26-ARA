# Raw Method Outputs

`outputs/raw_methods/` is the local mount point for preserved model predictions
used by the scoring scripts. Git tracks only the directory skeleton; run
`../scripts/setup_local_assets.sh` after configuring `../configs/paths.env` to
create local symlinks. The files are organized as:

- `evqa/{fixed,unfixed,augmented}/`
- `infoseek/{fixed,unfixed,augmented}/`

Fixed/unfixed splits include the five paper methods: EchoSight, IBA, CoMEM,
ReflectiVA, and Wiki-PRF. Augmented splits include IBA, EchoSight, and Wiki-PRF
predictions for `anchor`, `augmented_method1`, and `augmented_method2`.

Scored summaries are written to `../results/evaluation/`.

`scripts/run_comem_infoseek_sample.sh` is an end-to-end sanity test that writes
one generated CoMEM prediction here:
`generated_methods/CoMEM/infoseek/fixed/qwen2.5_CoMEM_custom_1.jsonl`, then
scores it into `../results/evaluation/infoseek/fixed_from_generated_comem_sample/`.
