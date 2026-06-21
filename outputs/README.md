# Raw Method Outputs

`outputs/raw_methods/` stores or links the preserved model predictions used by
the scoring scripts. They are organized as:

- `evqa/{fixed,unfixed,augmented}/`
- `infoseek/{fixed,unfixed,augmented}/`

Fixed/unfixed splits include all preserved baselines. Augmented splits include
IBA, EchoSight, and Wiki-PRF predictions for `anchor`,
`augmented_method1`, and `augmented_method2`.

The EVQA-unfixed scorer also knows about legacy CC-VQA and ReAG-7B outputs.
CC-VQA is linked here. ReAG-7B points to an older `/data3` location that is not
mounted in this workspace; the scorer treats it as optional and records a skip
in `results/evaluation/evqa/unfixed/summary.json`.

Scored summaries are written to `../results/evaluation/`.

`scripts/run_comem_infoseek_sample.sh` is an end-to-end sanity test that writes
one generated CoMEM prediction here:
`generated_methods/CoMEM/infoseek/fixed/qwen2.5_CoMEM_custom_1.jsonl`, then
scores it into `../results/evaluation/infoseek/fixed_from_generated_comem_sample/`.
