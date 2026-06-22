# Evaluation Scripts

One-command wrappers for the paper scoring scripts.

- `run_smoke.sh`: dependency-light InfoSeek scoring smoke test using committed
  sample outputs under `outputs/generated_methods/`.
- `run_all_evaluations.sh`: fixed, unfixed, and augmented scoring for E-VQA and InfoSeek.
- `run_evqa_*.sh`: E-VQA split-specific scoring.
- `run_infoseek_*.sh`: InfoSeek split-specific scoring with answer-reward style
  matching.

Run from the repository root, for example:

```bash
bash scripts/evaluation/run_smoke.sh
bash scripts/evaluation/run_infoseek_fixed.sh --max-samples 10
```

The split-specific and all-evaluation wrappers read full preserved outputs from
`outputs/raw_methods/`. For a fresh GitHub clone, configure
`configs/paths.env` and run `scripts/setup_local_assets.sh` before using those
full-output wrappers.
