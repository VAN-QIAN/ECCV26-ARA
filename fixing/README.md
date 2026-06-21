# Fixing Tools

This directory contains reusable scripts and prompts for the paper's
audit-repair-augment protocol. Scripts are grouped by dataset but follow the
same stages:

1. audit whether the question is clear and the answer is evidence-supported;
2. repair missing constraints, evidence, and answer mismatches;
3. run final consistency checks;
4. generate and materialize augmented multi-entity queries.

Large KB and image files are not copied into this folder. Use the symlinks under
`../data/kb` and `../data/images`, or pass explicit paths with each script's CLI.

## EVQA

Main audit/repair chain:

```bash
python fixing/evqa/evqa_add_data_id.py --help
python fixing/evqa/evqa_question_fix_audit.py --help
python fixing/evqa/evqa_evidence_supporting_audit.py --help
python fixing/evqa/evqa_final_check.py --help
```

Optional diagnostics:

```bash
python fixing/evqa/evqa_qa_alignment_audit.py --help
python fixing/evqa/evqa_answer_leak_audit.py --help
python fixing/evqa/evqa_context_length_analysis.py --help
```

Augmentation:

```bash
python fixing/evqa/evqa_generate_challenging_queries.py --help
python fixing/evqa/evqa_build_composite_images.py --help
```

Detailed notes remain in `fixing/evqa/README.md` and
`fixing/evqa/EVQA_Check_Plan.md`.

## InfoSeek

Main audit/repair chain:

```bash
python fixing/infoseek/infoseek_kb_extract.py --help
python fixing/infoseek/infoseek_question_fix_audit.py --help
python fixing/infoseek/infoseek_evidence_supporting_audit.py --help
python fixing/infoseek/infoseek_qa_fix_numeric_temporal.py --help
python fixing/infoseek/infoseek_final_check.py --help
python fixing/infoseek/infoseek_final_recheck.py --help
```

Optional diagnostics and post-processing:

```bash
python fixing/infoseek/infoseek_answer_leak_audit.py --help
python fixing/infoseek/infoseek_needs_evidence_reaudit.py --help
python fixing/infoseek/infoseek_sync_answers.py --help
python fixing/infoseek/infoseek_update_qtype.py --help
```

Augmentation:

```bash
python fixing/infoseek/infoseek_generate_challenging_queries.py --help
python fixing/infoseek/infoseek_build_composite_images.py --help
```

Detailed notes remain in `fixing/infoseek/InfoSeek_Check_Plan.md`.

## Final Camera-Ready Files

Evaluation reads the final CSVs copied into `../data/ground_truth`. Augmented
CSV files copied from FixKBVQA live in `../data/augmented`.
