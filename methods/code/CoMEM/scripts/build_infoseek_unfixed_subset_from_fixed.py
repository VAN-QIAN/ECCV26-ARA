#!/usr/bin/env python3
"""Build an Unfixed InfoSeek subset using data_ids from a Fixed InfoSeek CSV.

Default behavior:
- Select rows from the Unfixed CSV whose `data_id` appears in the Fixed CSV.
- Preserve the Unfixed CSV column order and row order.
- Write a new CSV subset file.

Optional strict mode:
- `--strict-question-match` additionally requires the question text to match
  (after whitespace normalization) for the same data_id.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_UNFIXED_CSV = str(REPO_ROOT / "data/ground_truth/infoseek_unfixed_subset.csv")
DEFAULT_FIXED_CSV = str(REPO_ROOT / "data/ground_truth/infoseek_fixed.csv")
DEFAULT_QTYPE_JSONL = str(REPO_ROOT / "data/retrieval/infoseek_val_qtype.jsonl")


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().split())


def infer_default_output(unfixed_csv: str, fixed_csv: str) -> str:
    unfixed_path = Path(unfixed_csv)
    fixed_stem = Path(fixed_csv).stem
    out_name = f"{unfixed_path.stem}_subset_by_{fixed_stem}_data_id.csv"
    return str(unfixed_path.with_name(out_name))


def load_fixed_index(
    fixed_csv: str,
    id_column: str,
    question_column: str,
) -> Tuple[Set[str], Dict[str, str], int]:
    """Return (fixed_ids, fixed_question_by_id, duplicate_id_count)."""
    fixed_ids: List[str] = []
    fixed_question_by_id: Dict[str, str] = {}

    with open(fixed_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {fixed_csv}")
        if id_column not in reader.fieldnames:
            raise KeyError(f"Column '{id_column}' not found in fixed CSV: {fixed_csv}")

        has_question_col = question_column in reader.fieldnames
        for row in reader:
            row_id = row.get(id_column, "")
            if not row_id:
                continue
            fixed_ids.append(row_id)
            if has_question_col and row_id not in fixed_question_by_id:
                fixed_question_by_id[row_id] = normalize_text(row.get(question_column))

    counter = Counter(fixed_ids)
    duplicate_id_count = sum(1 for v in counter.values() if v > 1)
    return set(counter.keys()), fixed_question_by_id, duplicate_id_count


def load_qtype_map(
    qtype_jsonl: str,
    id_column: str,
    qtype_column: str,
) -> Tuple[Dict[str, str], int]:
    """Return (qtype_by_id, duplicate_id_count) from jsonl."""
    qtype_by_id: Dict[str, str] = {}
    seen_ids: List[str] = []

    with open(qtype_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {qtype_jsonl}:{line_no}: {e}") from e

            if id_column not in obj:
                raise KeyError(
                    f"Column '{id_column}' not found in qtype jsonl object at line {line_no}"
                )
            if qtype_column not in obj:
                raise KeyError(
                    f"Column '{qtype_column}' not found in qtype jsonl object at line {line_no}"
                )

            row_id = str(obj.get(id_column, "")).strip()
            if not row_id:
                continue
            seen_ids.append(row_id)
            # Keep the last value if duplicated to reflect latest line.
            qtype_by_id[row_id] = str(obj.get(qtype_column, "")).strip()

    counter = Counter(seen_ids)
    duplicate_id_count = sum(1 for v in counter.values() if v > 1)
    return qtype_by_id, duplicate_id_count


def build_subset(
    unfixed_csv: str,
    fixed_ids: Set[str],
    fixed_question_by_id: Dict[str, str],
    qtype_by_id: Optional[Dict[str, str]],
    output_csv: str,
    id_column: str,
    question_column: str,
    qtype_column: str,
    strict_question_match: bool,
) -> Dict[str, int]:
    """Stream Unfixed CSV and write subset. Returns summary stats."""
    matched_ids: Set[str] = set()
    written_rows = 0
    question_same = 0
    question_diff = 0
    skipped_due_question = 0
    unfixed_duplicate_id_count = 0
    qtype_updated = 0
    qtype_missing = 0

    seen_unfixed_ids: Counter[str] = Counter()

    with open(unfixed_csv, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {unfixed_csv}")
        if id_column not in reader.fieldnames:
            raise KeyError(f"Column '{id_column}' not found in unfixed CSV: {unfixed_csv}")
        if qtype_by_id is not None and qtype_column not in reader.fieldnames:
            raise KeyError(f"Column '{qtype_column}' not found in unfixed CSV: {unfixed_csv}")

        has_unfixed_question_col = question_column in reader.fieldnames
        has_fixed_question_map = bool(fixed_question_by_id) and has_unfixed_question_col

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                row_id = row.get(id_column, "")
                if not row_id:
                    continue

                seen_unfixed_ids[row_id] += 1
                if row_id not in fixed_ids:
                    continue

                matched_ids.add(row_id)
                question_matches = True
                if has_fixed_question_map and row_id in fixed_question_by_id:
                    unfixed_q = normalize_text(row.get(question_column))
                    fixed_q = fixed_question_by_id[row_id]
                    question_matches = unfixed_q == fixed_q
                    if question_matches:
                        question_same += 1
                    else:
                        question_diff += 1

                if strict_question_match and not question_matches:
                    skipped_due_question += 1
                    continue

                if qtype_by_id is not None:
                    mapped_qtype = qtype_by_id.get(row_id)
                    if mapped_qtype:
                        row[qtype_column] = mapped_qtype
                        qtype_updated += 1
                    else:
                        qtype_missing += 1

                writer.writerow(row)
                written_rows += 1

    unfixed_duplicate_id_count = sum(1 for v in seen_unfixed_ids.values() if v > 1)
    return {
        "matched_unique_ids": len(matched_ids),
        "written_rows": written_rows,
        "question_same": question_same,
        "question_diff": question_diff,
        "skipped_due_question": skipped_due_question,
        "unfixed_duplicate_id_count": unfixed_duplicate_id_count,
        "qtype_updated": qtype_updated,
        "qtype_missing": qtype_missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "From an Unfixed InfoSeek CSV, select rows whose data_id appears in a "
            "Fixed InfoSeek CSV and export a new subset CSV."
        )
    )
    parser.add_argument("--unfixed-csv", default=DEFAULT_UNFIXED_CSV, help="Path to Unfixed InfoSeek CSV.")
    parser.add_argument("--fixed-csv", default=DEFAULT_FIXED_CSV, help="Path to Fixed InfoSeek CSV.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output subset CSV path. If omitted, auto-generated next to the Unfixed CSV.",
    )
    parser.add_argument("--id-column", default="data_id", help="ID column used for matching. Default: data_id")
    parser.add_argument(
        "--question-column",
        default="question",
        help="Question column used only for diagnostics / strict checking. Default: question",
    )
    parser.add_argument(
        "--strict-question-match",
        action="store_true",
        help=(
            "Only keep rows whose question text also matches the fixed CSV for the "
            "same data_id (after whitespace normalization)."
        ),
    )
    parser.add_argument(
        "--qtype-jsonl",
        default=None,
        help=(
            "Optional jsonl file for updating question_type in the output using data_id "
            f"(e.g. {DEFAULT_QTYPE_JSONL})."
        ),
    )
    parser.add_argument(
        "--qtype-column",
        default="question_type",
        help="Output CSV column to update from qtype jsonl. Default: question_type",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_csv = args.output_csv or infer_default_output(args.unfixed_csv, args.fixed_csv)
    qtype_by_id: Optional[Dict[str, str]] = None
    qtype_dup_count = 0

    try:
        fixed_ids, fixed_question_by_id, fixed_dup_count = load_fixed_index(
            fixed_csv=args.fixed_csv,
            id_column=args.id_column,
            question_column=args.question_column,
        )
        if args.qtype_jsonl:
            qtype_by_id, qtype_dup_count = load_qtype_map(
                qtype_jsonl=args.qtype_jsonl,
                id_column=args.id_column,
                qtype_column=args.qtype_column,
            )
        stats = build_subset(
            unfixed_csv=args.unfixed_csv,
            fixed_ids=fixed_ids,
            fixed_question_by_id=fixed_question_by_id,
            qtype_by_id=qtype_by_id,
            output_csv=output_csv,
            id_column=args.id_column,
            question_column=args.question_column,
            qtype_column=args.qtype_column,
            strict_question_match=args.strict_question_match,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    missing_ids = len(fixed_ids) - stats["matched_unique_ids"]

    print("Build completed.")
    print(f"  Unfixed CSV: {args.unfixed_csv}")
    print(f"  Fixed CSV:   {args.fixed_csv}")
    print(f"  Output CSV:  {output_csv}")
    if args.qtype_jsonl:
        print(f"  QType JSONL: {args.qtype_jsonl}")
    print(f"  Fixed unique {args.id_column}: {len(fixed_ids)}")
    print(f"  Matched unique {args.id_column} in Unfixed: {stats['matched_unique_ids']}")
    print(f"  Missing {args.id_column} from Unfixed: {missing_ids}")
    print(f"  Written rows: {stats['written_rows']}")

    if fixed_dup_count > 0:
        print(f"  Warning: fixed CSV has duplicate {args.id_column} values: {fixed_dup_count}")
    if stats["unfixed_duplicate_id_count"] > 0:
        print(
            f"  Warning: unfixed CSV has duplicate {args.id_column} values: "
            f"{stats['unfixed_duplicate_id_count']}"
        )
    if qtype_dup_count > 0:
        print(f"  Warning: qtype JSONL has duplicate {args.id_column} values: {qtype_dup_count}")

    if fixed_question_by_id:
        print(f"  Question text same (for matched IDs): {stats['question_same']}")
        print(f"  Question text different (for matched IDs): {stats['question_diff']}")
        if args.strict_question_match:
            print(f"  Skipped due to strict question mismatch: {stats['skipped_due_question']}")
    if qtype_by_id is not None:
        print(f"  Updated '{args.qtype_column}' from qtype JSONL: {stats['qtype_updated']}")
        print(f"  Missing qtype mapping for written rows: {stats['qtype_missing']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
