#!/usr/bin/env python3
"""Replace target CSV answer column using source CSV mapping by data_id."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TARGET_CSV = str(REPO_ROOT / "data/ground_truth/infoseek_unfixed_subset.csv")
DEFAULT_SOURCE_CSV = str(REPO_ROOT / "data/ground_truth/infoseek_fixed.csv")


def infer_default_output(target_csv: str) -> str:
    path = Path(target_csv)
    return str(path.with_name(f"{path.stem}_answer_replaced{path.suffix}"))


def load_answer_mapping(
    source_csv: str,
    id_column: str,
    answer_column: str,
) -> Tuple[Dict[str, str], int]:
    mapping: Dict[str, str] = {}
    duplicates = 0
    conflicts = 0
    counter: Counter[str] = Counter()

    with open(source_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty source CSV: {source_csv}")
        if id_column not in reader.fieldnames:
            raise KeyError(f"Column '{id_column}' not found in source CSV: {source_csv}")
        if answer_column not in reader.fieldnames:
            raise KeyError(f"Column '{answer_column}' not found in source CSV: {source_csv}")

        for row in reader:
            row_id = row.get(id_column, "").strip()
            if not row_id:
                continue
            answer = row.get(answer_column, "")
            counter[row_id] += 1

            if row_id in mapping:
                duplicates += 1
                if mapping[row_id] != answer:
                    conflicts += 1
                # Keep the last answer for duplicated IDs.
            mapping[row_id] = answer

    if conflicts > 0:
        raise ValueError(
            f"Source CSV has {conflicts} duplicated '{id_column}' rows with conflicting "
            f"'{answer_column}' values."
        )

    duplicate_ids = sum(1 for v in counter.values() if v > 1)
    return mapping, duplicate_ids


def replace_answers(
    target_csv: str,
    output_csv: str,
    mapping: Dict[str, str],
    id_column: str,
    answer_column: str,
    require_all_ids: bool,
) -> Dict[str, int]:
    written_rows = 0
    replaced_rows = 0
    unchanged_rows = 0
    missing_id_rows = 0
    missing_ids = set()

    with open(target_csv, "r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"Empty target CSV: {target_csv}")
        if id_column not in reader.fieldnames:
            raise KeyError(f"Column '{id_column}' not found in target CSV: {target_csv}")
        if answer_column not in reader.fieldnames:
            raise KeyError(f"Column '{answer_column}' not found in target CSV: {target_csv}")

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                row_id = row.get(id_column, "").strip()
                if not row_id:
                    writer.writerow(row)
                    written_rows += 1
                    continue

                new_answer = mapping.get(row_id)
                if new_answer is None:
                    missing_id_rows += 1
                    missing_ids.add(row_id)
                else:
                    old_answer = row.get(answer_column, "")
                    if old_answer != new_answer:
                        row[answer_column] = new_answer
                        replaced_rows += 1
                    else:
                        unchanged_rows += 1

                writer.writerow(row)
                written_rows += 1

    if require_all_ids and missing_ids:
        missing_preview = ", ".join(sorted(missing_ids)[:5])
        raise ValueError(
            f"{len(missing_ids)} unique IDs in target CSV not found in source CSV. "
            f"Examples: {missing_preview}"
        )

    return {
        "written_rows": written_rows,
        "replaced_rows": replaced_rows,
        "unchanged_rows": unchanged_rows,
        "missing_id_rows": missing_id_rows,
        "missing_unique_ids": len(missing_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace target CSV answer values with source CSV answers by data_id."
    )
    parser.add_argument("--target-csv", default=DEFAULT_TARGET_CSV, help="CSV to update.")
    parser.add_argument(
        "--source-csv",
        default=DEFAULT_SOURCE_CSV,
        help="CSV providing data_id -> answer mapping.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output path. If omitted, auto-generate next to target CSV unless --inplace is set.",
    )
    parser.add_argument(
        "--id-column",
        default="data_id",
        help="ID column used for matching. Default: data_id",
    )
    parser.add_argument(
        "--answer-column",
        default="answer",
        help="Answer column to replace. Default: answer",
    )
    parser.add_argument(
        "--require-all-ids",
        action="store_true",
        help="Fail if any target data_id is missing in source mapping.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite target CSV directly.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="When used with --inplace, keep a '.bak' backup of the target CSV before writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_csv = args.target_csv if args.inplace else (args.output_csv or infer_default_output(args.target_csv))

    try:
        mapping, duplicate_ids = load_answer_mapping(
            source_csv=args.source_csv,
            id_column=args.id_column,
            answer_column=args.answer_column,
        )

        target_path = Path(args.target_csv)
        temp_output = Path(output_csv)
        if args.inplace:
            if args.backup:
                backup_path = target_path.with_suffix(target_path.suffix + ".bak")
                shutil.copy2(target_path, backup_path)
                print(f"Backup created: {backup_path}")
            temp_output = target_path.with_suffix(target_path.suffix + ".tmp")

        stats = replace_answers(
            target_csv=args.target_csv,
            output_csv=str(temp_output),
            mapping=mapping,
            id_column=args.id_column,
            answer_column=args.answer_column,
            require_all_ids=args.require_all_ids,
        )

        if args.inplace:
            temp_output.replace(target_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print("Answer replacement completed.")
    print(f"  Target CSV: {args.target_csv}")
    print(f"  Source CSV: {args.source_csv}")
    print(f"  Output CSV: {output_csv}")
    print(f"  Mapping size: {len(mapping)}")
    print(f"  Source duplicate IDs: {duplicate_ids}")
    print(f"  Written rows: {stats['written_rows']}")
    print(f"  Replaced rows: {stats['replaced_rows']}")
    print(f"  Unchanged rows: {stats['unchanged_rows']}")
    print(f"  Missing-ID rows in target: {stats['missing_id_rows']}")
    print(f"  Missing unique IDs in target: {stats['missing_unique_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
