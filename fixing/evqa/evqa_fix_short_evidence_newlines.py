#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def clean_short_evidence(text: str) -> str:
    if text is None:
        return text
    if "\n" not in text and "\r" not in text:
        return text
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    return cleaned.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix stray newlines in a target column of an EVQA CSV."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="fixing/evqa/results_question_fix/evqa_question_fix.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to output CSV. Defaults to <input>.fixed.csv",
    )
    parser.add_argument(
        "--column",
        default="short_evidence",
        help="Column to clean. Default: short_evidence",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}.fixed{input_path.suffix}")
    )

    changed = 0
    total = 0
    target_column = args.column

    with input_path.open("r", newline="", encoding="utf-8") as fin, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")
        if target_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Column '{target_column}' not found in input CSV. Available columns: {available}"
            )

        writer = csv.DictWriter(
            fout, fieldnames=reader.fieldnames, lineterminator="\n"
        )
        writer.writeheader()

        for row in reader:
            total += 1
            original = row.get(target_column, "")
            cleaned = clean_short_evidence(original)
            if cleaned != original:
                changed += 1
                row[target_column] = cleaned
            writer.writerow(row)

    print(f"Processed {total} rows. Fixed {changed} rows in column '{target_column}'.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
