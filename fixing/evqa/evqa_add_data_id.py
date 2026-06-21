#!/usr/bin/env python3
import argparse
import csv
import os


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Add a data_id column to an E-VQA CSV."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--id-prefix", default="E-VQA_")
    parser.add_argument("--id-start", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing IDs instead of only filling empty values.",
    )
    args = parser.parse_args()

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if args.id_column not in fieldnames:
            fieldnames.insert(0, args.id_column)

        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for idx, row in enumerate(reader):
                current = normalize_text(row.get(args.id_column))
                if args.overwrite or not current:
                    row[args.id_column] = f"{args.id_prefix}{args.id_start + idx}"
                writer.writerow(row)


if __name__ == "__main__":
    main()
