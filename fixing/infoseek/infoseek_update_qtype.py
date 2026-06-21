#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def load_qtype_map(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data_id = normalize_text(obj.get("data_id"))
            qtype = normalize_text(obj.get("question_type"))
            if data_id and qtype:
                mapping[data_id] = qtype
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Add question_type from infoseek_val_qtype.jsonl into a CSV column."
    )
    parser.add_argument("--qtype-jsonl", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--output-column", default="question_type_qtype")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output column even if it already has a value.",
    )
    parser.add_argument(
        "--fallback-to-existing",
        action="store_true",
        help="If data_id not found, use fallback column value.",
    )
    parser.add_argument("--fallback-column", default="question_type")
    args = parser.parse_args()

    qtype_map = load_qtype_map(args.qtype_jsonl)
    if not qtype_map:
        print("No question_type entries loaded from JSONL.", file=sys.stderr)

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            print("Missing CSV header.", file=sys.stderr)
            sys.exit(1)
        if args.output_column not in fieldnames:
            insert_at = len(fieldnames)
            if args.fallback_column in fieldnames:
                insert_at = fieldnames.index(args.fallback_column) + 1
            fieldnames.insert(insert_at, args.output_column)

        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            missing_id = 0
            missing_map = 0
            filled = 0
            for row in reader:
                data_id = normalize_text(row.get(args.id_column))
                current = normalize_text(row.get(args.output_column))
                if not data_id:
                    missing_id += 1
                mapped = qtype_map.get(data_id, "")
                if mapped:
                    if args.overwrite or not current:
                        row[args.output_column] = mapped
                        filled += 1
                else:
                    missing_map += 1
                    if (args.overwrite or not current) and args.fallback_to_existing:
                        row[args.output_column] = normalize_text(row.get(args.fallback_column))
                writer.writerow(row)

    print(
        f"Wrote {args.output_csv}. filled={filled} missing_map={missing_map} missing_id={missing_id}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
