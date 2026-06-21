#!/usr/bin/env python3
import argparse
import csv
import json
import os


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def load_answers(path):
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
            if not data_id:
                continue
            answer = obj.get("answer")
            if answer is None:
                continue
            if isinstance(answer, list):
                answer_str = "|".join([normalize_text(a) for a in answer if normalize_text(a)])
            else:
                answer_str = normalize_text(answer)
            mapping[data_id] = answer_str
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Sync answers from infoseek_val.jsonl.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--answers-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--missing-output", default="")
    args = parser.parse_args()

    answers = load_answers(args.answers_jsonl)
    missing_ids = []

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    if args.missing_output:
        os.makedirs(os.path.dirname(args.missing_output) or ".", exist_ok=True)

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_csv, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            data_id = normalize_text(row.get(args.id_column))
            if not data_id or data_id not in answers:
                missing_ids.append(data_id)
                writer.writerow(row)
                continue
            row[args.answer_column] = answers[data_id]
            writer.writerow(row)

    if args.missing_output:
        with open(args.missing_output, "w", encoding="utf-8") as f:
            for data_id in missing_ids:
                f.write(f"{data_id}\n")

    print(f"Done. Updated {len(answers) - len(missing_ids)} rows. Missing {len(missing_ids)}.")


if __name__ == "__main__":
    main()
