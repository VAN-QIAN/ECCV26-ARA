"""Utility to measure Qwen identification accuracy from metadata JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def load_metadata(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_ground_truth(csv_path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    ground_truth: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_index = {name: idx for idx, name in enumerate(header)}
        required_fields = {"data_id", "wikipedia_title"}
        missing = required_fields - set(header_index)
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")
        for row in reader:
            data_id = row[header_index["data_id"]]
            dataset_name = (
                row[header_index["dataset_name"]]
                if "dataset_name" in header_index
                else "unknown"
            )
            ground_truth[data_id] = {
                "wikipedia_title": row[header_index["wikipedia_title"]],
                "dataset_name": dataset_name,
            }
    return ground_truth, header


def normalize(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return " ".join(text.strip().lower().split())


def evaluate_identification(
    metadata: Iterable[Dict[str, object]],
    ground_truth: Dict[str, Dict[str, str]],
) -> Dict[str, object]:
    total = 0
    correct = 0
    skipped = 0
    mismatched = []
    dataset_counter: Counter[str] = Counter()
    dataset_correct: Counter[str] = Counter()
    fallback_counter: Counter[str] = Counter()

    for record in metadata:
        data_id = record.get("data_id")
        if not isinstance(data_id, str):
            continue
        gt = ground_truth.get(data_id)
        if not gt:
            skipped += 1
            continue
        dataset_name = gt.get("dataset_name") or "unknown"
        dataset_counter[dataset_name] += 1
        total += 1
        pred_title = normalize(record.get("selected_title"))
        gt_title = normalize(gt.get("wikipedia_title"))
        if record.get("fallback_reason"):
            fallback_counter[str(record.get("fallback_reason"))] += 1
        if pred_title == gt_title and pred_title is not None:
            correct += 1
            dataset_correct[dataset_name] += 1
        else:
            mismatched.append(
                {
                    "data_id": data_id,
                    "predicted": record.get("selected_title"),
                    "ground_truth": gt.get("wikipedia_title"),
                    "fallback_reason": record.get("fallback_reason"),
                }
            )

    accuracy = correct / total if total else 0.0
    by_dataset = {
        name: {
            "total": dataset_counter[name],
            "correct": dataset_correct.get(name, 0),
            "accuracy": dataset_correct.get(name, 0) / dataset_counter[name]
            if dataset_counter[name]
            else 0.0,
        }
        for name in dataset_counter
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "skipped": skipped,
        "by_dataset": by_dataset,
        "fallback_counts": dict(fallback_counter),
        "mismatches": mismatched,
    }


def write_report(report_path: Path, report: Dict[str, object]) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen identification accuracy from metadata JSONL")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--test_file", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--max_mismatches", type=int, default=20)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    ground_truth, _ = load_ground_truth(args.test_file)
    report = evaluate_identification(metadata, ground_truth)

    print(f"Total evaluated: {report['total']}")
    print(f"Correct: {report['correct']}")
    print(f"Accuracy: {report['accuracy']:.3%}")
    if report["skipped"]:
        print(f"Skipped (missing GT): {report['skipped']}")

    if report["by_dataset"]:
        print("\nPer dataset:")
        for name, stats in sorted(report["by_dataset"].items()):
            print(
                f"  {name}: {stats['correct']}/{stats['total']}"
                f" ({stats['accuracy']:.3%})"
            )

    if report["fallback_counts"]:
        print("\nFallback reasons:")
        for reason, count in report["fallback_counts"].items():
            print(f"  {reason}: {count}")

    mismatches = report.get("mismatches", [])
    if mismatches:
        print("\nSample mismatches:")
        for item in mismatches[: args.max_mismatches]:
            print(
                f"  {item['data_id']}: predicted=\"{item['predicted']}\" | "
                f"ground_truth=\"{item['ground_truth']}\" | fallback={item['fallback_reason']}"
            )

    if args.output:
        write_report(args.output, report)
        print(f"\nDetailed report written to {args.output}")


if __name__ == "__main__":
    main()
