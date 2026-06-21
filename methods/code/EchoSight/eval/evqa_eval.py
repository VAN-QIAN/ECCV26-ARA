"""Minimal EVQA evaluation script following the original project flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from evqa_utils import evaluate_example
from utils import load_csv_data, get_test_question
import tensorflow as tf
from tqdm import tqdm
# tf.config.set_visible_devices([], "GPU")
# physical_devices = tf.config.list_physical_devices('GPU')

# tf.config.set_visible_devices(physical_devices, "GPU")

if tf is not None:
    physical_devices = tf.config.list_physical_devices("GPU")
    print(physical_devices)
    if physical_devices:
        tf.config.set_visible_devices(physical_devices[0], "GPU")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EVQA predictions")
    parser.add_argument("--test_file", required=True, type=Path, help="CSV with EVQA ground truth")
    parser.add_argument(
        "--prediction_file",
        required=True,
        type=Path,
        help="JSONL file produced by the model (with data_id/prediction fields)",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=0,
        help="If >0 print running average every N examples",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
            data_id = record.get("data_id")
            if not isinstance(data_id, str):
                raise ValueError(
                    f"Missing or invalid data_id on line {line_number} of {path}"
                )
            predictions[data_id] = record.get("prediction", "")
    return predictions


def iterate_examples(test_file: Path) -> Tuple[List[List[str]], List[str]]:
    test_list, header = load_csv_data(str(test_file))
    return test_list, header


def main() -> None:
    args = parse_args()

    test_list, test_header = iterate_examples(args.test_file)
    predictions = load_predictions(args.prediction_file)

    eval_score = 0.0
    evaluated = 0
    missing: List[str] = []

    for it, _ in tqdm(enumerate(test_list)):
        example = get_test_question(it, test_list, test_header)
        data_id = example.get("data_id", f"E-VQA_{it}")
        candidate = predictions.get(data_id)
        if candidate is None:
            missing.append(data_id)
            continue

        target_answers = [ans.strip() for ans in example["answer"].split("|") if ans.strip()]
        if not target_answers:
            continue

        score = evaluate_example(
            example["question"],
            reference_list=target_answers,
            candidate=candidate,
            question_type=example.get("question_type"),
        )
        eval_score += float(score)
        evaluated += 1

        if args.print_every and evaluated % args.print_every == 0:
            print(f"question_type={example.get('question_type')}: {example['question']}")
            print(f"  target_answers={target_answers}")
            print(f"  candidate={candidate}")
            print(
                f"Iter {evaluated}: score={score:.4f}, running average={eval_score / evaluated:.4f}"
            )

    if evaluated:
        print(f"Evaluated examples: {evaluated}")
        print(f"Average score: {eval_score / evaluated:.4f}")
    else:
        print("No examples evaluated (all predictions missing?)")

    if missing:
        print(f"Missing predictions for {len(missing)} examples. Example ids: {missing[:10]}")


if __name__ == "__main__":
    main()
