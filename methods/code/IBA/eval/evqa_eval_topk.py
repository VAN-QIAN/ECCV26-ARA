"""EVQA evaluation tailored for the top-k entity pipeline."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import tensorflow as tf
except ImportError:
    tf = None

from evqa_utils import evaluate_example
from utils import get_test_question, load_csv_data

if tf is not None:
    physical_devices = tf.config.list_physical_devices("GPU")
    print(physical_devices)
    if physical_devices:
        tf.config.set_visible_devices(physical_devices[0], "GPU")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EVQA predictions with top-k metadata")
    parser.add_argument("--test_file", required=True, type=Path, help="CSV with EVQA ground-truth rows")
    parser.add_argument("--prediction_file", required=True, type=Path, help="Predictions JSONL (data_id/prediction)")
    parser.add_argument(
        "--metadata_file",
        type=Path,
        default=None,
        help="Metadata JSON/JSONL from top-k prepare (omit if unavailable)",
    )
    parser.add_argument("--print_every", type=int, default=0, help="If >0, print running average every N examples")
    parser.add_argument("--grounded_csv", type=Path, default=None, help="Optional CSV enumerating grounded ids")
    parser.add_argument("--ungrounded_csv", type=Path, default=None, help="Optional CSV enumerating ungrounded ids")
    parser.add_argument("--details_output", type=Path, default=None, help="Optional per-question analysis JSONL output")
    return parser.parse_args()


def _coerce_prediction_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        for item in value:
            text = _coerce_prediction_string(item)
            if text:
                return text
        return None
    if isinstance(value, dict):
        for key in ("text", "answer", "prediction", "output", "content"):
            if key in value:
                text = _coerce_prediction_string(value[key])
                if text:
                    return text
        return None
    return None


def _extract_prediction(record: Dict[str, Any], context: str) -> str:
    for key in ("prediction", "answer", "answers", "output"):
        if key in record:
            text = _coerce_prediction_string(record[key])
            if text:
                return text
    raise ValueError(f"{context} is missing a usable prediction field")


def load_predictions(path: Path) -> Dict[str, str]:
    predictions: Dict[str, str] = {}
    raw_text = path.read_text(encoding="utf-8").strip()
    structured_records: Optional[List[Dict[str, Any]]] = None

    if raw_text:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            structured_records = []
            for idx, item in enumerate(parsed):
                if not isinstance(item, dict):
                    raise ValueError(f"{path} list element {idx} is not an object")
                structured_records.append(item)
        elif isinstance(parsed, dict):
            if "data_id" in parsed:
                structured_records = [parsed]
            else:
                for key in ("predictions", "data", "results", "items"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        structured_records = []
                        for idx, item in enumerate(value):
                            if not isinstance(item, dict):
                                raise ValueError(f"{path} {key} element {idx} is not an object")
                            structured_records.append(item)
                        break
                else:
                    if all(isinstance(value, dict) for value in parsed.values()):
                        structured_records = []
                        for key, value in parsed.items():
                            if not isinstance(value, dict):
                                continue
                            record = dict(value)
                            if "data_id" not in record and isinstance(key, str):
                                record["data_id"] = key
                            structured_records.append(record)

    if structured_records is not None:
        for idx, record in enumerate(structured_records):
            if not isinstance(record, dict):
                raise ValueError(f"{path} structured record {idx} is not an object")
            data_id = record.get("data_id")
            if not isinstance(data_id, str):
                raise ValueError(f"{path} structured record {idx} missing data_id")
            predictions[data_id] = _extract_prediction(record, context=f"{path} structured record {idx}")
        return predictions

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} of {path} is not a JSON object")
        data_id = record.get("data_id")
        if not isinstance(data_id, str):
            raise ValueError(f"Missing or invalid data_id on line {line_number} of {path}")
        predictions[data_id] = _extract_prediction(record, context=f"{path} line {line_number}")
    return predictions


def load_metadata(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if path is None:
        return {}
    metadata: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            data_id = record.get("data_id")
            if not isinstance(data_id, str):
                raise ValueError(f"Missing or invalid data_id on line {line_number} of {path}")
            metadata[data_id] = record
    return metadata


def iterate_examples(test_file: Path) -> Tuple[List[List[str]], List[str]]:
    rows, header = load_csv_data(str(test_file))
    return rows, header


def collect_ids_from_csv(path: Optional[Path]) -> Iterable[str]:
    if not path:
        return []
    rows, header = load_csv_data(str(path))
    idx_map = {name: idx for idx, name in enumerate(header)}
    ids: List[str] = []
    for row_idx, row in enumerate(rows):
        if "data_id" in idx_map:
            ids.append(row[idx_map["data_id"]])
        else:
            ids.append(f"E-VQA_{row_idx}")
    return ids


def main() -> None:
    args = parse_args()

    test_rows, test_header = iterate_examples(args.test_file)
    predictions = load_predictions(args.prediction_file)
    metadata = load_metadata(args.metadata_file)
    grounded_ids = set(collect_ids_from_csv(args.grounded_csv))
    ungrounded_ids = set(collect_ids_from_csv(args.ungrounded_csv))

    evaluated = 0
    eval_score = 0.0
    missing: List[str] = []

    primary_identified = 0
    context_identified = 0
    gt_in_topk = 0
    answer_correct_with_primary = 0
    answer_correct_without_primary = 0

    grounded_count = grounded_correct = 0
    grounded_primary = grounded_context = grounded_gt_topk = 0
    ungrounded_count = ungrounded_correct = 0

    analysis_rows: List[Dict[str, object]] = []

    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")

    for row_idx, _ in enumerate(test_rows):
        example = get_test_question(row_idx, test_rows, test_header)
        data_id = example.get("data_id", f"E-VQA_{row_idx}")
        prediction = predictions.get(data_id)
        if prediction is None:
            missing.append(data_id)
            continue

        answers = [item.strip() for item in example.get("answer", "").split("|") if item.strip()]
        if not answers:
            continue

        score = float(
            evaluate_example(
                example.get("question", ""),
                reference_list=answers,
                candidate=prediction,
                question_type=example.get("question_type"),
            )
        )
        eval_score += score
        evaluated += 1
        answer_correct = score > 0.0

        if args.print_every and evaluated % args.print_every == 0:
            print(f"Iter {evaluated}: score={score:.4f}, running avg={eval_score / evaluated:.4f}")

        record = metadata.get(data_id, {})
        ground_truth_url = example.get("wikipedia_url")

        selected_url = record.get("selected_url") if isinstance(record, dict) else None
        context_url = record.get("context_source_url") if isinstance(record, dict) else None
        if isinstance(record, dict):
            topk_flag = bool(
                record.get("ground_truth_entity_in_topk_identification")
                or record.get("ground_truth_in_topk_identification")
                or record.get("ground_truth_in_topk")
            )
        else:
            topk_flag = False

        primary_hit = bool(ground_truth_url) and selected_url == ground_truth_url
        context_hit = bool(ground_truth_url) and context_url == ground_truth_url

        if primary_hit:
            primary_identified += 1
        if context_hit:
            context_identified += 1
        if topk_flag:
            gt_in_topk += 1

        if answer_correct:
            if primary_hit:
                answer_correct_with_primary += 1
            else:
                answer_correct_without_primary += 1

        in_grounded = data_id in grounded_ids if grounded_ids else False
        in_ungrounded = data_id in ungrounded_ids if ungrounded_ids else False

        if in_grounded:
            grounded_count += 1
            grounded_correct += int(answer_correct)
            grounded_primary += int(primary_hit)
            grounded_context += int(context_hit)
            grounded_gt_topk += int(topk_flag)
        elif in_ungrounded:
            ungrounded_count += 1
            ungrounded_correct += int(answer_correct)

        analysis_rows.append(
            {
                "data_id": data_id,
                "question": example.get("question"),
                "prediction": prediction,
                "score": score,
                "answer_correct": answer_correct,
                "ground_truth_url": ground_truth_url,
                "selected_url": selected_url,
                "context_source_url": context_url,
                "ground_truth_in_topk": topk_flag,
                "primary_entity_identified": primary_hit,
                "context_entity_identified": context_hit,
                "is_grounded_split": in_grounded if grounded_ids else None,
                "is_ungrounded_split": in_ungrounded if ungrounded_ids else None,
            }
        )

    if evaluated:
        print(f"Evaluated examples: {evaluated}")
        print(f"Average score: {eval_score / evaluated:.4f}")
        print(
            f"Primary entity identified (selected_url == GT): {primary_identified}/{evaluated}"
        )
        print(
            f"Context entity identified (context_source_url == GT): {context_identified}/{evaluated}"
        )
        print(f"Ground truth surfaced in top-k list: {gt_in_topk}/{evaluated}")
        print(
            f"Answers correct w/ primary entity identified: {answer_correct_with_primary}/{primary_identified or 1}"
        )
        without_primary = evaluated - primary_identified
        if without_primary:
            print(
                f"Answers correct without primary entity identified: {answer_correct_without_primary}/{without_primary}"
            )
        if grounded_count:
            print(
                f"Grounded examples: {grounded_count}, avg score={grounded_correct/grounded_count:.4f}"
            )
            print(
                f"  Primary entity hit: {grounded_primary}/{grounded_count} ({grounded_primary/grounded_count:.3%})"
            )
            print(
                f"  Context entity hit: {grounded_context}/{grounded_count} ({grounded_context/grounded_count:.3%})"
            )
            print(
                f"  GT surfaced in top-k: {grounded_gt_topk}/{grounded_count} ({grounded_gt_topk/grounded_count:.3%})"
            )
        if ungrounded_count:
            print(
                f"Ungrounded examples: {ungrounded_count}, answer accuracy={ungrounded_correct/ungrounded_count:.3%}"
            )
    else:
        print("No examples evaluated (maybe all predictions missing?)")

    if missing:
        print(f"Missing predictions for {len(missing)} examples. Sample ids: {missing[:10]}")

    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        with args.details_output.open("w", encoding="utf-8") as f:
            for row in analysis_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote per-question analysis to {args.details_output}")

    end_time = datetime.datetime.now()
    print(f"End time: {end_time}")


if __name__ == "__main__":
    main()
