#!/usr/bin/env python3
"""
Build initial retrieval context for Wiki-PRF evaluation data.

This script:
1) Load input samples fro EchoSight CSV file
2) Runs EVA-CLIP image-to-image retrieval against a KB+FAISS index.
3) Writes a JSONL where each sample contains `entity_context` from top-1 KB article.

Output JSONL can be used as `json_path` in a new YAML for `test/test.py`.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _normalize_faiss_root(path: str) -> str:
    if path.endswith(".faiss"):
        path = os.path.dirname(path)
    if not path.endswith("/"):
        path += "/"
    return path


def _resolve_image_path(
    sample: Dict[str, Any],
    image_root: str,
    image_field: str = "image",
    image_path_field: str = "image_path",
) -> str:
    candidates: List[str] = []
    if image_path_field in sample and sample[image_path_field]:
        image_path = sample[image_path_field]
        if os.path.isabs(image_path):
            candidates.append(image_path)
        else:
            candidates.append(os.path.join(image_root, image_path))

    if image_field in sample and sample[image_field]:
        img_name = sample[image_field]
        if os.path.isabs(img_name):
            candidates.append(img_name)
        candidates.append(os.path.join(image_root, img_name))
        candidates.append(os.path.join(image_root, img_name.replace(".JPEG", ".jpg")))
        candidates.append(os.path.join(image_root, img_name.replace(".JPG", ".jpg")))
        candidates.append(os.path.join(image_root, img_name.replace(".png", ".jpg")))

    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot resolve image path for keys {image_field}/{image_path_field}. "
        f"candidates={candidates[:6]}"
    )


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_csv(path: str, qtype_filter: Optional[List[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        qtype_idx = header.index("question_type") if "question_type" in header else None

        for line_no, row in enumerate(reader, start=2):
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[: len(header)]

            if qtype_idx is not None and qtype_filter is not None:
                qtype = row[qtype_idx].strip().lower()
                if qtype not in qtype_filter:
                    continue

            sample = {header[i]: row[i] for i in range(len(header))}
            print(f"Loaded CSV line {line_no}: question_type={sample.get('question_type', '')}")
            sample["_csv_line_no"] = line_no
            rows.append(sample)
    return rows


def _apply_sampling(rows: List[Dict[str, Any]], sampling_strategy: str) -> List[Dict[str, Any]]:
    strategy = sampling_strategy or "all"
    if ":" not in strategy:
        return rows

    mode, amount = strategy.split(":", 1)
    mode = mode.strip().lower()
    amount = amount.strip()
    if "%" in amount:
        n = math.ceil(int(amount.replace("%", "")) * len(rows) / 100)
    else:
        n = int(amount)

    if mode == "first":
        return rows[:n]
    if mode == "end":
        return rows[-n:]
    if mode == "random":
        rows_copy = rows[:]
        random.shuffle(rows_copy)
        return rows_copy[:n]
    return rows


def _load_from_yaml(data_yaml: str) -> List[Dict[str, Any]]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    datasets = cfg.get("datasets", [])
    all_rows: List[Dict[str, Any]] = []
    for item in datasets:
        data_cfg = item.get("data", {})
        json_path = data_cfg.get("json_path")
        if not json_path or not json_path.endswith(".jsonl"):
            raise ValueError(f"Only .jsonl is supported in YAML data config, got: {json_path}")
        rows = _load_jsonl(json_path)
        rows = _apply_sampling(rows, data_cfg.get("sampling_strategy", "all"))
        print(f"Loaded {len(rows)} rows from {json_path}")
        all_rows.extend(rows)
    return all_rows


def _as_answer_eval(value: Any) -> List[Any]:
    if value is None:
        return [""]
    if isinstance(value, list):
        return value if value else [""]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return [""]
        if "|" in stripped:
            parts = [x.strip() for x in stripped.split("|") if x.strip()]
            return parts if parts else [stripped]
        return [stripped]
    return [value]


def _ensure_sample_for_wikiprf_test(
    sample: Dict[str, Any],
    image_path: str,
    image_root: str,
    question_field: str,
    answer_field: str,
    image_field: str,
) -> Dict[str, Any]:
    if "question" not in sample:
        if question_field not in sample:
            raise KeyError(f"Missing question field: '{question_field}'")
        sample["question"] = sample[question_field]

    if "answer_eval" not in sample:
        answer_value = sample.get(answer_field)
        if answer_value is None:
            answer_value = sample.get("answer")
        if answer_value is None:
            answer_value = sample.get("answers")
        sample["answer_eval"] = _as_answer_eval(answer_value)
    else:
        sample["answer_eval"] = _as_answer_eval(sample["answer_eval"])

    if image_field != "image" and "image" not in sample and image_field in sample:
        sample["image"] = sample[image_field]
    if "image" not in sample or not sample["image"]:
        rel = os.path.relpath(image_path, image_root)
        if not rel.startswith(".."):
            sample["image"] = rel
        else:
            # test.py can still load absolute path via os.path.join(image_root, abs_path) -> abs_path
            sample["image"] = image_path

    return sample


def _load_inat_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"Warning: iNaturalist mapping not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_by_dataset(
    sample: Dict[str, Any],
    image_root: str,
    inat_map: Dict[str, str],
    inaturalist_image_root: str,
    landmark_image_root: str,
    infoseek_image_root: str,
) -> Optional[str]:
    dataset_name = str(sample.get("dataset_name", "")).strip().lower()
    image_id = (
        str(sample.get("dataset_image_ids", "")).strip()
        or str(sample.get("image_id", "")).strip()
    )
    if not dataset_name or not image_id:
        return None

    if dataset_name == "inaturalist":
        if image_id not in inat_map:
            return None
        candidate = os.path.join(inaturalist_image_root, inat_map[image_id])
        return candidate if os.path.exists(candidate) else None

    if dataset_name == "landmarks":
        if len(image_id) < 3:
            return None
        candidate = os.path.join(
            landmark_image_root,
            image_id[0],
            image_id[1],
            image_id[2],
            image_id + ".jpg",
        )
        return candidate if os.path.exists(candidate) else None

    if dataset_name == "infoseek":
        base = infoseek_image_root or image_root
        direct = os.path.join(base, image_id)
        if os.path.exists(direct):
            return direct
        for ext in [".jpg", ".JPEG", ".jpeg", ".JPG", ".png", ".PNG"]:
            p = os.path.join(base, image_id + ext)
            if os.path.exists(p):
                return p
        return None

    return None


def _resolve_image_path_any(
    sample: Dict[str, Any],
    image_root: str,
    image_field: str,
    image_path_field: str,
    inat_map: Dict[str, str],
    inaturalist_image_root: str,
    landmark_image_root: str,
    infoseek_image_root: str,
) -> str:
    dataset_path = _resolve_by_dataset(
        sample=sample,
        image_root=image_root,
        inat_map=inat_map,
        inaturalist_image_root=inaturalist_image_root,
        landmark_image_root=landmark_image_root,
        infoseek_image_root=infoseek_image_root,
    )
    if dataset_path and os.path.exists(dataset_path):
        return dataset_path
    return _resolve_image_path(
        sample,
        image_root,
        image_field=image_field,
        image_path_field=image_path_field,
    )


def _deduplicate(raw_top_k: List[Dict[str, Any]], top_k: int) -> Tuple[List[str], List[float], List[Any]]:
    seen = set()
    urls: List[str] = []
    sims: List[float] = []
    entries: List[Any] = []
    for item in raw_top_k:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        score = item.get("similarity", 0.0)
        kb_entry = item.get("kb_entry")
        if kb_entry is None:
            continue
        urls.append(url)
        sims.append(score.item() if hasattr(score, "item") else float(score))
        entries.append(kb_entry)
        if len(urls) >= top_k:
            break
    return urls, sims, entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate initial retrieval entity_context with EVA-CLIP.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data_yaml", help="Path to Wiki-PRF data config YAML.")
    input_group.add_argument("--input_jsonl", help="Raw input JSONL path (no YAML required).")
    input_group.add_argument(
        "--input_csv",
        help="EchoSight-style input CSV (supports InfoSeek/E-VQA image lookup logic).",
    )

    parser.add_argument("--image_root", required=True, help="Root directory of images.")
    parser.add_argument("--knowledge_base", required=True, help="Path to wiki KB json.")
    parser.add_argument(
        "--faiss_root",
        required=True,
        help="Directory containing kb_index.faiss and kb_index_ids.pkl.",
    )
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--output_yaml",
        default=None,
        help="Output YAML path for test.py. Default: <output_jsonl_without_ext>.yaml",
    )
    parser.add_argument(
        "--input_sampling_strategy",
        default="all",
        help="Sampling strategy when --input_jsonl/--input_csv is used (e.g., all, first:100, random:10%%).",
    )
    parser.add_argument("--question_field", default="question", help="Question field name in input JSONL.")
    parser.add_argument("--answer_field", default="answer_eval", help="Answer field name in input JSONL.")
    parser.add_argument("--image_field", default="image", help="Image field name in input JSONL.")
    parser.add_argument("--image_path_field", default="image_path", help="Image path field name in input JSONL.")
    parser.add_argument(
        "--csv_question_type_filter",
        default="automatic,templated,multi_answer,String,Numerical,Time",
        help="Comma-separated question_type allowlist for CSV. Use empty string to keep all rows.",
    )
    parser.add_argument(
        "--inat_mapping",
        default="/data/qianMa/EchoSight/images/val_id2name.json",
        help="Path to iNaturalist val_id2name.json for CSV dataset_name=inaturalist.",
    )
    parser.add_argument(
        "--inaturalist_image_root",
        default=None,
        help="Root of inaturalist images. Default: --image_root",
    )
    parser.add_argument(
        "--landmark_image_root",
        default=None,
        help="Root of E-VQA landmark images. Default: --image_root",
    )
    parser.add_argument(
        "--infoseek_image_root",
        default=None,
        help="Root of InfoSeek val images. Default: --image_root",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id for retrieval and FAISS.")
    parser.add_argument("--retrieval_top_k", type=int, default=1, help="Top-k unique entries to keep.")
    parser.add_argument(
        "--overwrite_entity_context",
        action="store_true",
        help="If unset, existing entity_context is kept and retrieval is skipped for those samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from answer_generator import reconstruct_wiki_article
    from retriever import ClipRetriever

    random.seed(args.seed)

    if args.data_yaml:
        rows = _load_from_yaml(args.data_yaml)
    elif args.input_csv:
        qtype_filter = None
        if args.csv_question_type_filter.strip():
            qtype_filter = [
                x.strip().lower()
                for x in args.csv_question_type_filter.split(",")
                if x.strip()
            ]
        rows = _load_csv(args.input_csv, qtype_filter=qtype_filter)
        rows = _apply_sampling(rows, args.input_sampling_strategy)
        print(f"Loaded {len(rows)} rows from {args.input_csv}")
    else:
        rows = _load_jsonl(args.input_jsonl)
        rows = _apply_sampling(rows, args.input_sampling_strategy)
        print(f"Loaded {len(rows)} rows from {args.input_jsonl}")
    if not rows:
        raise ValueError("No rows loaded from input.")

    inat_map = _load_inat_map(args.inat_mapping)
    inaturalist_image_root = args.inaturalist_image_root or args.image_root
    landmark_image_root = args.landmark_image_root or args.image_root
    infoseek_image_root = args.infoseek_image_root or args.image_root

    retriever = ClipRetriever(device=f"cuda:{args.gpu_id}", model="eva-clip")
    retriever.load_knowledge_base(args.knowledge_base)
    retriever.load_faiss_index(_normalize_faiss_root(args.faiss_root))

    success = 0
    fail = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for sample in tqdm(rows, desc="Initial Retrieval"):
            if sample.get("entity_context") and not args.overwrite_entity_context:
                try:
                    image_path = _resolve_image_path_any(
                        sample,
                        args.image_root,
                        image_field=args.image_field,
                        image_path_field=args.image_path_field,
                        inat_map=inat_map,
                        inaturalist_image_root=inaturalist_image_root,
                        landmark_image_root=landmark_image_root,
                        infoseek_image_root=infoseek_image_root,
                    )
                    sample = _ensure_sample_for_wikiprf_test(
                        sample,
                        image_path=image_path,
                        image_root=args.image_root,
                        question_field=args.question_field,
                        answer_field=args.answer_field,
                        image_field=args.image_field,
                    )
                except Exception:
                    pass
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                success += 1
                continue

            # try:
            image_path = _resolve_image_path_any(
                sample,
                args.image_root,
                image_field=args.image_field,
                image_path_field=args.image_path_field,
                inat_map=inat_map,
                inaturalist_image_root=inaturalist_image_root,
                landmark_image_root=landmark_image_root,
                infoseek_image_root=infoseek_image_root,
            )
            image = Image.open(image_path).convert("RGB")
            raw_top_k = retriever.retrieve_image_faiss(image, top_k=max(1, args.retrieval_top_k))
            urls, sims, entries = _deduplicate(raw_top_k, args.retrieval_top_k)

            sample = _ensure_sample_for_wikiprf_test(
                sample,
                image_path=image_path,
                image_root=args.image_root,
                question_field=args.question_field,
                answer_field=args.answer_field,
                image_field=args.image_field,
            )
            sample["image_path"] = image_path
            sample["initial_retrieved_entries"] = urls
            sample["initial_retrieval_similarities"] = sims
            sample["initial_top1_url"] = urls[0] if urls else ""
            sample["entity_context"] = reconstruct_wiki_article(entries[0]) if entries else ""
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            success += 1
            # except Exception as exc:
            #     sample["initial_retrieval_error"] = str(exc)
            #     sample.setdefault("entity_context", "")
            #     sample.setdefault("answer_eval", _as_answer_eval(sample.get(args.answer_field)))
            #     if "question" not in sample and args.question_field in sample:
            #         sample["question"] = sample[args.question_field]
            #     out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            #     fail += 1

    print(f"Saved: {args.output_jsonl}")
    print(f"Processed success={success}, fail={fail}")

    output_yaml = args.output_yaml
    if output_yaml is None:
        output_yaml = os.path.splitext(os.path.abspath(args.output_jsonl))[0] + ".yaml"
    out_yaml = {
        "datasets": [
            {
                "data": {
                    "json_path": os.path.abspath(args.output_jsonl),
                    "sampling_strategy": "all",
                }
            }
        ]
    }
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"Saved YAML: {output_yaml}")


if __name__ == "__main__":
    main()
