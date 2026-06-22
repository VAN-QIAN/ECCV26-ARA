#!/usr/bin/env python3
"""
Build CoMEM-compatible retrieval dataset from custom CSV/JSONL input.

Output schema matches CoMEM infoseek inference expectation:
  - data_id: str
  - image_path: str
  - question: str
  - retrieval_info: json list of
      {"passage_content": str, "image": "<base64-jpeg>"}

Supports:
  - EchoSight-style CSV input (dataset_name + dataset_image_ids).
  - Generic JSONL input (custom field names).
  - Output as MDS (StreamingDataset) and/or JSONL.
"""

import argparse
import base64
import csv
import io
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../.."))


def _normalize_faiss_root(path: str) -> str:
    if path.endswith(".faiss"):
        path = os.path.dirname(path)
    if not path.endswith("/"):
        path += "/"
    return path


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        copied = rows[:]
        random.shuffle(copied)
        return copied[:n]
    return rows


def _load_inat_map(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


@dataclass
class Roots:
    image_root: str
    infoseek_root: str
    landmark_root: str
    inat_root: str


def _resolve_from_dataset_name(
    sample: Dict[str, Any],
    roots: Roots,
    inat_map: Dict[str, str],
) -> Optional[str]:
    dataset_name = str(sample.get("dataset_name", "")).strip().lower()
    image_id = str(sample.get("dataset_image_ids", "")).strip() or str(sample.get("image_id", "")).strip()
    if not dataset_name or not image_id:
        return None

    if dataset_name == "infoseek":
        direct = os.path.join(roots.infoseek_root, image_id)
        if os.path.exists(direct):
            return direct
        for ext in [".jpg", ".jpeg", ".JPEG", ".JPG", ".png", ".PNG"]:
            p = os.path.join(roots.infoseek_root, image_id + ext)
            if os.path.exists(p):
                return p
        return None

    if dataset_name == "landmarks":
        if len(image_id) < 3:
            return None
        p = os.path.join(roots.landmark_root, image_id[0], image_id[1], image_id[2], image_id + ".jpg")
        return p if os.path.exists(p) else None

    if dataset_name == "inaturalist":
        mapped = inat_map.get(image_id)
        if not mapped:
            return None
        p = os.path.join(roots.inat_root, mapped)
        return p if os.path.exists(p) else None

    return None


def _resolve_image_path_generic(
    sample: Dict[str, Any],
    image_root: str,
    image_field: str,
    image_path_field: str,
) -> Optional[str]:
    candidates: List[str] = []
    if sample.get(image_path_field):
        v = str(sample[image_path_field]).strip()
        if os.path.isabs(v):
            candidates.append(v)
        else:
            candidates.append(os.path.join(image_root, v))
    if sample.get(image_field):
        v = str(sample[image_field]).strip()
        if os.path.isabs(v):
            candidates.append(v)
        candidates.append(os.path.join(image_root, v))
        candidates.append(os.path.join(image_root, v.replace(".JPEG", ".jpg")))
        candidates.append(os.path.join(image_root, v.replace(".JPG", ".jpg")))
        candidates.append(os.path.join(image_root, v.replace(".png", ".jpg")))
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_image_path(
    sample: Dict[str, Any],
    roots: Roots,
    inat_map: Dict[str, str],
    image_field: str,
    image_path_field: str,
) -> str:
    p = _resolve_from_dataset_name(sample, roots, inat_map)
    if p:
        return p
    p = _resolve_image_path_generic(
        sample=sample,
        image_root=roots.image_root,
        image_field=image_field,
        image_path_field=image_path_field,
    )
    if p:
        return p
    raise FileNotFoundError(
        "Cannot resolve image path from dataset_name/dataset_image_ids or generic fields "
        f"{image_field}/{image_path_field}"
    )


def _image_to_b64_jpeg(img: Image.Image, max_side: int = 512) -> str:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _fetch_image_from_url(url: str, timeout: float = 8.0) -> Optional[Image.Image]:
    if not url:
        return None
    if not (url.startswith("http://") or url.startswith("https://")):
        if os.path.exists(url):
            try:
                return Image.open(url).convert("RGB")
            except Exception:
                return None
        return None
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=timeout, headers=headers)
        if resp.status_code != 200:
            return None
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _reconstruct_wiki_article_fallback(entry: Any) -> str:
    title = str(getattr(entry, "title", "") or "")
    section_titles = getattr(entry, "section_titles", []) or []
    section_texts = getattr(entry, "section_texts", []) or []
    article = "# Wiki Article: " + title + "\n"
    for i, section_title in enumerate(section_titles):
        section_title = str(section_title or "")
        if "external link" in section_title.lower() or "reference" in section_title.lower():
            continue
        section_text = str(section_texts[i] if i < len(section_texts) else "")
        article += "\n## Section Title: " + section_title + "\n" + section_text
    return article


def _build_passage(
    entry: Any,
    max_chars: int,
    reconstruct_wiki_article_fn: Optional[Any] = None,
) -> str:
    if reconstruct_wiki_article_fn is not None:
        try:
            txt = str(reconstruct_wiki_article_fn(entry) or "").strip()
        except Exception:
            txt = _reconstruct_wiki_article_fallback(entry).strip()
    else:
        txt = _reconstruct_wiki_article_fallback(entry).strip()

    if txt:
        if max_chars > 0 and len(txt) > max_chars:
            txt = txt[:max_chars]
        return txt

    # fallback for empty reconstructed text
    title = str(getattr(entry, "title", "") or "")
    sections = getattr(entry, "section_texts", []) or []
    section = ""
    for s in sections:
        s = str(s).strip()
        if s:
            section = s
            break
    if title and section:
        txt = f"{title}: {section}"
    elif title:
        txt = title
    else:
        txt = section
    txt = txt.strip()
    if max_chars > 0 and len(txt) > max_chars:
        txt = txt[:max_chars]
    return txt


def _deduplicate_retrieval(raw_top: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in raw_top:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def _iter_rows(
    input_csv: Optional[str],
    input_jsonl: Optional[str],
    input_sampling_strategy: str,
    csv_question_type_filter: str,
) -> List[Dict[str, Any]]:
    if input_csv:
        qf = None
        if csv_question_type_filter.strip():
            qf = [x.strip().lower() for x in csv_question_type_filter.split(",") if x.strip()]
        rows = _load_csv(input_csv, qf)
        rows = _apply_sampling(rows, input_sampling_strategy)
        return rows
    if input_jsonl:
        rows = _load_jsonl(input_jsonl)
        rows = _apply_sampling(rows, input_sampling_strategy)
        return rows
    raise ValueError("Either input_csv or input_jsonl must be provided.")


def _save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CoMEM retrieval dataset (MDS/JSONL) with EVA-CLIP.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_csv", default=None, help="EchoSight-style CSV input.")
    input_group.add_argument("--input_jsonl", default=None, help="Generic JSONL input.")

    parser.add_argument("--knowledge_base", required=True, help="Wiki KB json path used by EVA-CLIP retriever.")
    parser.add_argument("--faiss_root", required=True, help="Directory containing kb_index.faiss and kb_index_ids.pkl.")
    parser.add_argument(
        "--wiki_prf_test_dir",
        default=os.path.join(REPO_ROOT, "methods/code/Wiki-PRF"),
        help="Path for importing retriever.py.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--similar_num", type=int, default=10, help="Number of retrieval items in retrieval_info.")
    parser.add_argument("--retrieval_search_k", type=int, default=20, help="Raw top-k before de-dup.")
    parser.add_argument(
        "--max_passage_chars",
        type=int,
        default=0,
        help="Maximum chars for passage_content. Set 0 (or negative) to disable truncation.",
    )
    parser.add_argument("--pad_with_query", action="store_true", help="Pad retrieval_info with query image when fewer hits.")

    parser.add_argument("--image_root", required=True, help="Fallback generic image root.")
    parser.add_argument("--infoseek_image_root", default=None, help="Image root for dataset_name=infoseek.")
    parser.add_argument("--landmark_image_root", default=None, help="Image root for dataset_name=landmarks.")
    parser.add_argument("--inaturalist_image_root", default=None, help="Image root for dataset_name=inaturalist.")
    parser.add_argument(
        "--inat_mapping",
        default=os.path.join(REPO_ROOT, "data/images/echosight_inat_val_id2name.json"),
        help="val_id2name.json path.",
    )

    parser.add_argument("--data_id_field", default="data_id")
    parser.add_argument("--question_field", default="question")
    parser.add_argument("--image_field", default="image")
    parser.add_argument("--image_path_field", default="image_path")
    parser.add_argument(
        "--csv_question_type_filter",
        default="automatic,templated,multi_answer,String,Numerical,Time",
        help="Allowlist for CSV question_type. Empty means keep all.",
    )
    parser.add_argument("--input_sampling_strategy", default="all", help="all/first:100/end:100/random:10%%")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_format", choices=["mds", "jsonl", "both"], default="mds")
    parser.add_argument("--output_mds_dir", default=None, help="Output MDS directory.")
    parser.add_argument("--output_jsonl", default=None, help="Output JSONL path.")
    parser.add_argument("--overwrite_output", action="store_true", help="Remove output path if exists.")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.output_format in ("mds", "both") and not args.output_mds_dir:
        raise ValueError("--output_mds_dir is required when output_format is mds/both.")
    if args.output_format in ("jsonl", "both") and not args.output_jsonl:
        raise ValueError("--output_jsonl is required when output_format is jsonl/both.")

    sys.path.insert(0, args.wiki_prf_test_dir)
    from retriever import ClipRetriever  # type: ignore
    try:
        from answer_generator import reconstruct_wiki_article as reconstruct_wiki_article_fn  # type: ignore
    except Exception as exc:
        reconstruct_wiki_article_fn = None
        print(f"[WARN] Failed to import reconstruct_wiki_article, fallback will be used: {exc}")

    rows = _iter_rows(
        input_csv=args.input_csv,
        input_jsonl=args.input_jsonl,
        input_sampling_strategy=args.input_sampling_strategy,
        csv_question_type_filter=args.csv_question_type_filter,
    )
    print(f"Loaded {len(rows)} rows from input.")

    roots = Roots(
        image_root=args.image_root,
        infoseek_root=args.infoseek_image_root or args.image_root,
        landmark_root=args.landmark_image_root or args.image_root,
        inat_root=args.inaturalist_image_root or args.image_root,
    )
    inat_map = _load_inat_map(args.inat_mapping)

    retriever = ClipRetriever(device=f"cuda:{args.gpu_id}", model="eva-clip")
    retriever.load_knowledge_base(args.knowledge_base)
    retriever.load_faiss_index(_normalize_faiss_root(args.faiss_root))

    converted_rows: List[Dict[str, Any]] = []
    failed = 0
    for idx, sample in enumerate(tqdm(rows, desc="Building Retrieval Dataset")):
        try:
            image_path = _resolve_image_path(
                sample=sample,
                roots=roots,
                inat_map=inat_map,
                image_field=args.image_field,
                image_path_field=args.image_path_field,
            )
            query_image = Image.open(image_path).convert("RGB")
            raw_top = retriever.retrieve_image_faiss(query_image, top_k=max(args.retrieval_search_k, args.similar_num))
            top_items = _deduplicate_retrieval(raw_top, args.similar_num)

            retrieval_info: List[Dict[str, str]] = []
            for item in top_items:
                entry = item.get("kb_entry")
                if entry is None:
                    continue
                passage = _build_passage(
                    entry,
                    max_chars=args.max_passage_chars,
                    reconstruct_wiki_article_fn=reconstruct_wiki_article_fn,
                )

                retrieved_img = None
                if item.get("image_url"):
                    retrieved_img = _fetch_image_from_url(str(item["image_url"]))
                if retrieved_img is None:
                    entry_urls = getattr(entry, "image_urls", []) or []
                    for u in entry_urls[:3]:
                        retrieved_img = _fetch_image_from_url(str(u))
                        if retrieved_img is not None:
                            break
                if retrieved_img is None:
                    retrieved_img = query_image

                retrieval_info.append(
                    {
                        "passage_content": passage,
                        "image": _image_to_b64_jpeg(retrieved_img, max_side=512),
                    }
                )
                if len(retrieval_info) >= args.similar_num:
                    break

            if args.pad_with_query and len(retrieval_info) < args.similar_num:
                q_b64 = _image_to_b64_jpeg(query_image, max_side=512)
                while len(retrieval_info) < args.similar_num:
                    retrieval_info.append(
                        {
                            "passage_content": "",
                            "image": q_b64,
                        }
                    )

            data_id = str(sample.get(args.data_id_field) or f"custom_{idx:08d}")
            question = str(sample.get(args.question_field, "")).strip()
            converted_rows.append(
                {
                    "data_id": data_id,
                    "image_path": image_path,
                    "question": question,
                    "retrieval_info": retrieval_info,
                }
            )
        except Exception as exc:
            failed += 1
            print(f"[WARN] Skip sample idx={idx}, reason={exc}")

    print(f"Built {len(converted_rows)} rows, failed={failed}")

    if args.output_format in ("jsonl", "both"):
        if args.overwrite_output and os.path.exists(args.output_jsonl):
            os.remove(args.output_jsonl)
        _save_jsonl(args.output_jsonl, converted_rows)
        print(f"Saved JSONL: {args.output_jsonl}")

    if args.output_format in ("mds", "both"):
        if args.overwrite_output and os.path.exists(args.output_mds_dir):
            shutil.rmtree(args.output_mds_dir)
        os.makedirs(args.output_mds_dir, exist_ok=True)
        from streaming import MDSWriter

        columns = {
            "data_id": "str",
            "image_path": "str",
            "question": "str",
            "retrieval_info": "json",
        }
        with MDSWriter(out=args.output_mds_dir, columns=columns, compression=None) as out:
            for row in converted_rows:
                out.write(row)
        print(f"Saved MDS: {args.output_mds_dir}")


if __name__ == "__main__":
    main()
