#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional


METHOD_ROOT = Path(__file__).resolve().parents[1]
CAMERA_READY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CSV = CAMERA_READY_ROOT / "data/ground_truth/evqa_fixed_final_check_Feb12.csv"
DEFAULT_IMAGE_ROOT = CAMERA_READY_ROOT / "data/images/reflectiva_evqa_inference_images"
DEFAULT_INAT_MAP = CAMERA_READY_ROOT / "data/images/echosight_inat_val_id2name.json"
DEFAULT_OUTPUT = METHOD_ROOT / "data_evqa/test_one_hop_Feb14.json"
ALLOWED_TYPES = {"automatic", "templated", "multi_answer", "infoseek"}


def iter_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row:
                continue
            question_type = row.get("question_type", "").strip()
            if question_type in ALLOWED_TYPES:
                yield row


def landmarks_candidate_paths(image_root: Path, image_id: str):
    if len(image_id) < 3:
        return
    base = image_root / "Google_Landmarks_v2" / image_id[0] / image_id[1] / image_id[2]
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        yield base / f"{image_id}{ext}"


def resolve_related_image(
    image_root: Path,
    dataset_name: str,
    image_id: str,
    inat_id2name: Optional[dict],
) -> Optional[str]:
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "inaturalist":
        if not inat_id2name or image_id not in inat_id2name:
            return None
        rel_path = Path("iNaturalist_2021") / inat_id2name[image_id]
        if (image_root / rel_path).exists():
            return rel_path.as_posix()
        return None

    if dataset_name_lower == "landmarks":
        for candidate in landmarks_candidate_paths(image_root, image_id):
            if candidate.exists():
                return candidate.relative_to(image_root).as_posix()
        return None

    return None


def convert(args):
    inat_id2name = None
    if args.inat_map.exists():
        inat_id2name = json.loads(args.inat_map.read_text())

    records = []
    for idx, row in enumerate(iter_rows(args.csv)):
        image_ids = row["dataset_image_ids"].split("|")
        primary_image = image_ids[0].strip()
        related_rel = resolve_related_image(
            args.image_root,
            row["dataset_name"],
            primary_image,
            inat_id2name,
        )
        if related_rel is None:
            print(f"[WARN] Missing image for {row.get('data_id', primary_image)}")
            continue

        data_id = row.get("data_id") or f"E-VQA_{idx}"
        records.append(
            {
                "wikipedia_title": row["wikipedia_title"],
                "wikipedia_url": row["wikipedia_url"],
                "question": row["question"],
                "question_type": row["question_type"],
                "answer": row["answer"],
                "evidence": row.get("evidence") or row.get("final_evidence", ""),
                "evidence_section_id": row["evidence_section_id"],
                "evidence_section_title": row["evidence_section_title"],
                "dataset_name": row["dataset_name"],
                "encyclopedic_vqa_split": row["encyclopedic_vqa_split"],
                "dataset_image_ids": row["dataset_image_ids"],
                "related_images": related_rel,
                "retrieval": [],
                "unique_id": f"echosight_test_{idx:05d}",
                "data_id": data_id,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False))
    print(f"Wrote {len(records)} samples to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--inat_map", type=Path, default=DEFAULT_INAT_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
