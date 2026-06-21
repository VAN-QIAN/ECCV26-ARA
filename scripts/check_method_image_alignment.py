#!/usr/bin/env python3
"""Check whether ReflectiVA uses the same sample images as other methods.

The checker resolves, for each CSV row, the image that would be loaded by:

- ReflectiVA
- EchoSight / IBA
- Wiki-PRF

It compares resolved files with byte hashes and, when Pillow is available,
pixel hashes after RGB conversion. Pixel hashes make the check robust to
different filenames or container metadata for the same visible image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSVS = [
    REPO_ROOT / "data/ground_truth/evqa_fixed_final_check_Feb12.csv",
    REPO_ROOT / "data/ground_truth/evqa_unfixed_test_with_id.csv",
    REPO_ROOT / "data/ground_truth/infoseek_fixed_final_recheck_Feb7.csv",
    REPO_ROOT / "data/ground_truth/infoseek_unfixed_subset.csv",
]
DEFAULT_WIKIPRF_CONFIGS = {
    ("evqa", "fixed"): REPO_ROOT / "methods/code/Wiki-PRF/configs/evqa_fixed.yaml",
    ("evqa", "unfixed"): REPO_ROOT / "methods/code/Wiki-PRF/configs/evqa_unfixed.yaml",
    ("infoseek", "fixed"): REPO_ROOT / "methods/code/Wiki-PRF/configs/infoseek_fixed.yaml",
    ("infoseek", "unfixed"): REPO_ROOT / "methods/code/Wiki-PRF/configs/infoseek_unfixed.yaml",
}


@dataclass
class ImageInfo:
    path: Optional[str]
    exists: bool
    byte_sha256: Optional[str] = None
    pixel_sha256: Optional[str] = None
    size: Optional[Tuple[int, int]] = None
    mode: Optional[str] = None
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check ReflectiVA sample-image alignment against EchoSight and Wiki-PRF."
    )
    parser.add_argument(
        "--csv",
        dest="csv_paths",
        action="append",
        type=Path,
        help="CSV file to check. Can be repeated. Defaults to all four GT CSVs.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/image_alignment")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--skip-pixel-hash",
        action="store_true",
        help="Only compare file bytes; faster but less robust to metadata differences.",
    )
    parser.add_argument(
        "--reflectiva-evqa-manifest",
        type=Path,
        default=Path("/data/qianMa/ReflectiVA/data_evqa/test_one_hop_Feb14.json"),
        help="ReflectiVA EVQA JSON manifest containing data_id -> related_images.",
    )
    parser.add_argument(
        "--reflectiva-evqa-root",
        type=Path,
        default=REPO_ROOT / "data/images/reflectiva_evqa_inference_images",
    )
    parser.add_argument(
        "--reflectiva-infoseek-root",
        type=Path,
        default=REPO_ROOT / "data/images/reflectiva_infoseek_val_image",
    )
    parser.add_argument(
        "--echosight-inat-root",
        type=Path,
        default=REPO_ROOT / "data/images/echosight_images",
    )
    parser.add_argument(
        "--echosight-landmark-root",
        type=Path,
        default=REPO_ROOT / "data/images/evqa_landmark_images",
    )
    parser.add_argument(
        "--echosight-infoseek-root",
        type=Path,
        default=REPO_ROOT / "data/images/infoseek_val_images",
    )
    parser.add_argument(
        "--inat-id2name",
        type=Path,
        default=REPO_ROOT / "data/images/echosight_images/val_id2name.json",
    )
    parser.add_argument(
        "--wikiprf-config",
        type=Path,
        default=None,
        help=(
            "Optional Wiki-PRF YAML/JSONL data config for a single CSV. "
            "By default it is inferred from dataset/split."
        ),
    )
    parser.add_argument(
        "--sample-mismatches",
        type=int,
        default=20,
        help="Number of mismatch examples to include in each summary JSON.",
    )
    return parser.parse_args()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_image_id(row: Dict[str, Any]) -> str:
    image_ids = safe_text(row.get("dataset_image_ids") or row.get("image_id"))
    return image_ids.split("|")[0].strip() if image_ids else ""


def normalize_dataset_name(row: Dict[str, Any], csv_path: Path) -> str:
    dataset_name = safe_text(row.get("dataset_name")).lower()
    if dataset_name:
        return dataset_name
    name = csv_path.name.lower()
    if "infoseek" in name:
        return "infoseek"
    return ""


def infer_dataset_and_split(csv_path: Path) -> Tuple[str, str]:
    name = csv_path.name.lower()
    dataset = "infoseek" if "infoseek" in name else "evqa"
    split = "unfixed" if "unfixed" in name else "fixed"
    return dataset, split


def load_json_records_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.values()
    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        data_id = safe_text(record.get("data_id") or record.get("id") or record.get("unique_id"))
        if data_id:
            out[data_id] = record
    return out


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def resolve_json_path_from_simple_yaml(path: Path) -> Optional[Path]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"json_path\s*:\s*(.+)", text)
    if not match:
        return None
    value = match.group(1).strip().strip("'\"")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    return candidate


def load_wikiprf_records_by_id(config_or_jsonl: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if config_or_jsonl is None or not config_or_jsonl.exists():
        return {}
    if config_or_jsonl.suffix.lower() in {".yaml", ".yml"}:
        json_path = resolve_json_path_from_simple_yaml(config_or_jsonl)
        if json_path is None or not json_path.exists():
            return {}
    else:
        json_path = config_or_jsonl

    out: Dict[str, Dict[str, Any]] = {}
    for record in read_jsonl(json_path):
        data_id = safe_text(record.get("data_id") or record.get("id") or record.get("unique_id"))
        if data_id:
            out[data_id] = record
    return out


def candidate_existing_path(base: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = base / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def resolve_echosight_path(
    row: Dict[str, Any],
    *,
    dataset_name: str,
    image_id: str,
    inat_id2name: Dict[str, str],
    inat_root: Path,
    landmark_root: Path,
    infoseek_root: Path,
) -> Optional[Path]:
    if not image_id:
        return None
    if dataset_name == "inaturalist":
        rel = inat_id2name.get(image_id)
        return inat_root / rel if rel else None
    if dataset_name in {"landmarks", "landmark"}:
        if len(image_id) < 3:
            return None
        return landmark_root / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg"
    if dataset_name == "infoseek":
        return candidate_existing_path(infoseek_root, image_id)
    return None


def resolve_reflectiva_path(
    row: Dict[str, Any],
    *,
    dataset: str,
    dataset_name: str,
    image_id: str,
    data_id: str,
    evqa_manifest: Dict[str, Dict[str, Any]],
    evqa_root: Path,
    infoseek_root: Path,
    inat_id2name: Dict[str, str],
) -> Optional[Path]:
    if dataset == "infoseek" or dataset_name == "infoseek":
        return candidate_existing_path(infoseek_root, image_id)

    manifest_record = evqa_manifest.get(data_id)
    related = safe_text(manifest_record.get("related_images")) if manifest_record else ""
    if related:
        return evqa_root / related

    # Fallback: reproduce ReflectiVA/data_evqa/adapt_data.py rules.
    if dataset_name == "inaturalist":
        rel = inat_id2name.get(image_id)
        return evqa_root / "iNaturalist_2021" / rel if rel else None
    if dataset_name in {"landmarks", "landmark"} and len(image_id) >= 3:
        base = evqa_root / "Google_Landmarks_v2" / image_id[0] / image_id[1] / image_id[2]
        return candidate_existing_path(base, image_id)
    return None


def resolve_wikiprf_path(
    row: Dict[str, Any],
    *,
    data_id: str,
    wikiprf_records: Dict[str, Dict[str, Any]],
    fallback_echosight_path: Optional[Path],
) -> Optional[Path]:
    record = wikiprf_records.get(data_id)
    if record:
        raw_path = safe_text(record.get("image_path") or record.get("image"))
        if raw_path:
            return Path(raw_path)
    return fallback_echosight_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Optional[Path], *, skip_pixel_hash: bool) -> ImageInfo:
    if path is None:
        return ImageInfo(path=None, exists=False, error="unresolved")
    info = ImageInfo(path=str(path), exists=path.exists())
    if not path.exists():
        info.error = "missing"
        return info
    try:
        info.byte_sha256 = sha256_file(path)
    except Exception as exc:  # pragma: no cover - file-system dependent
        info.error = f"byte_hash_error: {exc}"
        return info

    if skip_pixel_hash or Image is None:
        return info
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            info.size = tuple(rgb.size)
            info.mode = rgb.mode
            info.pixel_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
    except Exception as exc:  # pragma: no cover - corrupt image dependent
        info.error = f"pixel_hash_error: {exc}"
    return info


def compare_images(left: ImageInfo, right: ImageInfo) -> Dict[str, Any]:
    if not left.path or not right.path:
        return {"same": False, "reason": "unresolved"}
    if not left.exists or not right.exists:
        return {"same": False, "reason": "missing"}
    if left.byte_sha256 and left.byte_sha256 == right.byte_sha256:
        return {"same": True, "reason": "byte_sha256_match"}
    if left.pixel_sha256 and left.pixel_sha256 == right.pixel_sha256:
        return {"same": True, "reason": "pixel_sha256_match"}
    return {
        "same": False,
        "reason": "hash_mismatch",
        "left_size": left.size,
        "right_size": right.size,
    }


def as_jsonable_info(info: ImageInfo) -> Dict[str, Any]:
    return {
        "path": info.path,
        "exists": info.exists,
        "byte_sha256": info.byte_sha256,
        "pixel_sha256": info.pixel_sha256,
        "size": list(info.size) if info.size else None,
        "mode": info.mode,
        "error": info.error,
    }


def read_csv_rows(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row["_row_index"] = str(idx)
            rows.append(row)
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def update_counter(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def check_csv(path: Path, args: argparse.Namespace, inat_id2name: Dict[str, str]) -> Dict[str, Any]:
    dataset, split = infer_dataset_and_split(path)
    evqa_manifest = load_json_records_by_id(args.reflectiva_evqa_manifest)
    wikiprf_config = args.wikiprf_config or DEFAULT_WIKIPRF_CONFIGS.get((dataset, split))
    wikiprf_records = load_wikiprf_records_by_id(wikiprf_config)

    records_dir = args.output_dir / path.stem
    records_dir.mkdir(parents=True, exist_ok=True)
    detail_path = records_dir / "image_alignment_records.jsonl"
    mismatch_path = records_dir / "image_alignment_mismatches.csv"
    summary_path = records_dir / "summary.json"

    counters: Dict[str, int] = {"total": 0}
    mismatch_rows: List[Dict[str, Any]] = []
    sample_mismatches: List[Dict[str, Any]] = []

    with detail_path.open("w", encoding="utf-8") as detail_f:
        for row in read_csv_rows(path, args.max_samples):
            counters["total"] += 1
            data_id = safe_text(row.get("data_id")) or f"row_{row['_row_index']}"
            dataset_name = normalize_dataset_name(row, path)
            image_id = first_image_id(row)

            echosight_path = resolve_echosight_path(
                row,
                dataset_name=dataset_name,
                image_id=image_id,
                inat_id2name=inat_id2name,
                inat_root=args.echosight_inat_root,
                landmark_root=args.echosight_landmark_root,
                infoseek_root=args.echosight_infoseek_root,
            )
            reflectiva_path = resolve_reflectiva_path(
                row,
                dataset=dataset,
                dataset_name=dataset_name,
                image_id=image_id,
                data_id=data_id,
                evqa_manifest=evqa_manifest,
                evqa_root=args.reflectiva_evqa_root,
                infoseek_root=args.reflectiva_infoseek_root,
                inat_id2name=inat_id2name,
            )
            wikiprf_path = resolve_wikiprf_path(
                row,
                data_id=data_id,
                wikiprf_records=wikiprf_records,
                fallback_echosight_path=echosight_path,
            )

            infos = {
                "reflectiva": inspect_image(reflectiva_path, skip_pixel_hash=args.skip_pixel_hash),
                "echosight": inspect_image(echosight_path, skip_pixel_hash=args.skip_pixel_hash),
                "wikiprf": inspect_image(wikiprf_path, skip_pixel_hash=args.skip_pixel_hash),
            }
            comparisons = {
                "reflectiva_vs_echosight": compare_images(infos["reflectiva"], infos["echosight"]),
                "reflectiva_vs_wikiprf": compare_images(infos["reflectiva"], infos["wikiprf"]),
                "echosight_vs_wikiprf": compare_images(infos["echosight"], infos["wikiprf"]),
            }

            for method, info in infos.items():
                update_counter(counters, f"{method}_resolved" if info.path else f"{method}_unresolved")
                update_counter(counters, f"{method}_exists" if info.exists else f"{method}_missing")

            row_has_mismatch = False
            for name, comparison in comparisons.items():
                update_counter(counters, f"{name}_same" if comparison["same"] else f"{name}_different")
                if not comparison["same"] and name.startswith("reflectiva_vs"):
                    row_has_mismatch = True

            record = {
                "data_id": data_id,
                "row_index": int(row["_row_index"]),
                "dataset": dataset,
                "split": split,
                "dataset_name": dataset_name,
                "image_id": image_id,
                "images": {name: as_jsonable_info(info) for name, info in infos.items()},
                "comparisons": comparisons,
            }
            detail_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if row_has_mismatch:
                mismatch_row = {
                    "data_id": data_id,
                    "row_index": row["_row_index"],
                    "dataset_name": dataset_name,
                    "image_id": image_id,
                    "reflectiva_path": infos["reflectiva"].path or "",
                    "echosight_path": infos["echosight"].path or "",
                    "wikiprf_path": infos["wikiprf"].path or "",
                    "reflectiva_vs_echosight": comparisons["reflectiva_vs_echosight"]["reason"],
                    "reflectiva_vs_wikiprf": comparisons["reflectiva_vs_wikiprf"]["reason"],
                }
                mismatch_rows.append(mismatch_row)
                if len(sample_mismatches) < args.sample_mismatches:
                    sample_mismatches.append(record)

    if mismatch_rows:
        with mismatch_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(mismatch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mismatch_rows)
    else:
        mismatch_path.write_text("", encoding="utf-8")

    summary = {
        "csv": str(path),
        "dataset": dataset,
        "split": split,
        "wikiprf_config": str(wikiprf_config) if wikiprf_config else None,
        "reflectiva_evqa_manifest": str(args.reflectiva_evqa_manifest),
        "detail_path": str(detail_path),
        "mismatch_path": str(mismatch_path),
        "counters": counters,
        "sample_mismatches": sample_mismatches,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    csv_paths = args.csv_paths or DEFAULT_CSVS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inat_id2name: Dict[str, str] = {}
    if args.inat_id2name.exists():
        inat_id2name = json.loads(args.inat_id2name.read_text(encoding="utf-8"))

    all_summaries = []
    for csv_path in csv_paths:
        summary = check_csv(csv_path, args, inat_id2name)
        all_summaries.append(summary)
        counters = summary["counters"]
        print(
            f"{csv_path}: total={counters.get('total', 0)} "
            f"ReflectiVA-vs-EchoSight different="
            f"{counters.get('reflectiva_vs_echosight_different', 0)} "
            f"ReflectiVA-vs-Wiki-PRF different="
            f"{counters.get('reflectiva_vs_wikiprf_different', 0)}"
        )

    aggregate_path = args.output_dir / "summary.json"
    aggregate_path.write_text(
        json.dumps({"summaries": all_summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved aggregate summary to {aggregate_path}")


if __name__ == "__main__":
    main()
