#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from typing import Dict, List

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None


DEFAULT_LANDMARKS_IMAGE_ROOT = "/data/qianMa/EchoSight/E-VQA/landmark"
DEFAULT_INAT_IMAGE_ROOT = "/data/qianMa/EchoSight/images"
DEFAULT_INAT_ID2NAME_PATH = "/data/qianMa/EchoSight/images/val_id2name.json"


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def parse_first_image_id(raw):
    text = normalize_text(raw)
    if not text:
        return ""
    parts = [p.strip() for p in re.split(r"[|,;]", text) if p.strip()]
    if not parts:
        return ""
    return parts[0]


def safe_token(text, fallback="na"):
    token = normalize_text(text)
    if not token:
        return fallback
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    return token[:120] if token else fallback


def load_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def build_data_id_to_image_meta(source_rows):
    mapping = {}
    for row in source_rows:
        data_id = normalize_text(row.get("data_id"))
        image_id = parse_first_image_id(row.get("dataset_image_ids"))
        dataset_name = normalize_text(row.get("dataset_name"))
        if data_id and image_id:
            mapping[data_id] = (image_id, dataset_name)
    return mapping


def normalize_dataset_name(dataset_name):
    ds = normalize_text(dataset_name).lower()
    if ds == "landmark":
        return "landmarks"
    return ds


def load_inat_id2name(path):
    mapping: Dict[str, str] = {}
    p = normalize_text(path)
    if not p:
        return mapping
    if not os.path.exists(p):
        raise FileNotFoundError(f"iNaturalist id2name file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for k, v in raw.items():
        sk = normalize_text(k)
        sv = normalize_text(v)
        if sk and sv:
            mapping[sk] = sv
    return mapping


def get_landmarks_image_path(image_id, landmarks_image_root):
    image_id = normalize_text(image_id)
    if len(image_id) < 3:
        return ""

    candidates = [
        os.path.join(landmarks_image_root, image_id[0], image_id[1], image_id[2], image_id + ".jpg"),
        os.path.join(landmarks_image_root, image_id[0], image_id[1], image_id[2], image_id + ".jpeg"),
        os.path.join(landmarks_image_root, image_id + ".jpg"),
        os.path.join(landmarks_image_root, image_id + ".jpeg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def get_inaturalist_image_path(image_id, inat_image_root, inat_id2name):
    image_id = normalize_text(image_id)
    if not image_id:
        return ""

    rel_path = normalize_text(inat_id2name.get(image_id, ""))
    candidates = []
    if rel_path:
        candidates.append(os.path.join(inat_image_root, rel_path))
    candidates.extend(
        [
            os.path.join(inat_image_root, image_id + ".jpg"),
            os.path.join(inat_image_root, image_id + ".jpeg"),
            os.path.join(inat_image_root, image_id + ".png"),
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def get_evqa_image_path(image_id, dataset_name, inat_image_root, landmarks_image_root, inat_id2name):
    ds = normalize_dataset_name(dataset_name)
    if ds == "landmarks":
        return get_landmarks_image_path(image_id, landmarks_image_root)
    if ds == "inaturalist":
        return get_inaturalist_image_path(image_id, inat_image_root, inat_id2name)
    return ""


def load_rgb_image(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def resize_keep_aspect(img, target_height):
    if target_height <= 0:
        return img
    if img.height == target_height:
        return img
    new_width = max(1, int(round(img.width * target_height / float(img.height))))
    return img.resize((new_width, target_height), Image.Resampling.LANCZOS)


def parse_rgb_color(raw):
    text = normalize_text(raw)
    if not text:
        return (255, 255, 255)
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB color must have format R,G,B")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB color values must be integers") from exc
    if any(v < 0 or v > 255 for v in values):
        raise argparse.ArgumentTypeError("RGB color values must be in [0, 255]")
    return values


def compose_side_by_side(left_img, right_img, target_height, gap=0, bg_color=(0, 0, 0)):
    left = resize_keep_aspect(left_img, target_height)
    right = resize_keep_aspect(right_img, target_height)
    width = left.width + gap + right.width
    canvas = Image.new("RGB", (width, target_height), color=bg_color)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_pair_image(anchor_img, pair_img_path, control_mode, blank_color):
    if control_mode == "distractor":
        if not pair_img_path:
            return None, "missing_pair_image"
        return load_rgb_image(pair_img_path), "distractor"
    if control_mode == "blank":
        if not pair_img_path:
            return None, "missing_pair_image"
        with Image.open(pair_img_path) as pair_meta:
            pair_meta = ImageOps.exif_transpose(pair_meta)
            width, height = pair_meta.size
        return Image.new("RGB", (width, height), color=blank_color), "blank"
    if control_mode == "double-anchor":
        return anchor_img.copy(), "double_anchor"
    return None, f"unsupported_control_mode:{control_mode}"


def build_composite(
    anchor_img_path,
    pair_img_path,
    target_side,
    output_path,
    target_height,
    gap,
    overwrite,
    control_mode,
    blank_color,
):
    if not anchor_img_path:
        return False, "missing_input_image"
    if (not overwrite) and os.path.exists(output_path):
        return True, "exists"

    anchor_img = load_rgb_image(anchor_img_path)
    pair_img, pair_status = make_pair_image(
        anchor_img=anchor_img,
        pair_img_path=pair_img_path,
        control_mode=control_mode,
        blank_color=blank_color,
    )
    if pair_img is None:
        return False, pair_status

    if normalize_text(target_side).lower() == "left":
        left_img, right_img = anchor_img, pair_img
    else:
        left_img, right_img = pair_img, anchor_img

    composite = compose_side_by_side(
        left_img=left_img,
        right_img=right_img,
        target_height=target_height,
        gap=gap,
        bg_color=(0, 0, 0),
    )
    composite.save(output_path, format="JPEG", quality=95)
    return True, f"saved_{pair_status}"


def add_output_fields(base_fields):
    extra = [
        "anchor_image_id",
        "anchor_image_path",
        "method1_pair_image_id",
        "method1_pair_image_path",
        "method2_pair_image_id",
        "method2_pair_image_path",
        "method1_composite_image_path",
        "method2_composite_image_path",
        "method1_image_status",
        "method2_image_status",
        "composite_control_mode",
        "blank_color_rgb",
        "image_error",
    ]
    fields = list(base_fields)
    for f in extra:
        if f not in fields:
            fields.append(f)
    return fields


def resolve_meta(data_to_img, data_id):
    if data_id in data_to_img:
        return data_to_img[data_id]
    return "", ""


def main():
    parser = argparse.ArgumentParser(
        description="Build side-by-side composite images for EVQA challenging query rows."
    )
    parser.add_argument(
        "--challenging-csv",
        required=True,
        help="CSV created by evqa_generate_challenging_queries.py",
    )
    parser.add_argument(
        "--source-csv",
        required=True,
        help=(
            "Source EVQA CSV containing data_id, dataset_image_ids, and optional "
            "dataset_name (e.g., evqa_final_check_Feb12.csv)."
        ),
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV with composite image path columns.")
    parser.add_argument("--output-image-dir", required=True, help="Folder to save generated composite images.")
    parser.add_argument(
        "--inat-image-root",
        default=DEFAULT_INAT_IMAGE_ROOT,
        help="iNaturalist root folder (used with --inat-id2name mapping).",
    )
    parser.add_argument(
        "--landmarks-image-root",
        default=DEFAULT_LANDMARKS_IMAGE_ROOT,
        help="GLD/landmarks root folder. Uses a/b/c/image_id.jpg indexing.",
    )
    parser.add_argument(
        "--inat-id2name",
        default=DEFAULT_INAT_ID2NAME_PATH,
        help="JSON mapping from iNaturalist image_id to relative file path (e.g. val_id2name.json).",
    )
    parser.add_argument("--height", type=int, default=512, help="Composite image height.")
    parser.add_argument("--gap", type=int, default=0, help="Gap (pixels) between two images.")
    parser.add_argument(
        "--control-mode",
        choices=["distractor", "blank", "double-anchor"],
        default="distractor",
        help=(
            "Composite branch to build. 'distractor' preserves the original augmentation; "
            "'blank' replaces each distractor with a same-size blank panel; "
            "'double-anchor' duplicates the anchor image as the other panel."
        ),
    )
    parser.add_argument(
        "--blank-color",
        type=parse_rgb_color,
        default=(255, 255, 255),
        help="RGB color for --control-mode blank, formatted as R,G,B.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing composite images.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows (0 means all).")
    args = parser.parse_args()

    if Image is None:
        raise RuntimeError("Pillow is required. Install with: pip install pillow")

    challenging_rows, challenging_fields = load_rows(args.challenging_csv)
    source_rows, _ = load_rows(args.source_csv)
    data_to_img = build_data_id_to_image_meta(source_rows)
    inat_id2name = load_inat_id2name(args.inat_id2name)

    method1_dir = os.path.join(args.output_image_dir, "method1")
    method2_dir = os.path.join(args.output_image_dir, "method2")
    ensure_dir(method1_dir)
    ensure_dir(method2_dir)

    out_fields = add_output_fields(challenging_fields)
    out_rows = []

    m1_ok = 0
    m2_ok = 0
    skipped = 0

    for idx, row in enumerate(challenging_rows):
        if args.limit and idx >= args.limit:
            break

        anchor_data_id = normalize_text(row.get("anchor_data_id"))
        method1_data_id = normalize_text(row.get("method1_pair_data_id"))
        method2_data_id = normalize_text(row.get("method2_pair_data_id"))
        target_side = normalize_text(row.get("target_side")).lower()
        if target_side not in {"left", "right"}:
            target_side = "left"

        anchor_image_id, anchor_dataset_name = resolve_meta(data_to_img, anchor_data_id)
        method1_image_id, method1_dataset_name = resolve_meta(data_to_img, method1_data_id)
        method2_image_id, method2_dataset_name = resolve_meta(data_to_img, method2_data_id)

        anchor_image_path = get_evqa_image_path(
            image_id=anchor_image_id,
            dataset_name=anchor_dataset_name,
            inat_image_root=args.inat_image_root,
            landmarks_image_root=args.landmarks_image_root,
            inat_id2name=inat_id2name,
        )
        method1_pair_image_path = get_evqa_image_path(
            image_id=method1_image_id,
            dataset_name=method1_dataset_name,
            inat_image_root=args.inat_image_root,
            landmarks_image_root=args.landmarks_image_root,
            inat_id2name=inat_id2name,
        )
        method2_pair_image_path = get_evqa_image_path(
            image_id=method2_image_id,
            dataset_name=method2_dataset_name,
            inat_image_root=args.inat_image_root,
            landmarks_image_root=args.landmarks_image_root,
            inat_id2name=inat_id2name,
        )

        base_name = f"{idx:04d}_{safe_token(anchor_data_id)}"
        if args.control_mode == "distractor":
            method1_filename = f"{base_name}__m1__{safe_token(method1_data_id)}.jpg"
            method2_filename = f"{base_name}__m2__{safe_token(method2_data_id)}.jpg"
        else:
            mode_token = safe_token(args.control_mode)
            method1_filename = f"{base_name}__m1__{mode_token}__{safe_token(method1_data_id)}.jpg"
            method2_filename = f"{base_name}__m2__{mode_token}__{safe_token(method2_data_id)}.jpg"
        method1_file = os.path.join(method1_dir, method1_filename)
        method2_file = os.path.join(method2_dir, method2_filename)

        image_errors: List[str] = []

        ok1, status1 = build_composite(
            anchor_img_path=anchor_image_path,
            pair_img_path=method1_pair_image_path,
            target_side=target_side,
            output_path=method1_file,
            target_height=args.height,
            gap=args.gap,
            overwrite=args.overwrite,
            control_mode=args.control_mode,
            blank_color=args.blank_color,
        )
        if ok1:
            m1_ok += 1
        else:
            image_errors.append(f"method1:{status1}")

        ok2, status2 = build_composite(
            anchor_img_path=anchor_image_path,
            pair_img_path=method2_pair_image_path,
            target_side=target_side,
            output_path=method2_file,
            target_height=args.height,
            gap=args.gap,
            overwrite=args.overwrite,
            control_mode=args.control_mode,
            blank_color=args.blank_color,
        )
        if ok2:
            m2_ok += 1
        else:
            image_errors.append(f"method2:{status2}")

        if image_errors:
            skipped += 1

        out_row = dict(row)
        out_row["anchor_image_id"] = anchor_image_id
        out_row["anchor_image_path"] = anchor_image_path
        out_row["method1_pair_image_id"] = method1_image_id
        out_row["method1_pair_image_path"] = method1_pair_image_path
        out_row["method2_pair_image_id"] = method2_image_id
        out_row["method2_pair_image_path"] = method2_pair_image_path
        out_row["method1_composite_image_path"] = method1_file if ok1 else ""
        out_row["method2_composite_image_path"] = method2_file if ok2 else ""
        out_row["method1_image_status"] = status1
        out_row["method2_image_status"] = status2
        out_row["composite_control_mode"] = args.control_mode
        out_row["blank_color_rgb"] = ",".join(str(v) for v in args.blank_color)
        out_row["image_error"] = " | ".join(image_errors)
        out_rows.append(out_row)

    ensure_dir(os.path.dirname(args.output_csv) or ".")
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Rows processed: {len(out_rows)}")
    print(f"Control mode: {args.control_mode}")
    print(f"Method1 composites OK: {m1_ok}")
    print(f"Method2 composites OK: {m2_ok}")
    print(f"Rows with any image issue: {skipped}")
    print(f"Composite image dir: {args.output_image_dir}")


if __name__ == "__main__":
    main()
