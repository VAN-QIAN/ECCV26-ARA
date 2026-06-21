#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import sys


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def parse_section_ids(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split("|") if p.strip()]
    ids = []
    for part in parts:
        try:
            ids.append(int(float(part)))
        except ValueError:
            continue
    return ids


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    n = len(sorted_vals)
    pos = (p / 100.0) * (n - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_vals[lower]
    frac = pos - lower
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def summarize_lengths(lengths):
    if not lengths:
        return {}
    vals = sorted(lengths)
    total = sum(vals)
    count = len(vals)
    return {
        "count": count,
        "min": vals[0],
        "max": vals[-1],
        "mean": total / count,
        "p50": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
    }


def bucketize(lengths, buckets):
    counts = {f"<= {b}": 0 for b in buckets}
    counts["> {last}".format(last=buckets[-1])] = 0
    for val in lengths:
        placed = False
        for b in buckets:
            if val <= b:
                counts[f"<= {b}"] += 1
                placed = True
                break
        if not placed:
            counts["> {last}".format(last=buckets[-1])] += 1
    return counts


class EvqaKnowledgeBase:
    def __init__(self, path):
        self.path = path
        self.data = self._load_json(path)

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_entry(self, url):
        if url in self.data:
            return url, self.data[url]
        if url.startswith("https://"):
            alt = "http://" + url[len("https://") :]
            if alt in self.data:
                return alt, self.data[alt]
        if url.startswith("http://"):
            alt = "https://" + url[len("http://") :]
            if alt in self.data:
                return alt, self.data[alt]
        return None, None

    def _extract_sections(self, entry):
        sections = entry.get("sections")
        if isinstance(sections, list) and sections:
            if isinstance(sections[0], dict):
                texts = []
                for sec in sections:
                    if isinstance(sec, dict):
                        if "text" in sec:
                            texts.append(sec["text"])
                        elif "content" in sec:
                            texts.append(sec["content"])
                        elif "paragraphs" in sec and isinstance(sec["paragraphs"], list):
                            texts.append("\n".join(sec["paragraphs"]))
                        else:
                            texts.append(json.dumps(sec, ensure_ascii=True))
                    else:
                        texts.append(str(sec))
                return texts
            return [str(sec) for sec in sections]
        section_texts = entry.get("section_texts")
        if isinstance(section_texts, list):
            return ["" if sec is None else str(sec) for sec in section_texts]
        return None

    def get_sections(self, url):
        _, entry = self._resolve_entry(url)
        if not entry:
            return None
        return self._extract_sections(entry)


def main():
    parser = argparse.ArgumentParser(description="Analyze EVQA KB section length stats.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--kb-path", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--row-output", default="")
    parser.add_argument("--section-output", default="")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--section-id-column", default="evidence_section_id")
    parser.add_argument("--question-type-column", default="question_type")
    parser.add_argument("--skip-question-types", default="")
    parser.add_argument("--notes-column", default="")
    parser.add_argument("--notes-contains", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    kb = EvqaKnowledgeBase(args.kb_path)
    skip_types = {t.strip() for t in args.skip_question_types.split(",") if t.strip()}

    section_lengths = []
    row_lengths = []
    missing_urls = 0
    missing_sections = 0
    rows = 0
    rows_with_sections = 0

    row_records = []
    section_records = []

    notes_contains = args.notes_contains.lower().strip()
    notes_re = re.compile(re.escape(notes_contains)) if notes_contains else None

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if args.limit and rows >= args.limit:
                break
            qtype = normalize_text(row.get(args.question_type_column))
            if qtype in skip_types:
                continue
            if args.notes_column and notes_re:
                notes = normalize_text(row.get(args.notes_column)).lower()
                if not notes_re.search(notes):
                    continue
            rows += 1

            url = normalize_text(row.get(args.url_column))
            section_ids = parse_section_ids(row.get(args.section_id_column))
            if not url or not section_ids:
                missing_sections += 1
                continue
            sections = kb.get_sections(url)
            if not sections:
                missing_urls += 1
                continue

            rows_with_sections += 1
            per_row_lengths = []
            for sid in section_ids:
                if sid < 0 or sid >= len(sections):
                    continue
                text = sections[sid] or ""
                length = len(text)
                section_lengths.append(length)
                per_row_lengths.append(length)
                if args.section_output:
                    section_records.append(
                        {
                            "row_index": idx,
                            "url": url,
                            "section_id": sid,
                            "length": length,
                        }
                    )

            if per_row_lengths:
                row_total = sum(per_row_lengths)
                row_lengths.append(row_total)
                if args.row_output:
                    row_records.append(
                        {
                            "row_index": idx,
                            "url": url,
                            "sections": "|".join([str(s) for s in section_ids]),
                            "section_count": len(per_row_lengths),
                            "row_total_length": row_total,
                            "row_max_length": max(per_row_lengths),
                        }
                    )

    stats = {
        "rows_considered": rows,
        "rows_with_sections": rows_with_sections,
        "missing_urls": missing_urls,
        "missing_sections": missing_sections,
        "section_length_stats": summarize_lengths(section_lengths),
        "row_total_length_stats": summarize_lengths(row_lengths),
        "section_length_buckets": bucketize(
            section_lengths, [500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
        ),
        "row_total_length_buckets": bucketize(
            row_lengths, [500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
        ),
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=True, indent=2)
    else:
        print(json.dumps(stats, ensure_ascii=True, indent=2))

    if args.row_output and row_records:
        os.makedirs(os.path.dirname(args.row_output) or ".", exist_ok=True)
        with open(args.row_output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_records[0].keys())
            writer.writeheader()
            writer.writerows(row_records)

    if args.section_output and section_records:
        os.makedirs(os.path.dirname(args.section_output) or ".", exist_ok=True)
        with open(args.section_output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=section_records[0].keys())
            writer.writeheader()
            writer.writerows(section_records)


if __name__ == "__main__":
    main()
