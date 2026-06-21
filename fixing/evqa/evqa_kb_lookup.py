#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def resolve_entry(kb, url):
    if url in kb:
        return url, kb[url]
    if url.startswith("https://"):
        alt = "http://" + url[len("https://") :]
        if alt in kb:
            return alt, kb[alt]
    if url.startswith("http://"):
        alt = "https://" + url[len("http://") :]
        if alt in kb:
            return alt, kb[alt]
    return None, None


def trim_text(text, max_chars):
    if max_chars <= 0:
        return text
    if text is None:
        return text
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


def trim_sections(entry, max_sections, max_section_chars):
    if not entry:
        return entry
    out = dict(entry)
    if isinstance(out.get("section_titles"), list) and max_sections > 0:
        out["section_titles"] = out["section_titles"][:max_sections]
    if isinstance(out.get("section_texts"), list):
        texts = out["section_texts"]
        if max_sections > 0:
            texts = texts[:max_sections]
        if max_section_chars > 0:
            texts = [trim_text(t, max_section_chars) for t in texts]
        out["section_texts"] = texts
    if isinstance(out.get("sections"), list):
        sections = out["sections"]
        if max_sections > 0:
            sections = sections[:max_sections]
        trimmed = []
        for sec in sections:
            if isinstance(sec, dict):
                sec = dict(sec)
                if "text" in sec:
                    sec["text"] = trim_text(sec["text"], max_section_chars)
                if "content" in sec:
                    sec["content"] = trim_text(sec["content"], max_section_chars)
                if "paragraphs" in sec and isinstance(sec["paragraphs"], list):
                    sec["paragraphs"] = [
                        trim_text(p, max_section_chars) for p in sec["paragraphs"]
                    ]
            trimmed.append(sec)
        out["sections"] = trimmed
    return out


def load_urls_from_csv(path, url_column):
    urls = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = normalize_text(row.get(url_column))
            if url:
                urls.append(url)
    return urls


def main():
    parser = argparse.ArgumentParser(description="Lookup EVQA KB entry by URL.")
    parser.add_argument("--kb-path", required=True)
    parser.add_argument("--url", action="append", default=[], help="Wikipedia URL (repeatable).")
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--max-sections", type=int, default=0)
    parser.add_argument("--max-section-chars", type=int, default=0)
    args = parser.parse_args()

    urls = []
    if args.input_csv:
        urls.extend(load_urls_from_csv(args.input_csv, args.url_column))
    urls.extend([u for u in args.url if u])
    if not urls:
        print("No URLs provided. Use --url or --input-csv.", file=sys.stderr)
        sys.exit(1)

    with open(args.kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    urls = list(dict.fromkeys(urls))

    out_f = None
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        out_f = open(args.output_jsonl, "w", encoding="utf-8")

    for url in urls:
        resolved_url, entry = resolve_entry(kb, url)
        if entry is None:
            payload = {"url": url, "resolved_url": None, "found": False}
        else:
            trimmed = trim_sections(entry, args.max_sections, args.max_section_chars)
            payload = {"url": url, "resolved_url": resolved_url, "found": True, "data": trimmed}

        if out_f:
            out_f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        else:
            if args.pretty:
                print(json.dumps(payload, ensure_ascii=True, indent=2))
            else:
                print(json.dumps(payload, ensure_ascii=True))

    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
