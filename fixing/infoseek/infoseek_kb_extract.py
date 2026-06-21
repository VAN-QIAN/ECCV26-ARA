#!/usr/bin/env python3
import argparse
import csv
import json
import os


def load_kb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_sections(entry):
    sections = entry.get("sections")
    if isinstance(sections, list) and sections:
        if isinstance(sections[0], dict):
            texts = []
            titles = []
            for sec in sections:
                if isinstance(sec, dict):
                    titles.append(sec.get("title", ""))
                    if "text" in sec:
                        texts.append(sec["text"])
                    elif "content" in sec:
                        texts.append(sec["content"])
                    elif "paragraphs" in sec and isinstance(sec["paragraphs"], list):
                        texts.append("\n".join(sec["paragraphs"]))
                    else:
                        texts.append(json.dumps(sec, ensure_ascii=True))
                else:
                    titles.append("")
                    texts.append(str(sec))
            return titles, texts
        return [], [str(sec) for sec in sections]

    section_texts = entry.get("section_texts")
    section_titles = entry.get("section_titles")
    if isinstance(section_texts, list):
        texts = ["" if sec is None else str(sec) for sec in section_texts]
        titles = []
        if isinstance(section_titles, list):
            titles = ["" if t is None else str(t) for t in section_titles]
        return titles, texts
    return [], []


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


def main():
    parser = argparse.ArgumentParser(description="Extract a subset KB for InfoSeek.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--kb-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", default="")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--title-column", default="wikipedia_title")
    args = parser.parse_args()

    urls = []
    url_to_title = {}
    total_questions = 0

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_questions += 1
            url = (row.get(args.url_column) or "").strip()
            title = (row.get(args.title_column) or "").strip()
            if url:
                urls.append(url)
                if url not in url_to_title:
                    url_to_title[url] = title

    unique_urls = list(dict.fromkeys(urls))

    kb = load_kb(args.kb_path)
    subset = {}

    missing_questions = 0
    missing_urls = set()
    url_cache = {}

    for url in unique_urls:
        resolved_url, entry = resolve_entry(kb, url)
        if entry is None:
            missing_urls.add(url)
            continue
        titles, texts = extract_sections(entry)
        subset[url] = {
            "title": entry.get("title", url_to_title.get(url, "")),
            "url": url,
            "section_titles": titles,
            "section_texts": texts,
        }
        url_cache[url] = resolved_url

    for url in urls:
        if url in missing_urls:
            missing_questions += 1

    stats = {
        "total_questions": total_questions,
        "missing_questions": missing_questions,
        "total_entities": len(unique_urls),
        "missing_entities": len(missing_urls),
        "matched_entities": len(unique_urls) - len(missing_urls),
        "missing_urls": sorted(missing_urls),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=True)

    if args.stats_output:
        os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=True, indent=2)

    print(
        "Done. Questions:",
        total_questions,
        "Missing questions:",
        missing_questions,
        "Entities:",
        len(unique_urls),
        "Missing entities:",
        len(missing_urls),
    )


if __name__ == "__main__":
    main()
