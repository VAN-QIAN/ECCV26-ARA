#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request

LABELS = {"entailed", "contradicted", "not_supported", "missing_kb"}


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def split_pipe_field(text):
    if text is None:
        return []
    parts = [p.strip() for p in str(text).split("|")]
    seen = set()
    out = []
    for p in parts:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def clean_evidence(text):
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_ids(value):
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


def extract_json_block(text):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned)
        cleaned = cleaned.strip("` \n")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_messages(question, answers, section_text, section_title):
    system = (
        "You are an evidence extractor for KB-VQA. "
        "Use only the section text. "
        "Return the minimal span (1-2 sentences) that supports ANY candidate answer. "
        "If no support, return an empty string. "
        "Return JSON with keys: evidence."
    )
    user = {
        "question": question,
        "candidate_answers": answers,
        "section_title": section_title,
        "section_text": section_text,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
    ]


def call_openai_compat(api_base, api_key, model, messages, temperature, max_tokens):
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def call_llm(provider, api_base, api_key, model, messages, temperature, max_tokens):
    if provider in {"openai", "openai_compat", "qwen", "qwen3"}:
        return call_openai_compat(api_base, api_key, model, messages, temperature, max_tokens)
    raise ValueError(f"Unsupported provider: {provider}")


class InfoSeekKnowledgeBase:
    def __init__(self, path):
        self.path = path
        self.data = self._load_json(path)
        self.cache = {}

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
            titles = []
            texts = []
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
        section_texts = entry.get("section_texts")
        section_titles = entry.get("section_titles")
        if isinstance(section_texts, list):
            texts = ["" if sec is None else str(sec) for sec in section_texts]
            titles = []
            if isinstance(section_titles, list):
                titles = ["" if t is None else str(t) for t in section_titles]
            return titles, texts
        return [], []

    def get_sections(self, url):
        if url in self.cache:
            return self.cache[url]
        resolved_url, entry = self._resolve_entry(url)
        if entry is None:
            self.cache[url] = None
            return None
        titles, texts = self._extract_sections(entry)
        payload = {"url": resolved_url, "titles": titles, "texts": texts}
        self.cache[url] = payload
        return payload


def load_verbose(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data_id = obj.get("data_id")
            if not data_id:
                continue
            out[data_id] = obj
    return out


def main():
    parser = argparse.ArgumentParser(description="Regenerate InfoSeek CSV from verbose JSONL.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--verbose-jsonl", required=True)
    parser.add_argument("--kb-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--not-supported-output", default="")
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--evidence-column", default="evidence")
    parser.add_argument("--evidence-id-column", default="evidence_section_id")
    parser.add_argument("--evidence-title-column", default="evidence_section_title")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-evidence-chars", type=int, default=800)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    if not args.no_llm:
        if args.provider in {"qwen", "qwen3"} and not args.api_base:
            args.api_base = os.environ.get("QWEN_API_BASE", "") or os.environ.get(
                "OPENAI_API_BASE", ""
            )
        if args.provider in {"qwen", "qwen3"} and not args.api_key:
            args.api_key = os.environ.get("QWEN_API_KEY", "") or os.environ.get(
                "OPENAI_API_KEY", ""
            )
        if not args.api_base:
            print("Missing --api-base (or OPENAI_API_BASE/QWEN_API_BASE).", file=sys.stderr)
            sys.exit(1)
        if not args.api_key:
            print("Missing --api-key (or OPENAI_API_KEY/QWEN_API_KEY).", file=sys.stderr)
            sys.exit(1)

    kb = InfoSeekKnowledgeBase(args.kb_path)
    verbose_map = load_verbose(args.verbose_jsonl)

    stats = {
        "total_rows": 0,
        "updated_rows": 0,
        "entailed_rows": 0,
        "not_entailed_rows": 0,
        "missing_verbose_rows": 0,
        "missing_kb_rows": 0,
        "entailed_missing_evidence_rows": 0,
        "llm_calls": 0,
        "llm_errors": 0,
    }

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    not_supported_f = None
    not_supported_writer = None
    if args.not_supported_output:
        os.makedirs(os.path.dirname(args.not_supported_output) or ".", exist_ok=True)
        not_supported_f = open(args.not_supported_output, "w", encoding="utf-8", newline="")

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_csv, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        if not_supported_f:
            not_supported_writer = csv.DictWriter(not_supported_f, fieldnames=reader.fieldnames)
            not_supported_writer.writeheader()

        for row in reader:
            stats["total_rows"] += 1
            data_id = normalize_text(row.get(args.id_column))
            verbose = verbose_map.get(data_id)

            if not verbose:
                stats["missing_verbose_rows"] += 1
                row[args.evidence_column] = "0"
                row[args.evidence_id_column] = "0"
                row[args.evidence_title_column] = "0"
                writer.writerow(row)
                if not_supported_writer:
                    not_supported_writer.writerow(row)
                continue

            label = normalize_text(verbose.get("label"))
            if label not in LABELS:
                label = "not_supported"

            if label != "entailed":
                stats["not_entailed_rows"] += 1
                row[args.evidence_column] = "0"
                row[args.evidence_id_column] = "0"
                row[args.evidence_title_column] = "0"
                writer.writerow(row)
                if not_supported_writer:
                    not_supported_writer.writerow(row)
                continue

            selected_ids = parse_ids(verbose.get("selected_section_id"))
            if not selected_ids:
                stats["entailed_missing_evidence_rows"] += 1
                row[args.evidence_column] = "0"
                row[args.evidence_id_column] = "0"
                row[args.evidence_title_column] = "0"
                writer.writerow(row)
                continue

            url = normalize_text(row.get(args.url_column))
            sections = kb.get_sections(url) if url else None
            if not sections:
                stats["missing_kb_rows"] += 1
                stats["entailed_missing_evidence_rows"] += 1
                row[args.evidence_column] = "0"
                row[args.evidence_id_column] = "0"
                row[args.evidence_title_column] = "0"
                writer.writerow(row)
                continue

            question = normalize_text(row.get(args.question_column))
            answer_raw = normalize_text(row.get(args.answer_column))
            answers = [a for a in split_pipe_field(answer_raw) if a]

            evidence_list = []
            title_list = []
            id_list = []

            for sid in selected_ids:
                if sid < 0 or sid >= len(sections["texts"]):
                    continue
                section_text = sections["texts"][sid]
                section_title = sections["titles"][sid] if sid < len(sections["titles"]) else ""

                evidence = section_text
                if not args.no_llm:
                    messages = build_messages(question, answers, section_text, section_title)
                    try:
                        content = call_llm(
                            args.provider,
                            args.api_base,
                            args.api_key,
                            args.model,
                            messages,
                            args.temperature,
                            args.max_tokens,
                        )
                        stats["llm_calls"] += 1
                        parsed = extract_json_block(content) or {}
                        evidence = normalize_text(parsed.get("evidence"))
                        if not evidence:
                            evidence = section_text
                    except Exception:
                        stats["llm_errors"] += 1
                        evidence = section_text

                evidence = clean_evidence(evidence)
                if args.max_evidence_chars > 0 and len(evidence) > args.max_evidence_chars:
                    evidence = evidence[: args.max_evidence_chars].rstrip()

                if evidence:
                    evidence_list.append(evidence)
                    title_list.append(clean_evidence(section_title) or "0")
                    id_list.append(str(sid))

                if args.sleep:
                    time.sleep(args.sleep)

            if not evidence_list and selected_ids:
                for sid in selected_ids:
                    if sid < 0 or sid >= len(sections["texts"]):
                        continue
                    section_text = clean_evidence(sections["texts"][sid])
                    if not section_text:
                        continue
                    section_title = (
                        clean_evidence(sections["titles"][sid])
                        if sid < len(sections["titles"])
                        else "0"
                    )
                    evidence_list.append(section_text)
                    title_list.append(section_title or "0")
                    id_list.append(str(sid))

            if not evidence_list:
                stats["entailed_missing_evidence_rows"] += 1
                row[args.evidence_column] = "0"
                row[args.evidence_id_column] = "0"
                row[args.evidence_title_column] = "0"
            else:
                stats["entailed_rows"] += 1
                stats["updated_rows"] += 1
                row[args.evidence_column] = "|".join(evidence_list)
                row[args.evidence_id_column] = "|".join(id_list)
                row[args.evidence_title_column] = "|".join(title_list)

            writer.writerow(row)

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)
    if not_supported_f:
        not_supported_f.close()


if __name__ == "__main__":
    main()
