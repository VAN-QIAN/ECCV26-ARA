#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LABELS = {"entailed", "contradicted", "not_supported"}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "by",
    "at",
    "from",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
}


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


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def tokenize(text, min_len=2):
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        if len(tok) < min_len:
            continue
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
    return set(tokens)


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


def build_messages(question, answers, evidence, section_title):
    system = (
        "You are a strict verifier for KB-VQA. "
        "Use only the evidence text. "
        "Decide if the evidence supports the answer to the question. "
        "Label must be one of: entailed, contradicted, not_supported. "
        "If any candidate answer is supported, label entailed. "
        "If evidence directly conflicts with the answer, label contradicted. "
        "If evidence is insufficient, label not_supported. "
        "Return JSON with keys: label, reason."
    )
    user = {
        "question": question,
        "candidate_answers": answers,
        "evidence_section_title": section_title,
        "evidence_section_text": evidence,
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
        tokens = [tokenize(t) for t in texts]
        payload = {"url": resolved_url, "titles": titles, "texts": texts, "tokens": tokens}
        self.cache[url] = payload
        return payload


def select_candidate_sections(sections, q_tokens, a_tokens, top_k, answer_weight, fallback_top_k):
    scored = []
    for idx, sec_tokens in enumerate(sections["tokens"]):
        if not sec_tokens:
            score = 0
        else:
            score = len(q_tokens & sec_tokens) + answer_weight * len(a_tokens & sec_tokens)
        scored.append((score, idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [sid for score, sid in scored if score > 0]
    if top_k and candidates:
        candidates = candidates[:top_k]
    if not candidates:
        fallback = list(range(min(fallback_top_k, len(scored))))
        return fallback, scored
    return candidates, scored


def main():
    parser = argparse.ArgumentParser(description="Judge InfoSeek grounding with LLM.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--kb-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--verbose-output", default="")
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
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=0,
        help="Truncate evidence if >0; 0 disables truncation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Candidate sections to judge; 0 means traverse all sections.",
    )
    parser.add_argument("--fallback-top-k", type=int, default=3)
    parser.add_argument("--answer-weight", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel LLM requests per question. Use >1 for concurrency.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    if args.provider in {"qwen", "qwen3"} and not args.api_base:
        args.api_base = os.environ.get("QWEN_API_BASE", "") or os.environ.get("OPENAI_API_BASE", "")
    if args.provider in {"qwen", "qwen3"} and not args.api_key:
        args.api_key = os.environ.get("QWEN_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

    if not args.api_base:
        print("Missing --api-base (or OPENAI_API_BASE/QWEN_API_BASE).", file=sys.stderr)
        sys.exit(1)
    if not args.api_key:
        print("Missing --api-key (or OPENAI_API_KEY/QWEN_API_KEY).", file=sys.stderr)
        sys.exit(1)

    kb = InfoSeekKnowledgeBase(args.kb_path)

    stats = {
        "total_questions": 0,
        "grounded_questions": 0,
        "not_grounded_questions": 0,
        "missing_questions": 0,
        "missing_entities": 0,
        "entities_total": 0,
        "labels": {"entailed": 0, "contradicted": 0, "not_supported": 0},
        "llm_calls": 0,
        "errors": 0,
        "missing_urls": [],
    }

    missing_urls = set()
    all_urls = set()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    if args.verbose_output:
        os.makedirs(os.path.dirname(args.verbose_output) or ".", exist_ok=True)
        verbose_f = open(args.verbose_output, "w", encoding="utf-8")
    else:
        verbose_f = None

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_csv, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader):
            if idx < args.start:
                continue
            if args.end is not None and idx >= args.end:
                break
            stats["total_questions"] += 1
            data_id = normalize_text(row.get(args.id_column)) or f"row_{idx}"
            url = normalize_text(row.get(args.url_column))
            all_urls.add(url)

            question = normalize_text(row.get(args.question_column))
            answer_raw = normalize_text(row.get(args.answer_column))
            answers = split_pipe_field(answer_raw)

            sections = kb.get_sections(url) if url else None
            if not sections:
                stats["missing_questions"] += 1
                missing_urls.add(url)
                row[args.evidence_column] = "0"
                row[args.evidence_id_column] = "0"
                row[args.evidence_title_column] = "0"
                writer.writerow(row)
                if verbose_f:
                    verbose_f.write(
                        json.dumps(
                            {
                                "data_id": data_id,
                                "label": "missing_kb",
                                "url": url,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                continue

            if args.top_k <= 0:
                candidate_ids = list(range(len(sections["texts"])))
                scored = []
            else:
                q_tokens = tokenize(question)
                a_tokens = tokenize(" ".join(answers))
                candidate_ids, scored = select_candidate_sections(
                    sections, q_tokens, a_tokens, args.top_k, args.answer_weight, args.fallback_top_k
                )

            found = False
            labels_seen = []
            verbose_candidates = []
            results = []

            def judge_section(sid):
                evidence = sections["texts"][sid] if sid < len(sections["texts"]) else ""
                title = sections["titles"][sid] if sid < len(sections["titles"]) else ""
                if not evidence:
                    return {
                        "section_id": sid,
                        "label": "not_supported",
                        "reason": "empty_section",
                        "section_title": title,
                        "called": False,
                        "error": False,
                        "evidence": "",
                    }
                if args.max_evidence_chars > 0 and len(evidence) > args.max_evidence_chars:
                    evidence = evidence[: args.max_evidence_chars] + " ..."
                messages = build_messages(question, answers, evidence, title)
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
                except Exception as exc:
                    return {
                        "section_id": sid,
                        "label": "not_supported",
                        "reason": f"error:{exc}",
                        "section_title": title,
                        "called": True,
                        "error": True,
                        "evidence": evidence,
                    }

                parsed = extract_json_block(content)
                if not parsed or parsed.get("label") not in LABELS:
                    parsed = {"label": "not_supported", "reason": "unparseable_or_missing"}

                return {
                    "section_id": sid,
                    "label": parsed.get("label"),
                    "reason": normalize_text(parsed.get("reason")),
                    "section_title": title,
                    "called": True,
                    "error": False,
                    "evidence": evidence,
                }

            if args.workers > 1:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    results = list(executor.map(judge_section, candidate_ids))
            else:
                results = [judge_section(sid) for sid in candidate_ids]

            entailed_candidates = []
            for res in results:
                if res["called"]:
                    stats["llm_calls"] += 1
                if res["error"]:
                    stats["errors"] += 1
                labels_seen.append(res["label"])
                verbose_candidates.append(
                    {
                        "section_id": res["section_id"],
                        "label": res["label"],
                        "reason": res["reason"],
                        "section_title": res["section_title"],
                    }
                )
                if res["label"] == "entailed":
                    entailed_candidates.append(res)

            if entailed_candidates:
                entailed_candidates.sort(key=lambda r: r["section_id"])
                row[args.evidence_column] = "|".join([c["evidence"] for c in entailed_candidates])
                row[args.evidence_id_column] = "|".join(
                    [str(c["section_id"]) for c in entailed_candidates]
                )
                row[args.evidence_title_column] = "|".join(
                    [c["section_title"] or "0" for c in entailed_candidates]
                )
                found = True

            if args.sleep:
                time.sleep(args.sleep)

            if found:
                stats["grounded_questions"] += 1
                stats["labels"]["entailed"] += 1
                row_label = "entailed"
            else:
                row_label = "not_supported"
                if labels_seen and all(l == "contradicted" for l in labels_seen):
                    row_label = "contradicted"
                stats["not_grounded_questions"] += 1
                stats["labels"][row_label] += 1
                row[args.evidence_column] = "0"
                row[args.evidence_id_column] = "0"
                row[args.evidence_title_column] = "0"

            writer.writerow(row)
            if verbose_f:
                verbose_f.write(
                    json.dumps(
                        {
                            "data_id": data_id,
                            "label": row_label,
                            "url": url,
                            "candidates": verbose_candidates,
                            "selected_section_id": row[args.evidence_id_column],
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            if args.sleep and found:
                time.sleep(args.sleep)

    stats["entities_total"] = len(all_urls)
    stats["missing_entities"] = len(missing_urls)
    stats["missing_urls"] = sorted(u for u in missing_urls if u)

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    if verbose_f:
        verbose_f.close()


if __name__ == "__main__":
    main()
