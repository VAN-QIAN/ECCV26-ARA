#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request


TAGS = {
    "Q_clear",
    "Missing_Attribute_Constraint",
    "Missing_Temporal_Scope",
    "Missing_Spatial_Reference",
}


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def split_answer_candidates(text, multi_delim="&&"):
    if text is None:
        return []
    parts = [p.strip() for p in str(text).split("|")]
    seen = set()
    out = []
    for part in parts:
        if not part:
            continue
        norm = normalize_candidate(part, multi_delim=multi_delim)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(part)
    return out


def normalize_candidate(candidate, multi_delim="&&"):
    parts = [p.strip() for p in str(candidate).split(multi_delim) if p.strip()]
    seen = set()
    norm_parts = []
    for part in parts:
        key = normalize_text(part).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        norm_parts.append(key)
    if not norm_parts:
        return ""
    if len(norm_parts) == 1:
        return norm_parts[0]
    return multi_delim.join(sorted(norm_parts))


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


def split_sentences(text):
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def answer_leaks_in_question(question, answers, multi_delim="&&"):
    q_norm = normalize_text(question).lower()
    if not q_norm:
        return False
    for ans in answers:
        if not ans:
            continue
        parts = [p.strip() for p in str(ans).split(multi_delim) if p.strip()]
        for part in parts:
            part_norm = normalize_text(part).lower()
            if not part_norm:
                continue
            if part_norm in q_norm:
                return True
    return False


class EvqaKnowledgeBase:
    def __init__(self, path):
        self.path = path
        self.data = self._load_json(path)

    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_sections(self, entry):
        sections = entry.get("sections")
        if isinstance(sections, list) and sections:
            if isinstance(sections[0], dict):
                texts = []
                for sec in sections:
                    if isinstance(sec, str):
                        texts.append(sec)
                    elif isinstance(sec, dict):
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

    def _extract_titles(self, entry, count):
        titles = entry.get("section_titles")
        if isinstance(titles, list):
            return ["" if t is None else str(t) for t in titles]
        sections = entry.get("sections")
        if isinstance(sections, list):
            title_list = []
            for sec in sections:
                if isinstance(sec, dict) and "title" in sec:
                    title_list.append(sec["title"])
                else:
                    title_list.append("")
            return title_list
        if count is not None:
            return ["" for _ in range(count)]
        return []

    def get_sections(self, url, section_ids):
        if not url or not section_ids:
            return [], {"missing_url": False, "missing_ids": section_ids}
        entry = self.data.get(url)
        if not entry:
            return [], {"missing_url": True, "missing_ids": section_ids}
        sections = self._extract_sections(entry)
        if not sections:
            return [], {"missing_url": False, "missing_ids": section_ids}
        titles = self._extract_titles(entry, len(sections))
        selected = []
        missing = []
        for sid in section_ids:
            if sid < 0 or sid >= len(sections):
                missing.append(sid)
                continue
            selected.append(
                {
                    "section_id": sid,
                    "title": titles[sid] if sid < len(titles) else "",
                    "text": sections[sid],
                }
            )
        return selected, {"missing_url": False, "missing_ids": missing}


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(template, question, annotated_answer, optional_allowed):
    prompt = template
    prompt = prompt.replace("{QUESTION}", question)
    prompt = prompt.replace("{ANNOTATED_ANSWER}", annotated_answer)
    prompt = prompt.replace("{OPTIONAL_ALLOWED_ANSWERS}", optional_allowed)
    return prompt


def build_messages(prompt_text):
    return [
        {"role": "system", "content": "You are a strict auditor. Return JSON only."},
        {"role": "user", "content": prompt_text},
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


def load_seen_ids(path):
    seen = set()
    if not path or not os.path.exists(path):
        return seen
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
            if data_id:
                seen.add(data_id)
    return seen


def get_fieldnames(input_path):
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or []


def iter_rows(input_path):
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            yield idx, row


def resolve_question_column(fieldnames, requested):
    if not fieldnames:
        return ""
    if requested and requested != "auto":
        if requested in fieldnames:
            return requested
        print(
            f"Warning: question column '{requested}' not found; falling back.",
            file=sys.stderr,
        )
    if "question_original" in fieldnames:
        return "question_original"
    if "question" in fieldnames:
        return "question"
    return ""


def default_prompt_path():
    return os.path.join(os.path.dirname(__file__), "evqa_question_fix_prompt.txt")


def default_kb_path():
    return os.path.join(os.path.dirname(__file__), "encyclopedic_kb_wiki.json")


def default_output_jsonl_path():
    return os.path.join("results_question_fix", "evqa_question_fix.jsonl")


def default_output_csv_path():
    return os.path.join("results_question_fix", "evqa_question_fix.csv")


def parse_response(content):
    parsed = extract_json_block(content)
    if not parsed:
        return "Q_clear", "unparseable_response", ""
    tag = normalize_text(parsed.get("question_clarity_tag"))
    reason = normalize_text(parsed.get("clarity_reason"))
    suggested = normalize_text(parsed.get("suggested_question"))
    if tag not in TAGS:
        tag = "Q_clear"
    return tag, reason, suggested


def select_short_evidence(sections, annotated_answer, max_chars):
    answer_text = normalize_text(annotated_answer).lower()
    if answer_text:
        answer_text = answer_text.split("&&")[0].strip().lower()
    for section in sections:
        text = normalize_text(section.get("text"))
        if not text:
            continue
        sentences = split_sentences(text)
        if not sentences:
            continue
        if answer_text:
            for idx, sent in enumerate(sentences):
                if answer_text in sent.lower():
                    chosen = sent
                    if idx + 1 < len(sentences):
                        chosen = chosen + " " + sentences[idx + 1]
                    if max_chars > 0 and len(chosen) > max_chars:
                        chosen = chosen[:max_chars].rstrip() + " ..."
                    return chosen, section
        fallback = " ".join(sentences[:2])
        if max_chars > 0 and len(fallback) > max_chars:
            fallback = fallback[:max_chars].rstrip() + " ..."
        return fallback, section
    return "", None


def main():
    parser = argparse.ArgumentParser(
        description="Audit EVQA question clarity and suggest fixes."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", default=default_output_jsonl_path())
    parser.add_argument("--output-csv", default=default_output_csv_path())
    parser.add_argument("--prompt-path", default=default_prompt_path())
    parser.add_argument(
        "--question-column",
        default="question",
        help="Column name for question text; use 'auto' to prefer question_original.",
    )
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--evidence-column", default="evidence")
    parser.add_argument("--section-id-column", default="evidence_section_id")
    parser.add_argument("--section-title-column", default="evidence_section_title")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--kb-path", default=default_kb_path())
    parser.add_argument("--no-kb", action="store_true")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--short-evidence-max-chars", type=int, default=320)
    parser.add_argument("--multi-answer-delim", default="&&")
    parser.add_argument("--split-answers", action="store_true", default=None)
    parser.add_argument("--no-split-answers", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--dump-raw",
        default="",
        help="Write raw LLM responses to this JSONL file for debugging.",
    )
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

    fieldnames = get_fieldnames(args.input_csv)
    question_column = resolve_question_column(fieldnames, args.question_column)
    if not question_column:
        print("Missing question column.", file=sys.stderr)
        sys.exit(1)
    if args.id_column not in fieldnames:
        print(f"Missing {args.id_column} column; run evqa_add_data_id.py first.", file=sys.stderr)
        sys.exit(1)

    split_answers = args.split_answers
    if split_answers is None:
        split_answers = args.answer_column == "answer"
    if args.no_split_answers:
        split_answers = False

    kb = None
    if not args.no_kb and args.kb_path:
        kb = EvqaKnowledgeBase(args.kb_path)

    prompt_template = load_prompt(args.prompt_path)

    raw_f = None
    if args.dump_raw:
        os.makedirs(os.path.dirname(args.dump_raw) or ".", exist_ok=True)
        raw_f = open(args.dump_raw, "w", encoding="utf-8")

    seen = load_seen_ids(args.output_jsonl) if args.resume else set()
    processed = 0
    written = 0

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_mode = "a" if args.resume else "w"
    csv_mode = "a" if args.resume else "w"

    with open(args.output_jsonl, out_mode, encoding="utf-8") as out_f, open(
        args.output_csv, csv_mode, encoding="utf-8", newline=""
    ) as out_csv:
        writer = None
        extra_cols = [
            "question_clarity_tag",
            "clarity_reason",
            "suggested_question",
            "short_evidence",
            "short_evidence_source",
            "short_evidence_section_id",
            "short_evidence_section_title",
        ]
        write_header = True
        if args.resume and os.path.exists(args.output_csv):
            write_header = os.path.getsize(args.output_csv) == 0

        for idx, row in iter_rows(args.input_csv):
            if idx < args.start:
                continue
            if args.end is not None and idx >= args.end:
                break
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1

            data_id = normalize_text(row.get(args.id_column))
            if not data_id:
                print(
                    f"Missing {args.id_column} at row {idx}; run evqa_add_data_id.py first.",
                    file=sys.stderr,
                )
                continue
            if data_id in seen:
                continue

            question = normalize_text(row.get(question_column))
            answer_raw = normalize_text(row.get(args.answer_column))
            evidence_raw = normalize_text(row.get(args.evidence_column))
            if evidence_raw == "0":
                evidence_raw = ""

            if split_answers:
                answers = split_answer_candidates(answer_raw, multi_delim=args.multi_answer_delim)
            else:
                answers = [answer_raw] if answer_raw else []
            annotated_answer = answers[0] if answers else ""
            optional_allowed = answers[1:] if len(answers) > 1 else []
            optional_allowed_json = json.dumps(optional_allowed, ensure_ascii=True)

            prompt_text = build_prompt(
                prompt_template, question, annotated_answer, optional_allowed_json
            )
            messages = build_messages(prompt_text)

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
                error_obj = {
                    "data_id": data_id,
                    "error": str(exc),
                    "model": args.model,
                }
                out_f.write(json.dumps(error_obj, ensure_ascii=True) + "\n")
                out_f.flush()
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            if raw_f is not None:
                raw_f.write(
                    json.dumps(
                        {
                            "data_id": data_id,
                            "question": question,
                            "raw_response": content,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            clarity_tag, clarity_reason, suggested_question = parse_response(content)
            if suggested_question and answer_leaks_in_question(
                suggested_question, [annotated_answer] + optional_allowed, args.multi_answer_delim
            ):
                suggested_question = ""
                if clarity_reason:
                    clarity_reason = clarity_reason + " | removed_suggested_question_due_to_leak"
                else:
                    clarity_reason = "removed_suggested_question_due_to_leak"

            short_evidence = ""
            short_evidence_source = "none"
            short_evidence_section_id = ""
            short_evidence_section_title = ""
            kb_meta = {"missing_url": False, "missing_ids": []}

            if clarity_tag == "Q_clear" and not evidence_raw and kb is not None:
                section_ids = parse_section_ids(row.get(args.section_id_column))
                sections, kb_meta = kb.get_sections(
                    normalize_text(row.get(args.url_column)), section_ids
                )
                short_evidence, section = select_short_evidence(
                    sections, annotated_answer, args.short_evidence_max_chars
                )
                if short_evidence:
                    short_evidence_source = "kb"
                    if section:
                        short_evidence_section_id = str(section.get("section_id", ""))
                        short_evidence_section_title = normalize_text(section.get("title"))

            result = {
                "data_id": data_id,
                "question_clarity_tag": clarity_tag,
                "clarity_reason": clarity_reason,
                "suggested_question": suggested_question,
                "short_evidence": short_evidence,
                "short_evidence_source": short_evidence_source,
                "short_evidence_section_id": short_evidence_section_id,
                "short_evidence_section_title": short_evidence_section_title,
                "model": args.model,
                "kb_missing_url": kb_meta.get("missing_url", False),
                "kb_missing_ids": kb_meta.get("missing_ids", []),
            }
            out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
            out_f.flush()

            row = dict(row)
            row["question_clarity_tag"] = clarity_tag
            row["clarity_reason"] = clarity_reason
            row["suggested_question"] = suggested_question
            row["short_evidence"] = short_evidence
            row["short_evidence_source"] = short_evidence_source
            row["short_evidence_section_id"] = short_evidence_section_id
            row["short_evidence_section_title"] = short_evidence_section_title

            if writer is None:
                fieldnames = list(row.keys())
                for col in extra_cols:
                    if col not in fieldnames:
                        fieldnames.append(col)
                writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
            writer.writerow(row)
            written += 1

            if args.sleep:
                time.sleep(args.sleep)

    if raw_f is not None:
        raw_f.close()

    print(f"Processed {processed} rows, wrote {written} outputs.")


if __name__ == "__main__":
    main()
