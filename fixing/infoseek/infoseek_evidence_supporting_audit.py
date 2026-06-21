#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request


EVIDENCE_TAGS = {"E_supporting", "E_unsupporting"}
MATCH_TAGS = {"Match", "NoMatch", "NoExtraction"}


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


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


def is_zero_field(value):
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if not parts:
        return True
    return all(p == "0" for p in parts)


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


def split_paragraphs(text):
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def decode_unicode_escapes(text):
    if text is None:
        return ""
    text = str(text)

    def replace_u(match):
        return chr(int(match.group(1), 16))

    def replace_U(match):
        return chr(int(match.group(1), 16))

    text = re.sub(r"\\u([0-9a-fA-F]{4})", replace_u, text)
    text = re.sub(r"\\U([0-9a-fA-F]{8})", replace_U, text)
    return text


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


class InfoSeekKnowledgeBase:
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

    def get_section_texts(self, url, section_ids):
        if not url or not section_ids:
            return [], {"missing_url": False, "missing_ids": section_ids}
        _, entry = self._resolve_entry(url)
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


def build_prompt(template, question, annotated_answer, optional_allowed, evidence):
    prompt = template
    prompt = prompt.replace("{QUESTION}", question)
    prompt = prompt.replace("{ANNOTATED_ANSWER}", annotated_answer)
    prompt = prompt.replace("{OPTIONAL_ALLOWED_ANSWERS}", optional_allowed)
    prompt = prompt.replace("{EVIDENCE}", evidence)
    return prompt


def build_messages(prompt_text):
    return [
        {
            "role": "system",
            "content": "You are a strict auditor. Return JSON only.",
        },
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


def resolve_qtype_column(fieldnames, requested):
    if not fieldnames:
        return ""
    if requested and requested != "auto":
        if requested in fieldnames:
            return requested
        print(
            f"Warning: question type column '{requested}' not found; falling back.",
            file=sys.stderr,
        )
    if "question_type_qtype" in fieldnames:
        return "question_type_qtype"
    if "question_type" in fieldnames:
        return "question_type"
    return ""


def default_prompt_path():
    return os.path.join(os.path.dirname(__file__), "infoseek_evidence_supporting_prompt.txt")


def default_kb_path():
    return os.path.join(os.path.dirname(__file__), "infoseek_kb_subset.json")


def default_output_jsonl_path():
    return os.path.join("results_evidence_supporting", "infoseek_evidence_supporting.jsonl")


def default_output_csv_path():
    return os.path.join("results_evidence_supporting", "infoseek_evidence_supporting.csv")


def parse_response(content):
    parsed = extract_json_block(content)
    if not parsed:
        return "E_unsupporting", None, "NoExtraction", "unparseable_response"
    evidence_tag = normalize_text(parsed.get("evidence_sufficiency_tag"))
    extraction_answer = parsed.get("extraction_answer")
    matches = normalize_text(parsed.get("matches_annotated_answer"))
    explanation = normalize_text(parsed.get("explanation"))

    if evidence_tag not in EVIDENCE_TAGS:
        evidence_tag = "E_unsupporting"
    if matches not in MATCH_TAGS:
        matches = "NoExtraction" if evidence_tag == "E_unsupporting" else "NoMatch"

    if evidence_tag == "E_unsupporting":
        matches = "NoExtraction"
        extraction_answer = None
    elif matches == "NoExtraction":
        extraction_answer = None

    if extraction_answer is not None:
        extraction_answer = normalize_text(extraction_answer)
        if not extraction_answer:
            extraction_answer = None

    return evidence_tag, extraction_answer, matches, explanation


def audit_and_parse(
    prompt_template,
    question,
    annotated_answer,
    optional_allowed_json,
    evidence_text,
    provider,
    api_base,
    api_key,
    model,
    temperature,
    max_tokens,
    max_evidence_chars,
):
    if evidence_text is None:
        evidence_text = ""
    if max_evidence_chars > 0 and len(evidence_text) > max_evidence_chars:
        evidence_text = evidence_text[:max_evidence_chars] + " ..."
    prompt_text = build_prompt(
        prompt_template, question, annotated_answer, optional_allowed_json, evidence_text
    )
    messages = build_messages(prompt_text)
    content = call_llm(
        provider,
        api_base,
        api_key,
        model,
        messages,
        temperature,
        max_tokens,
    )
    evidence_tag, extraction_answer, matches, explanation = parse_response(content)
    return evidence_tag, extraction_answer, matches, explanation, evidence_text, content


def main():
    parser = argparse.ArgumentParser(
        description="Audit InfoSeek evidence sufficiency and answer match with LLM."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", default=default_output_jsonl_path())
    parser.add_argument("--output-csv", default=default_output_csv_path())
    parser.add_argument("--prompt-path", default=default_prompt_path())
    parser.add_argument(
        "--question-column",
        default="auto",
        help="Column name for question text; use 'auto' to prefer question_original.",
    )
    parser.add_argument(
        "--question-type-column",
        default="auto",
        help="Column for question type; use 'auto' to prefer question_type_qtype.",
    )
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--evidence-column", default="evidence")
    parser.add_argument("--section-id-column", default="evidence_section_id")
    parser.add_argument("--section-title-column", default="evidence_section_title")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument(
        "--skip-question-types",
        default="",
        help="Comma-separated question_type values to skip.",
    )
    parser.add_argument("--kb-path", default=default_kb_path())
    parser.add_argument("--no-kb", action="store_true")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-evidence-chars", type=int, default=0)
    parser.add_argument("--multi-answer-delim", default="&&")
    parser.add_argument("--split-answers", action="store_true", default=None)
    parser.add_argument("--no-split-answers", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--include-question",
        dest="include_question",
        action="store_true",
        default=True,
        help="Include question in output (default).",
    )
    parser.add_argument(
        "--no-include-question",
        dest="include_question",
        action="store_false",
        help="Omit question from output.",
    )
    parser.add_argument(
        "--include-evidence",
        dest="include_evidence",
        action="store_true",
        default=True,
        help="Include evidence_used in output (default).",
    )
    parser.add_argument(
        "--no-include-evidence",
        dest="include_evidence",
        action="store_false",
        help="Omit evidence_used from output.",
    )
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
    qtype_column = resolve_qtype_column(fieldnames, args.question_type_column)
    if not question_column:
        print("Missing question column.", file=sys.stderr)
        sys.exit(1)
    if args.id_column not in fieldnames:
        print(f"Missing {args.id_column} column.", file=sys.stderr)
        sys.exit(1)

    split_answers = args.split_answers
    if split_answers is None:
        split_answers = args.answer_column == "answer"
    if args.no_split_answers:
        split_answers = False

    skip_types = {t.strip().lower() for t in args.skip_question_types.split(",") if t.strip()}

    kb = None
    if not args.no_kb and args.kb_path:
        kb = InfoSeekKnowledgeBase(args.kb_path)

    prompt_template = load_prompt(args.prompt_path)

    raw_f = None
    if args.dump_raw:
        os.makedirs(os.path.dirname(args.dump_raw) or ".", exist_ok=True)
        raw_f = open(args.dump_raw, "w", encoding="utf-8")

    seen = load_seen_ids(args.output_jsonl) if args.resume else set()
    processed = 0
    written = 0
    skipped_all_zero = 0

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_mode = "a" if args.resume else "w"
    csv_mode = "a" if args.resume else "w"
    with open(args.output_jsonl, out_mode, encoding="utf-8") as out_f, open(
        args.output_csv, csv_mode, encoding="utf-8", newline=""
    ) as out_csv:
        writer = None
        extra_cols = [
            "evidence_sufficiency_tag",
            "matches_annotated_answer",
            "extraction_answer",
            "evidence_supporting_explanation",
            "evidence_used",
            "evidence_used_source",
            "evidence_used_section_id",
            "evidence_used_section_title",
            "kb_missing_url",
            "kb_missing_ids",
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

            data_id = normalize_text(row.get(args.id_column)) or f"row_{idx}"
            if data_id in seen:
                continue

            question_type = normalize_text(row.get(qtype_column)) if qtype_column else ""
            if skip_types and question_type.lower() in skip_types:
                continue

            question = normalize_text(row.get(question_column))
            answer_raw = normalize_text(row.get(args.answer_column))
            evidence_raw = normalize_text(row.get(args.evidence_column))
            evidence_id_raw = normalize_text(row.get(args.section_id_column))
            evidence_title_raw = normalize_text(row.get(args.section_title_column))
            section_ids = parse_section_ids(row.get(args.section_id_column))

            if split_answers:
                answers = split_answer_candidates(answer_raw, multi_delim=args.multi_answer_delim)
            else:
                answers = [answer_raw] if answer_raw else []

            annotated_answer = answers[0] if answers else ""
            optional_allowed = answers[1:] if len(answers) > 1 else []
            optional_allowed_json = json.dumps(optional_allowed, ensure_ascii=True)

            all_zero = (
                is_zero_field(evidence_raw)
                and is_zero_field(evidence_id_raw)
                and is_zero_field(evidence_title_raw)
            )
            if all_zero:
                evidence_tag = "E_unsupporting"
                matches = "NoExtraction"
                extraction_answer = None
                explanation = "missing evidence according to initial matching"
                evidence_used = ""
                evidence_used_source = "none"
                evidence_used_section_id = ""
                evidence_used_section_title = ""
                kb_meta = {"missing_url": False, "missing_ids": []}

                result = {
                    "data_id": data_id,
                    "evidence_sufficiency_tag": evidence_tag,
                    "matches_annotated_answer": matches,
                    "extraction_answer": extraction_answer,
                    "explanation": explanation,
                    "model": args.model,
                    "evidence_used": evidence_used,
                    "evidence_used_source": evidence_used_source,
                    "evidence_used_section_id": evidence_used_section_id,
                    "evidence_used_section_title": evidence_used_section_title,
                    "kb_missing_url": kb_meta.get("missing_url", False),
                    "kb_missing_ids": kb_meta.get("missing_ids", []),
                }
                if args.include_question:
                    result["question"] = question
                if args.include_evidence:
                    result["evidence"] = evidence_used
                result["annotated_answer"] = annotated_answer
                result["optional_allowed_answers"] = optional_allowed

                out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
                out_f.flush()

                row = dict(row)
                row["evidence_sufficiency_tag"] = evidence_tag
                row["matches_annotated_answer"] = matches
                row["extraction_answer"] = ""
                row["evidence_supporting_explanation"] = explanation
                row["evidence_used"] = evidence_used
                row["evidence_used_source"] = evidence_used_source
                row["evidence_used_section_id"] = evidence_used_section_id
                row["evidence_used_section_title"] = evidence_used_section_title
                row["kb_missing_url"] = str(kb_meta.get("missing_url", False))
                row["kb_missing_ids"] = "|".join([str(i) for i in kb_meta.get("missing_ids", [])])

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
                skipped_all_zero += 1
                continue

            evidence_csv = decode_unicode_escapes(evidence_raw).replace("|", "\n\n") if evidence_raw else ""
            section_titles = normalize_text(row.get(args.section_title_column))

            evidence_tag = None
            extraction_answer = None
            matches = "NoExtraction"
            explanation = ""
            evidence_used = evidence_csv if evidence_csv else ""
            evidence_used_source = "csv" if evidence_csv else "none"
            evidence_used_section_id = (
                "|".join([str(sid) for sid in section_ids]) if evidence_csv and section_ids else ""
            )
            evidence_used_section_title = section_titles if evidence_csv else ""
            kb_meta = {"missing_url": False, "missing_ids": []}

            if evidence_csv:
                try:
                    (
                        evidence_tag,
                        extraction_answer,
                        matches,
                        explanation,
                        evidence_used,
                        content,
                    ) = audit_and_parse(
                        prompt_template,
                        question,
                        annotated_answer,
                        optional_allowed_json,
                        evidence_csv,
                        args.provider,
                        args.api_base,
                        args.api_key,
                        args.model,
                        args.temperature,
                        args.max_tokens,
                        args.max_evidence_chars,
                    )
                except Exception as exc:
                    error_obj = {
                        "data_id": data_id,
                        "error": str(exc),
                        "model": args.model,
                        "evidence_source": "csv",
                    }
                    if args.include_question:
                        error_obj["question"] = question
                    if args.include_evidence:
                        error_obj["evidence_used"] = evidence_used
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
                                "evidence_source": "csv",
                                "raw_response": content,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

            if evidence_tag != "E_supporting" and kb is not None and section_ids:
                sections, kb_meta = kb.get_section_texts(
                    normalize_text(row.get(args.url_column)), section_ids
                )
                found_support = False
                audit_error = False

                for section in sections:
                    text = normalize_text(section.get("text"))
                    paragraphs = split_paragraphs(text)
                    if not paragraphs and text:
                        paragraphs = [text]
                    for p_idx, paragraph in enumerate(paragraphs):
                        try:
                            (
                                kb_tag,
                                kb_extraction,
                                kb_matches,
                                kb_explanation,
                                kb_used,
                                content,
                            ) = audit_and_parse(
                                prompt_template,
                                question,
                                annotated_answer,
                                optional_allowed_json,
                                paragraph,
                                args.provider,
                                args.api_base,
                                args.api_key,
                                args.model,
                                args.temperature,
                                args.max_tokens,
                                args.max_evidence_chars,
                            )
                        except Exception as exc:
                            error_obj = {
                                "data_id": data_id,
                                "error": str(exc),
                                "model": args.model,
                                "evidence_source": "kb",
                                "section_id": section.get("section_id"),
                                "paragraph_index": p_idx,
                            }
                            if args.include_question:
                                error_obj["question"] = question
                            out_f.write(json.dumps(error_obj, ensure_ascii=True) + "\n")
                            out_f.flush()
                            if args.sleep:
                                time.sleep(args.sleep)
                            audit_error = True
                            break

                        if raw_f is not None:
                            raw_f.write(
                                json.dumps(
                                    {
                                        "data_id": data_id,
                                        "question": question,
                                        "evidence_source": "kb",
                                        "section_id": section.get("section_id"),
                                        "paragraph_index": p_idx,
                                        "raw_response": content,
                                    },
                                    ensure_ascii=True,
                                )
                                + "\n"
                            )

                        if kb_tag == "E_supporting":
                            evidence_tag = kb_tag
                            extraction_answer = kb_extraction
                            matches = kb_matches
                            explanation = kb_explanation
                            evidence_used = kb_used
                            evidence_used_source = "kb"
                            evidence_used_section_id = str(section.get("section_id", ""))
                            evidence_used_section_title = normalize_text(section.get("title"))
                            found_support = True
                            break

                        if args.sleep:
                            time.sleep(args.sleep)
                    if found_support or audit_error:
                        break

                if audit_error:
                    continue

                if not found_support:
                    evidence_tag = evidence_tag or "E_unsupporting"
                    matches = "NoExtraction"
                    extraction_answer = None
                    explanation = (
                        "no_supporting_evidence_in_kb"
                        if sections
                        else "no_kb_sections"
                    )
                    if not evidence_used:
                        explanation = "no_supporting_evidence_available"

            if evidence_tag is None:
                evidence_tag = "E_unsupporting"
                matches = "NoExtraction"
                extraction_answer = None
                explanation = "no_evidence"

            result = {
                "data_id": data_id,
                "evidence_sufficiency_tag": evidence_tag,
                "matches_annotated_answer": matches,
                "extraction_answer": extraction_answer,
                "explanation": explanation,
                "model": args.model,
                "evidence_used": evidence_used,
                "evidence_used_source": evidence_used_source,
                "evidence_used_section_id": evidence_used_section_id,
                "evidence_used_section_title": evidence_used_section_title,
                "kb_missing_url": kb_meta.get("missing_url", False),
                "kb_missing_ids": kb_meta.get("missing_ids", []),
            }
            if args.include_question:
                result["question"] = question
            if args.include_evidence:
                result["evidence"] = evidence_used
            result["annotated_answer"] = annotated_answer
            result["optional_allowed_answers"] = optional_allowed

            out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
            out_f.flush()

            row = dict(row)
            row["evidence_sufficiency_tag"] = evidence_tag
            row["matches_annotated_answer"] = matches
            row["extraction_answer"] = extraction_answer or ""
            row["evidence_supporting_explanation"] = explanation
            row["evidence_used"] = evidence_used
            row["evidence_used_source"] = evidence_used_source
            row["evidence_used_section_id"] = evidence_used_section_id
            row["evidence_used_section_title"] = evidence_used_section_title
            row["kb_missing_url"] = str(kb_meta.get("missing_url", False))
            row["kb_missing_ids"] = "|".join([str(i) for i in kb_meta.get("missing_ids", [])])

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

    print(
        f"Processed {processed} rows, wrote {written} audits, "
        f"flagged {skipped_all_zero} all-zero rows."
    )


if __name__ == "__main__":
    main()
