#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request

LABELS = {"good", "improvable", "incorrect", "missing_evidence"}
IMPROVE_TYPES = {
    "precision",
    "unit",
    "range",
    "entity",
    "normalization",
    "evidence_span",
}


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


def clean_text(text):
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_paragraphs(text):
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def score_label(label):
    if label == "good":
        return 3
    if label == "improvable":
        return 2
    if label == "incorrect":
        return 1
    return 0


def load_filter_ids(path, label_filter, reason_filter):
    if not path:
        return None
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data_id = normalize_text(obj.get("data_id"))
            if not data_id:
                continue
            if label_filter:
                if normalize_text(obj.get("label")) != label_filter:
                    continue
            if reason_filter:
                if normalize_text(obj.get("reason")) != reason_filter:
                    continue
            ids.add(data_id)
    return ids


def load_seen_ids(path):
    if not path or not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data_id = normalize_text(obj.get("data_id"))
            if data_id:
                seen.add(data_id)
    return seen


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
        section_texts = entry.get("section_texts")
        if isinstance(section_texts, list):
            return ["" if sec is None else str(sec) for sec in section_texts]
        return None

    def get_evidence(self, url, section_ids):
        if not url or not section_ids:
            return None
        _, entry = self._resolve_entry(url)
        if not entry:
            return None
        sections = self._extract_sections(entry)
        if not sections:
            return None
        texts = []
        for sid in section_ids:
            if sid < 0 or sid >= len(sections):
                continue
            texts.append(sections[sid])
        if not texts:
            return None
        return "\n\n".join([t for t in texts if t])


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


def build_messages(
    question,
    answers,
    evidence,
    reason_limit,
    evidence_limit,
    question_type,
    strict_types,
    short_response=False,
):
    qtype_norm = normalize_text(question_type).lower()
    strict_mode = qtype_norm in strict_types
    system = (
        "You are an annotation quality auditor for KB-VQA. "
        "Use only the provided evidence text. "
        "Decide whether the annotation (answer) is correct and sufficiently precise. "
        "Return JSON with keys: label, reason, improve_type, suggested_answer, suggested_evidence. "
        "Label must be one of: good, improvable, incorrect, missing_evidence. "
        "Use 'improvable' only if the answer is supported but can be made more precise "
        "(e.g., add units, exact range, exact entity name). "
        "If incorrect or unsupported, label 'incorrect' and leave suggested_* empty. "
        "If evidence is empty, use 'missing_evidence'. "
        "improve_type must be one of: precision, unit, range, entity, normalization, evidence_span. "
        f"Keep reason <= {reason_limit} chars. "
        f"Keep suggested_evidence <= {evidence_limit} chars. "
        "If needed, return empty suggested_evidence."
    )
    if strict_mode:
        system += (
            " This is a Numerical/Temporal question. Be extremely strict: the answer must "
            "exactly match the evidence, including units. For ranges, both endpoints must "
            "match the evidence. If the evidence supports a convertible value, you may "
            "convert and accept only if exact. Minor formatting differences can be "
            "improvable with improve_type 'normalization' or 'unit'."
        )
    if short_response:
        system += " Return JSON only, no markdown or extra text."
    user = {
        "question": question,
        "answer": answers,
        "evidence": evidence,
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


def main():
    parser = argparse.ArgumentParser(description="Audit InfoSeek annotation quality with LLM.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--question-column", default="question")
    parser.add_argument(
        "--question-type-column",
        default="auto",
        help="Column for question type; use 'auto' to prefer question_type_qtype.",
    )
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--evidence-column", default="evidence")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--section-id-column", default="evidence_section_id")
    parser.add_argument("--section-title-column", default="evidence_section_title")
    parser.add_argument("--kb-path", default="")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-evidence-chars", type=int, default=0)
    parser.add_argument("--max-reason-chars", type=int, default=300)
    parser.add_argument("--max-suggested-evidence-chars", type=int, default=300)
    parser.add_argument("--retry-on-parse-fail", action="store_true")
    parser.add_argument("--retry-max-tokens", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--strict-qtypes",
        default="numerical,temporal",
        help="Comma-separated question types to enforce strict numeric/temporal rules.",
    )
    parser.add_argument(
        "--only-qtypes",
        default="",
        help="Comma-separated question types to audit; others are skipped.",
    )
    parser.add_argument(
        "--filter-jsonl",
        default="",
        help="Only process rows whose data_id appears in this JSONL file.",
    )
    parser.add_argument(
        "--filter-label",
        default="",
        help="When used with --filter-jsonl, only include entries with this label.",
    )
    parser.add_argument(
        "--filter-reason",
        default="",
        help="When used with --filter-jsonl, only include entries with this reason.",
    )
    parser.add_argument(
        "--apply-suggestions",
        action="store_true",
        help="If label=improvable, overwrite answer/evidence in the output CSV.",
    )
    parser.add_argument(
        "--skip-empty-evidence",
        action="store_true",
        help="If evidence is empty/0, mark label as missing_evidence and skip LLM.",
    )
    parser.add_argument(
        "--dump-raw",
        default="",
        help="Write raw LLM responses to this JSONL file for debugging.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output JSONL exists, skip data_ids already processed and append outputs.",
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

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    kb = InfoSeekKnowledgeBase(args.kb_path) if args.kb_path else None

    raw_f = None
    if args.dump_raw:
        os.makedirs(os.path.dirname(args.dump_raw) or ".", exist_ok=True)
        raw_f = open(args.dump_raw, "w", encoding="utf-8")

    seen_ids = load_seen_ids(args.output_jsonl) if args.resume else set()

    filter_ids = load_filter_ids(
        args.filter_jsonl, normalize_text(args.filter_label), normalize_text(args.filter_reason)
    )

    jsonl_mode = "a" if args.resume else "w"
    csv_mode = "a" if args.resume else "w"
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_jsonl, jsonl_mode, encoding="utf-8"
    ) as f_jsonl, open(args.output_csv, csv_mode, encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        qtype_column = resolve_qtype_column(reader.fieldnames or [], args.question_type_column)
        if not qtype_column:
            print(
                "Warning: no question type column found; strict numeric/temporal rules disabled.",
                file=sys.stderr,
            )
        strict_types = {
            t.strip().lower() for t in args.strict_qtypes.split(",") if t.strip()
        }
        only_qtypes = {
            t.strip().lower() for t in args.only_qtypes.split(",") if t.strip()
        }
        extra_cols = [
            "annotation_quality_label",
            "annotation_quality_reason",
            "annotation_quality_improve_type",
            "suggested_answer",
            "suggested_evidence",
        ]
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames + extra_cols)
        write_header = True
        if args.resume and os.path.exists(args.output_csv):
            write_header = os.path.getsize(args.output_csv) == 0
        if write_header:
            writer.writeheader()

        for idx, row in enumerate(reader):
            data_id = normalize_text(row.get(args.id_column)) or f"row_{idx}"
            if filter_ids is not None and data_id not in filter_ids:
                continue
            if args.resume and data_id in seen_ids:
                continue
            question = normalize_text(row.get(args.question_column))
            question_type = normalize_text(row.get(qtype_column)) if qtype_column else ""
            if only_qtypes and question_type.lower() not in only_qtypes:
                continue
            answer_raw = normalize_text(row.get(args.answer_column))
            evidence_raw = normalize_text(row.get(args.evidence_column))
            evidence_id_raw = normalize_text(row.get(args.section_id_column))
            evidence_title_raw = normalize_text(row.get(args.section_title_column))
            section_ids = parse_section_ids(row.get(args.section_id_column))
            evidence_source = "csv"

            answers = split_pipe_field(answer_raw) if "|" in answer_raw else [answer_raw] if answer_raw else []

            all_zero = (
                is_zero_field(evidence_raw)
                and is_zero_field(evidence_id_raw)
                and is_zero_field(evidence_title_raw)
            )
            evidence_input = ""
            if not all_zero and kb and section_ids:
                kb_evidence = kb.get_evidence(normalize_text(row.get(args.url_column)), section_ids)
                if kb_evidence:
                    evidence_input = kb_evidence
                    evidence_source = "kb"

            if not evidence_input and not all_zero and not kb:
                evidence_decoded = decode_unicode_escapes(evidence_raw)
                evidence_input = evidence_decoded.replace("|", "\n\n")

            if args.max_evidence_chars > 0 and len(evidence_input) > args.max_evidence_chars:
                evidence_input = evidence_input[: args.max_evidence_chars] + " ..."

            if not answers:
                label = "incorrect"
                reason = "empty_answer"
                improve_type = ""
                suggested_answer = ""
                suggested_evidence = ""
                answer_audits = []
            elif all_zero:
                label = "missing_evidence"
                reason = "all_zero_evidence_fields"
                improve_type = ""
                suggested_answer = ""
                suggested_evidence = ""
                answer_audits = [
                    {
                        "answer": ans,
                        "label": label,
                        "reason": reason,
                        "improve_type": "",
                        "suggested_answer": "",
                        "suggested_evidence": "",
                    }
                    for ans in answers
                ]
            elif args.skip_empty_evidence and (not evidence_input or evidence_input == "0"):
                label = "missing_evidence"
                reason = "empty_or_zero_evidence"
                improve_type = ""
                suggested_answer = ""
                suggested_evidence = ""
                answer_audits = [
                    {
                        "answer": ans,
                        "label": label,
                        "reason": reason,
                        "improve_type": "",
                        "suggested_answer": "",
                        "suggested_evidence": "",
                    }
                    for ans in answers
                ]
            else:
                evidence_paragraphs = split_paragraphs(evidence_input)
                if not evidence_paragraphs:
                    evidence_paragraphs = [evidence_input]
                answer_audits = []
                for ans in answers:
                    best = None
                    best_score = -1
                    for p_idx, paragraph in enumerate(evidence_paragraphs):
                        messages = build_messages(
                            question,
                            ans,
                            paragraph,
                            args.max_reason_chars,
                            args.max_suggested_evidence_chars,
                            question_type,
                            strict_types,
                            False,
                        )
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
                            label = "incorrect"
                            reason = f"llm_error:{exc}"
                            improve_type = ""
                            suggested_answer = ""
                            suggested_evidence = ""
                        else:
                            if raw_f is not None:
                                raw_f.write(
                                    json.dumps(
                                        {
                                            "data_id": data_id,
                                            "question": question,
                                            "answer": ans,
                                            "paragraph_index": p_idx,
                                            "raw_response": content,
                                        },
                                        ensure_ascii=True,
                                    )
                                    + "\n"
                                )
                            parsed = extract_json_block(content)
                            if not parsed and args.retry_on_parse_fail:
                                retry_messages = build_messages(
                                    question,
                                    ans,
                                    paragraph,
                                    args.max_reason_chars,
                                    args.max_suggested_evidence_chars,
                                    question_type,
                                    strict_types,
                                    True,
                                )
                                retry_tokens = args.retry_max_tokens or args.max_tokens
                                try:
                                    content = call_llm(
                                        args.provider,
                                        args.api_base,
                                        args.api_key,
                                        args.model,
                                        retry_messages,
                                        args.temperature,
                                        retry_tokens,
                                    )
                                    parsed = extract_json_block(content)
                                except Exception:
                                    parsed = None

                            if not parsed:
                                label = "incorrect"
                                reason = "unparseable_response"
                                improve_type = ""
                                suggested_answer = ""
                                suggested_evidence = ""
                            else:
                                label = normalize_text(parsed.get("label")).lower()
                                reason = normalize_text(parsed.get("reason"))
                                improve_type = normalize_text(parsed.get("improve_type")).lower()
                                suggested_answer = normalize_text(parsed.get("suggested_answer"))
                                suggested_evidence = clean_text(parsed.get("suggested_evidence"))
                                if label not in LABELS:
                                    label = "incorrect"
                                    reason = "invalid_label"
                                    improve_type = ""
                                    suggested_answer = ""
                                    suggested_evidence = ""
                                if label != "improvable":
                                    improve_type = ""
                                elif improve_type not in IMPROVE_TYPES:
                                    improve_type = "precision"

                                if args.max_reason_chars > 0:
                                    reason = reason[: args.max_reason_chars]
                                if args.max_suggested_evidence_chars > 0 and suggested_evidence:
                                    suggested_evidence = suggested_evidence[
                                        : args.max_suggested_evidence_chars
                                    ]

                        score = score_label(label)
                        if score > best_score:
                            best_score = score
                            best = {
                                "answer": ans,
                                "label": label,
                                "reason": reason,
                                "improve_type": improve_type,
                                "suggested_answer": suggested_answer,
                                "suggested_evidence": suggested_evidence,
                            }
                            if score == 3:
                                break

                    if best is None:
                        best = {
                            "answer": ans,
                            "label": "incorrect",
                            "reason": "no_paragraphs",
                            "improve_type": "",
                            "suggested_answer": "",
                            "suggested_evidence": "",
                        }
                    answer_audits.append(best)

                labels = [a["label"] for a in answer_audits]
                if any(l == "incorrect" for l in labels):
                    label = "incorrect"
                elif any(l == "improvable" for l in labels):
                    label = "improvable"
                elif all(l == "missing_evidence" for l in labels):
                    label = "missing_evidence"
                else:
                    label = "good"
                reason = " | ".join([a["reason"] for a in answer_audits if a["reason"]])
                improve_types = [a["improve_type"] for a in answer_audits if a["improve_type"]]
                improve_type = " | ".join(improve_types)
                suggested_answers = [
                    a["suggested_answer"] if a["suggested_answer"] else a["answer"]
                    for a in answer_audits
                ]
                suggested_evidences = [
                    a["suggested_evidence"] for a in answer_audits if a["suggested_evidence"]
                ]
                suggested_answer = "|".join(suggested_answers)
                suggested_evidence = "|".join(suggested_evidences)

            if label == "improvable" and args.apply_suggestions:
                if suggested_answer:
                    row[args.answer_column] = suggested_answer
                if suggested_evidence:
                    row[args.evidence_column] = suggested_evidence

            row["annotation_quality_label"] = label
            row["annotation_quality_reason"] = reason
            row["annotation_quality_improve_type"] = improve_type
            row["suggested_answer"] = suggested_answer
            row["suggested_evidence"] = suggested_evidence

            writer.writerow(row)
            f_jsonl.write(
                json.dumps(
                    {
                        "data_id": data_id,
                        "label": label,
                        "reason": reason,
                        "improve_type": improve_type,
                        "suggested_answer": suggested_answer,
                        "suggested_evidence": suggested_evidence,
                        "answer_audits": answer_audits,
                        "evidence_source": evidence_source,
                        "model": args.model,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

            if args.sleep:
                time.sleep(args.sleep)

    if raw_f is not None:
        raw_f.close()


if __name__ == "__main__":
    main()
