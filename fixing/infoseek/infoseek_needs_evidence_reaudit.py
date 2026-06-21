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


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(template, question, answer, qtype, evidence):
    prompt = template
    prompt = prompt.replace("{QUESTION}", question)
    prompt = prompt.replace("{ANNOTATED_ANSWER}", answer)
    prompt = prompt.replace("{QUESTION_TYPE}", qtype)
    prompt = prompt.replace("{EVIDENCE}", evidence)
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


def parse_response(content):
    parsed = extract_json_block(content)
    if not parsed:
        return "E_unsupporting", "", "NoExtraction", "", "unparseable_response"
    evidence_tag = normalize_text(parsed.get("evidence_sufficiency_tag"))
    extracted = normalize_text(parsed.get("extracted_answer"))
    matches = normalize_text(parsed.get("matches_annotated_answer"))
    fixed = normalize_text(parsed.get("fixed_answer"))
    reason = normalize_text(parsed.get("reason"))

    if evidence_tag not in EVIDENCE_TAGS:
        evidence_tag = "E_unsupporting"
    if matches not in MATCH_TAGS:
        matches = "NoExtraction" if evidence_tag == "E_unsupporting" else "NoMatch"

    if evidence_tag == "E_unsupporting":
        extracted = ""
        fixed = ""
        matches = "NoExtraction"

    return evidence_tag, extracted, matches, fixed, reason


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
    parser = argparse.ArgumentParser(
        description="Re-audit InfoSeek needs_evidence with LLM and suggest answer fixes."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="annotated_answer")
    parser.add_argument(
        "--question-type-column",
        default="auto",
        help="Column for question type; use 'auto' to prefer question_type_qtype.",
    )
    parser.add_argument("--evidence-column", default="evidence_used")
    parser.add_argument("--fallback-evidence-column", default="evidence")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--prompt-path", default="infoseek_needs_evidence_reaudit_prompt.txt")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
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

    prompt_path = args.prompt_path
    if not os.path.isabs(prompt_path):
        prompt_path = os.path.join(os.path.dirname(__file__), prompt_path)
    prompt_template = load_prompt(prompt_path)

    raw_f = None
    if args.dump_raw:
        os.makedirs(os.path.dirname(args.dump_raw) or ".", exist_ok=True)
        raw_f = open(args.dump_raw, "w", encoding="utf-8")

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    seen = set()
    if args.resume and os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data_id = normalize_text(obj.get(args.id_column))
                if data_id:
                    seen.add(data_id)

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_jsonl, "a" if args.resume else "w", encoding="utf-8"
    ) as f_jsonl, open(
        args.output_csv, "a" if args.resume else "w", encoding="utf-8", newline=""
    ) as f_csv:
        reader = csv.DictReader(f_in)
        qtype_column = resolve_qtype_column(reader.fieldnames or [], args.question_type_column)
        if not qtype_column:
            print("Missing question type column.", file=sys.stderr)
            sys.exit(1)

        extra_cols = [
            "reaudit_evidence_tag",
            "reaudit_extracted_answer",
            "reaudit_matches_annotated_answer",
            "reaudit_fixed_answer",
            "reaudit_reason",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=(reader.fieldnames or []) + extra_cols)
        write_header = True
        if args.resume and os.path.exists(args.output_csv):
            write_header = os.path.getsize(args.output_csv) == 0
        if write_header:
            writer.writeheader()

        processed = 0
        written = 0
        for idx, row in enumerate(reader):
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

            question = normalize_text(row.get(args.question_column))
            answer = normalize_text(row.get(args.answer_column))
            qtype = normalize_text(row.get(qtype_column))
            evidence = normalize_text(row.get(args.evidence_column))
            if not evidence:
                evidence = normalize_text(row.get(args.fallback_evidence_column))
            if not evidence:
                continue

            prompt_text = build_prompt(prompt_template, question, answer, qtype, evidence)
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
                f_jsonl.write(json.dumps(error_obj, ensure_ascii=True) + "\n")
                f_jsonl.flush()
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            if raw_f is not None:
                raw_f.write(
                    json.dumps(
                        {
                            "data_id": data_id,
                            "question": question,
                            "answer": answer,
                            "raw_response": content,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            evidence_tag, extracted, matches, fixed, reason = parse_response(content)

            result = {
                "data_id": data_id,
                "question": question,
                "annotated_answer": answer,
                "question_type": qtype,
                "evidence_used": evidence,
                "reaudit_evidence_tag": evidence_tag,
                "reaudit_extracted_answer": extracted,
                "reaudit_matches_annotated_answer": matches,
                "reaudit_fixed_answer": fixed,
                "reaudit_reason": reason,
                "model": args.model,
            }
            f_jsonl.write(json.dumps(result, ensure_ascii=True) + "\n")
            f_jsonl.flush()

            row = dict(row)
            row["reaudit_evidence_tag"] = evidence_tag
            row["reaudit_extracted_answer"] = extracted
            row["reaudit_matches_annotated_answer"] = matches
            row["reaudit_fixed_answer"] = fixed
            row["reaudit_reason"] = reason
            writer.writerow(row)
            written += 1

            if args.sleep:
                time.sleep(args.sleep)

    if raw_f is not None:
        raw_f.close()

    print(f"Processed {processed} rows, wrote {written} reaudits.")


if __name__ == "__main__":
    main()
