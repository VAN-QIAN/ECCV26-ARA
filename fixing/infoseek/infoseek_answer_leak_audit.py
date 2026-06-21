#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request


TAGS = {"A_leaks", "A_inferrable", "A_ok"}


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


def select_question(row, base_column):
    suggested = normalize_text(row.get("suggested_question"))
    if suggested:
        return suggested
    return normalize_text(row.get(base_column))


def default_prompt_path():
    return os.path.join(os.path.dirname(__file__), "infoseek_answer_leak_prompt.txt")


def default_output_jsonl_path():
    return os.path.join("results_answer_leak", "infoseek_answer_leak.jsonl")


def default_output_csv_path():
    return os.path.join("results_answer_leak", "infoseek_answer_leak.csv")


def parse_response(content):
    parsed = extract_json_block(content)
    if not parsed:
        return "A_ok", "unparseable_response"
    tag = normalize_text(parsed.get("answer_leak_tag"))
    reason = normalize_text(parsed.get("reason"))
    if tag not in TAGS:
        tag = "A_ok"
    return tag, reason


def main():
    parser = argparse.ArgumentParser(description="Audit answer leakage for InfoSeek.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", default=default_output_jsonl_path())
    parser.add_argument("--output-csv", default=default_output_csv_path())
    parser.add_argument("--prompt-path", default=default_prompt_path())
    parser.add_argument(
        "--question-column",
        default="auto",
        help="Column name for question text; use 'auto' to prefer question_original.",
    )
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
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
        print(f"Missing {args.id_column} column.", file=sys.stderr)
        sys.exit(1)

    split_answers = args.split_answers
    if split_answers is None:
        split_answers = args.answer_column == "answer"
    if args.no_split_answers:
        split_answers = False

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
        extra_cols = ["answer_leak_tag", "answer_leak_reason"]
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
                print(f"Missing {args.id_column} at row {idx}.", file=sys.stderr)
                continue
            if data_id in seen:
                continue

            question = select_question(row, question_column)
            answer_raw = normalize_text(row.get(args.answer_column))
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

            tag, reason = parse_response(content)

            result = {
                "data_id": data_id,
                "answer_leak_tag": tag,
                "answer_leak_reason": reason,
                "model": args.model,
                "question": question,
                "annotated_answer": annotated_answer,
                "optional_allowed_answers": optional_allowed,
            }
            out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
            out_f.flush()

            row = dict(row)
            row["answer_leak_tag"] = tag
            row["answer_leak_reason"] = reason

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

    print(f"Processed {processed} rows, wrote {written} audits.")


if __name__ == "__main__":
    main()
