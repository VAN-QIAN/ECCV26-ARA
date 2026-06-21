#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request


TAGS = {"Q_clear", "Q_redundant", "Q_under-specified"}


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


def build_prompt(template, question):
    if "{QUESTION}" in template:
        return template.replace("{QUESTION}", question)
    return template.rstrip() + "\n\nQUESTION: " + question


def build_messages(prompt_text):
    return [
        {
            "role": "system",
            "content": "You are a strict auditor. Follow the user instructions exactly.",
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


def default_prompt_path():
    return os.path.join(os.path.dirname(__file__), "question_clarity_prompt.txt")


def default_output_path():
    return os.path.join("results_question_clarity", "evqa_question_clarity.jsonl")


def parse_response(content):
    parsed = extract_json_block(content)
    if not parsed:
        return "Q_under-specified", "unparseable_response"
    tag = normalize_text(parsed.get("question_clarity_tag"))
    explanation = normalize_text(parsed.get("clarity_explanation"))
    if tag not in TAGS:
        return "Q_under-specified", "invalid_tag"
    return tag, explanation


def main():
    parser = argparse.ArgumentParser(description="Judge question clarity for EVQA.")
    parser.add_argument("--input", required=True, help="Input CSV with EVQA fields.")
    parser.add_argument("--output", default=default_output_path(), help="Output JSONL path.")
    parser.add_argument("--prompt-path", default=default_prompt_path())
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--id-column", default="data_id")
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

    fieldnames = get_fieldnames(args.input)
    if args.id_column not in fieldnames:
        print(
            f"Missing {args.id_column} column; run evqa_add_data_id.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt_template = load_prompt(args.prompt_path)

    raw_f = None
    if args.dump_raw:
        os.makedirs(os.path.dirname(args.dump_raw) or ".", exist_ok=True)
        raw_f = open(args.dump_raw, "w", encoding="utf-8")

    seen = load_seen_ids(args.output) if args.resume else set()
    processed = 0
    written = 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out_f:
        for idx, row in iter_rows(args.input):
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

            question = normalize_text(row.get(args.question_column))
            prompt_text = build_prompt(prompt_template, question)
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
                if args.include_question:
                    error_obj["question"] = question
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

            tag, explanation = parse_response(content)

            result = {
                "data_id": data_id,
                "question_clarity_tag": tag,
                "clarity_explanation": explanation,
                "model": args.model,
            }
            if args.include_question:
                result["question"] = question
            out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
            out_f.flush()
            written += 1

            if args.sleep:
                time.sleep(args.sleep)

    if raw_f is not None:
        raw_f.close()

    print(f"Processed {processed} rows, wrote {written} judgments.")


if __name__ == "__main__":
    main()
