#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request

NUMERIC_HINTS = [
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bhow long\b",
    r"\bhow tall\b",
    r"\bhow high\b",
    r"\bhow big\b",
    r"\bhow far\b",
    r"\bhow old\b",
    r"\bhow deep\b",
    r"\bhow wide\b",
    r"\bhow large\b",
    r"\bwhat year\b",
    r"\bwhat is the (?:length|height|width|depth|distance|diameter|radius|area|size|"
    r"weight|mass|population|elevation|altitude|speed|velocity|temperature|pressure|"
    r"volume|rate|percentage|percent|density|age)\b",
    r"\bwhat are the (?:dimensions|measurements)\b",
    r"\bwhen was\b",
    r"\bwhen were\b",
    r"\bwhen did\b",
]
NUMERIC_HINT_RE = re.compile("|".join(NUMERIC_HINTS), re.I)


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def clean_evidence(text):
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_numeric_candidate(question, answer, evidence):
    if re.search(r"\d", answer):
        return True
    if NUMERIC_HINT_RE.search(question):
        return True
    if re.search(r"\d", evidence):
        return True
    return False


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


def build_messages(question, answers, evidence):
    system = (
        "You are a precise numeric answer extractor. "
        "Use only the evidence text. "
        "Return the most precise numeric answer with units exactly as written in the evidence. "
        "If the answer is a range, keep the range and include units. "
        "If multiple numeric answers are required, join them with '|'. "
        "Also return the minimal supporting span (1-2 sentences). "
        "If the evidence does not support a numeric answer, return empty strings."
    )
    user = {
        "question": question,
        "candidate_answers": answers,
        "evidence_text": evidence,
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


def main():
    parser = argparse.ArgumentParser(description="Refine numeric answers with units.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--evidence-column", default="evidence")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-evidence-chars", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
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

    stats = {
        "total_rows": 0,
        "numeric_candidates": 0,
        "updated_rows": 0,
        "skipped_no_evidence": 0,
        "llm_calls": 0,
        "llm_errors": 0,
    }

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_csv, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            stats["total_rows"] += 1
            question = normalize_text(row.get(args.question_column))
            answer = normalize_text(row.get(args.answer_column))
            evidence = normalize_text(row.get(args.evidence_column))
            data_id = normalize_text(row.get(args.id_column))

            if not evidence or evidence == "0":
                stats["skipped_no_evidence"] += 1
                writer.writerow(row)
                continue

            if not is_numeric_candidate(question, answer, evidence):
                writer.writerow(row)
                continue

            stats["numeric_candidates"] += 1
            evidence_input = evidence.replace("|", "\n\n")
            if args.max_evidence_chars > 0 and len(evidence_input) > args.max_evidence_chars:
                evidence_input = evidence_input[: args.max_evidence_chars] + " ..."

            messages = build_messages(question, answer, evidence_input)
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
            except Exception:
                stats["llm_errors"] += 1
                writer.writerow(row)
                continue

            parsed = extract_json_block(content) or {}
            refined_answer = normalize_text(parsed.get("answer"))
            refined_evidence = clean_evidence(parsed.get("evidence"))

            if refined_answer:
                row[args.answer_column] = refined_answer
                if refined_evidence:
                    row[args.evidence_column] = refined_evidence
                else:
                    row[args.evidence_column] = clean_evidence(evidence)
                stats["updated_rows"] += 1

            writer.writerow(row)

            if args.sleep:
                time.sleep(args.sleep)

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
