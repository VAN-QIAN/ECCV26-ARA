#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


FINAL_TAGS = {"OK", "Needs_revision", "Not_answerable"}


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(template, row):
    prompt = template
    prompt = prompt.replace("{WIKIPEDIA_TITLE}", normalize_text(row.get("wikipedia_title")))
    prompt = prompt.replace("{QUESTION_TYPE}", normalize_text(row.get("question_type")))
    prompt = prompt.replace("{QUESTION_ORIGINAL}", normalize_text(row.get("question_original")))
    prompt = prompt.replace("{FINAL_QUESTION}", normalize_text(row.get("final_question")))
    prompt = prompt.replace("{FINAL_ANSWER}", normalize_text(row.get("final_answer")))
    prompt = prompt.replace("{FINAL_EVIDENCE}", normalize_text(row.get("final_evidence")))
    prompt = prompt.replace("{EVIDENCE}", normalize_text(row.get("evidence")))
    return prompt


def build_messages(prompt_text):
    return [
        {"role": "system", "content": "You are a strict auditor. Return JSON only."},
        {"role": "user", "content": prompt_text},
    ]


_CLIENT_CACHE = {}


def get_openai_client(api_base, api_key):
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip3 install openai")
    cache_key = (api_base or "", api_key or "")
    client = _CLIENT_CACHE.get(cache_key)
    if client is not None:
        return client
    if api_base:
        client = OpenAI(api_key=api_key, base_url=api_base)
    else:
        client = OpenAI(api_key=api_key)
    _CLIENT_CACHE[cache_key] = client
    return client


def call_openai_sdk(api_base, api_key, model, messages, temperature, max_tokens):
    client = get_openai_client(api_base, api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].message.content


def call_llm(provider, api_base, api_key, model, messages, temperature, max_tokens):
    if provider in {"openai", "openai_compat", "qwen", "qwen3", "deepseek"}:
        return call_openai_sdk(api_base, api_key, model, messages, temperature, max_tokens)
    raise ValueError(f"Unsupported provider: {provider}")


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


def parse_response(content, row):
    parsed = extract_json_block(content)
    if not parsed:
        return {
            "final_check_tag": "Needs_revision",
            "final_question": normalize_text(row.get("final_question")),
            "final_answer": normalize_text(row.get("final_answer")),
            "final_evidence": normalize_text(row.get("final_evidence")),
            "evidence": normalize_text(row.get("evidence")),
            "reason": "unparseable_response",
        }
    tag = normalize_text(parsed.get("final_check_tag"))
    if tag not in FINAL_TAGS:
        tag = "Needs_revision"
    out = {
        "final_check_tag": tag,
        "final_question": normalize_text(parsed.get("final_question"))
        or normalize_text(row.get("final_question")),
        "final_answer": normalize_text(parsed.get("final_answer"))
        or normalize_text(row.get("final_answer")),
        "final_evidence": normalize_text(parsed.get("final_evidence"))
        or normalize_text(row.get("final_evidence")),
        "evidence": normalize_text(parsed.get("evidence"))
        or normalize_text(row.get("evidence")),
        "reason": normalize_text(parsed.get("reason")),
    }
    if out["final_check_tag"] == "OK":
        out["final_question"] = normalize_text(row.get("final_question"))
        out["final_answer"] = normalize_text(row.get("final_answer"))
        out["final_evidence"] = normalize_text(row.get("final_evidence"))
        out["evidence"] = normalize_text(row.get("evidence"))
    # Heuristic: if answer changed from range to single value, avoid "range/between/from-to" phrasing.
    if out["final_check_tag"] == "Needs_revision":
        original_answer = normalize_text(row.get("final_answer"))
        if out["final_answer"] and out["final_answer"] != original_answer:
            q = out["final_question"]
            if q == normalize_text(row.get("final_question")):
                q_lower = q.lower()
                single_value = all(tok not in out["final_answer"] for tok in ["-", "to", "–", "—"])
                if single_value and any(t in q_lower for t in ["range", "between", "from"]):
                    q = re.sub(r"\brange\b", "", q, flags=re.IGNORECASE)
                    q = re.sub(r"\bbetween\b.*", "", q, flags=re.IGNORECASE)
                    q = re.sub(r"\bfrom\b.*", "", q, flags=re.IGNORECASE)
                    q = re.sub(r"\s{2,}", " ", q).strip()
                    if q.endswith("?") is False:
                        q = q + "?"
                    out["final_question"] = q
                    if out["final_check_tag"] == "OK":
                        out["final_check_tag"] = "Needs_revision"
                    if out["reason"]:
                        out["reason"] += " | adjusted_question_single_value"
                    else:
                        out["reason"] = "adjusted_question_single_value"
    if out["final_evidence"] != out["evidence"]:
        out["evidence"] = out["final_evidence"]
    return out


def default_prompt_path():
    return os.path.join(os.path.dirname(__file__), "infoseek_final_recheck_prompt.txt")


def load_done_ids(path, id_column):
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data_id = normalize_text(obj.get(id_column))
            if data_id:
                done.add(data_id)
    return done


def main():
    parser = argparse.ArgumentParser(description="Recheck InfoSeek final QA with evidence.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--prompt-path", default=default_prompt_path())
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    prompt_template = load_prompt(args.prompt_path)
    done_ids = load_done_ids(args.output_jsonl, args.id_column) if args.resume else set()

    jsonl_mode = "a" if args.resume else "w"
    csv_mode = "a" if (args.resume and os.path.exists(args.output_csv)) else "w"
    total = 0

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        input_fields = reader.fieldnames or []
        output_fields = list(input_fields)
        for extra in [
            "final_check_tag",
            "final_check_reason",
            "final_question",
            "final_answer",
            "final_evidence",
            "evidence",
            "final_model",
        ]:
            if extra not in output_fields:
                output_fields.append(extra)

        with open(args.output_jsonl, jsonl_mode, encoding="utf-8") as jf, open(
            args.output_csv, csv_mode, encoding="utf-8", newline=""
        ) as cf:
            writer = csv.DictWriter(cf, fieldnames=output_fields)
            if csv_mode == "w":
                writer.writeheader()

            for idx, row in enumerate(reader):
                data_id = normalize_text(row.get(args.id_column))
                if done_ids and data_id in done_ids:
                    continue
                total += 1
                if args.limit and total > args.limit:
                    break
                prompt = build_prompt(prompt_template, row)
                messages = build_messages(prompt)
                content = call_llm(
                    args.provider,
                    args.api_base,
                    args.api_key,
                    args.model,
                    messages,
                    args.temperature,
                    args.max_tokens,
                )
                parsed = parse_response(content, row)
                record = dict(row)
                record.update(
                    {
                        "final_check_tag": parsed["final_check_tag"],
                        "final_check_reason": parsed["reason"],
                        "final_question": parsed["final_question"],
                        "final_answer": parsed["final_answer"],
                        "final_evidence": parsed["final_evidence"],
                        "evidence": parsed["evidence"],
                        "final_model": args.model,
                    }
                )
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.writerow(record)
                if args.sleep:
                    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
