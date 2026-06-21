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


FINAL_TAGS = {"OK", "Needs_revision", "Answer_leak", "Not_answerable"}


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
        return suggested, "suggested_question"
    fixed = normalize_text(row.get("fixed_question"))
    if fixed:
        return fixed, "fixed_question"
    return normalize_text(row.get(base_column)), base_column


def select_answer(row, answer_column):
    fixed = normalize_text(row.get("fixed_answer"))
    if fixed:
        return fixed, "fixed_answer"
    ans = normalize_text(row.get(answer_column))
    if ans:
        return ans, answer_column
    ann = normalize_text(row.get("annotated_answer"))
    if ann:
        return ann, "annotated_answer"
    return "", "empty"


def select_evidence(row):
    evidence = normalize_text(row.get("evidence_used"))
    if evidence:
        return evidence, "evidence_used"
    evidence = normalize_text(row.get("evidence"))
    if evidence:
        return evidence, "evidence"
    return "", "none"


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(
    template, question_original, question_suggested, question_fixed, question_used, answer, evidence
):
    prompt = template
    prompt = prompt.replace("{QUESTION_ORIGINAL}", question_original)
    prompt = prompt.replace("{QUESTION_SUGGESTED}", question_suggested)
    prompt = prompt.replace("{QUESTION_FIXED}", question_fixed)
    prompt = prompt.replace("{QUESTION_USED}", question_used)
    prompt = prompt.replace("{ANSWER}", answer)
    prompt = prompt.replace("{EVIDENCE}", evidence)
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


def parse_response(content):
    parsed = extract_json_block(content)
    if not parsed:
        return "Needs_revision", "unparseable_response", ""
    tag = normalize_text(parsed.get("final_check_tag"))
    reason = normalize_text(parsed.get("reason"))
    revised = normalize_text(parsed.get("revised_question"))
    if tag not in FINAL_TAGS:
        tag = "Needs_revision"
    if tag == "OK":
        revised = ""
    return tag, reason, revised


def default_prompt_path():
    return os.path.join(os.path.dirname(__file__), "infoseek_final_check_prompt.txt")


def default_output_jsonl_path():
    return os.path.join("results_final_check", "infoseek_final_check.jsonl")


def default_output_csv_path():
    return os.path.join("results_final_check", "infoseek_final_check.csv")


def iter_rows(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                yield path, idx, row, reader.fieldnames or []


def main():
    parser = argparse.ArgumentParser(description="Final check for InfoSeek QA.")
    parser.add_argument("--input-csv", action="append", required=True)
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
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--multi-answer-delim", default="&&")
    parser.add_argument(
        "--skip-non-supporting",
        action="store_true",
        default=True,
        help="Skip rows where evidence_sufficiency_tag != E_supporting.",
    )
    parser.add_argument(
        "--no-skip-non-supporting",
        dest="skip_non_supporting",
        action="store_false",
        help="Process non-supporting rows too.",
    )
    parser.add_argument("--max-evidence-chars", type=int, default=320)
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
    if args.provider == "deepseek" and not args.api_base:
        args.api_base = os.environ.get("DEEPSEEK_API_BASE", "") or os.environ.get("OPENAI_API_BASE", "")
    if args.provider == "deepseek" and not args.api_key:
        args.api_key = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

    if not args.api_base:
        print("Missing --api-base (or OPENAI_API_BASE/QWEN_API_BASE/DEEPSEEK_API_BASE).", file=sys.stderr)
        sys.exit(1)
    if not args.api_key:
        print("Missing --api-key (or OPENAI_API_KEY/QWEN_API_KEY/DEEPSEEK_API_KEY).", file=sys.stderr)
        sys.exit(1)

    if args.provider == "deepseek":
        model_lower = args.model.strip().lower()
        if model_lower in {"deepseek-v3.2", "deepseek_v3.2", "deepseek-v3", "deepseek_v3"}:
            print(
                "Warning: mapping DeepSeek v3.* to 'deepseek-chat' per DeepSeek API docs. "
                "Override with --model if needed.",
                file=sys.stderr,
            )
            args.model = "deepseek-chat"

    prompt_template = load_prompt(args.prompt_path)

    raw_f = None
    if args.dump_raw:
        os.makedirs(os.path.dirname(args.dump_raw) or ".", exist_ok=True)
        raw_f = open(args.dump_raw, "w", encoding="utf-8")

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

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_mode = "a" if args.resume else "w"
    csv_mode = "a" if args.resume else "w"

    with open(args.output_jsonl, out_mode, encoding="utf-8") as out_f, open(
        args.output_csv, csv_mode, encoding="utf-8", newline=""
    ) as out_csv:
        writer = None
        extra_cols = [
            "final_check_tag",
            "final_check_reason",
            "final_revised_question",
            "final_question_used",
            "final_question_source",
            "final_answer",
            "final_answer_source",
            "final_evidence_used",
            "final_evidence_source",
            "final_answer_leak",
            "final_model",
            "input_source",
        ]
        write_header = True
        if args.resume and os.path.exists(args.output_csv):
            write_header = os.path.getsize(args.output_csv) == 0

        processed = 0
        written = 0
        for path, idx, row, fieldnames in iter_rows(args.input_csv):
            if idx < args.start:
                continue
            if args.end is not None and idx >= args.end:
                break
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1

            data_id = normalize_text(row.get(args.id_column))
            if not data_id:
                continue
            if data_id in seen:
                continue

            if args.skip_non_supporting:
                evidence_tag = normalize_text(row.get("evidence_sufficiency_tag"))
                if evidence_tag and evidence_tag != "E_supporting":
                    continue

            question_column = resolve_question_column(fieldnames, args.question_column)
            if not question_column:
                continue

            question_original = normalize_text(row.get(question_column))
            question_suggested = normalize_text(row.get("suggested_question"))
            question_fixed = normalize_text(row.get("fixed_question"))
            question_used, question_source = select_question(row, question_column)

            answer_raw, answer_source = select_answer(row, args.answer_column)
            candidates = split_answer_candidates(answer_raw, multi_delim=args.multi_answer_delim)
            answer_eval = candidates[0] if candidates else answer_raw

            evidence_text, evidence_source = select_evidence(row)
            if args.max_evidence_chars > 0 and len(evidence_text) > args.max_evidence_chars:
                evidence_text = evidence_text[: args.max_evidence_chars].rstrip() + " ..."

            prompt_text = build_prompt(
                prompt_template,
                question_original,
                question_suggested,
                question_fixed,
                question_used,
                answer_eval,
                evidence_text,
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
                error_obj = {"data_id": data_id, "error": str(exc), "model": args.model}
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
                            "question_used": question_used,
                            "answer": answer_eval,
                            "raw_response": content,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            final_tag, final_reason, final_revised = parse_response(content)
            final_answer_leak = False

            if answer_leaks_in_question(question_used, [answer_eval], args.multi_answer_delim):
                final_answer_leak = True
                if final_tag != "Answer_leak":
                    final_tag = "Answer_leak"
                    if final_reason:
                        final_reason = final_reason + " | detected_answer_leak_in_question"
                    else:
                        final_reason = "detected_answer_leak_in_question"

            if final_revised and answer_leaks_in_question(
                final_revised, [answer_eval], args.multi_answer_delim
            ):
                final_revised = ""
                if final_reason:
                    final_reason = final_reason + " | removed_revised_question_due_to_leak"
                else:
                    final_reason = "removed_revised_question_due_to_leak"

            result = {
                "data_id": data_id,
                "final_check_tag": final_tag,
                "final_check_reason": final_reason,
                "final_revised_question": final_revised,
                "final_question_used": question_used,
                "final_question_source": question_source,
                "final_answer": answer_eval,
                "final_answer_source": answer_source,
                "final_evidence_used": evidence_text,
                "final_evidence_source": evidence_source,
                "final_answer_leak": final_answer_leak,
                "final_model": args.model,
                "input_source": os.path.basename(path),
            }
            out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
            out_f.flush()

            row = dict(row)
            row["final_check_tag"] = final_tag
            row["final_check_reason"] = final_reason
            row["final_revised_question"] = final_revised
            row["final_question_used"] = question_used
            row["final_question_source"] = question_source
            row["final_answer"] = answer_eval
            row["final_answer_source"] = answer_source
            row["final_evidence_used"] = evidence_text
            row["final_evidence_source"] = evidence_source
            row["final_answer_leak"] = final_answer_leak
            row["final_model"] = args.model
            row["input_source"] = os.path.basename(path)

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
