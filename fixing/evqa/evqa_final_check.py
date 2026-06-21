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


def is_multi_answer(candidates, multi_delim="&&"):
    for cand in candidates:
        if multi_delim in cand:
            return True
    return False


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


def tokenize(text):
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def candidate_match_score(candidate, evidence_text):
    if not candidate or not evidence_text:
        return 0.0
    cand_norm = normalize_text(candidate).lower()
    ev_norm = normalize_text(evidence_text).lower()
    if cand_norm and cand_norm in ev_norm:
        return 1.0
    cand_tokens = tokenize(cand_norm)
    if not cand_tokens:
        return 0.0
    ev_tokens = set(tokenize(ev_norm))
    if not ev_tokens:
        return 0.0
    overlap = sum(1 for t in cand_tokens if t in ev_tokens)
    return overlap / max(1, len(cand_tokens))


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


def build_prompt(template, question_original, question_suggested, question_used, answer, evidence):
    prompt = template
    prompt = prompt.replace("{QUESTION_ORIGINAL}", question_original)
    prompt = prompt.replace("{QUESTION_SUGGESTED}", question_suggested)
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
        return suggested, "suggested_question"
    return normalize_text(row.get(base_column)), base_column


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


def trim_text(text, max_chars):
    if max_chars <= 0:
        return text
    if text is None:
        return text
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def select_evidence_text(row, answer_index, answer_eval, kb, args):
    evidence_raw = normalize_text(row.get(args.evidence_column))
    if evidence_raw == "0":
        evidence_raw = ""
    if evidence_raw:
        parts = [p.strip() for p in evidence_raw.split("|") if p.strip()]
        if parts:
            idx = answer_index if answer_index < len(parts) else 0
            evidence = parts[idx]
            return trim_text(evidence, args.max_evidence_chars), "evidence", {}

    short_evidence = normalize_text(row.get("short_evidence"))
    if short_evidence:
        return trim_text(short_evidence, args.max_evidence_chars), "short_evidence", {}

    kb_meta = {"missing_url": False, "missing_ids": []}
    if kb is None:
        return "", "none", kb_meta

    section_ids = parse_section_ids(row.get(args.section_id_column))
    sections, kb_meta = kb.get_sections(normalize_text(row.get(args.url_column)), section_ids)
    evidence, section = select_short_evidence(sections, answer_eval, args.max_evidence_chars)
    if evidence:
        return evidence, "kb", kb_meta
    return "", "none", kb_meta


def choose_best_candidate(candidates, row, args, kb):
    if not candidates:
        return "", "empty", 0

    evidence_raw = normalize_text(row.get(args.evidence_column))
    if evidence_raw == "0":
        evidence_raw = ""
    evidence_parts = [p.strip() for p in evidence_raw.split("|") if p.strip()] if evidence_raw else []

    short_evidence = normalize_text(row.get("short_evidence"))
    scores = []
    evidence_source = ""

    if evidence_parts:
        evidence_source = "evidence"
        for idx, cand in enumerate(candidates):
            if idx < len(evidence_parts):
                ev = evidence_parts[idx]
            else:
                ev = " ".join(evidence_parts)
            scores.append(candidate_match_score(cand, ev))
    elif short_evidence:
        evidence_source = "short_evidence"
        for cand in candidates:
            scores.append(candidate_match_score(cand, short_evidence))
    else:
        scores = [0.0 for _ in candidates]

    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    best_score = scores[best_idx] if scores else 0.0
    if best_score > 0:
        return candidates[best_idx], f"best_match_{evidence_source}", best_idx

    if kb is not None:
        section_ids = parse_section_ids(row.get(args.section_id_column))
        sections, _kb_meta = kb.get_sections(
            normalize_text(row.get(args.url_column)), section_ids
        )
        kb_scores = []
        for cand in candidates:
            ev_text, _section = select_short_evidence(
                sections, cand, args.max_evidence_chars
            )
            kb_scores.append(candidate_match_score(cand, ev_text))
        kb_best_idx = max(range(len(candidates)), key=lambda i: kb_scores[i])
        kb_best_score = kb_scores[kb_best_idx] if kb_scores else 0.0
        if kb_best_score > 0:
            return candidates[kb_best_idx], "best_match_kb", kb_best_idx

    return candidates[0], "first_candidate", 0


def default_prompt_path():
    return os.path.join(os.path.dirname(__file__), "evqa_final_check_prompt.txt")


def default_kb_path():
    return os.path.join(os.path.dirname(__file__), "encyclopedic_kb_wiki.json")


def default_output_jsonl_path():
    return os.path.join("EVQA_results_final_check", "evqa_final_check.jsonl")


def default_output_csv_path():
    return os.path.join("EVQA_results_final_check", "evqa_final_check.csv")


def main():
    parser = argparse.ArgumentParser(description="Final check for EVQA question fixes.")
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
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--q-clear-column", default="question_clarity_tag")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--kb-path", default=default_kb_path())
    parser.add_argument("--no-kb", action="store_true")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--multi-answer-delim", default="&&")
    parser.add_argument("--skip-q-clear", action="store_true", default=True)
    parser.add_argument(
        "--no-skip-q-clear",
        dest="skip_q_clear",
        action="store_false",
        help="Process Q_clear rows too.",
    )
    parser.add_argument("--overwrite-answer", action="store_true", default=True)
    parser.add_argument(
        "--no-overwrite-answer",
        dest="overwrite_answer",
        action="store_false",
        help="Keep original answer column; write final answer to final_answer.",
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

    fieldnames = get_fieldnames(args.input_csv)
    question_column = resolve_question_column(fieldnames, args.question_column)
    if not question_column:
        print("Missing question column.", file=sys.stderr)
        sys.exit(1)
    if args.id_column not in fieldnames:
        print(f"Missing {args.id_column} column; run evqa_add_data_id.py first.", file=sys.stderr)
        sys.exit(1)

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
            "kb_missing_url",
            "kb_missing_ids",
            "answer_original",
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

            question_original = normalize_text(row.get(question_column))
            question_suggested = normalize_text(row.get("suggested_question"))
            question_used, question_source = select_question(row, question_column)

            answer_raw = normalize_text(row.get(args.answer_column))
            candidates = split_answer_candidates(answer_raw, multi_delim=args.multi_answer_delim)
            multi_answer = is_multi_answer(candidates, multi_delim=args.multi_answer_delim)
            answer_primary = candidates[0] if candidates else ""
            chosen_index = 0

            if not candidates:
                final_answer = ""
                answer_source = "empty"
            elif not multi_answer:
                if len(candidates) == 1:
                    final_answer = answer_primary
                    answer_source = "single_candidate"
                else:
                    final_answer, answer_source, chosen_index = choose_best_candidate(
                        candidates, row, args, kb
                    )
            else:
                final_answer = answer_raw
                answer_source = "multi_answer_preserved"

            if not multi_answer and final_answer:
                answer_eval = final_answer
            else:
                answer_eval = answer_primary if answer_primary else final_answer
            evidence_text, evidence_source, kb_meta = select_evidence_text(
                row, chosen_index, answer_eval, kb, args
            )

            skip_row = False
            clarity_tag = normalize_text(row.get(args.q_clear_column))
            if args.skip_q_clear and clarity_tag == "Q_clear":
                skip_row = True

            final_tag = "Q_clear" if skip_row else ""
            final_reason = "skipped_q_clear" if skip_row else ""
            final_revised = ""
            final_answer_leak = False

            if not skip_row:
                prompt_text = build_prompt(
                    prompt_template,
                    question_original,
                    question_suggested,
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
                                "question_used": question_used,
                                "answer": answer_eval,
                                "raw_response": content,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

                final_tag, final_reason, final_revised = parse_response(content)

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
                "final_answer": final_answer,
                "final_answer_source": answer_source,
                "final_evidence_used": evidence_text,
                "final_evidence_source": evidence_source,
                "final_answer_leak": final_answer_leak,
                "final_model": args.model,
                "kb_missing_url": kb_meta.get("missing_url", False),
                "kb_missing_ids": kb_meta.get("missing_ids", []),
            }
            out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
            out_f.flush()

            row = dict(row)
            if "answer_original" not in row:
                row["answer_original"] = answer_raw
            if args.overwrite_answer:
                row[args.answer_column] = final_answer

            row["final_check_tag"] = final_tag
            row["final_check_reason"] = final_reason
            row["final_revised_question"] = final_revised
            row["final_question_used"] = question_used
            row["final_question_source"] = question_source
            row["final_answer"] = final_answer
            row["final_answer_source"] = answer_source
            row["final_evidence_used"] = evidence_text
            row["final_evidence_source"] = evidence_source
            row["final_answer_leak"] = final_answer_leak
            row["final_model"] = args.model
            row["kb_missing_url"] = kb_meta.get("missing_url", False)
            row["kb_missing_ids"] = kb_meta.get("missing_ids", [])

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
