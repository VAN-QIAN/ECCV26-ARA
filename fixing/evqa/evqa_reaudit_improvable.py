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


def chunk_text(text, window_size, stride, max_windows):
    if not text:
        return []
    if window_size <= 0:
        return [text]
    if stride <= 0:
        stride = window_size
    if len(text) <= window_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + window_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        if max_windows > 0 and len(chunks) >= max_windows:
            break
        start += stride
    return chunks


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
            return [str(sec) for sec in sections]
        section_texts = entry.get("section_texts")
        if isinstance(section_texts, list):
            return ["" if sec is None else str(sec) for sec in section_texts]
        return None

    def get_evidence(self, url, section_ids):
        if not url or not section_ids:
            return None
        entry = self.data.get(url)
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


def build_messages(question, answer, evidence, reason_limit, evidence_limit):
    system = (
        "You are a strict annotation quality auditor for KB-VQA. "
        "Use only the provided evidence text. "
        "Decide whether the annotation (answer) is correct and sufficiently precise. "
        "IMPORTANT: If the answer already fully answers the question, even if short, label it GOOD. "
        "Do NOT require repeating subject, location, or context already in the question. "
        "Use 'improvable' only if the answer is supported but missing essential specificity "
        "(e.g., missing unit required by the question, missing exact range/value, or ambiguous entity). "
        "If incorrect or unsupported, label 'incorrect' and leave suggested_* empty. "
        "If evidence is empty, use 'missing_evidence'. "
        "Return JSON with keys: label, reason, improve_type, suggested_answer, suggested_evidence. "
        "Label must be one of: good, improvable, incorrect, missing_evidence. "
        "improve_type must be one of: precision, unit, range, entity, normalization, evidence_span. "
        f"Keep reason <= {reason_limit} chars. "
        f"Keep suggested_evidence <= {evidence_limit} chars. "
        "Return JSON only, no extra text."
    )
    user = {
        "question": question,
        "answer": answer,
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


def load_ids(path, label_filter):
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
            if label_filter:
                if normalize_text(obj.get("label")) != label_filter:
                    continue
            data_id = normalize_text(obj.get("data_id"))
            if data_id:
                ids.add(data_id)
    return ids


def best_label(labels):
    if not labels:
        return "incorrect"
    if any(l == "good" for l in labels):
        return "good"
    if any(l == "improvable" for l in labels):
        return "improvable"
    if any(l == "incorrect" for l in labels):
        return "incorrect"
    return "missing_evidence"


def main():
    parser = argparse.ArgumentParser(description="Reaudit improvable EVQA annotations.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--evidence-column", default="evidence")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--section-id-column", default="evidence_section_id")
    parser.add_argument("--question-type-column", default="question_type")
    parser.add_argument("--skip-question-types", default="")
    parser.add_argument("--kb-path", default="")
    parser.add_argument("--kb-for-templated", action="store_true")
    parser.add_argument("--label-filter", default="improvable")
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-evidence-chars", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=1500)
    parser.add_argument("--window-stride", type=int, default=800)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--max-reason-chars", type=int, default=200)
    parser.add_argument("--max-suggested-evidence-chars", type=int, default=200)
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

    target_ids = load_ids(args.audit_jsonl, args.label_filter)
    skip_types = {t.strip() for t in args.skip_question_types.split(",") if t.strip()}
    kb = EvqaKnowledgeBase(args.kb_path) if args.kb_path else None

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    raw_f = None
    if args.dump_raw:
        os.makedirs(os.path.dirname(args.dump_raw) or ".", exist_ok=True)
        raw_f = open(args.dump_raw, "w", encoding="utf-8")

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as f_jsonl, open(args.output_csv, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        extra_cols = [
            "reaudit_label",
            "reaudit_reason",
            "reaudit_improve_type",
            "reaudit_suggested_answer",
            "reaudit_suggested_evidence",
        ]
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames + extra_cols)
        writer.writeheader()

        for idx, row in enumerate(reader):
            data_id = normalize_text(row.get(args.id_column)) or f"row_{idx}"
            if data_id not in target_ids:
                continue

            question = normalize_text(row.get(args.question_column))
            answer_raw = normalize_text(row.get(args.answer_column))
            evidence_raw = normalize_text(row.get(args.evidence_column))
            question_type = normalize_text(row.get(args.question_type_column))
            section_ids = parse_section_ids(row.get(args.section_id_column))

            if question_type in skip_types:
                continue

            if "|" in answer_raw:
                answers = split_pipe_field(answer_raw)
            else:
                answers = [answer_raw] if answer_raw else []

            evidence_source = "csv"
            evidence_input = evidence_raw.replace("|", "\n\n")
            if kb and section_ids and (question_type != "templated" or args.kb_for_templated):
                kb_evidence = kb.get_evidence(normalize_text(row.get(args.url_column)), section_ids)
                if kb_evidence:
                    evidence_input = kb_evidence
                    evidence_source = "kb"

            if args.max_evidence_chars > 0 and len(evidence_input) > args.max_evidence_chars:
                evidence_input = evidence_input[: args.max_evidence_chars] + " ..."

            windows = chunk_text(
                evidence_input, args.window_size, args.window_stride, args.max_windows
            )
            if not windows:
                windows = [evidence_input]

            answer_audits = []
            for ans in answers:
                best = None
                best_score = -1
                for w_idx, window in enumerate(windows):
                    messages = build_messages(
                        question,
                        ans,
                        window,
                        args.max_reason_chars,
                        args.max_suggested_evidence_chars,
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
                                        "window_index": w_idx,
                                        "raw_response": content,
                                    },
                                    ensure_ascii=True,
                                )
                                + "\n"
                            )
                        parsed = extract_json_block(content)
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
                            suggested_evidence = suggested_evidence[: args.max_suggested_evidence_chars]

                    score = 0
                    if label == "good":
                        score = 3
                    elif label == "improvable":
                        score = 2
                    elif label == "incorrect":
                        score = 1
                    elif label == "missing_evidence":
                        score = 0

                    if score > best_score:
                        best_score = score
                        best = {
                            "label": label,
                            "reason": reason,
                            "improve_type": improve_type,
                            "suggested_answer": suggested_answer,
                            "suggested_evidence": suggested_evidence,
                        }
                        if label == "good":
                            break

                if best is None:
                    best = {
                        "label": "incorrect",
                        "reason": "no_windows",
                        "improve_type": "",
                        "suggested_answer": "",
                        "suggested_evidence": "",
                    }
                best["answer"] = ans
                answer_audits.append(best)

            labels = [a["label"] for a in answer_audits]
            final_label = best_label(labels)
            reasons = [a["reason"] for a in answer_audits if a.get("reason")]
            improve_types = [a["improve_type"] for a in answer_audits if a.get("improve_type")]
            suggested_answers = [
                a["suggested_answer"] if a.get("suggested_answer") else a.get("answer", "")
                for a in answer_audits
            ]
            suggested_evidences = [
                a["suggested_evidence"] for a in answer_audits if a.get("suggested_evidence")
            ]

            row["reaudit_label"] = final_label
            row["reaudit_reason"] = " | ".join(reasons)
            row["reaudit_improve_type"] = " | ".join(improve_types)
            row["reaudit_suggested_answer"] = "|".join([a for a in suggested_answers if a])
            row["reaudit_suggested_evidence"] = "|".join(suggested_evidences)

            writer.writerow(row)
            f_jsonl.write(
                json.dumps(
                    {
                        "data_id": data_id,
                        "label": final_label,
                        "reason": row["reaudit_reason"],
                        "improve_type": row["reaudit_improve_type"],
                        "suggested_answer": row["reaudit_suggested_answer"],
                        "suggested_evidence": row["reaudit_suggested_evidence"],
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
