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
    "partial",
}


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def split_pipe_field(text):
    return split_answer_candidates(text)


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


def split_multi_answer(text, multi_delim="&&"):
    parts = [p.strip() for p in str(text).split(multi_delim) if p.strip()]
    seen = set()
    out = []
    for part in parts:
        key = normalize_text(part).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def normalize_for_match(text):
    text = normalize_text(text).lower()
    return re.sub(r"\s+", " ", text)


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


def chunk_text_plain(text, window_size, stride, max_windows):
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


def chunk_text(text, window_size, stride, max_windows, prefer_paragraphs=False):
    if not text:
        return []
    if window_size <= 0:
        return [text]
    if prefer_paragraphs and "\n" in text:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if paragraphs:
            chunks = []
            current = []
            current_len = 0
            for para in paragraphs:
                if len(para) > window_size:
                    if current:
                        chunks.append("\n\n".join(current))
                        if max_windows > 0 and len(chunks) >= max_windows:
                            return chunks
                        current = []
                        current_len = 0
                    remaining = max_windows - len(chunks) if max_windows > 0 else 0
                    if remaining == 0 and max_windows > 0:
                        return chunks
                    para_chunks = chunk_text_plain(para, window_size, stride, remaining)
                    chunks.extend(para_chunks)
                    if max_windows > 0 and len(chunks) >= max_windows:
                        return chunks
                    continue
                if not current:
                    current = [para]
                    current_len = len(para)
                    continue
                if current_len + 2 + len(para) <= window_size:
                    current.append(para)
                    current_len += 2 + len(para)
                else:
                    chunks.append("\n\n".join(current))
                    if max_windows > 0 and len(chunks) >= max_windows:
                        return chunks
                    current = [para]
                    current_len = len(para)
            if current and (max_windows <= 0 or len(chunks) < max_windows):
                chunks.append("\n\n".join(current))
            return chunks
    return chunk_text_plain(text, window_size, stride, max_windows)


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


def parse_audit_response(content):
    parsed = extract_json_block(content)
    if not parsed:
        return {
            "label": "incorrect",
            "reason": "unparseable_response",
            "improve_type": "",
            "suggested_answer": "",
            "suggested_evidence": "",
        }
    label = normalize_text(parsed.get("label")).lower()
    reason = normalize_text(parsed.get("reason"))
    improve_type = normalize_text(parsed.get("improve_type")).lower()
    suggested_answer = normalize_text(parsed.get("suggested_answer"))
    suggested_evidence = clean_text(parsed.get("suggested_evidence"))
    if label not in LABELS:
        return {
            "label": "incorrect",
            "reason": "invalid_label",
            "improve_type": "",
            "suggested_answer": "",
            "suggested_evidence": "",
        }
    if label != "improvable":
        improve_type = ""
    elif improve_type not in IMPROVE_TYPES:
        improve_type = "precision"
    return {
        "label": label,
        "reason": reason,
        "improve_type": improve_type,
        "suggested_answer": suggested_answer,
        "suggested_evidence": suggested_evidence,
    }


def score_label(label):
    if label == "improvable":
        return 3
    if label == "good":
        return 2
    if label == "incorrect":
        return 1
    return 0


def combine_part_results(parts, results):
    supported = []
    missing = []
    contradicted = []
    reasons = []
    suggested_evidence = ""
    improve_type = ""
    any_improvable = False

    for part, res in zip(parts, results):
        label = res["label"]
        if label in {"good", "improvable"}:
            supported.append(res["suggested_answer"] or part)
            if label == "improvable":
                any_improvable = True
                if not improve_type:
                    improve_type = res["improve_type"] or "precision"
        elif label == "incorrect":
            contradicted.append(part)
        else:
            missing.append(part)
        if res["reason"]:
            reasons.append(f"{part}: {res['reason']}")
        if not suggested_evidence and res["suggested_evidence"]:
            suggested_evidence = res["suggested_evidence"]

    if supported and len(supported) == len(parts):
        label = "improvable" if any_improvable else "good"
        if label == "improvable" and not improve_type:
            improve_type = "precision"
        suggested_answer = "&&".join(supported)
    elif supported:
        label = "improvable"
        improve_type = improve_type or "partial"
        suggested_answer = "&&".join(supported)
    else:
        if contradicted:
            label = "incorrect"
        else:
            label = "missing_evidence"
        improve_type = ""
        suggested_answer = ""
        suggested_evidence = ""

    reason = "; ".join(reasons)
    if label == "improvable" and not reason:
        reason = "partial_support"
    return {
        "label": label,
        "reason": reason,
        "improve_type": improve_type,
        "suggested_answer": suggested_answer,
        "suggested_evidence": suggested_evidence,
    }


def check_evidence_in_kb(csv_evidence, kb_evidence):
    if not csv_evidence or not kb_evidence:
        return "unknown"
    kb_norm = normalize_for_match(kb_evidence)
    parts = split_pipe_field(csv_evidence) or [csv_evidence]
    matched = 0
    for part in parts:
        part_norm = normalize_for_match(part)
        if part_norm and part_norm in kb_norm:
            matched += 1
    if matched == 0:
        return "missing"
    if matched == len(parts):
        return "match"
    return "partial"


def build_messages(question, answers, evidence):
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
        "Short answers can be correct if they directly answer the question. "
        "For multi-answer strings containing '&&', all parts must be supported to be good; "
        "if only some parts are supported, label 'improvable' with improve_type 'partial' "
        "and suggest only the supported parts. "
        "improve_type must be one of: precision, unit, range, entity, normalization, evidence_span, partial."
    )
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


def main():
    parser = argparse.ArgumentParser(description="Audit annotation quality with LLM.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--evidence-column", default="evidence")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--section-id-column", default="evidence_section_id")
    parser.add_argument("--section-title-column", default="evidence_section_title")
    parser.add_argument("--question-type-column", default="question_type")
    parser.add_argument(
        "--skip-question-types",
        default="",
        help="Comma-separated question_type values to skip (e.g. 2_hop).",
    )
    parser.add_argument("--kb-path", default="")
    parser.add_argument("--kb-for-templated", action="store_true")
    parser.add_argument(
        "--extend-templated-evidence",
        action="store_true",
        help="Append KB evidence to CSV evidence for templated questions.",
    )
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-evidence-chars", type=int, default=2000)
    parser.add_argument("--window-size", type=int, default=1500)
    parser.add_argument("--window-stride", type=int, default=800)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--multi-answer-delim", default="&&")
    parser.add_argument(
        "--multi-answer-mode",
        choices=["per_part", "single"],
        default="per_part",
        help="How to score answers containing multi-part delimiters.",
    )
    parser.add_argument(
        "--dump-raw",
        default="",
        help="Write raw LLM responses to this JSONL file for debugging.",
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
        "--no-chunk-by-paragraph",
        action="store_true",
        help="Disable paragraph-based chunking for long evidence.",
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

    chunk_by_paragraph = not args.no_chunk_by_paragraph
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    skip_types = {t.strip() for t in args.skip_question_types.split(",") if t.strip()}
    kb = EvqaKnowledgeBase(args.kb_path) if args.kb_path else None

    raw_f = None
    if args.dump_raw:
        os.makedirs(os.path.dirname(args.dump_raw) or ".", exist_ok=True)
        raw_f = open(args.dump_raw, "w", encoding="utf-8")

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f_in, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as f_jsonl, open(args.output_csv, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        input_fieldnames = reader.fieldnames or []
        if args.id_column not in input_fieldnames:
            print(
                f"Missing {args.id_column} column; run evqa_add_data_id.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        fieldnames = list(input_fieldnames)
        extra_cols = [
            "annotation_quality_label",
            "annotation_quality_reason",
            "annotation_quality_improve_type",
            "suggested_answer",
            "suggested_evidence",
            "evidence_source",
            "csv_evidence_in_kb",
        ]
        extra_cols = [col for col in extra_cols if col not in fieldnames]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames + extra_cols)
        writer.writeheader()

        for idx, row in enumerate(reader):
            data_id = normalize_text(row.get(args.id_column))
            if not data_id:
                print(
                    f"Missing {args.id_column} at row {idx}; run evqa_add_data_id.py first.",
                    file=sys.stderr,
                )
                continue
            question = normalize_text(row.get(args.question_column))
            answer = normalize_text(row.get(args.answer_column))
            evidence = normalize_text(row.get(args.evidence_column))
            if evidence == "0":
                evidence = ""
            question_type = normalize_text(row.get(args.question_type_column))
            section_ids = parse_section_ids(row.get(args.section_id_column))

            if question_type in skip_types:
                continue

            if "|" in answer:
                answer = "|".join(
                    split_answer_candidates(answer, multi_delim=args.multi_answer_delim)
                )

            evidence_source = "csv"
            csv_evidence_in_kb = "unknown"
            evidence_input = evidence.replace("|", "\n\n") if evidence else ""
            kb_evidence = None
            if kb and section_ids:
                kb_evidence = kb.get_evidence(
                    normalize_text(row.get(args.url_column)), section_ids
                )
                if kb_evidence and evidence:
                    csv_evidence_in_kb = check_evidence_in_kb(evidence, kb_evidence)

            csv_has_evidence = bool(evidence)
            if question_type == "templated" or (question_type == "multi_answer" and csv_has_evidence):
                evidence_input = evidence.replace("|", "\n\n") if evidence else ""
                evidence_source = "csv"
                if args.extend_templated_evidence and kb_evidence:
                    if evidence_input:
                        evidence_input = evidence_input + "\n\n" + kb_evidence
                        evidence_source = "csv+kb"
                    else:
                        evidence_input = kb_evidence
                        evidence_source = "kb"
                elif args.kb_for_templated and kb_evidence:
                    evidence_input = kb_evidence
                    evidence_source = "kb"
            elif question_type == "multi_answer":
                if kb_evidence:
                    evidence_input = kb_evidence
                    evidence_source = "kb"
                else:
                    evidence_input = evidence.replace("|", "\n\n") if evidence else ""
                    evidence_source = "csv"
            else:
                if kb_evidence:
                    evidence_input = kb_evidence
                    evidence_source = "kb"
                else:
                    evidence_input = evidence.replace("|", "\n\n") if evidence else ""
                    evidence_source = "csv"

            if args.max_evidence_chars > 0 and len(evidence_input) > args.max_evidence_chars:
                evidence_input = evidence_input[: args.max_evidence_chars] + " ..."

            candidates = (
                split_answer_candidates(answer, multi_delim=args.multi_answer_delim) if answer else []
            )
            use_multi_parts = (
                args.multi_answer_mode == "per_part"
                and (question_type == "multi_answer" or args.multi_answer_delim in answer)
            )
            multi_candidates = []
            if use_multi_parts:
                for candidate in candidates or [answer]:
                    parts = split_multi_answer(candidate, multi_delim=args.multi_answer_delim)
                    if parts:
                        multi_candidates.append(parts)
                if not multi_candidates:
                    use_multi_parts = False

            if args.skip_empty_evidence and (not evidence_input or evidence_input == "0"):
                label = "missing_evidence"
                reason = "empty_or_zero_evidence"
                improve_type = ""
                suggested_answer = ""
                suggested_evidence = ""
            else:
                windows = chunk_text(
                    evidence_input,
                    args.window_size,
                    args.window_stride,
                    args.max_windows,
                    prefer_paragraphs=chunk_by_paragraph,
                )
                if not windows:
                    windows = [evidence_input]

                best = None
                best_score = -1
                for w_idx, window in enumerate(windows):
                    if not use_multi_parts:
                        messages = build_messages(question, answer, window)
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
                            result = {
                                "label": "incorrect",
                                "reason": f"llm_error:{exc}",
                                "improve_type": "",
                                "suggested_answer": "",
                                "suggested_evidence": "",
                            }
                        else:
                            if raw_f is not None:
                                raw_f.write(
                                    json.dumps(
                                        {
                                            "data_id": data_id,
                                            "question": question,
                                            "answer": answer,
                                            "window_index": w_idx,
                                            "raw_response": content,
                                        },
                                        ensure_ascii=True,
                                    )
                                    + "\n"
                                )
                            result = parse_audit_response(content)

                        score = score_label(result["label"])
                        if score > best_score:
                            best_score = score
                            best = result
                            if result["label"] == "improvable":
                                break

                        if args.sleep:
                            time.sleep(args.sleep)
                        continue

                    window_best = None
                    window_best_score = -1
                    for c_idx, candidate_parts in enumerate(multi_candidates):
                        part_results = []
                        for p_idx, part in enumerate(candidate_parts):
                            messages = build_messages(question, part, window)
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
                                part_result = {
                                    "label": "incorrect",
                                    "reason": f"llm_error:{exc}",
                                    "improve_type": "",
                                    "suggested_answer": "",
                                    "suggested_evidence": "",
                                }
                            else:
                                if raw_f is not None:
                                    raw_f.write(
                                        json.dumps(
                                            {
                                                "data_id": data_id,
                                                "question": question,
                                                "answer": answer,
                                                "answer_part": part,
                                                "candidate_index": c_idx,
                                                "part_index": p_idx,
                                                "window_index": w_idx,
                                                "raw_response": content,
                                            },
                                            ensure_ascii=True,
                                        )
                                        + "\n"
                                    )
                                part_result = parse_audit_response(content)

                            part_results.append(part_result)

                            if args.sleep:
                                time.sleep(args.sleep)

                        combined = combine_part_results(candidate_parts, part_results)
                        score = score_label(combined["label"])
                        if score > window_best_score:
                            window_best_score = score
                            window_best = combined
                            if combined["label"] == "improvable":
                                break

                    if window_best and window_best_score > best_score:
                        best_score = window_best_score
                        best = window_best
                        if window_best["label"] == "improvable":
                            break

                if best:
                    label = best["label"]
                    reason = best["reason"]
                    improve_type = best["improve_type"]
                    suggested_answer = best["suggested_answer"]
                    suggested_evidence = best["suggested_evidence"]

            if label == "improvable" and args.apply_suggestions:
                if suggested_answer:
                    row[args.answer_column] = suggested_answer
                if suggested_evidence:
                    row[args.evidence_column] = suggested_evidence

            row[args.id_column] = data_id
            row["annotation_quality_label"] = label
            row["annotation_quality_reason"] = reason
            row["annotation_quality_improve_type"] = improve_type
            row["suggested_answer"] = suggested_answer
            row["suggested_evidence"] = suggested_evidence
            row["evidence_source"] = evidence_source
            row["csv_evidence_in_kb"] = csv_evidence_in_kb

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
                        "evidence_source": evidence_source,
                        "csv_evidence_in_kb": csv_evidence_in_kb,
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
