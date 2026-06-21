#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request

LABELS = {"entailed", "contradicted", "not_supported"}


def split_pipe_field(text):
    return split_answer_candidates(text)


def split_pipe_list(text):
    if text is None:
        return []
    return [p.strip() for p in str(text).split("|") if p.strip()]


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


def parse_judge_response(content):
    parsed = extract_json_block(content)
    if not parsed or parsed.get("label") not in LABELS:
        return "not_supported", "unparseable_or_missing"
    label = parsed.get("label")
    reason = normalize_text(parsed.get("reason"))
    return label, reason


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


def normalize_title(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compare_titles(csv_titles, kb_titles):
    if not csv_titles or not kb_titles:
        return "unknown"
    csv_norm = [normalize_title(t) for t in csv_titles if t]
    kb_norm = [normalize_title(t) for t in kb_titles if t]
    if not csv_norm or not kb_norm:
        return "unknown"
    if len(csv_norm) == len(kb_norm):
        if all(c == k for c, k in zip(csv_norm, kb_norm)):
            return "match"
        return "mismatch"
    if set(csv_norm) & set(kb_norm):
        return "partial"
    return "mismatch"


def check_evidence_in_kb(csv_evidence, kb_evidence):
    if not csv_evidence or not kb_evidence:
        return "unknown"
    kb_norm = normalize_for_match(kb_evidence)
    parts = split_pipe_list(csv_evidence) or [csv_evidence]
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

    def get_evidence(self, url, section_ids):
        if not url or not section_ids:
            return None, [], {"missing_url": False, "missing_ids": section_ids}
        entry = self.data.get(url)
        if not entry:
            return None, [], {"missing_url": True, "missing_ids": section_ids}
        sections = self._extract_sections(entry)
        if not sections:
            return None, [], {"missing_url": False, "missing_ids": section_ids}
        titles = self._extract_titles(entry, len(sections))
        texts = []
        selected_titles = []
        missing = []
        for sid in section_ids:
            if sid < 0 or sid >= len(sections):
                missing.append(sid)
                continue
            texts.append(sections[sid])
            title = titles[sid] if sid < len(titles) else ""
            selected_titles.append(title)
        evidence = "\n\n".join([t for t in texts if t])
        return evidence, selected_titles, {"missing_url": False, "missing_ids": missing}


def build_messages(question, answers, evidence, section_title):
    system = (
        "You are a strict verifier for KB-VQA. "
        "Use only the evidence text. "
        "Decide if the evidence supports the answer to the question. "
        "Label must be one of: entailed, contradicted, not_supported. "
        "If any candidate answer is supported, label entailed. "
        "If a candidate answer contains '&&', it represents multiple required parts. "
        "Such a candidate is entailed only if all parts are supported by the evidence. "
        "If evidence directly conflicts with the answer, label contradicted. "
        "If evidence is insufficient, label not_supported. "
        "Return JSON with keys: label, reason."
    )
    user = {
        "question": question,
        "candidate_answers": answers,
        "evidence_section_title": section_title,
        "evidence_section_text": evidence,
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


def iter_rows(input_path):
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            yield idx, row


def get_fieldnames(input_path):
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or []


def main():
    parser = argparse.ArgumentParser(description="Judge E-VQA evidence with LLM.")
    parser.add_argument("--input", required=True, help="Input CSV with E-VQA fields.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--evidence-column", default="evidence")
    parser.add_argument("--section-title-column", default="evidence_section_title")
    parser.add_argument("--section-id-column", default="evidence_section_id")
    parser.add_argument("--id-column", default="data_id")
    parser.add_argument("--url-column", default="wikipedia_url")
    parser.add_argument("--question-type-column", default="question_type")
    parser.add_argument(
        "--skip-question-types",
        default="",
        help="Comma-separated question_type values to skip (e.g. 2_hop,multi_answer).",
    )
    parser.add_argument("--provider", default="openai_compat")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-evidence-chars", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=1500)
    parser.add_argument("--window-stride", type=int, default=800)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--kb-path", default="")
    parser.add_argument("--kb-for-templated", action="store_true")
    parser.add_argument(
        "--extend-templated-evidence",
        action="store_true",
        help="Append KB evidence to CSV evidence for templated questions.",
    )
    parser.add_argument("--split-answers", action="store_true", default=None)
    parser.add_argument("--no-split-answers", action="store_true")
    parser.add_argument("--multi-answer-delim", default="&&")
    parser.add_argument(
        "--multi-answer-mode",
        choices=["per_part", "aggregate"],
        default="per_part",
        help="How to score answers containing multi-part delimiters.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--include-evidence", action="store_true")
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

    fieldnames = get_fieldnames(args.input)
    if args.id_column not in fieldnames:
        print(
            f"Missing {args.id_column} column; run evqa_add_data_id.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    chunk_by_paragraph = not args.no_chunk_by_paragraph
    split_answers = args.split_answers
    if split_answers is None:
        split_answers = args.answer_column == "answer"
    if args.no_split_answers:
        split_answers = False

    skip_types = {t.strip() for t in args.skip_question_types.split(",") if t.strip()}

    kb = None
    if args.kb_path:
        kb = EvqaKnowledgeBase(args.kb_path)

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
            answer_raw = normalize_text(row.get(args.answer_column))
            evidence_raw = normalize_text(row.get(args.evidence_column))
            if evidence_raw == "0":
                evidence_raw = ""
            csv_section_title = normalize_text(row.get(args.section_title_column))
            section_ids = parse_section_ids(row.get(args.section_id_column))
            question_type = normalize_text(row.get(args.question_type_column))
            if question_type in skip_types:
                continue
            evidence_source = "csv"
            kb_titles = []
            kb_meta = {"missing_url": False, "missing_ids": []}
            title_check = "unknown"
            csv_evidence_in_kb = "unknown"
            kb_evidence = None

            if split_answers:
                answers = split_answer_candidates(answer_raw, multi_delim=args.multi_answer_delim)
            else:
                answers = [answer_raw] if answer_raw else []

            use_multi_parts = False
            multi_candidates = []
            if args.multi_answer_mode == "per_part" and answers:
                if question_type == "multi_answer" or any(
                    args.multi_answer_delim in a for a in answers
                ):
                    use_multi_parts = True
                    for candidate in answers:
                        parts = split_multi_answer(candidate, multi_delim=args.multi_answer_delim)
                        if parts:
                            multi_candidates.append(parts)
                    if not multi_candidates:
                        use_multi_parts = False

            section_title = csv_section_title
            if kb is not None and section_ids:
                kb_evidence, kb_titles, kb_meta = kb.get_evidence(
                    normalize_text(row.get(args.url_column)), section_ids
                )
                if kb_evidence and evidence_raw:
                    csv_evidence_in_kb = check_evidence_in_kb(evidence_raw, kb_evidence)
                csv_titles = split_pipe_list(csv_section_title)
                title_check = compare_titles(csv_titles, kb_titles)

            csv_has_evidence = bool(evidence_raw)
            if question_type == "templated" or (question_type == "multi_answer" and csv_has_evidence):
                evidence = evidence_raw
                evidence_source = "csv"
                if args.extend_templated_evidence and kb_evidence:
                    if evidence:
                        evidence = evidence + "\n\n" + kb_evidence
                        evidence_source = "csv+kb"
                    else:
                        evidence = kb_evidence
                        evidence_source = "kb"
                elif args.kb_for_templated and kb_evidence:
                    evidence = kb_evidence
                    evidence_source = "kb"
            elif question_type == "multi_answer":
                if kb_evidence:
                    evidence = kb_evidence
                    evidence_source = "kb"
                else:
                    evidence = evidence_raw
                    evidence_source = "csv"
            else:
                if kb_evidence:
                    evidence = kb_evidence
                    evidence_source = "kb"
                else:
                    evidence = evidence_raw
                    evidence_source = "csv"

            if evidence_source != "csv" and kb_titles:
                section_title = " | ".join(kb_titles)

            if args.max_evidence_chars > 0 and len(evidence) > args.max_evidence_chars:
                evidence = evidence[: args.max_evidence_chars] + " ..."

            windows = chunk_text(
                evidence,
                args.window_size,
                args.window_stride,
                args.max_windows,
                prefer_paragraphs=chunk_by_paragraph,
            )
            if not windows:
                windows = [evidence]

            final_label = "not_supported"
            final_reason = ""
            saw_contradicted = False
            saw_not_supported_reason = ""
            saw_non_contradiction = False
            best_partial = None
            best_contradicted = None

            for w_idx, window in enumerate(windows):
                if not use_multi_parts:
                    messages = build_messages(question, answers, window, section_title)
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
                            "answer_source": args.answer_column,
                            "model": args.model,
                            "window_index": w_idx,
                        }
                        out_f.write(json.dumps(error_obj, ensure_ascii=True) + "\n")
                        out_f.flush()
                        if args.sleep:
                            time.sleep(args.sleep)
                        continue

                    label, reason = parse_judge_response(content)

                    if label == "entailed":
                        final_label = "entailed"
                        final_reason = reason
                        break
                    if label == "contradicted":
                        saw_contradicted = True
                        if not final_reason:
                            final_reason = reason
                    if label == "not_supported" and reason and not saw_not_supported_reason:
                        saw_not_supported_reason = reason

                    if args.sleep:
                        time.sleep(args.sleep)
                    continue

                window_failed = False
                window_any_entail = False
                window_any_non_contradiction = False
                window_any_contradiction = False
                window_partial = None
                window_contradicted = None

                for candidate_parts in multi_candidates:
                    supported = []
                    missing = []
                    contradicted = []
                    part_labels = []
                    for part in candidate_parts:
                        messages = build_messages(question, [part], window, section_title)
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
                                "answer_source": args.answer_column,
                                "model": args.model,
                                "window_index": w_idx,
                                "answer_part": part,
                            }
                            out_f.write(json.dumps(error_obj, ensure_ascii=True) + "\n")
                            out_f.flush()
                            if args.sleep:
                                time.sleep(args.sleep)
                            window_failed = True
                            break

                        label, _reason = parse_judge_response(content)
                        part_labels.append(label)
                        if label == "entailed":
                            supported.append(part)
                        elif label == "contradicted":
                            contradicted.append(part)
                        else:
                            missing.append(part)

                        if args.sleep:
                            time.sleep(args.sleep)

                    if window_failed:
                        break

                    if part_labels and all(lbl == "entailed" for lbl in part_labels):
                        window_any_entail = True
                        final_label = "entailed"
                        final_reason = "all_parts_supported"
                        break

                    if contradicted:
                        window_any_contradiction = True
                        if not window_contradicted or len(contradicted) > len(
                            window_contradicted
                        ):
                            window_contradicted = contradicted
                    else:
                        window_any_non_contradiction = True

                    if supported:
                        if not window_partial or len(supported) > len(window_partial["supported"]):
                            window_partial = {"supported": supported, "missing": missing}

                if window_failed:
                    continue
                if window_any_entail:
                    if window_partial and (
                        not best_partial
                        or len(window_partial["supported"]) > len(best_partial["supported"])
                    ):
                        best_partial = window_partial
                    break

                if window_any_contradiction:
                    saw_contradicted = True
                    if window_contradicted and (
                        not best_contradicted or len(window_contradicted) > len(best_contradicted)
                    ):
                        best_contradicted = window_contradicted
                if window_any_non_contradiction:
                    saw_non_contradiction = True
                if window_partial and (
                    not best_partial
                    or len(window_partial["supported"]) > len(best_partial["supported"])
                ):
                    best_partial = window_partial

            if final_label != "entailed" and saw_contradicted:
                if not saw_non_contradiction:
                    final_label = "contradicted"
            if final_label == "contradicted" and not final_reason:
                if best_contradicted:
                    final_reason = "contradicted_parts=" + ",".join(best_contradicted)
                else:
                    final_reason = "contradicted"
            if final_label == "not_supported" and not final_reason:
                if best_partial:
                    final_reason = (
                        "partial_support: supported="
                        + ",".join(best_partial["supported"])
                        + "; missing="
                        + ",".join(best_partial["missing"])
                    )
                else:
                    final_reason = saw_not_supported_reason or "not_supported"

            result = {
                "data_id": data_id,
                "label": final_label,
                "reason": normalize_text(final_reason),
                "answer_source": args.answer_column,
                "evidence_source": evidence_source,
                "section_ids": section_ids,
                "kb_title_check": title_check,
                "csv_evidence_in_kb": csv_evidence_in_kb,
                "kb_missing_url": kb_meta.get("missing_url", False),
                "kb_missing_ids": kb_meta.get("missing_ids", []),
                "model": args.model,
                "multi_answer": use_multi_parts,
                "multi_supported_parts": best_partial.get("supported", []) if best_partial else [],
                "multi_missing_parts": best_partial.get("missing", []) if best_partial else [],
            }
            if args.include_evidence:
                result.update(
                    {
                        "question": question,
                        "answers": answers,
                        "evidence_section_title": section_title,
                        "evidence": evidence,
                    }
                )

            out_f.write(json.dumps(result, ensure_ascii=True) + "\n")
            out_f.flush()
            written += 1
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Processed {processed} rows, wrote {written} judgments.")


if __name__ == "__main__":
    main()
