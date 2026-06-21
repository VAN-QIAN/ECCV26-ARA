import argparse
import csv
import os
import sys
from typing import List, Optional

import gc
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import ujson
from PIL import Image, ImageFile
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    collate_and_pad_input_ids,
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from rag_evaluation.encyclopedic.Retriever import ClipRetriever

IMAGE_TOKEN = f"{DEFAULT_IMAGE_TOKEN}\n\n"
RET_TOKEN = 128251
REL_TOKEN = 128253
TEMPLATE = "Consider this paragraph: "


ImageFile.LOAD_TRUNCATED_IMAGES = True


def concat_paragraph(paragraphs_list, question, sections, conv_mode, config, tokenizer):
    template = TEMPLATE
    qs = IMAGE_TOKEN + question
    conv_relevant = conv_templates[conv_mode].copy()
    conv_relevant.append_message(conv_relevant.roles[0], qs)
    qs = "[Retrieval]"
    conv_relevant.append_message(conv_relevant.roles[1], qs)
    qs = template + "<paragraph>"
    for idx in paragraphs_list:
        qs += sections[idx]
    qs += "</paragraph>"

    if args.short_prompt:
        qs += ". Give a short answer."

    conv_relevant.append_message(conv_relevant.roles[0], qs)
    conv_relevant.append_message(conv_relevant.roles[1], None)
    prompt = conv_relevant.get_prompt()
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[
        1:
    ].unsqueeze(0).to(device="cuda:0", non_blocking=True)


def inference(args, model, input_ids, images, image_sizes):
    output_ids = model.generate(
        input_ids,
        images=images.to(dtype=torch.float16, device="cuda:0", non_blocking=True),
        image_sizes=image_sizes,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )

    return output_ids


def resolve_image_path(image_root: str, image_id: str) -> Optional[str]:
    """Try a handful of extensions to locate an image on disk."""
    candidates = [
        f"{image_id}.jpg",
        f"{image_id}.jpeg",
        f"{image_id}.png",
        f"{image_id}.JPG",
        f"{image_id}.JPEG",
        f"{image_id}.PNG",
    ]
    for candidate in candidates:
        candidate_path = os.path.join(image_root, candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def split_text_by_length(text: str, max_chars: int) -> List[str]:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks or [text]


ALLOWED_TYPES = {"automatic", "templated", "multi_answer", "infoseek", "String", "Time", "Numerical"}


def load_csv_entries(path: str) -> List[dict]:
    entries: List[dict] = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        index_question_type = header.index("question_type")
        for row in reader:
            if not row:
                continue
            q_type = row[index_question_type].strip()
            if q_type not in ALLOWED_TYPES:
                continue
            entry = {header[i]: row[i] for i in range(len(header))}
            entries.append(entry)
    return entries


def load_question_filter(csv_path: str):
    questions = set()
    data_ids = set()

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} does not contain a header row.")

        for row in reader:
            if not row:
                continue

            for question_key in ("question", "question_original"):
                question = row.get(question_key)
                if question:
                    question = question.strip()
                    if question:
                        questions.add(question)

            data_id = row.get("data_id")
            if data_id:
                data_id = data_id.strip()
                if data_id:
                    data_ids.add(data_id)

            dataset_image_ids = row.get("dataset_image_ids")
            if dataset_image_ids:
                for dataset_image_id in dataset_image_ids.split("|"):
                    dataset_image_id = dataset_image_id.strip()
                    if dataset_image_id:
                        data_ids.add(dataset_image_id)

    return questions, data_ids


def filter_entries(entries: List[dict], args) -> List[dict]:
    if not args.has_question_filter:
        return entries

    filtered = []
    for entry in entries:
        question = entry.get("question", "")
        data_id = entry.get("data_id", "")
        keep_sample = False
        if args.question_whitelist and question.strip() in args.question_whitelist:
            keep_sample = True
        if not keep_sample and args.data_id_whitelist and data_id in args.data_id_whitelist:
            keep_sample = True
        if keep_sample:
            filtered.append(entry)
    return filtered


def prepare_sample(
    entry,
    retriever,
    tokenizer,
    image_processor,
    model_config,
    image_root: str,
) -> Optional[dict]:
    question = entry.get("question")
    if not question:
        return None

    data_id = entry.get("data_id", "")
    image_id = entry.get("dataset_image_ids") or entry.get("image_id")
    if not image_id:
        return None
    image_id = image_id.split("|")[0].strip()

    image_path = resolve_image_path(image_root, image_id)
    if image_path is None:
        print(f"[Warning] Could not locate image for {data_id} (image_id={image_id}). Skipping.")
        return None
    # print(f"Loading image from {image_path}")
    pil_img = Image.open(image_path).convert("RGB")

    max_candidates = len(retriever.faiss_index_ids) if getattr(retriever, "faiss_index_ids", None) else 0
    # print(f"Max candidates in FAISS index: {max_candidates}")
    if max_candidates == 0:
        return None

    request_k = min(2 * args.entity_k, max_candidates)
    with torch.no_grad():
        kb_candidates = retriever.retrieve_image_faiss(
            pil_img,
            top_k=request_k,
        )
    # print(f"Retrieved {len(kb_candidates)} candidates from retriever.")

    unique_entries = []
    seen_urls = set()
    for candidate in kb_candidates:
        if not isinstance(candidate, dict):
            continue
        entry_obj = candidate.get("kb_entry")
        entry_url = candidate.get("url")
        if entry_obj is None:
            continue
        if entry_url is None:
            entry_url = id(entry_obj)
        if entry_url in seen_urls:
            continue
        seen_urls.add(entry_url)
        unique_entries.append(entry_obj)
        if len(unique_entries) >= args.entity_k:
            break
    kb_entries = unique_entries

    sections = []
    for entry_obj in kb_entries:
        for section_text in entry_obj.section_texts:
            section_text = section_text.strip()
            if not section_text:
                continue
            if args.max_section_chars:
                sections.extend(split_text_by_length(section_text, args.max_section_chars))
            else:
                sections.append(section_text)

    sections = [section.strip() for section in sections if section.strip()]
    if len(sections) == 0:
        sections = [""]

    total_qs_relevant = []
    for section in sections:
        qs_retrieval = IMAGE_TOKEN + question
        conv_relevant = conv_templates[args.conv_mode].copy()
        qs_relevant = IMAGE_TOKEN + question
        conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
        qs_relevant = "[Retrieval]"
        conv_relevant.append_message(conv_relevant.roles[1], qs_relevant)
        qs_relevant = TEMPLATE + "<paragraph>"
        qs_relevant += section + "</paragraph>"

        conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
        conv_relevant.append_message(conv_relevant.roles[1], None)
        prompt_relevant = conv_relevant.get_prompt()

        input_ids_relevant = tokenizer_image_token(
            prompt_relevant, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        total_qs_relevant.append(input_ids_relevant[1:])

    conv_retrieval = conv_templates[args.conv_mode].copy()
    qs_retrieval = IMAGE_TOKEN + question
    conv_retrieval.append_message(conv_retrieval.roles[0], qs_retrieval)
    conv_retrieval.append_message(conv_retrieval.roles[1], None)
    prompt_retrieval = conv_retrieval.get_prompt()

    input_ids_retrieval = tokenizer_image_token(
        prompt_retrieval, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )

    image_tensor = process_images([pil_img], image_processor, model_config)[0]

    return {
        "img": image_tensor,
        "img_size": pil_img.size,
        "retrieval_input_ids": input_ids_retrieval[1:],
        "relevant_input_ids": total_qs_relevant,
        "question": question,
        "sections": sections,
        "data_id": data_id,
    }


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name if args.model_name else get_model_name_from_path(model_path)

    # tokenizer, model, image_processor, _ = load_pretrained_model(
    #     model_path, None, model_name, device_map="cuda:0", use_flash_attn=True
    # )
    # model.to("cuda:0")
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    model.cuda()

    entries = load_csv_entries(args.input_csv)
    total_entries = len(entries)

    if args.question_csv:
        question_whitelist, data_id_whitelist = load_question_filter(args.question_csv)
        args.question_whitelist = question_whitelist if question_whitelist else None
        args.data_id_whitelist = data_id_whitelist if data_id_whitelist else None
        args.has_question_filter = bool(args.question_whitelist or args.data_id_whitelist)
    else:
        args.question_whitelist = None
        args.data_id_whitelist = None
        args.has_question_filter = False

    if args.has_question_filter:
        entries = filter_entries(entries, args)
        total_entries = len(entries)

    part_env = os.environ.get("PART")
    total_part_env = os.environ.get("TOTAL_PART")
    part = int(part_env) if part_env is not None else 0
    if args.samples_per_part:
        slicing = args.samples_per_part
        total_part = max(1, (total_entries + slicing - 1) // slicing)
    else:
        total_part = int(total_part_env) + 1 if total_part_env is not None else 1
        slicing = max(1, total_entries // total_part) if total_part > 0 else total_entries
    print(f"Computing split {part} of the dataset (total entries: {total_entries})...")
    if args.samples_per_part:
        args.start_idx = slicing * part
        args.end_idx = min(total_entries, args.start_idx + slicing)
    else:
        if (part + 1) == total_part:
            args.start_idx = slicing * part
            args.end_idx = total_entries
        else:
            args.start_idx = slicing * part
            args.end_idx = slicing * part + slicing
    print(f"Processing elements from {args.start_idx} to {args.end_idx}...")

    entries = entries[args.start_idx : args.end_idx]

    device = torch.device("cuda:0")
    retriever_model = "eva-clip" if args.use_eva_to_retrieve else "clip"
    retriever = ClipRetriever(model=retriever_model, device=device)
    retriever.load_knowledge_base(args.kb_wikipedia_path)
    index_dir = args.index_path
    if index_dir.endswith("kb_index.faiss"):
        index_dir = os.path.dirname(index_dir)
    if index_dir and not index_dir.endswith(os.sep):
        index_dir += os.sep
    retriever.load_faiss_index(index_dir)

    out_data = []
    for entry in tqdm(entries, mininterval=1):
        print(entry)
        print(f"Processing data_id: {entry.get('data_id', '')}")
        sample = prepare_sample(
            entry,
            retriever,
            tokenizer,
            image_processor,
            model.config,
            args.image_root,
        )
        print(f"Prepared sample for data_id: {entry.get('data_id', '')}")
        if sample is None:
            continue

        retrieval_ids = collate_and_pad_input_ids(
            [sample["retrieval_input_ids"]], tokenizer.pad_token_id, "left"
        ).to(device="cuda:0", non_blocking=True)
        relevant_ids_list = [
            collate_and_pad_input_ids([el], tokenizer.pad_token_id, "left")
            for el in sample["relevant_input_ids"]
        ]
        images = sample["img"].unsqueeze(0)
        image_sizes = sample["img_size"]
        sections = sample["sections"]
        
        print(f"Running retrieval inference for data_id: {entry.get('data_id', '')}")

        with torch.inference_mode():
            output_ids = inference(args, model, retrieval_ids, images, image_sizes)

        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        torch.cuda.empty_cache()
        gc.collect()

        relevant_paragraphs_detected = []
        if RET_TOKEN == output_ids[0][0].item() or args.fix_ret_token:
            for id_context, element in enumerate(relevant_ids_list):
                relevant_ids = element.to(device="cuda:0", non_blocking=True)

                with torch.inference_mode():
                    output_ids = inference(args, model, relevant_ids, images, image_sizes)

                if REL_TOKEN == output_ids[0][0].item():
                    relevant_paragraphs_detected.append(id_context)

            if len(relevant_paragraphs_detected) == 0:
                relevant_paragraphs_detected = [random.randint(0, len(sections) - 1)]

            with torch.inference_mode():
                concat_relevant_paragraphs = concat_paragraph(
                    relevant_paragraphs_detected,
                    sample["question"],
                    sections,
                    args.conv_mode,
                    model.config,
                    tokenizer,
                )
                output_ids = inference(args, model, concat_relevant_paragraphs, images, image_sizes)

        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        del output_ids
        torch.cuda.empty_cache()
        gc.collect()

        out_data.append({"data_id": sample["data_id"], "prediction": answers[0].strip()})

    with open(args.answers_file, "w") as f:
        f.write(ujson.dumps(out_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="llama_3_1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)  # kept for parity

    parser.add_argument("--entity_k", type=int, default=5)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--short_prompt", action="store_true")
    parser.add_argument("--kb_wikipedia_path", type=str)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--max_section_chars", type=int, default=4096)
    parser.add_argument("--use_clip_to_retrieve", action="store_true")
    parser.add_argument("--use_eva_to_retrieve", action="store_true")
    parser.add_argument("--retriever_path", type=str)
    parser.add_argument("--retriever_processor_path", type=str)
    parser.add_argument("--fix_ret_token", type=bool, default=True)
    parser.add_argument("--question_csv", type=str, default=None)
    parser.add_argument("--samples_per_part", type=int, default=None)
    parser.add_argument("--answers_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.use_eva_to_retrieve:
        retriever_model = "eva"
    elif args.use_clip_to_retrieve:
        retriever_model = "clip"
    else:
        retriever_model = "unknown"

    index = "Image2Image"
    part_str = os.environ.get("PART", "0")
    if args.answers_file:
        answers_dir = os.path.dirname(args.answers_file)
        if answers_dir:
            os.makedirs(answers_dir, exist_ok=True)
    else:
        output_dir = args.output_dir or f"output/Reflectiva_infoseek_echo_kb_{retriever_model}_index_{index}"
        os.makedirs(output_dir, exist_ok=True)
        args.answers_file = os.path.join(output_dir, f"split_{part_str}_k{args.entity_k}.json")
    print(f"Saving answers to {args.answers_file}")

    eval_model(args)
