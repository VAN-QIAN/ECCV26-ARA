import argparse
import gc
import json
import os
import random
import re
import math
import time

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
import yaml

from retriever import ClipRetriever

import torch.distributed as dist
import torch.multiprocessing as mp

def extract_bbox(data):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None

    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, dict):
            return first_item.get("bbox_2d")
    elif isinstance(data, dict):
        return data.get("bbox_2d")

    return None

def setup(rank, world_size, master_port):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def dedupe_list_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def eval_recall(candidates, ground_truth, top_ks=[1, 5]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def extract_urls_from_retrieval_items(retrieved_items):
    urls = []
    if not isinstance(retrieved_items, list):
        return urls
    for item in retrieved_items:
        if isinstance(item, dict):
            item_url = (item.get("url") or "").strip()
            if item_url:
                urls.append(item_url)
        elif isinstance(item, str):
            item_url = item.strip()
            if item_url:
                urls.append(item_url)
    return dedupe_list_keep_order(urls)


def update_tool_recall_hits(
    stats: dict,
    tool_name: str,
    target_url: str,
    retrieved_entries: list,
    top_k: int,
) -> None:
    target_url = (target_url or "").strip()
    if not target_url:
        return

    retrieved_urls = extract_urls_from_retrieval_items(retrieved_entries)
    if not retrieved_urls:
        return

    recall = eval_recall(retrieved_urls, target_url, top_ks=[1, top_k])
    stats[f"{tool_name}_top1_hits"] += recall[1]
    stats[f"{tool_name}_topk_hits"] += recall[top_k]


def _to_json_safe(value, max_depth=6):
    if max_depth < 0:
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v, max_depth=max_depth - 1) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _to_json_safe(v, max_depth=max_depth - 1)
            for k, v in value.items()
        }
    if hasattr(value, "item"):
        try:
            return _to_json_safe(value.item(), max_depth=max_depth - 1)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _to_json_safe(value.tolist(), max_depth=max_depth - 1)
        except Exception:
            pass
    return str(value)


def serialize_kb_entry_metadata(kb_entry):
    if kb_entry is None:
        return None
    if isinstance(kb_entry, dict):
        return {
            "title": _to_json_safe(kb_entry.get("title")),
            "url": _to_json_safe(kb_entry.get("url")),
            "image_urls": _to_json_safe(kb_entry.get("image_urls", [])),
            "image_reference_descriptions": _to_json_safe(kb_entry.get("image_reference_descriptions", [])),
            "image_section_indices": _to_json_safe(kb_entry.get("image_section_indices", [])),
            "section_titles": _to_json_safe(kb_entry.get("section_titles", [])),
            "section_texts": _to_json_safe(kb_entry.get("section_texts", [])),
        }

    return {
        "title": _to_json_safe(getattr(kb_entry, "title", "")),
        "url": _to_json_safe(getattr(kb_entry, "url", "")),
        "image_urls": _to_json_safe(getattr(kb_entry, "image_urls", [])),
        "image_reference_descriptions": _to_json_safe(getattr(kb_entry, "image_reference_descriptions", [])),
        "image_section_indices": _to_json_safe(getattr(kb_entry, "image_section_indices", [])),
        "section_titles": _to_json_safe(getattr(kb_entry, "section_titles", [])),
        "section_texts": _to_json_safe(getattr(kb_entry, "section_texts", [])),
    }


def serialize_retrieval_items_for_metadata(retrieved_items):
    serialized_items = []
    if not isinstance(retrieved_items, list):
        return serialized_items

    for item in retrieved_items:
        if isinstance(item, dict):
            serialized = {}
            for k, v in item.items():
                if k == "kb_entry":
                    serialized[k] = serialize_kb_entry_metadata(v)
                else:
                    serialized[k] = _to_json_safe(v)
            serialized_items.append(serialized)
        else:
            serialized_items.append(_to_json_safe(item))
    return serialized_items


def build_section_candidates_with_provenance(retrieved_items):
    candidates = []
    seen_sections = set()

    if not isinstance(retrieved_items, list):
        return candidates

    for rank_idx, item in enumerate(retrieved_items, start=1):
        if not isinstance(item, dict):
            continue
        entry = item.get("kb_entry")
        if entry is None:
            continue

        section_titles = list(getattr(entry, "section_titles", []) or [])
        section_texts = list(getattr(entry, "section_texts", []) or [])
        entry_title = getattr(entry, "title", "")
        entry_url = getattr(entry, "url", "") or item.get("url", "")
        retrieval_similarity = _to_json_safe(item.get("similarity"))
        knowledge_base_index = _to_json_safe(item.get("knowledge_base_index"))

        for sec_idx, sec_text in enumerate(section_texts):
            if sec_text is None:
                continue
            sec_text = str(sec_text).strip()
            if not sec_text:
                continue

            sec_title = ""
            if sec_idx < len(section_titles) and section_titles[sec_idx] is not None:
                sec_title = str(section_titles[sec_idx])

            if (
                "external links" in sec_title.lower()
                or "references" in sec_title.lower()
            ):
                continue

            if sec_text in seen_sections:
                continue
            seen_sections.add(sec_text)

            candidates.append(
                {
                    "section_text": sec_text,
                    "section_index": sec_idx,
                    "section_title": sec_title,
                    "source_url": str(entry_url).strip(),
                    "source_title": str(entry_title).strip(),
                    "source_image_url": _to_json_safe(item.get("image_url", item.get("image_urls"))),
                    "source_retrieval_rank": rank_idx,
                    "source_retrieval_similarity": retrieval_similarity,
                    "source_knowledge_base_index": knowledge_base_index,
                }
            )

    return candidates

class LazySupervisedDataset_wRAG(Dataset):
    def __init__(self, data_path: str, image_root: str = ""):
        super(LazySupervisedDataset_wRAG, self).__init__()
        self.image_root = image_root or ""
        self.list_data_dict = []
        data_path = os.path.abspath(data_path)

        if data_path.endswith(".yaml"):
            with open(data_path, "r", encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                for data in datasets:
                    data_dict = data.get("data")
                    json_path = data_dict.get("json_path")
                    if not os.path.isabs(json_path):
                        json_path = os.path.join(os.path.dirname(data_path), json_path)
                    sampling_strategy = data_dict.get("sampling_strategy", "all")
                    sampling_number = data_dict.get("sampling_number", None)
                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r", encoding="utf-8") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def _resolve_image_path(self, example, index):
        candidates = []
        for key in ("image_path", "image"):
            value = example.get(key)
            if not value:
                continue
            value = str(value)
            if os.path.isabs(value):
                candidates.append(value)
            if self.image_root:
                candidates.append(os.path.join(self.image_root, value))
            candidates.append(value)

            root = self.image_root if self.image_root else os.path.dirname(value)
            name = value if not os.path.isabs(value) else os.path.basename(value)
            stem, ext = os.path.splitext(name)
            for alt_ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                if ext and ext == alt_ext:
                    continue
                candidates.append(os.path.join(root, stem + alt_ext))

        deduped = dedupe_list_keep_order([p for p in candidates if p])
        for candidate in deduped:
            if os.path.exists(candidate):
                return candidate

        data_id = example.get("data_id", str(index))
        raise FileNotFoundError(
            f"Cannot resolve image for data_id={data_id}. Tried: {deduped[:8]}"
        )

    def __getitem__(self, i):
        example = self.list_data_dict[i]
        document = example['entity_context']
        initial_urls = []
        initial_top1 = (example.get("initial_top1_url") or "").strip()
        if initial_top1:
            initial_urls.append(initial_top1)
        initial_entries = example.get("initial_retrieved_entries")
        if isinstance(initial_entries, list):
            for url in initial_entries:
                if isinstance(url, str) and url.strip():
                    initial_urls.append(url.strip())
        initial_urls = dedupe_list_keep_order(initial_urls)
        
        image_path = self._resolve_image_path(example, i)
        image = Image.open(image_path).convert("RGB")

        return {
            'image': image,
            'image_path': image_path,
            'data_id': example.get('data_id', str(i)),
            'problem': example['question'],
            'solution': example['answer_eval'],
            'document': document,
            'target_wikipedia_url': example.get('wikipedia_url', ''),
            'initial_retrieved_urls': initial_urls,
        }

def make_pre_conversation_grounding_retrieval(search):
    return { 
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Locate {object}, output its bbox coordinates using JSON format.".format(object=search)},
        ],
    }

def make_pre_conversation_caption_retrieval(question, caption):
    return { 
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Here is the question: {Question}. Here is the caption: {Caption}. Please combine them to generate a new concise caption.".format(Question=question, Caption=caption)},
        ],
    }


TOOL_ID_TO_NAME = {
    "1": "caption",
    "2": "grounding",
    "3": "flip",
}


def normalize_tool_key(raw_key):
    if raw_key is None:
        return None
    key = str(raw_key).strip().lower().strip("`'\"")
    if not key:
        return None
    if key in TOOL_ID_TO_NAME:
        return TOOL_ID_TO_NAME[key]

    alpha_key = re.sub(r"[^a-z]", "", key)
    if alpha_key in {"c", "ca", "cap"} or alpha_key.startswith("caption"):
        return "caption"
    if alpha_key in {"g", "gr", "ground"} or alpha_key.startswith("ground") or alpha_key in {"bbox", "box", "crop"}:
        return "grounding"
    if alpha_key in {"f", "fl"} or alpha_key.startswith("flip") or alpha_key.startswith("mirror"):
        return "flip"
    return None


def parse_tool_actions_from_text(tool_text):
    if tool_text is None:
        return []
    tool_text = str(tool_text).strip()
    if not tool_text:
        return []

    actions = []
    lines = [line.strip() for line in tool_text.splitlines() if line.strip()]

    for line in lines:
        # Support compact id list, e.g. "1,2,3" / "1 2 3"
        if ":" not in line and ("," in line or " " in line):
            tokens = [tok.strip() for tok in re.split(r"[,\s]+", line) if tok.strip()]
            if len(tokens) > 1 and all(normalize_tool_key(tok) for tok in tokens):
                for tok in tokens:
                    actions.append({"key": normalize_tool_key(tok), "value": ""})
                continue

        # Support "1. caption: xxx" / "1. caption" / "1"
        m = re.match(r"^\s*(\d+)\s*[\.\)]?\s*([A-Za-z_-]+)?\s*(?::\s*(.*))?$", line)
        if m:
            idx = m.group(1).strip()
            key = TOOL_ID_TO_NAME.get(idx)
            if key is not None:
                value = (m.group(3) or "").strip()
                actions.append({"key": key, "value": value})
                continue

        # Support "caption: xxx" / "Caption" / "Grounding" / "Flip"
        m = re.match(r"^\s*([A-Za-z_-]+)\s*(?::\s*(.*))?$", line)
        if m:
            key = normalize_tool_key(m.group(1))
            if key is not None:
                value = (m.group(2) or "").strip()
                actions.append({"key": key, "value": value})
                continue

        # Support isolated ids inside line, e.g. "use 1 and 2"
        if ":" not in line:
            ids = re.findall(r"(?<!\d)([1-3])(?!\d)", line)
            if ids:
                for idx in ids:
                    actions.append({"key": TOOL_ID_TO_NAME[idx], "value": ""})
                continue

    deduped = []
    seen = set()
    for action in actions:
        marker = (action["key"], action["value"])
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(action)
    return deduped


def extract_tag_text(text, tag_name="answer", default="None"):
    if text is None:
        return default
    text = str(text)
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return default
    extracted = match.group(1).strip()
    return extracted if extracted else default

# NOTE: Where is the recall recorded?
def evaluate_batch(
    rank,
    model,
    model1,
    retriever_text_actor,
    processor,
    batch_messages,
    batch_images,
    batch_question,
    batch_document,
    batch_target_urls,
    max_new_tokens=1024,
    recall_top_k=3,
):
    caption_count = 0
    grounding_count = 0
    caption_time = 0.0
    grounding_time = 0.0
    filter_time = 0.0
    answer_time = 0.0
    tool_recall_stats = {
        "caption_calls": 0,
        "caption_top1_hits": 0,
        "caption_topk_hits": 0,
        "grounding_calls": 0,
        "grounding_top1_hits": 0,
        "grounding_topk_hits": 0,
        "flip_calls": 0,
        "flip_top1_hits": 0,
        "flip_topk_hits": 0,
    }
    caption_retrieved_urls = [[] for _ in range(len(batch_messages))]
    grounding_retrieved_urls = [[] for _ in range(len(batch_messages))]
    flip_retrieved_urls = [[] for _ in range(len(batch_messages))]
    parsed_tool_actions = [[] for _ in range(len(batch_messages))]
    tool_retrieval_traces = [[] for _ in range(len(batch_messages))]
    search_result_evidence_sections = [[] for _ in range(len(batch_messages))]

    def _select_top_sections_with_provenance(section_candidates, sim_scores, top_n=3):
        if not section_candidates:
            return [], []
        sorted_indices = torch.argsort(sim_scores, descending=True)
        top_indices = sorted_indices[:top_n].cpu().numpy().tolist()
        selected_texts = []
        selected_meta = []
        for idx in top_indices:
            if idx >= len(section_candidates):
                continue
            meta = dict(section_candidates[idx])
            try:
                meta["section_similarity"] = _to_json_safe(sim_scores[idx].item())
            except Exception:
                meta["section_similarity"] = _to_json_safe(sim_scores[idx])
            selected_texts.append(meta["section_text"])
            selected_meta.append(meta)
        return selected_texts, selected_meta

    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    inputs = processor(text=text, images=batch_images, padding=True, padding_side="left", return_tensors="pt")
    inputs = inputs.to(model1.device)

    with torch.no_grad():
        generated_ids = model1.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    completions_first = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    completions_first_with_special = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    del generated_ids_trimmed

    pattern = r'<tool>(.*?:.*?)</tool>'
    matches_all = [re.search(pattern, completion, re.DOTALL | re.IGNORECASE) for completion in completions_first]
    pattern1 = r'<tool>(.*?)</tool>'
    tool_blocks_all = [re.findall(pattern1, completion, re.DOTALL | re.IGNORECASE) for completion in completions_first]
    tool_blocks = ["\n".join([blk.strip() for blk in blocks if blk and blk.strip()]) for blocks in tool_blocks_all]
    strict_tool_parse_match = [match is not None for match in matches_all]

    search_result = []
    for i in range(len(batch_messages)):
        sections_search = []
        actions = []
        for block in tool_blocks_all[i]:
            actions.extend(parse_tool_actions_from_text(block))

        if actions:
            parsed_tool_actions[i].extend(actions)
        else:
            search_result.append("None")
            continue

        for action in actions:
            key = action["key"]
            value = action["value"]
            if key == "caption":
                tool_trace = {
                    "tool": "caption",
                    "call_index_within_sample": len(tool_retrieval_traces[i]) + 1,
                    "action_value": value,
                    "retrieval_top_k": recall_top_k,
                    "status": "started",
                }
                try:
                    start_time = time.time()
                    tool_recall_stats["caption_calls"] += 1
                    caption_hint = value if value else "Describe the image in detail."
                    tool_trace["caption_hint"] = caption_hint
                    prompt_caption = [make_pre_conversation_caption_retrieval(question=batch_question[i], caption=caption_hint)]
                    prompt_captioning = processor.apply_chat_template(prompt_caption, tokenize=False, add_generation_prompt=True)
                    caption_prompt_inputs = processor(text=prompt_captioning, images=batch_images[i], return_tensors="pt", padding=True, padding_side="left")
                    caption_prompt_inputs = caption_prompt_inputs.to(model.device)
                    caption_prompt_completion_ids = model.generate(**caption_prompt_inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
                    caption_length = caption_prompt_inputs["input_ids"].size(1)
                    completion_ids = caption_prompt_completion_ids[:, caption_length:]
                    caption_completions = processor.batch_decode(completion_ids, skip_special_tokens=True)
                    del caption_prompt_inputs

                    tool_trace["caption_query"] = caption_completions[0] if caption_completions else ""
                    top_1 = retriever_text_actor.retrieve_text_faiss(caption_completions[0], top_k=recall_top_k) #align with paper
                    tool_trace["retrieved_items"] = serialize_retrieval_items_for_metadata(top_1)
                    caption_urls = extract_urls_from_retrieval_items(top_1)
                    caption_retrieved_urls[i].extend(caption_urls)
                    update_tool_recall_hits(
                        tool_recall_stats,
                        "caption",
                        batch_target_urls[i],
                        top_1,
                        recall_top_k,
                    )
                    section_candidates = build_section_candidates_with_provenance(top_1)
                    text_section = [cand["section_text"] for cand in section_candidates]
                    top5_texts = []
                    selected_sections_meta = []
                    if text_section:
                        sim = retriever_text_actor.similarity_section_text(caption_completions[0], text_section)
                        top5_texts, selected_sections_meta = _select_top_sections_with_provenance(
                            section_candidates,
                            sim,
                            top_n=3,
                        )
                    sections_search.extend(top5_texts)
                    if selected_sections_meta:
                        for sec_meta in selected_sections_meta:
                            sec_record = dict(sec_meta)
                            sec_record["tool"] = "caption"
                            search_result_evidence_sections[i].append(sec_record)
                    tool_trace["selected_sections_for_search"] = selected_sections_meta
                    tool_trace["status"] = "ok"
                    end_time = time.time()
                    caption_count += 1
                    caption_time += (end_time - start_time)
                except Exception as e:
                    end_time = time.time()
                    caption_time += (end_time - start_time)
                    tool_trace["status"] = "error"
                    tool_trace["error"] = str(e)
                    print(f"Error in caption: {str(e)}")
                tool_retrieval_traces[i].append(tool_trace)

            elif key == "grounding":
                tool_trace = {
                    "tool": "grounding",
                    "call_index_within_sample": len(tool_retrieval_traces[i]) + 1,
                    "action_value": value,
                    "retrieval_top_k": recall_top_k,
                    "status": "started",
                }
                start_time = time.time()
                grounding_count += 1
                tool_recall_stats["grounding_calls"] += 1
                target_object = value if value else "the main object in the image"
                tool_trace["target_object"] = target_object
                prompt_ground = [make_pre_conversation_grounding_retrieval(target_object)]
                prompt_grounding = processor.apply_chat_template(prompt_ground, tokenize=False, add_generation_prompt=True)
                grounding_prompt_inputs = processor(text=prompt_grounding, images=batch_images[i], return_tensors="pt", padding=True, padding_side="left")
                grounding_prompt_inputs = grounding_prompt_inputs.to(model.device)
                grounding_prompt_completion_ids = model.generate(**grounding_prompt_inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
                grounding_length = grounding_prompt_inputs["input_ids"].size(1)
                grounding_completion_ids = grounding_prompt_completion_ids[:, grounding_length:]
                grounding_completions = processor.batch_decode(grounding_completion_ids, skip_special_tokens=True)
                del grounding_prompt_inputs

                try:
                    tool_trace["grounding_raw_output"] = grounding_completions[0] if grounding_completions else ""
                    json_str = grounding_completions[0].strip('```json\n').strip('```').strip()
                    tool_trace["grounding_json_candidate"] = json_str
                    data_bbox = json.loads(json_str)
                    bbox = data_bbox[0]["bbox_2d"]
                    tool_trace["parsed_bbox"] = _to_json_safe(bbox)
                    if bbox:    
                        image = batch_images[i]
                        cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        cropped_img = cropped_img.resize((224,224), Image.Resampling.LANCZOS)
                        top_1 = retriever_text_actor.retrieve_image_faiss(cropped_img, top_k=recall_top_k)
                        tool_trace["retrieved_items"] = serialize_retrieval_items_for_metadata(top_1)
                        grounding_urls = extract_urls_from_retrieval_items(top_1)
                        grounding_retrieved_urls[i].extend(grounding_urls)
                        update_tool_recall_hits(
                            tool_recall_stats,
                            "grounding",
                            batch_target_urls[i],
                            top_1,
                            recall_top_k,
                        )
                        section_candidates = build_section_candidates_with_provenance(top_1)
                        text_section = [cand["section_text"] for cand in section_candidates]
                        top5_texts = []
                        selected_sections_meta = []
                        if text_section:
                            sim = retriever_text_actor.similarity_section_image(cropped_img, text_section)
                            top5_texts, selected_sections_meta = _select_top_sections_with_provenance(
                                section_candidates,
                                sim,
                                top_n=3,
                            )
                        sections_search.extend(top5_texts)
                        if selected_sections_meta:
                            for sec_meta in selected_sections_meta:
                                sec_record = dict(sec_meta)
                                sec_record["tool"] = "grounding"
                                search_result_evidence_sections[i].append(sec_record)
                        tool_trace["selected_sections_for_search"] = selected_sections_meta
                        tool_trace["status"] = "ok"
                        torch.cuda.empty_cache()
                    else:
                        tool_trace["status"] = "no_bbox"
                    end_time = time.time()
                    grounding_time += (end_time - start_time)
                except Exception as e:
                    end_time = time.time()
                    grounding_time += (end_time - start_time)
                    tool_trace["status"] = "error"
                    tool_trace["error"] = str(e)
                    print(f"Error in grounding: {str(e)}")
                tool_retrieval_traces[i].append(tool_trace)

            elif key == "flip":
                tool_trace = {
                    "tool": "flip",
                    "call_index_within_sample": len(tool_retrieval_traces[i]) + 1,
                    "action_value": value,
                    "retrieval_top_k": recall_top_k,
                    "status": "started",
                }
                try:
                    tool_recall_stats["flip_calls"] += 1
                    batch_images[i] = batch_images[i].transpose(Image.FLIP_LEFT_RIGHT)
                    tool_trace["image_flipped"] = True
                    flip_top = retriever_text_actor.retrieve_image_faiss(
                        batch_images[i],
                        top_k=recall_top_k,
                    )
                    tool_trace["retrieved_items"] = serialize_retrieval_items_for_metadata(flip_top)
                    flip_urls = extract_urls_from_retrieval_items(flip_top)
                    flip_retrieved_urls[i].extend(flip_urls)
                    update_tool_recall_hits(
                        tool_recall_stats,
                        "flip",
                        batch_target_urls[i],
                        flip_top,
                        recall_top_k,
                    )
                    tool_trace["status"] = "ok"
                except Exception as e:
                    tool_trace["status"] = "error"
                    tool_trace["error"] = str(e)
                    print(f"Error in flip: {str(e)}")
                tool_retrieval_traces[i].append(tool_trace)

        sections_search = list(set(sections_search))
        search = ".".join(sections_search)
        search_result.append(search)

    curr_search_template = (
        "Here is the user question: <question>{Question}</question>. "
        "Here is the relevant information retrieved through image retrieval: <retrieved_information>{Document}</retrieved_information>. "
        "Here is the relevant information through <tool>{Search}</tool>: <search_result>{Search_result}</search_result>. "
        "To obtain useful information, you must conduct reasoning inside <think></think> first. "
        "After reasoning, provide the filtered information inside <answer></answer>."
    )

    def make_pre_conversation_image_with_gt_retrieval(example, document, search, search_result):
        return {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": curr_search_template.format(Question=example, Document=document, Search=search, Search_result=search_result)},
            ],
        }

    prompt_new_text = [
        [make_pre_conversation_image_with_gt_retrieval(batch_question[i], batch_document[i], tool_blocks[i] if tool_blocks[i] else "None", search_result[i])]
        for i in range(len(batch_messages))
    ]
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in prompt_new_text]
    inputs = processor(text=text, images=batch_images, padding=True, padding_side="left", return_tensors="pt")
    inputs = inputs.to(model1.device)

    try:
        start_filter_time = time.time()
        with torch.no_grad():
            generated_ids = model1.generate(**inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        batch_output_first = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        batch_output_first_with_special = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        del generated_ids_trimmed
        end_filter_time = time.time()
        filter_time += (end_filter_time - start_filter_time)
    except Exception as e:
        end_filter_time = time.time()
        print(f"Error in first stage generation: {str(e)}")
        batch_output_first = [""] * len(batch_messages)
        batch_output_first_with_special = [""] * len(batch_messages)
        filter_time += (end_filter_time - start_filter_time)

    AFTER_QUESTION_TEMPLATE = "Context: {Document}\nQuestion: {Question}\nShort answer:"

    def _make_after_conversation_image(question, document):
        document = extract_tag_text(document, tag_name="answer", default="None")
        return {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": AFTER_QUESTION_TEMPLATE.format(Question=question, Document=document)},
            ],
        }

    second_stage_messages = [[_make_after_conversation_image(q, d)] for q, d in zip(batch_question, batch_output_first)]
    second_text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in second_stage_messages]
    second_inputs = processor(text=second_text, images=batch_images, padding=True, padding_side="left", return_tensors="pt")
    second_inputs = second_inputs.to(model.device)

    try:
        start_answer_time = time.time()
        with torch.no_grad():
            second_generated_ids = model.generate(**second_inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False)
        second_generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(second_inputs.input_ids, second_generated_ids)]
        batch_output_final = processor.batch_decode(second_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        end_answer_time = time.time()
        answer_time += (end_answer_time - start_answer_time)
    except Exception as e:
        end_answer_time = time.time()
        answer_time += (end_answer_time - start_answer_time)
        print(f"Error in second stage generation: {str(e)}")
        batch_output_final = [""] * len(batch_messages)

    per_sample_tool_urls = []
    per_sample_raw_outputs = []
    for i in range(len(batch_messages)):
        per_sample_tool_urls.append(
            {
                "caption_urls": dedupe_list_keep_order(caption_retrieved_urls[i]),
                "grounding_urls": dedupe_list_keep_order(grounding_retrieved_urls[i]),
                "flip_urls": dedupe_list_keep_order(flip_retrieved_urls[i]),
            }
        )
        per_sample_raw_outputs.append(
            {
                "planner_raw_output": completions_first[i] if i < len(completions_first) else "",
                "planner_raw_output_with_special_tokens": completions_first_with_special[i] if i < len(completions_first_with_special) else "",
                "tool_block_raw": tool_blocks[i] if i < len(tool_blocks) else "",
                "strict_tool_parse_match": strict_tool_parse_match[i] if i < len(strict_tool_parse_match) else False,
                "parsed_tool_actions": parsed_tool_actions[i] if i < len(parsed_tool_actions) else [],
                "tool_retrieval_traces": tool_retrieval_traces[i] if i < len(tool_retrieval_traces) else [],
                "search_result_text": search_result[i] if i < len(search_result) else "",
                "search_result_evidence_sections": search_result_evidence_sections[i] if i < len(search_result_evidence_sections) else [],
                "filter_raw_output": batch_output_first[i] if i < len(batch_output_first) else "",
                "filter_raw_output_with_special_tokens": batch_output_first_with_special[i] if i < len(batch_output_first_with_special) else "",
                "filter_answer_extracted": extract_tag_text(
                    batch_output_first[i] if i < len(batch_output_first) else "",
                    tag_name="answer",
                    default="None",
                ),
                "final_raw_output": batch_output_final[i] if i < len(batch_output_final) else "",
            }
        )

    return (
        batch_output_final,
        caption_count,
        grounding_count,
        caption_time,
        grounding_time,
        filter_time,
        answer_time,
        tool_recall_stats,
        per_sample_tool_urls,
        per_sample_raw_outputs,
    )

def eval_RAG(
    rank,
    world_size,
    steps,
    dataset,
    MODEL_PATH,
    PEFT_MODEL_PATH,
    OUTPUT_PATH,
    BSZ,
    TEST_DATASETS,
    MODEL_GPU_ID,
    MODEL1_GPU_ID,
    RETRIEVER_GPU_ID,
    KNOWLEDGE_BASE_PATH,
    FAISS_ROOT,
    MASTER_PORT,
    MAX_NEW_TOKENS,
    RECALL_TOP_K,
    ATTN_IMPLEMENTATION,
):
    setup(rank, world_size, MASTER_PORT)
    all_start_time = time.time()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank} started")
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    caption_total = 0
    grounding_total = 0
    caption_total_time = 0.0
    grounding_total_time = 0.0
    filter_total_time = 0.0
    answer_total_time = 0.0
    caption_tool_calls_total = 0
    caption_tool_top1_hits_total = 0
    caption_tool_topk_hits_total = 0
    grounding_tool_calls_total = 0
    grounding_tool_top1_hits_total = 0
    grounding_tool_topk_hits_total = 0
    flip_tool_calls_total = 0
    flip_tool_top1_hits_total = 0
    flip_tool_topk_hits_total = 0
    initial_eval_total = 0
    initial_top1_hits_total = 0
    initial_topk_hits_total = 0
    aggregate_eval_total = 0
    aggregate_top1_hits_total = 0
    aggregate_topk_hits_total = 0
    tool_recall_top_k = RECALL_TOP_K
    topk_label = f"top{tool_recall_top_k}"

    cuda_count = torch.cuda.device_count()
    max_gpu_id = max(MODEL_GPU_ID, MODEL1_GPU_ID, RETRIEVER_GPU_ID)
    if max_gpu_id >= cuda_count:
        raise ValueError(
            f"GPU id out of range. got max id={max_gpu_id}, available cuda devices={cuda_count}"
        )

    model_device = f"cuda:{MODEL_GPU_ID}"
    model1_device = f"cuda:{MODEL1_GPU_ID}"
    retriever_device = f"cuda:{RETRIEVER_GPU_ID}"

    if rank == 0:
        print(
            f"Device map -> model: {model_device}, model1: {model1_device}, retriever: {retriever_device}"
        )

    retriever_text_actor = ClipRetriever(device=retriever_device, model="eva-clip")
    retriever_text_actor.load_knowledge_base(knowledge_base_path=KNOWLEDGE_BASE_PATH)
    retriever_text_actor.load_faiss_index(load_index_path=FAISS_ROOT)

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if ATTN_IMPLEMENTATION:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        **model_kwargs,
    ).eval().to(model_device)
    model1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        PEFT_MODEL_PATH,
        **model_kwargs,
    ).eval().to(model1_device)#PeftModel.from_pretrained(model, PEFT_MODEL_PATH)
    # model1 = peft_model.merge_and_unload().eval().to(device)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    QUESTION_TEMPLATE = (
        "Given a question whose answer is within a knowledge base, you need to utilize one or more tools to query the knowledge base. "
        "Choose from: 1. Caption: detailed description. 2. Grounding: identify core subject. 3. Flip: flip image. "
        "Enclose reasoning in <think></think> and tool calls in <tool></tool>. "
        "Here is the user question: {Question}."
    )

    batch_messages = []
    batch_images = []
    batch_questions = []
    batch_solution = []
    batch_document = []
    batch_target_urls = []
    batch_initial_urls = []
    batch_data_ids = []
    for x in dataset:
        batch_messages.append([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=x["problem"])}
            ]
        }])
        img = x["image"] if "image" in x else Image.open(x["image_path"])
        w, h = img.size
        if w < 28 or h < 28:
            if w < h:
                new_w, new_h = 28, int(h * (28 / w))
            else:
                new_h, new_w = 28, int(w * (28 / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        batch_images.append(img)
        batch_questions.append(x["problem"])
        batch_solution.append(x["solution"])
        batch_document.append(x["document"])
        batch_target_urls.append(x.get("target_wikipedia_url", ""))
        batch_initial_urls.append(x.get("initial_retrieved_urls", []))
        batch_data_ids.append(x.get("data_id", ""))

    chunk_size = len(batch_messages) // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank != world_size - 1 else len(batch_messages)
    local_messages = batch_messages[start_idx:end_idx]
    local_images = batch_images[start_idx:end_idx]
    local_questions = batch_questions[start_idx:end_idx]
    local_solution = batch_solution[start_idx:end_idx]
    local_document = batch_document[start_idx:end_idx]
    local_target_urls = batch_target_urls[start_idx:end_idx]
    local_initial_urls = batch_initial_urls[start_idx:end_idx]
    local_data_ids = batch_data_ids[start_idx:end_idx]

    all_outputs = []
    all_raw_outputs = []
    recall_detail_records = []
    generation_detail_records = []
    if rank == 0:
        pbar = tqdm(total=len(local_messages), desc="Inference")

    for i in range(0, len(local_messages), BSZ):
        (
            batch_output,
            caption_count,
            grounding_count,
            caption_time,
            grounding_time,
            filter_time,
            answer_time,
            tool_recall_stats,
            batch_tool_retrieved_urls,
            batch_raw_outputs,
        ) = evaluate_batch(
            rank, model, model1, retriever_text_actor, processor,
            local_messages[i:i+BSZ], local_images[i:i+BSZ],
            local_questions[i:i+BSZ], local_document[i:i+BSZ],
            local_target_urls[i:i+BSZ],
            max_new_tokens=MAX_NEW_TOKENS,
            recall_top_k=tool_recall_top_k,
        )
        all_outputs.extend(batch_output)
        all_raw_outputs.extend(batch_raw_outputs)
        caption_total += caption_count
        grounding_total += grounding_count
        caption_total_time += caption_time
        grounding_total_time += grounding_time
        filter_total_time += filter_time
        answer_total_time += answer_time
        caption_tool_calls_total += tool_recall_stats["caption_calls"]
        caption_tool_top1_hits_total += tool_recall_stats["caption_top1_hits"]
        caption_tool_topk_hits_total += tool_recall_stats["caption_topk_hits"]
        grounding_tool_calls_total += tool_recall_stats["grounding_calls"]
        grounding_tool_top1_hits_total += tool_recall_stats["grounding_top1_hits"]
        grounding_tool_topk_hits_total += tool_recall_stats["grounding_topk_hits"]
        flip_tool_calls_total += tool_recall_stats["flip_calls"]
        flip_tool_top1_hits_total += tool_recall_stats["flip_top1_hits"]
        flip_tool_topk_hits_total += tool_recall_stats["flip_topk_hits"]

        current_batch_target_urls = local_target_urls[i:i+BSZ]
        current_batch_initial_urls = local_initial_urls[i:i+BSZ]
        current_batch_questions = local_questions[i:i+BSZ]
        current_batch_solutions = local_solution[i:i+BSZ]
        current_batch_data_ids = local_data_ids[i:i+BSZ]
        current_batch_predictions = batch_output
        for j, target_url in enumerate(current_batch_target_urls):
            target_url = (target_url or "").strip()

            initial_urls = []
            if j < len(current_batch_initial_urls):
                initial_urls = extract_urls_from_retrieval_items(current_batch_initial_urls[j])
            initial_recall = {1: 0, tool_recall_top_k: 0}
            if target_url:
                initial_eval_total += 1
                initial_recall = eval_recall(initial_urls, target_url, top_ks=[1, tool_recall_top_k])
                initial_top1_hits_total += initial_recall[1]
                initial_topk_hits_total += initial_recall[tool_recall_top_k]

            tool_urls = []
            caption_urls = []
            grounding_urls = []
            flip_urls = []
            if j < len(batch_tool_retrieved_urls):
                sample_tool_urls = batch_tool_retrieved_urls[j]
                if isinstance(sample_tool_urls, dict):
                    caption_urls = extract_urls_from_retrieval_items(sample_tool_urls.get("caption_urls", []))
                    grounding_urls = extract_urls_from_retrieval_items(sample_tool_urls.get("grounding_urls", []))
                    flip_urls = extract_urls_from_retrieval_items(sample_tool_urls.get("flip_urls", []))
                    tool_urls.extend(caption_urls)
                    tool_urls.extend(grounding_urls)
                    tool_urls.extend(flip_urls)
            tool_urls = extract_urls_from_retrieval_items(tool_urls)
            aggregate_urls = dedupe_list_keep_order(initial_urls + tool_urls)
            aggregate_recall = {1: 0, tool_recall_top_k: 0}
            if target_url:
                aggregate_eval_total += 1
                aggregate_recall = eval_recall(aggregate_urls, target_url, top_ks=[1, tool_recall_top_k])
                aggregate_top1_hits_total += aggregate_recall[1]
                aggregate_topk_hits_total += aggregate_recall[tool_recall_top_k]

            caption_recall = eval_recall(caption_urls, target_url, top_ks=[1, tool_recall_top_k]) if target_url else {1: 0, tool_recall_top_k: 0}
            grounding_recall = eval_recall(grounding_urls, target_url, top_ks=[1, tool_recall_top_k]) if target_url else {1: 0, tool_recall_top_k: 0}
            flip_recall = eval_recall(flip_urls, target_url, top_ks=[1, tool_recall_top_k]) if target_url else {1: 0, tool_recall_top_k: 0}

            sample_raw_debug = batch_raw_outputs[j] if j < len(batch_raw_outputs) else {}
            parsed_tool_actions = sample_raw_debug.get("parsed_tool_actions", [])
            tool_call_counts = {"caption": 0, "grounding": 0, "flip": 0}
            for action in parsed_tool_actions:
                if isinstance(action, dict):
                    tool_key = action.get("key")
                    if tool_key in tool_call_counts:
                        tool_call_counts[tool_key] += 1

            data_id = current_batch_data_ids[j] if j < len(current_batch_data_ids) else ""
            question = current_batch_questions[j] if j < len(current_batch_questions) else ""
            prediction = current_batch_predictions[j] if j < len(current_batch_predictions) else ""
            solution = current_batch_solutions[j] if j < len(current_batch_solutions) else ""

            recall_detail_records.append(
                {
                    "data_id": data_id,
                    "question": question,
                    "wikipedia_url": target_url,
                    "retrieved_urls": {
                        "initial": initial_urls,
                        "caption": caption_urls,
                        "grounding": grounding_urls,
                        "flip": flip_urls,
                        "aggregate_initial_plus_tools": aggregate_urls,
                    },
                    "hits": {
                        "initial": {"top1": int(initial_recall[1]), topk_label: int(initial_recall[tool_recall_top_k])},
                        "caption": {"top1": int(caption_recall[1]), topk_label: int(caption_recall[tool_recall_top_k])},
                        "grounding": {"top1": int(grounding_recall[1]), topk_label: int(grounding_recall[tool_recall_top_k])},
                        "flip": {"top1": int(flip_recall[1]), topk_label: int(flip_recall[tool_recall_top_k])},
                        "aggregate_initial_plus_tools": {"top1": int(aggregate_recall[1]), topk_label: int(aggregate_recall[tool_recall_top_k])},
                    },
                    "tool_calls": {
                        "tool_block_raw": sample_raw_debug.get("tool_block_raw", ""),
                        "strict_tool_parse_match": bool(sample_raw_debug.get("strict_tool_parse_match", False)),
                        "parsed_tool_actions": parsed_tool_actions,
                        "call_counts": tool_call_counts,
                    },
                    "tool_retrieval_traces": sample_raw_debug.get("tool_retrieval_traces", []),
                    "search_result_text": sample_raw_debug.get("search_result_text", ""),
                    "search_result_evidence_sections": sample_raw_debug.get("search_result_evidence_sections", []),
                    "filter_answer_extracted": sample_raw_debug.get("filter_answer_extracted", "None"),
                }
            )

            generation_detail_records.append(
                {
                    "data_id": data_id,
                    "question": question,
                    "wikipedia_url": target_url,
                    "prediction": prediction,
                    "solution": solution,
                    "retrieved_urls": {
                        "initial": initial_urls,
                        "caption": caption_urls,
                        "grounding": grounding_urls,
                        "flip": flip_urls,
                        "aggregate_initial_plus_tools": aggregate_urls,
                    },
                    "hits": {
                        "initial": {"top1": int(initial_recall[1]), topk_label: int(initial_recall[tool_recall_top_k])},
                        "aggregate_initial_plus_tools": {"top1": int(aggregate_recall[1]), topk_label: int(aggregate_recall[tool_recall_top_k])},
                    },
                    "tool_retrieval_traces": sample_raw_debug.get("tool_retrieval_traces", []),
                    "search_result_text": sample_raw_debug.get("search_result_text", ""),
                    "search_result_evidence_sections": sample_raw_debug.get("search_result_evidence_sections", []),
                    "filter_answer_extracted": sample_raw_debug.get("filter_answer_extracted", "None"),
                    "debug_raw": sample_raw_debug,
                }
            )

        if rank == 0:
            pbar.update(len(batch_output))

    # Reduce metrics across ranks
    caption_total_t = torch.tensor([caption_total], device=device, dtype=torch.float64)
    grounding_total_t = torch.tensor([grounding_total], device=device, dtype=torch.float64)
    caption_total_time_t = torch.tensor([caption_total_time], device=device, dtype=torch.float64)
    grounding_total_time_t = torch.tensor([grounding_total_time], device=device, dtype=torch.float64)
    filter_total_time_t = torch.tensor([filter_total_time], device=device, dtype=torch.float64)
    answer_total_time_t = torch.tensor([answer_total_time], device=device, dtype=torch.float64)
    caption_tool_calls_t = torch.tensor([caption_tool_calls_total], device=device, dtype=torch.float64)
    caption_tool_top1_hits_t = torch.tensor([caption_tool_top1_hits_total], device=device, dtype=torch.float64)
    caption_tool_topk_hits_t = torch.tensor([caption_tool_topk_hits_total], device=device, dtype=torch.float64)
    grounding_tool_calls_t = torch.tensor([grounding_tool_calls_total], device=device, dtype=torch.float64)
    grounding_tool_top1_hits_t = torch.tensor([grounding_tool_top1_hits_total], device=device, dtype=torch.float64)
    grounding_tool_topk_hits_t = torch.tensor([grounding_tool_topk_hits_total], device=device, dtype=torch.float64)
    flip_tool_calls_t = torch.tensor([flip_tool_calls_total], device=device, dtype=torch.float64)
    flip_tool_top1_hits_t = torch.tensor([flip_tool_top1_hits_total], device=device, dtype=torch.float64)
    flip_tool_topk_hits_t = torch.tensor([flip_tool_topk_hits_total], device=device, dtype=torch.float64)
    initial_eval_total_t = torch.tensor([initial_eval_total], device=device, dtype=torch.float64)
    initial_top1_hits_t = torch.tensor([initial_top1_hits_total], device=device, dtype=torch.float64)
    initial_topk_hits_t = torch.tensor([initial_topk_hits_total], device=device, dtype=torch.float64)
    aggregate_eval_total_t = torch.tensor([aggregate_eval_total], device=device, dtype=torch.float64)
    aggregate_top1_hits_t = torch.tensor([aggregate_top1_hits_total], device=device, dtype=torch.float64)
    aggregate_topk_hits_t = torch.tensor([aggregate_topk_hits_total], device=device, dtype=torch.float64)

    reduce_tensors = [
        caption_total_t,
        grounding_total_t,
        caption_total_time_t,
        grounding_total_time_t,
        filter_total_time_t,
        answer_total_time_t,
        caption_tool_calls_t,
        caption_tool_top1_hits_t,
        caption_tool_topk_hits_t,
        grounding_tool_calls_t,
        grounding_tool_top1_hits_t,
        grounding_tool_topk_hits_t,
        flip_tool_calls_t,
        flip_tool_top1_hits_t,
        flip_tool_topk_hits_t,
        initial_eval_total_t,
        initial_top1_hits_t,
        initial_topk_hits_t,
        aggregate_eval_total_t,
        aggregate_top1_hits_t,
        aggregate_topk_hits_t,
    ]
    for t in reduce_tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    output_path = OUTPUT_PATH.format(DATASET=TEST_DATASETS[0], STEPS=steps)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_middle_path = output_path + f"_{rank}.json"
    with open(output_middle_path, 'w') as f:
        json.dump({
            "metadata": {"rank": rank, "num_samples": len(all_outputs)},
            "data_id": local_data_ids,
            "question": local_questions,
            "results": all_outputs,
            "debug_raw_outputs": all_raw_outputs,
            "solution": local_solution
        }, f)

    raw_output_path = output_path + f"_{rank}_raw_outputs.jsonl"
    with open(raw_output_path, "w") as raw_f:
        for data_id, question, pred, solution, raw_debug in zip(local_data_ids, local_questions, all_outputs, local_solution, all_raw_outputs):
            raw_f.write(
                json.dumps(
                    {
                        "data_id": data_id,
                        "question": question,
                        "prediction": pred,
                        "solution": solution,
                        "debug_raw": raw_debug,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    recall_detail_path = output_path + f"_{rank}_recall_details.jsonl"
    with open(recall_detail_path, "w") as recall_f:
        for record in recall_detail_records:
            recall_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    generation_detail_path = output_path + f"_{rank}_generation_details.jsonl"
    with open(generation_detail_path, "w") as generation_f:
        for record in generation_detail_records:
            generation_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if rank == 0:
        all_end_time = time.time()
        all_time = all_end_time - all_start_time
        caption_calls = int(caption_tool_calls_t.item())
        caption_top1_hits = int(caption_tool_top1_hits_t.item())
        caption_topk_hits = int(caption_tool_topk_hits_t.item())
        grounding_calls = int(grounding_tool_calls_t.item())
        grounding_top1_hits = int(grounding_tool_top1_hits_t.item())
        grounding_topk_hits = int(grounding_tool_topk_hits_t.item())
        flip_calls = int(flip_tool_calls_t.item())
        flip_top1_hits = int(flip_tool_top1_hits_t.item())
        flip_topk_hits = int(flip_tool_topk_hits_t.item())
        initial_samples = int(initial_eval_total_t.item())
        initial_top1_hits = int(initial_top1_hits_t.item())
        initial_topk_hits = int(initial_topk_hits_t.item())
        aggregate_samples = int(aggregate_eval_total_t.item())
        aggregate_top1_hits = int(aggregate_top1_hits_t.item())
        aggregate_topk_hits = int(aggregate_topk_hits_t.item())

        caption_top1_recall = caption_top1_hits / caption_calls if caption_calls > 0 else 0.0
        caption_topk_recall = caption_topk_hits / caption_calls if caption_calls > 0 else 0.0
        grounding_top1_recall = grounding_top1_hits / grounding_calls if grounding_calls > 0 else 0.0
        grounding_topk_recall = grounding_topk_hits / grounding_calls if grounding_calls > 0 else 0.0
        flip_top1_recall = flip_top1_hits / flip_calls if flip_calls > 0 else 0.0
        flip_topk_recall = flip_topk_hits / flip_calls if flip_calls > 0 else 0.0
        initial_top1_recall = initial_top1_hits / initial_samples if initial_samples > 0 else 0.0
        initial_topk_recall = initial_topk_hits / initial_samples if initial_samples > 0 else 0.0
        aggregate_top1_recall = aggregate_top1_hits / aggregate_samples if aggregate_samples > 0 else 0.0
        aggregate_topk_recall = aggregate_topk_hits / aggregate_samples if aggregate_samples > 0 else 0.0

        print("\n===== Initial / Aggregated Recall (Target URL Match) =====")
        print(
            f"[Initial] samples={initial_samples}, top1_hits={initial_top1_hits}, "
            f"top{tool_recall_top_k}_hits={initial_topk_hits}, top1_recall={initial_top1_recall:.4f}, "
            f"top{tool_recall_top_k}_recall={initial_topk_recall:.4f}"
        )
        print(
            f"[Aggregated initial+tools] samples={aggregate_samples}, top1_hits={aggregate_top1_hits}, "
            f"top{tool_recall_top_k}_hits={aggregate_topk_hits}, top1_recall={aggregate_top1_recall:.4f}, "
            f"top{tool_recall_top_k}_recall={aggregate_topk_recall:.4f}"
        )
        print("=========================================================\n")

        print("\n===== Tool Recall (Target URL Match) =====")
        print(
            f"[Caption] calls={caption_calls}, top1_hits={caption_top1_hits}, "
            f"top{tool_recall_top_k}_hits={caption_topk_hits}, top1_recall={caption_top1_recall:.4f}, "
            f"top{tool_recall_top_k}_recall={caption_topk_recall:.4f}"
        )
        print(
            f"[Grounding] calls={grounding_calls}, top1_hits={grounding_top1_hits}, "
            f"top{tool_recall_top_k}_hits={grounding_topk_hits}, top1_recall={grounding_top1_recall:.4f}, "
            f"top{tool_recall_top_k}_recall={grounding_topk_recall:.4f}"
        )
        print(
            f"[Flip] calls={flip_calls}, top1_hits={flip_top1_hits}, "
            f"top{tool_recall_top_k}_hits={flip_topk_hits}, top1_recall={flip_top1_recall:.4f}, "
            f"top{tool_recall_top_k}_recall={flip_topk_recall:.4f}"
        )
        print("==========================================\n")

        with open(output_middle_path, 'a') as f:
            f.write(f"\nTotal Time: {all_time:.2f} s\n")
            f.write(
                "Initial/Aggregated Recall Summary: "
                f"initial_samples={initial_samples}, initial_top1_hits={initial_top1_hits}, "
                f"initial_top{tool_recall_top_k}_hits={initial_topk_hits}, initial_top1_recall={initial_top1_recall:.4f}, "
                f"initial_top{tool_recall_top_k}_recall={initial_topk_recall:.4f}, "
                f"aggregate_samples={aggregate_samples}, aggregate_top1_hits={aggregate_top1_hits}, "
                f"aggregate_top{tool_recall_top_k}_hits={aggregate_topk_hits}, aggregate_top1_recall={aggregate_top1_recall:.4f}, "
                f"aggregate_top{tool_recall_top_k}_recall={aggregate_topk_recall:.4f}\n"
            )
            f.write(
                "Tool Recall Summary: "
                f"caption_calls={caption_calls}, caption_top1_hits={caption_top1_hits}, "
                f"caption_top{tool_recall_top_k}_hits={caption_topk_hits}, caption_top1_recall={caption_top1_recall:.4f}, "
                f"caption_top{tool_recall_top_k}_recall={caption_topk_recall:.4f}, "
                f"grounding_calls={grounding_calls}, grounding_top1_hits={grounding_top1_hits}, "
                f"grounding_top{tool_recall_top_k}_hits={grounding_topk_hits}, grounding_top1_recall={grounding_top1_recall:.4f}, "
                f"grounding_top{tool_recall_top_k}_recall={grounding_topk_recall:.4f}, "
                f"flip_calls={flip_calls}, flip_top1_hits={flip_top1_hits}, "
                f"flip_top{tool_recall_top_k}_hits={flip_topk_hits}, flip_top1_recall={flip_top1_recall:.4f}, "
                f"flip_top{tool_recall_top_k}_recall={flip_topk_recall:.4f}\n"
            )
            # Note: Full metrics aggregation would require gathering all rank outputs

    if rank == 0:
        pbar.close()

    cleanup()
    del model, model1, processor, retriever_text_actor
    torch.cuda.empty_cache()
    gc.collect()

def build_parser():
    method_root = os.path.dirname(os.path.abspath(__file__))
    camera_ready_root = os.path.abspath(os.path.join(method_root, "..", "..", ".."))
    default_output_dir = os.path.join(camera_ready_root, "outputs", "generated_methods", "Wiki-PRF")

    parser = argparse.ArgumentParser(description="Run camera-ready Wiki-PRF evaluation.")
    parser.add_argument(
        "--dataset",
        choices=["evqa_fixed", "evqa_unfixed", "infoseek_fixed", "infoseek_unfixed"],
        default="infoseek_fixed",
        help="Camera-ready dataset config to run.",
    )
    parser.add_argument("--data_yaml", default=None, help="Override YAML data config path.")
    parser.add_argument("--dataset_name", default=None, help="Name written into output filenames.")
    parser.add_argument("--image_root", default="", help="Optional root for relative image names.")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--peft_model_path",
        default=os.path.join(camera_ready_root, "data", "checkpoints", "Wiki-PRF"),
        help="Merged Wiki-PRF checkpoint path.",
    )
    parser.add_argument(
        "--knowledge_base",
        default=os.path.join(camera_ready_root, "data", "kb", "infoseek_wiki_100_dict_v4.json"),
        help="Wiki KB JSON used by dynamic tool retrieval.",
    )
    parser.add_argument(
        "--faiss_root",
        default=os.path.join(camera_ready_root, "data", "kb", "KB_infoseek"),
        help="Directory containing kb_index.faiss and kb_index_ids.pkl.",
    )
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output path template. Supports {DATASET} and {STEPS}.",
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--model_gpu_id", type=int, default=0)
    parser.add_argument("--filter_gpu_id", type=int, default=0)
    parser.add_argument("--retriever_gpu_id", type=int, default=0)
    parser.add_argument("--master_port", type=int, default=int(os.environ.get("MASTER_PORT", "13188")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--recall_top_k", type=int, default=3)
    parser.add_argument(
        "--attn_implementation",
        default=os.environ.get("WIKIPRF_ATTN_IMPLEMENTATION", "flash_attention_2"),
        help="Set to an empty string to use transformers default attention.",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Optional first-N sample limit.")
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Validate paths without loading models.")
    return parser


def main():
    method_root = os.path.dirname(os.path.abspath(__file__))
    camera_ready_root = os.path.abspath(os.path.join(method_root, "..", "..", ".."))
    configs = {
        "evqa_fixed": os.path.join(method_root, "configs", "evqa_fixed.yaml"),
        "evqa_unfixed": os.path.join(method_root, "configs", "evqa_unfixed.yaml"),
        "infoseek_fixed": os.path.join(method_root, "configs", "infoseek_fixed.yaml"),
        "infoseek_unfixed": os.path.join(method_root, "configs", "infoseek_unfixed.yaml"),
    }

    parser = build_parser()
    args = parser.parse_args()
    data_root = os.path.abspath(args.data_yaml or configs[args.dataset])
    dataset_name = args.dataset_name or args.dataset
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(
            args.output_dir,
            args.dataset,
            f"results_{{DATASET}}_step{{STEPS}}_topk{args.recall_top_k}.json",
        )
    output_path = os.path.abspath(output_path)

    random.seed(args.seed)
    dataset = LazySupervisedDataset_wRAG(data_root, args.image_root)
    if args.max_samples > 0:
        dataset.list_data_dict = dataset.list_data_dict[: args.max_samples]
    if len(dataset) == 0:
        raise ValueError(f"No samples loaded from {data_root}")

    first_sample = dataset[0]
    first_image_size = first_sample["image"].size if first_sample.get("image") is not None else None
    if first_sample.get("image") is not None:
        first_sample["image"].close()

    resolved = {
        "camera_ready_root": camera_ready_root,
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "data_yaml": data_root,
        "num_samples": len(dataset),
        "first_data_id": first_sample.get("data_id", ""),
        "first_image_path": first_sample.get("image_path", ""),
        "first_image_size": first_image_size,
        "model_path": args.model_path,
        "peft_model_path": os.path.abspath(args.peft_model_path),
        "knowledge_base": os.path.abspath(args.knowledge_base),
        "faiss_root": os.path.abspath(args.faiss_root),
        "output_path_template": output_path,
        "devices": {
            "model_gpu_id": args.model_gpu_id,
            "filter_gpu_id": args.filter_gpu_id,
            "retriever_gpu_id": args.retriever_gpu_id,
            "world_size": args.world_size,
        },
        "recall_top_k": args.recall_top_k,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
    }

    if args.dry_run:
        print(json.dumps(resolved, indent=2, ensure_ascii=False))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Wiki-PRF inference requires CUDA. Use --dry_run to validate paths only.")
    if args.world_size < 1:
        raise ValueError("--world_size must be >= 1")

    print(json.dumps(resolved, indent=2, ensure_ascii=False))
    print(f"Using {args.world_size} process(es) for evaluation")

    mp.spawn(
        eval_RAG,
        args=(
            args.world_size,
            args.steps,
            dataset,
            args.model_path,
            os.path.abspath(args.peft_model_path),
            output_path,
            args.batch_size,
            [dataset_name],
            args.model_gpu_id,
            args.filter_gpu_id,
            args.retriever_gpu_id,
            os.path.abspath(args.knowledge_base),
            os.path.abspath(args.faiss_root),
            args.master_port,
            args.max_new_tokens,
            args.recall_top_k,
            args.attn_implementation.strip() or None,
        ),
        nprocs=args.world_size,
        join=True,
    )

if __name__ == "__main__":
    main()
