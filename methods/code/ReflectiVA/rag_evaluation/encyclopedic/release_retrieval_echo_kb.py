import os
import sys
import argparse
import torch
import ujson
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, collate_and_pad_input_ids
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from functools import partial
import gc
import random
from spacy.lang.en import English
from pathlib import Path

from rag_evaluation.encyclopedic.Retriever import ClipRetriever

IMAGE_TOKEN = f"{DEFAULT_IMAGE_TOKEN}\n\n"
RET_TOKEN = 128251
REL_TOKEN = 128253
TEMPLATE = "Consider this paragraph: "


def uniform_passages_of_sentences(paragraphs, n=100):
    spacy_model = English()
    spacy_model.add_pipe("sentencizer")
    text = paragraphs

    sentences = spacy_model(text).sents

    passages = []
    passage = []
    tokens_in_passage = 0
    for sent in sentences:
        if tokens_in_passage + len(sent) > n:
            if len(passage) > 0:
                passages.append(' '.join(passage))
                passage = [sent.text]
                tokens_in_passage = len(sent)
            else:
                passages.append(sent.text)
        else:
            passage.append(sent.text)
            tokens_in_passage += len(sent)

    if len(passage) > 0:
        passages.append(' '.join(passage))

    return passages

class CustomDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor, model_config):
        self.args = args
        self.data_path = args.data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = args.conv_mode
        self.template = TEMPLATE
        self.image_root = Path(args.image_root)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.debug_dump = getattr(args, "debug_dump", False)
        self.debug_max_section_chars = 120
        retriever_model = "eva-clip" if args.use_eva_to_retrieve else "clip"
        self.retriever = ClipRetriever(model=retriever_model, device=device)
        self.retriever.load_knowledge_base(args.kb_wikipedia_path)
        index_dir = args.index_path
        if index_dir.endswith("kb_index.faiss"):
            index_dir = os.path.dirname(index_dir)
        if index_dir and not index_dir.endswith(os.sep):
            index_dir += os.sep
        self.retriever.load_faiss_index(index_dir)
            
        print(f'Loading from {self.data_path}...')
        with open(self.data_path, 'r') as f:
            self.all_samples = ujson.load(f)
        print(f'Loading completed...')
        
        self.entities = self.all_samples
        part = int(os.environ.get('PART', "0"))
        total_part = int(os.environ.get('TOTAL_PART', "0")) + 1
        print(f'Computing split {part} of the dataset...')
        slicing = len(self.all_samples) // total_part
        if (part+1) == total_part:
            self.all_samples = self.all_samples[slicing * part:]
        else:
            self.all_samples = self.all_samples[slicing * part:slicing * part + slicing]

    def __getitem__(self, index):
        sample = self.all_samples[index]
        image = Image.open(self.image_root / sample['related_images']).convert('RGB')
        sample['dataset_image_ids'] = sample['dataset_image_ids']
        question = sample['question']
        data_id = sample['data_id']

        total_qs_relevant = []
        debug_info = {} if self.debug_dump else None
        with torch.no_grad():
            precomputed_features = None
            if self.debug_dump:
                processor = self.retriever.processor
                inputs = processor(images=image, return_tensors="pt").pixel_values.to(self.retriever.device).half()
                if self.retriever.model_type == "clip":
                    image_features = self.retriever.model.get_image_features(inputs)
                else:
                    image_features = self.retriever.model.encode_image(inputs)
                raw_features = image_features.detach().float().cpu().reshape(-1)
                debug_info.update({
                    "data_id": data_id,
                    "question": question,
                    "raw_feature_norm": float(raw_features.norm().item()),
                    "raw_feature_head": raw_features[:min(8, raw_features.numel())].tolist()
                })
                normalized = torch.nn.functional.normalize(image_features.float()).cpu().reshape(-1)
                debug_info["normalized_feature_norm"] = float(normalized.norm().item())
                precomputed_features = image_features
            kb_entries = self.retriever.retrieve_image_faiss(
                image, top_k=2*self.args.entity_k, return_entry_list=True # , precomputed_features=precomputed_features
            )
            unique_entries = []
            seen_urls = set()
            for entry in kb_entries:
                entry_url = getattr(entry, "url", None)
                if entry_url is None:
                    entry_url = id(entry)
                if entry_url in seen_urls:
                    continue
                seen_urls.add(entry_url)
                unique_entries.append(entry)
                if len(unique_entries) >= self.args.entity_k:
                    break
            kb_entries = unique_entries[:self.args.entity_k]

        sections = []
        if self.debug_dump:
            debug_info["entity_urls"] = [entry.url for entry in kb_entries]
        for entry in kb_entries:
            for section in entry.section_texts:
                if len(section.strip()) < 5:
                    continue
                sections.append(section)

        qs_retrieval = IMAGE_TOKEN + question
        for section in sections:
            section = section.strip()
            conv_relevant = conv_templates[self.conv_mode].copy()
            qs_relevant = IMAGE_TOKEN + question
            conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
            qs_relevant = '[Retrieval]'
            conv_relevant.append_message(conv_relevant.roles[1], qs_relevant)
            if 'rank' in self.model_config.template_paragraph:
                qs_relevant = self.template['pre'] + '<paragraph>'
                qs_relevant += section + '</paragraph>' + self.template['post']
            else:
                qs_relevant = self.template + '<paragraph>'
                qs_relevant += section + '</paragraph>'

            conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
            conv_relevant.append_message(conv_relevant.roles[1], None)
            prompt_relevant = conv_relevant.get_prompt()

            input_ids_relevant = tokenizer_image_token(
                prompt_relevant, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            )
            total_qs_relevant.append(input_ids_relevant[1:])

        conv_retrieval = conv_templates[self.conv_mode].copy()
        conv_retrieval.append_message(conv_retrieval.roles[0], qs_retrieval)
        conv_retrieval.append_message(conv_retrieval.roles[1], None)
        prompt_retrieval = conv_retrieval.get_prompt()

        input_ids_retrieval = tokenizer_image_token(prompt_retrieval, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        sample['image_size'] = image.size
        sample['image_tensor'] = process_images([image], self.image_processor, self.model_config)[0]
        sample['retrieval_input_ids'] = input_ids_retrieval[1:]
        sample['relevant_input_ids'] = total_qs_relevant
        sample['sections'] = sections
        if self.debug_dump and debug_info is not None:
            debug_info["sections_preview"] = [sec.strip()[:self.debug_max_section_chars] for sec in sections]
            sample['debug_info'] = debug_info

        return sample

    def __len__(self):
        return len(self.all_samples)


def create_data_loader(args, tokenizer, image_processor, model_config):
    dataset = CustomDataset(args, tokenizer, image_processor, model_config)

    def collate_fn(tokenizer, batch):
        out = {k: [example[k] for example in batch]
               for k in list(batch[0].keys())}
        out['retrieval_input_ids'] = collate_and_pad_input_ids(out['retrieval_input_ids'], tokenizer.pad_token_id, 'left')
        out['relevant_input_ids'] = [collate_and_pad_input_ids(el.unsqueeze(0), tokenizer.pad_token_id, 'left') for el in out['relevant_input_ids'][0]]
        out['image_tensor'] = torch.stack(out['image_tensor'], dim=0)
        return out

    collate_fn = partial(collate_fn, tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def concat_paragraph(paragraphs_list, question, sections, conv_mode, config, tokenizer):
    template = TEMPLATE
    qs = IMAGE_TOKEN + question
    conv_relevant = conv_templates[conv_mode].copy()
    conv_relevant.append_message(conv_relevant.roles[0], qs)
    qs = '[Retrieval]'
    conv_relevant.append_message(conv_relevant.roles[1], qs)
    qs = template + '<paragraph>'
    for idx in paragraphs_list:
        qs += sections[idx]
    qs +='</paragraph>'
    
    if args.short_prompt:
        qs += ". Give a short answer."
        
    conv_relevant.append_message(conv_relevant.roles[0], qs)
    conv_relevant.append_message(conv_relevant.roles[1], None)
    prompt = conv_relevant.get_prompt()
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[1:].unsqueeze(0).to(device='cuda:0', non_blocking=True)


def inference(args, model, input_ids, images, image_sizes):
    output_ids = model.generate(
        input_ids,
        images=images.to(dtype=torch.float16, device='cuda:0', non_blocking=True),
        image_sizes=image_sizes,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True)
    
    return output_ids


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name if args.model_name else get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map="cuda:0",
        use_flash_attn=args.use_flash_attn,
    )
    model.cuda()

    data_loader = create_data_loader(args, tokenizer, image_processor, model.config)

    out_data = []
    debug_records = [] if args.debug_dump else None
    for batch in tqdm(data_loader, mininterval=1, total=len(data_loader)):
        question_raw = batch['question']
        data_id = batch['data_id']
        retrieval_ids = batch['retrieval_input_ids'].to(device='cuda:0', non_blocking=True)
        relevant_ids_list = batch['relevant_input_ids']
        images = batch['image_tensor']
        image_sizes = batch['image_size']
        reference = batch['answer']
        question_type = batch['question_type']
        sections = batch['sections'][0]

        # Retrieval Forward [FIRST STAGE]
        with torch.inference_mode():
            output_ids = inference(args, model, retrieval_ids, images, image_sizes)

        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
        torch.cuda.empty_cache()
        gc.collect()
        
        relevant_paragraphs_detected = []
        if 128251 == output_ids[0][0].item() or args.fix_ret_token:
            for id_context, element in enumerate(relevant_ids_list):
                relevant_ids = element.to(device='cuda:0', non_blocking=True)
            
                # Relevant Forward [SECOND STAGE]
                with torch.inference_mode():
                    output_ids = inference(args, model, relevant_ids, images, image_sizes)

                if 128253 == output_ids[0][0].item():
                    relevant_paragraphs_detected.append(id_context)
            
            if len(relevant_paragraphs_detected) == 0:
                relevant_paragraphs_detected = [ random.randint(0,len(sections)-1) ]         

            with torch.inference_mode():
                concat_relevant_paraghraphs = concat_paragraph(relevant_paragraphs_detected, question_raw[0], sections, args.conv_mode, model.config, tokenizer)
                output_ids = inference(args, model, concat_relevant_paraghraphs, images, image_sizes)
        
        answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
        del output_ids
        torch.cuda.empty_cache()
        gc.collect()
        batch_debug_infos = batch.get('debug_info') if args.debug_dump else None

        for i in range(len(answers)):
            sample = {"question": question_raw[i],
                      "reference": reference[i].split('|'),
                      "answers": answers[i],
                      "question_type": question_type[i],
                      "data_id": data_id[i]
                      }
            out_data.append(sample)
            if debug_records is not None:
                debug_entry = batch_debug_infos[i] if batch_debug_infos else None
                record = {
                    "data_id": data_id[i],
                    "question": question_raw[i],
                    "reference": sample["reference"],
                    "answer": answers[i],
                    "question_type": question_type[i]
                }
                if isinstance(debug_entry, dict):
                    record["retrieval_debug"] = debug_entry
                debug_records.append(record)

    with open(args.answers_file, "w") as f:
        f.write(ujson.dumps(out_data))
    if args.debug_dump:
        if args.debug_dump_path:
            debug_dir = os.path.dirname(args.debug_dump_path)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
            with open(args.debug_dump_path, "w") as f:
                f.write(ujson.dumps(debug_records))
        else:
            for record in debug_records:
                print(ujson.dumps(record, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # ReflectiVA hyperparameters
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None) # this is to load the test file
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default="llama_3_1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_flash_attn", action="store_true")
    
    # ReflectiVA evaluation parameteres
    parser.add_argument("--entity_k", type=int, default=1)
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--index_path_json", type=str)
    parser.add_argument('--short_prompt', action='store_true')
    parser.add_argument("--kb_wikipedia_path", type=str)
    parser.add_argument('--use_clip_to_retrieve', action='store_true')
    parser.add_argument('--use_eva_to_retrieve', action='store_true')
    parser.add_argument("--retriever_path", type=str)
    parser.add_argument("--retriever_processor_path", type=str)
    parser.add_argument("--fix_ret_token", type=bool, default=True)
    parser.add_argument("--answers_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument('--debug_dump', action='store_true', help='Enable verbose retrieval debugging for a handful of samples.')
    parser.add_argument('--debug_dump_path', type=str, default=None, help='Optional path to save retrieval debug records as JSON.')
    args = parser.parse_args()
    
    if args.use_eva_to_retrieve:
        retriever_model = 'eva'
    elif args.use_clip_to_retrieve:
        retriever_model = 'clip'
    else:
        retriever_model = 'unknown'
        
    index = 'I2I'

    if args.answers_file:
        answers_dir = os.path.dirname(args.answers_file)
        if answers_dir:
            os.makedirs(answers_dir, exist_ok=True)
    else:
        output_dir = args.output_dir or f'output/Reflectiva_evqa_echo_kb_{retriever_model}_index_{index}'
        part = os.environ.get('PART', "0")
        os.makedirs(output_dir, exist_ok=True)
        input_file = os.path.basename(args.data_path).replace('.json', '')
        args.answers_file = os.path.join(output_dir, f'split_{part}_{input_file}_k{args.entity_k}.json')
    print(f'Saving answers to {args.answers_file}')
    eval_model(args)
