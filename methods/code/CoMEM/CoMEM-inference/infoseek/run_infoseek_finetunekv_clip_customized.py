"""zeroshot Infoseek inference script."""
import os
import sys
module_path = "CoMEM-train"
sys.path.append(module_path)
from src_vlm.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
from src_vlm.training.qwenVL_inference2 import Qwen2VLForConditionalGeneration_new
import json
import csv
import torch
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
import time
sys.path.insert(0, "CoMEM-inference")
from src.load_model_test import *

from io import BytesIO
from streaming import StreamingDataset
import base64 
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CAMERA_READY_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_METADATA_PATH = CAMERA_READY_ROOT / "data/ground_truth/infoseek_fixed.csv"
DEFAULT_MDS_DIR = SCRIPT_DIR / "Custom_test_full"
DEFAULT_IMAGE_ROOT = CAMERA_READY_ROOT / "data/images/infoseek_val_images"

def load_mds(mds_dir, max_samples=0):
    dataset = StreamingDataset(local=mds_dir,
                           remote=None,
                           shuffle=False,
                           batch_size=1)
    records = []
    for sample in tqdm(dataset, desc="Loading MDS files"):
        records.append(sample)
        if max_samples and len(records) >= max_samples:
            break
    return records

def load_question_map(data_path):
    if not data_path or not os.path.exists(data_path):
        return {}

    question_map = {}
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                data_id = row.get("data_id")
                question = row.get("question")
                if data_id and question:
                    question_map[str(data_id)] = question
    elif data_path.endswith(".csv"):
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_id = row.get("data_id")
                question = row.get("question") or row.get("question_original")
                if data_id and question:
                    question_map[str(data_id)] = question
    return question_map

def load_and_process_image(item, image_root):
    # Load and preprocess the image
    path = os.path.join(image_root, item['image_path'].split("/")[-1])
    raw_image = Image.open(path).convert("RGB")     
    if raw_image.size[0] > 512 or raw_image.size[1] > 512:
        raw_image = raw_image.resize((512, 512), Image.LANCZOS)       
    return raw_image, item["question"]

def process_images_in_batches(batch_data, question_ids, batch_size, prompt, args):
    ########## Get output saving path ###########
    file_path = os.path.join(args.output_dir, "{}_{}_{}_{}.jsonl".format(
                    args.model_name, args.model_type, args.split, args.similar_num
                    ))
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            output = [json.loads(line) for line in f]
    else:
        output = []
    batch_data = batch_data[len(output):]
    question_ids = question_ids[len(output):]
    if not batch_data:
        print("Existing output already covers requested samples; skip model loading.")
        return output
    # setup device to use
    max_memory = { 
        0: "23GiB",
        1: "23GiB"
    }
    print("Load pretrained model...")
    if 'qwen2.5' in args.model_name:
        checkpoint_path = args.checkpoint_path
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        tokenizer = processor.tokenizer
        model = Qwen2_5_VLForConditionalGeneration_new.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True)
    elif 'qwen2' in args.model_name:
        checkpoint_path = args.checkpoint_path
        print('load qwen2 model')
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
        tokenizer = processor.tokenizer
        model = Qwen2VLForConditionalGeneration_new.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True)
    
    print("Generate predictions...")
    print(f"Total samples to process: {len(batch_data)}")
    # Process images in batches
    for idx, i in enumerate(range(0, len(batch_data), batch_size)):
        if (idx + 1) % 100 == 0:
            print(f"Processing batch {idx}/{len(batch_data)/batch_size}")
        # Subset results for the current batch
        batch_subset = batch_data[i:i+batch_size]
        question_ids_subset = question_ids[i:i+batch_size]

        # Separate the images, questions, and ids
        batch_ids, answers = [], []

        # Load and preprocess the images
        start_time = time.time()
        for tmp_id, item in zip(question_ids_subset, batch_subset):
            tmp_img, tmp_q = load_and_process_image(item, args.image_root)
            batch_ids.append(tmp_id)
            tmp_q = prompt.format(tmp_q)
            ####### Find Similar Images #######
            def process_similar_infos(item, similar_num):
                similar_infos = item["retrieval_info"][:similar_num]
                similar_infos_dict = {}
                for idx, info in enumerate(similar_infos):
                    key = idx
                    image_data = base64.b64decode(info['image'])
                    fact_img = Image.open(BytesIO(image_data)).convert("RGB")   
                    if fact_img.size[0] > 512 or fact_img.size[1] > 512:
                        fact_img = fact_img.resize((512, 512), Image.LANCZOS)
                    fact_text = info["passage_content"] or ""
                    similar_infos_dict[key] = {"image": fact_img, "desc": fact_text}
                return similar_infos_dict
            similar_infos = process_similar_infos(item, args.similar_num)
            texts = [item['desc'] for item in similar_infos.values()]
            images = [item['image'] for item in similar_infos.values()]
            ans = generate_response_knowledge(args.model_name, processor, model, tmp_img, tmp_q, texts, images)
            print(ans)
            answers.append(ans)

        print(f"Time for batch {idx}: {time.time() - start_time}")
        for idx, ans in zip(batch_ids, answers):
            output.append({"data_id": idx, "prediction": ans})
        # save output into jsonl
        with open(file_path, 'w') as f:
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
       
    return output
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="custom", help="custom, val, test, or human")
    parser.add_argument("--model_name", type=str, default="qwen2.5_clip", help="blip2_t5 | blip2_vicuna_instruct | blip2_t5_instruct")
    parser.add_argument("--model_type", type=str, default="CoMEM", help="pretrain_flant5xxl | vicuna13b | flant5xxl")
    parser.add_argument("--output_dir", type=str, default="CoMEM-inference/infoseek/result", help="output directory")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--similar_num", type=int, default=10, help="number of similar samples")
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
    parser.add_argument("--metadata_path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--mds_dir", type=str, default=str(DEFAULT_MDS_DIR))
    parser.add_argument("--image_root", type=str, default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--max_samples", type=int, default=0, help="debug option: process only first N MDS rows")
    parser.add_argument(
        "--disable_metadata_override",
        action="store_true",
        help="Use question from MDS directly and skip CSV/JSONL metadata override.",
    )
    args = parser.parse_args()

    split2data = {
        "val": "CoMEM-inference/infoseek/val_dataset/infoseek_val.jsonl",
        "custom": args.metadata_path,
        # "val": "CoMEM-inference/infoseek/Infoseek_test_full",
        "spanish": "CoMEM-inference/infoseek/val_dataset/infoseek_val_spanish.jsonl",
        "portuguese": "CoMEM-inference/infoseek/val_dataset/infoseek_val_portuguese.jsonl",
        "chinese": "CoMEM-inference/infoseek/val_dataset/infoseek_val_chinese.jsonl",
        "russian": "CoMEM-inference/infoseek/val_dataset/infoseek_val_russian.jsonl",
        "bulgarian": "CoMEM-inference/infoseek/val_dataset/infoseek_val_bulgarian.jsonl",
    }

    # Read the input MDS dataset
    print("Read the input MDS dataset")
    batch_data = load_mds(args.mds_dir, max_samples=args.max_samples)
    if args.disable_metadata_override:
        print("Metadata override disabled. Use question from MDS directly.")
    else:
        metadata_path = split2data.get(args.split, "")
        question_map = load_question_map(metadata_path)
        if question_map:
            updated = 0
            for idx, item in enumerate(batch_data):
                key = item["data_id"]
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                else:
                    key = str(key)
                if key in question_map:
                    item["question"] = question_map[key]
                    batch_data[idx] = item
                    updated += 1
            print(f"Updated question field from metadata: {updated}/{len(batch_data)} ({metadata_path})")
        else:
            print("No metadata override applied. Use question from MDS directly.")

    # double check data exists:
    not_exist = []
    clean_batch_data = []
    clean_question_ids = []
    for idx, item in enumerate(batch_data):
        if idx % 10000 == 0:
            print(f"Processing {idx}/{len(batch_data)}")
        qid = item['data_id']
        path = os.path.join(args.image_root, item['image_path'].split("/")[-1])
        # check path exists
        if not os.path.exists(path):
            not_exist.append(qid)
        else:
            clean_batch_data.append(item)
            clean_question_ids.append(qid)
    # print(f"path {path} not exist, total {len(not_exist)}")
    print(len(not_exist))
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            
    # Desired batch size
    batch_size = args.batch_size

    templa_dict = {
        "val": """Question: {} 
    For this question, please reference to the given information and perform step-by-step reasoning, to obtain the final answer. 
    Note that the final answer should be formatted as:
    Reasoning Process: all thinking steps
    Final answer: \\boxed{{your short answer here}}""",
        "custom": """Question: {} 
    For this question, please reference to the given information and perform step-by-step reasoning, to obtain the final answer. 
    Note that the final answer should be formatted as:
    Reasoning Process: all thinking steps
    Final answer: \\boxed{{your short answer here}}""",
"spanish": """Pregunta: {}  
Para esta pregunta, por favor realiza un razonamiento paso a paso para obtener la respuesta final. Ten en cuenta que la respuesta final debe estar formateada de la siguiente manera:  
Proceso de razonamiento: todos los pasos del pensamiento  
Respuesta final: \boxed{{tu respuesta corta aquí}}""",
    "chinese": """问题: {}  
请参考所给信息，并进行逐步推理，以得出最终答案。  
请注意，最终答案应采用以下格式：  
推理过程: 所有思考步骤  
最终答案: \\boxed{{你的简短答案}}""",
"spanish": """Pregunta: {}  
Para esta pregunta, por favor consulta la información proporcionada y realiza un razonamiento paso a paso para llegar a la respuesta final.  
Ten en cuenta que la respuesta final debe tener el siguiente formato:  
Proceso de razonamiento: todos los pasos del pensamiento  
Respuesta final: \\boxed{{tu respuesta breve aquí}}""",
"russian": """Вопрос: {}  
Пожалуйста, опирайтесь на предоставленную информацию и выполните пошаговое рассуждение, чтобы получить окончательный ответ.  
Обратите внимание, что окончательный ответ должен быть в следующем формате:  
Ход рассуждений: все шаги размышлений  
Окончательный ответ: \\boxed{{ваш краткий ответ здесь}}""",
"portuguese": """Pergunta: {}  
Para esta pergunta, por favor, consulte as informações fornecidas e realize um raciocínio passo a passo para chegar à resposta final.  
Note que a resposta final deve estar formatada da seguinte forma:  
Processo de raciocínio: todos os passos do pensamento  
Resposta final: \\boxed{{sua resposta curta aqui}}""",
"bulgarian": """Въпрос: {}
За този въпрос, моля, използвайте предоставената информация и извършете поетапно разсъждение, за да достигнете до крайния отговор.  
Обърнете внимание, че крайният отговор трябва да бъде във формат:
Процес на разсъждение: всички мисловни стъпки  
Краен отговор: \\boxed{{вашият кратък отговор тук}}""",
    }
    
    PROMPT = templa_dict[args.split]
    
    # Run the batch processing function
    output = process_images_in_batches(clean_batch_data, clean_question_ids, batch_size, prompt=PROMPT, args=args)

    # save output into jsonl
    with open(os.path.join(args.output_dir, "{}_{}_{}_{}.jsonl".format(
                args.model_name, args.model_type, args.split, args.similar_num
                )), 'w') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
