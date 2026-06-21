from argparse import ArgumentParser
import csv, json
from typing import Optional
from model import (
    MistralAnswerGenerator,
    LLaMA3AnswerGenerator,
    PaLMAnswerGenerator,
    GPT4AnswerGenerator,
)
from utils import evaluate_example, load_csv_data, get_test_question
import tensorflow as tf
import tqdm
import PIL

# If using CPU for tensorflow
# tf.config.set_visible_devices([], "GPU")
from utils import load_csv_data, get_test_question, get_image, seed_everything


def run_vqa(
    question_generator,
    test_file,
    retrieval_results_file,
    output_file,
    dataset_name: Optional[str] = None,
):
    test_list, test_header = load_csv_data(test_file)
    if retrieval_results_file:
        retrieval_results = json.load(open(retrieval_results_file, "r"))
        print(f"Loaded {len(retrieval_results)} retrieval results")
        print(f"retrieved example: {retrieval_results.keys()}")
    else:
        retrieval_results = None
    result_list = []
    for it, example in tqdm.tqdm(enumerate(test_list)):
        question = get_test_question(it, test_list, test_header)
        candidate_ids = []
        if "data_id" in question and question["data_id"]:
            candidate_ids.append(question["data_id"])
        dataset_from_row = question.get("dataset_name", "")
        effective_dataset = dataset_name or dataset_from_row
        evqa_candidate = f"E-VQA_{it:08d}"
        candidate_ids.extend(
            [
                evqa_candidate,
                f"E-VQA_{it}",
                f"{dataset_name}_{question.get('dataset_image_ids', '')}".rstrip("_"),
                str(question.get("dataset_image_ids", "")),
            ]
        )
        seen = set()
        candidate_ids = [cid for cid in candidate_ids if cid and not (cid in seen or seen.add(cid))]
        data_id = candidate_ids[0] if candidate_ids else evqa_candidate
        print(f"Processing {data_id}: {question['question']}")
        if retrieval_results:
            matched_id = None
            for cid in candidate_ids:
                if cid in retrieval_results:
                    matched_id = cid
                    break
            if matched_id is None:
                sample_keys = list(retrieval_results.keys())[:5]
                raise KeyError(
                    f"No retrieval result found for candidates {candidate_ids}. "
                    f"Sample keys: {sample_keys}"
                )
            data_id = matched_id
            entry_section = retrieval_results[data_id]["reranked_sections"][0]
            answer = question_generator.llm_answering(
                question=question["question"],
                entry_section=entry_section,
                dataset_name=effective_dataset,
            )
        else:
            answer = question_generator.llm_answering(
                question=question["question"],
                dataset_name=effective_dataset,
            )

        result_list.append(
            {
                "data_id": data_id,
                "prediction": answer,
            }
        )
    with open(output_file, "w") as f:
        for result in result_list:
            json.dump(result, f)
            f.write('\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--retrieval_results", type=str)
    parser.add_argument("--answer_generator", type=str)
    parser.add_argument("--llm_checkpoint", type=str)
    parser.add_argument("--output_file", type=str, default="answer.json")
    parser.add_argument("--dataset_name", type=str, default=None)
    
    args = parser.parse_args()
    test_file = args.test_file
    retrieval_results = args.retrieval_results
    print("#"*50)
    seed_everything(42)
    print(f"Seeded everything with 42")
    print("#"*50)
    # vqa_results = args.vqa_results
    output_file = args.output_file
    if args.answer_generator.lower() == "mistral":
        answer_generator = MistralAnswerGenerator(model_path=args.llm_checkpoint,device="cuda")
    elif args.answer_generator.lower() == "llama3":
        answer_generator = LLaMA3AnswerGenerator(model_path=args.llm_checkpoint,device="cuda")
    elif args.answer_generator.lower() == "gpt4":
        answer_generator = GPT4AnswerGenerator()
    elif args.answer_generator.lower() == "palm":
        answer_generator = PaLMAnswerGenerator()
    else:
        raise ValueError("Invalid Answer Generator, Please choose from Mistral, LLaMA3, GPT4, PaLM")
    run_vqa(
        answer_generator,
        test_file,
        retrieval_results,
        output_file,
        dataset_name=args.dataset_name,
    )
