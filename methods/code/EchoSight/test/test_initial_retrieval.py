from argparse import ArgumentParser
import csv
import json, tqdm
import torch
from model import (
    ClipRetriever,
    MistralAnswerGenerator,
    GPT4AnswerGenerator,
    reconstruct_wiki_article,
    PaLMAnswerGenerator,
    reconstruct_wiki_sections,
    WikipediaKnowledgeBaseEntry,
)
from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates
import PIL

iNat_image_path = "/data/qianMa/EchoSight/images" #"/PATH/TO/INAT_ID2NAME"


def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def run_test(
    test_file_path: str,
    knowledge_base_path: str,
    faiss_index_path: str,
    top_ks: list,
    retrieval_top_k: int,
    **kwargs
):
    test_list, test_header = load_csv_data(test_file_path)
    print(f"######\nLoaded {len(test_list)} test examples from {test_file_path}")
    with open(iNat_image_path + "/val_id2name.json", "r") as f:
        iNat_id2name = json.load(f)

    if kwargs["resume_from"] is not None:
        resumed_results = json.load(open(kwargs["resume_from"], "r"))
        kb_dict = json.load(open(knowledge_base_path, "r"))
    else:
        retriever = ClipRetriever(device="cuda:1", model=kwargs["retriever_vit"])
        # retriever.save_knowledge_base_faiss(knowledge_base_path, scores_path=score_dict, save_path=faiss_index_path)
        retriever.load_knowledge_base(knowledge_base_path)
        retriever.load_faiss_index(faiss_index_path)

    recalls = {k: 0 for k in top_ks}
    grounded_rows = []
    ungrounded_rows = []
    header_has_data_id = "data_id" in test_header
    if header_has_data_id:
        csv_header = list(test_header)
        data_id_index = test_header.index("data_id")
    else:
        csv_header = list(test_header) + ["data_id"]

    if kwargs["perform_vqa"]:
        from utils import evaluate_example
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")  # disable GPU for tensorflow
        question_generator = MistralAnswerGenerator(
            model_path="/remote-home/share/huggingface_model/Mistral-7B-Instruct-v0.2",
            device="cuda:0",
            use_embedding_model=False,
        )
        eval_score = 0

    metric = "url matching"

    retrieval_result = {}
    for it, test_example in tqdm.tqdm(enumerate(test_list)):
        example = get_test_question(it, test_list, test_header)
        row_data = list(test_list[it])
        # print("Example: ", example) # example["data_id"], .split("|")[0]
        print("\n", example["dataset_image_ids"], example["dataset_name"])
        image_path_find = get_image(
                example["dataset_image_ids"],
                example["dataset_name"],
                iNat_id2name,
            )
        if image_path_find is None:
            print(f"Image not found for {example['dataset_image_ids']} in {example['dataset_name']}, skipping...")
            # continue
            break
        image = PIL.Image.open(
            image_path_find
        )
        ground_truth = example["wikipedia_url"]
        target_answer = example["answer"].split("|")
        if example["dataset_name"] == "infoseek":
            data_id = example["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)
        if header_has_data_id:
            row_data[data_id_index] = data_id
            row_with_id = row_data
        else:
            row_with_id = row_data + [data_id]
        print("wiki_url: ", example["wikipedia_url"])
        print("question: ", example["question"])
        if kwargs["resume_from"] is not None:
            resumed_result = resumed_results[data_id]
            retrieved_entries = resumed_result["retrieved_entries"]
            if (
                isinstance(retrieved_entries, list)
                and retrieved_entries
                and isinstance(retrieved_entries[0], list)
            ):
                top_k_wiki = retrieved_entries[0]
                retrieval_simlarities = retrieved_entries[1]
            else:
                top_k_wiki = retrieved_entries
                retrieval_simlarities = resumed_result.get(
                    "retrieval_similarities", []
                )
            entries = [WikipediaKnowledgeBaseEntry(kb_dict[url]) for url in top_k_wiki]
        else:
            top_k = retriever.retrieve_image_faiss(image, top_k=retrieval_top_k)
            top_k_wiki = [retrieved_entry["url"] for retrieved_entry in top_k]
            top_k_wiki = remove_list_duplicates(top_k_wiki)
            entries = [retrieved_entry["kb_entry"] for retrieved_entry in top_k]
            entries = remove_list_duplicates(entries)
            seen = set()
            retrieval_simlarities = [
                top_k[i]["similarity"]
                for i in range(retrieval_top_k)
                if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))
            ]

        candidate_sections = []
        if metric == "answer matching" or kwargs.get("perform_vqa", False):
            for entry in entries:
                candidate_sections.extend(reconstruct_wiki_sections(entry))
            candidate_sections = remove_list_duplicates(candidate_sections)

        is_grounded = ground_truth in top_k_wiki[:retrieval_top_k]
        if kwargs["save_result"]:
            retrieval_result[data_id] = {
                "retrieved_entries": [entry.url for entry in entries[:20]],
                "retrieval_similarities": [
                    sim.item() if hasattr(sim, "item") else sim
                    for sim in retrieval_simlarities[:20]
                ],
                "is_grounded": is_grounded,
            }
        if is_grounded:
            grounded_rows.append(row_with_id)
        else:
            ungrounded_rows.append(row_with_id)
        if metric == "answer matching":
            entry_articles = [reconstruct_wiki_article(entry) for entry in entries]
            found = False
            for i, entry in enumerate(entry_articles):
                for answer in target_answer:
                    if answer.strip().lower() in entry.strip().lower():
                        found = True
                        break
                if found:
                    break
            if found:
                for k in top_ks:
                    if i < k:
                        recalls[k] += 1

        else:
            # in url_matching
            recall = eval_recall(top_k_wiki, ground_truth, top_ks)
            for k in top_ks:
                recalls[k] += recall[k]
        for k in top_ks:
            print("Avg Recall@{}: ".format(k), recalls[k] / (it + 1))

        if kwargs["perform_vqa"]:
            if not candidate_sections:
                print("No candidate sections available for VQA, skipping this example.")
            else:
                answer = question_generator.llm_answering(
                    question=example["question"], entry_section=candidate_sections[0]
                )

                print("answer: ", answer)
                print("target answer: ", target_answer)
                score = evaluate_example(
                    example["question"],
                    reference_list=target_answer,
                    candidate=answer,
                    question_type=example["question_type"],
                )

                eval_score += score
                print("score: ", score, "iter: ", it + 1)
                print("eval score: ", eval_score / (it + 1))
    if kwargs["save_result"]:
        with open(kwargs["save_result_path"], "w") as f:
            json.dump(retrieval_result, f, indent=4)

    if kwargs.get("grounded_csv"):
        with open(kwargs["grounded_csv"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(grounded_rows)
        print(f"Saved {len(grounded_rows)} grounded examples to {kwargs['grounded_csv']}")
    else:
        print(f"Total grounded examples: {len(grounded_rows)}")

    if kwargs.get("ungrounded_csv"):
        with open(kwargs["ungrounded_csv"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(ungrounded_rows)
        print(
            f"Saved {len(ungrounded_rows)} ungrounded examples to {kwargs['ungrounded_csv']}"
        )
    else:
        print(f"Total ungrounded examples: {len(ungrounded_rows)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1,5,10,20",
        help="comma separated list of top k values, e.g. 1,5,10,20",
    )
    parser.add_argument("--perform_vqa", action="store_true")
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--save_result_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--retriever_vit", type=str, default="clip", help="clip or eva-clip"
    )
    parser.add_argument("--grounded_csv", type=str, default=None)
    parser.add_argument("--ungrounded_csv", type=str, default=None)
    args = parser.parse_args()

    test_config = {
        "test_file_path": args.test_file,
        "knowledge_base_path": args.knowledge_base,
        "faiss_index_path": args.faiss_index,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "perform_vqa": args.perform_vqa,
        "save_result": args.save_result,
        "save_result_path": args.save_result_path,
        "resume_from": args.resume_from,
        "retriever_vit": args.retriever_vit,
        "grounded_csv": args.grounded_csv,
        "ungrounded_csv": args.ungrounded_csv,
    }
    print("test_config: ", test_config)
    run_test(**test_config)
