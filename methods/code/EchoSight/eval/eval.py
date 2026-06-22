""" Infoseek Validation Set Evaluation script."""
from pathlib import Path

from infoseek_eval import evaluate

REPO_ROOT = Path(__file__).resolve().parents[4]

if __name__ == "__main__":
    # for split in ["val"]:
    split = "val"
    print(f"===Evaluating===")
    pred_path = REPO_ROOT / "outputs/raw_methods/infoseek/unfixed/EchoSight.jsonl"
    print(f"Pred path: {pred_path}")
    reference_path = REPO_ROOT / "data/retrieval/infoseek_val.jsonl"
    reference_qtype_path = REPO_ROOT / "data/retrieval/infoseek_val_qtype.jsonl"

    result = evaluate(str(pred_path), str(reference_path), str(reference_qtype_path))
    # print(result.keys())
    final_score = result["final_score"]
    unseen_question_score = result["unseen_question_score"]["score"]
    unseen_entity_score = result["unseen_entity_score"]["score"]
    print(f"{split} final score: {final_score}")
    print(f"{split} unseen question score: {unseen_question_score}")
    print(f"{split} unseen entity score: {unseen_entity_score}")
