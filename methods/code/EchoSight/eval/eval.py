""" Infoseek Validation Set Evaluation script."""
from infoseek_eval import evaluate

if __name__ == "__main__":
    # for split in ["val"]:
    split = "val"
    print(f"===Evaluating===")
    # pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_short.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_5.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_10.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_5_1000.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_10_1000.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/qwen_answers_k5_1000_20250921_103848.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/qwen_answers_k10_1000_20250921_103836.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/qwen_answers_k10_20250920_214804.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/qwen_answers_k10_withoutimage_20250920_224251.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_10.jsonl" # 5,10,20
    # pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_k20_1000.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/qwen_pipeline/topk/outputs/answers_infoseek_top3_Oct15_bge_reranker/infoseek_llama3_answers_bge_reranker_0.5_0.5_1.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_k20_full_Oct15_llama3.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/qwen_pipeline/topk/outputs/answers_infoseek_top3_Oct19_bge_reranker/infoseek_llama3_answers_echo_reranker.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/qwen_pipeline/topk/outputs/answers_infoseek_top3_Oct19_bge_reranker/infoseek_llama3_answers_bge_reranker_0.5_0.5_1.jsonl"
    pred_path = f"/data/qianMa/EchoSight/infoseek_to_eval_5_Nov2.jsonl"
    # pred_path=f"/data/qianMa/EchoSight/qwen_pipeline/topk/outputs/answers_infoseek_top3_Oct19_bge_reranker/infoseek_llama3_answers_bge_reranker_0.5_0.5_1_chuncked.jsonl"
    # pred_path = f"/data/qianMa/EchoSight/outputs/infoseek_to_eval_k20_full_Oct29_I2T_llama3.jsonl"
    print(f"Pred path: {pred_path}")
    reference_path = f"/data/qianMa/EchoSight/InfoSeek/infoseek_val.jsonl"
    reference_qtype_path = f"/data/qianMa/EchoSight/infoseek_val_qtype.jsonl"

    result = evaluate(pred_path, reference_path, reference_qtype_path)
    # print(result.keys())
    final_score = result["final_score"]
    unseen_question_score = result["unseen_question_score"]["score"]
    unseen_entity_score = result["unseen_entity_score"]["score"]
    print(f"{split} final score: {final_score}")
    print(f"{split} unseen question score: {unseen_question_score}")
    print(f"{split} unseen entity score: {unseen_entity_score}")