"""
Evaluate GPT-2 Small (124M) on the same 20-question physics QA benchmark
used for TinyLM evaluation. Uses continuation log-probability ranking
(same protocol as Phase 1/Phase 2 evaluation).

Scoring:
  - exact match (rank 1) = 1.0
  - semantic match (rank 2) = 0.5
  - otherwise = 0.0
"""
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

def compute_avg_logprob(model, tokenizer, prompt, answer, device):
    """Compute average log-probability of answer tokens given prompt."""
    full_text = f"{prompt} {answer}"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    
    prompt_len = prompt_ids.shape[1]
    answer_len = full_ids.shape[1] - prompt_len
    
    if answer_len <= 0:
        return -100.0
    
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)
    
    # Get log probs for answer tokens
    log_probs = torch.log_softmax(logits[0], dim=-1)
    
    total_logprob = 0.0
    for i in range(prompt_len, full_ids.shape[1]):
        token_id = full_ids[0, i].item()
        total_logprob += log_probs[i - 1, token_id].item()
    
    return total_logprob / answer_len


def main():
    print("=" * 60)
    print("GPT-2 Small (124M) Physics QA Evaluation")
    print("=" * 60)
    
    # Load dataset
    with open("data/physics_qa_dataset.json", "r") as f:
        dataset = json.load(f)
    
    # Load GPT-2 Small
    print("\nLoading GPT-2 Small (124M parameters)...")
    device = "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params / 1e6:.1f}M parameters on {device}")
    print(f"Evaluating {len(dataset['questions'])} questions...\n")
    
    start_time = time.time()
    results = []
    
    for q in dataset["questions"]:
        question = q["question"]
        expected = q["expected_answer"]
        all_options = [expected] + q["distractors"]
        
        # Score each option
        scored = []
        for option in all_options:
            avg_lp = compute_avg_logprob(model, tokenizer, question, option, device)
            scored.append({"answer": option, "avg_logprob": avg_lp})
        
        # Rank by log-probability (higher is better)
        scored.sort(key=lambda x: x["avg_logprob"], reverse=True)
        
        # Find rank of expected answer
        expected_rank = next(i + 1 for i, s in enumerate(scored) if s["answer"] == expected)
        
        # Score
        if expected_rank == 1:
            score = 1.0
        elif expected_rank == 2:
            score = 0.5
        else:
            score = 0.0
        
        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": question,
            "expected": expected,
            "expected_rank": expected_rank,
            "score": score,
            "best_answer": scored[0]["answer"],
            "ranked_options": scored
        })
        
        status = "EXACT" if score == 1.0 else ("SEMANTIC" if score == 0.5 else "MISS")
        print(f"  [{status}] {q['id']}: rank={expected_rank}, score={score:.1f}")
    
    elapsed = time.time() - start_time
    
    # Compute summary
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores)
    exact_rate = sum(1 for s in scores if s == 1.0) / len(scores)
    semantic_or_better = sum(1 for s in scores if s >= 0.5) / len(scores)
    
    # By category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["score"])
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - GPT-2 Small (124M)")
    print(f"{'=' * 60}")
    print(f"  Average QA Score:      {avg_score:.3f}")
    print(f"  Exact Match Rate:      {exact_rate:.1%}")
    print(f"  Semantic-or-Better:    {semantic_or_better:.1%}")
    print(f"  Elapsed Time:          {elapsed:.1f}s")
    print(f"\n  By Category:")
    for cat, cat_scores in sorted(categories.items()):
        cat_avg = sum(cat_scores) / len(cat_scores)
        cat_exact = sum(1 for s in cat_scores if s == 1.0) / len(cat_scores)
        print(f"    {cat:30s} avg={cat_avg:.3f} exact={cat_exact:.0%}")
    
    # Save results
    output = {
        "metadata": {
            "model": "gpt2 (124M)",
            "task": "physics_qa_baseline_comparison",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device,
            "num_questions": len(results),
            "elapsed_seconds": elapsed
        },
        "summary": {
            "avg_score": avg_score,
            "exact_match_rate": exact_rate,
            "semantic_or_better_rate": semantic_or_better,
            "by_category": {
                cat: {"avg_score": sum(s)/len(s), "exact_rate": sum(1 for x in s if x == 1.0)/len(s)}
                for cat, s in categories.items()
            }
        },
        "results": results
    }
    
    output_path = "results/gpt2_physics_qa_results.json"
    import os
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Comparison table
    print(f"\n{'=' * 60}")
    print(f"COMPARISON TABLE (for paper)")
    print(f"{'=' * 60}")
    print(f"  Random Baseline:       0.250")
    print(f"  TinyLM Phase 1 (35M):  0.325")
    print(f"  GPT-2 Small (124M):    {avg_score:.3f}")
    print(f"  TinyLM Phase 2 (35M):  0.450")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
