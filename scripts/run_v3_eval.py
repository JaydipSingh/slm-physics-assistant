#!/usr/bin/env python3
"""
Evaluate TinyLM v3 on physics QA.
Usage:
  python scripts/run_v3_eval.py --checkpoint checkpoints/v3_long50k_final.pt
  python scripts/run_v3_eval.py --checkpoint checkpoints/v3_long50k_final.pt --lora checkpoints/phase2_lora_v3/best_lora_adapter.pt
"""
import json, sys, time, argparse
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import stream_train_v3 as st3

def compute_avg_logprob(model, q_tokens, a_tokens, device):
    full = q_tokens + a_tokens
    if not a_tokens:
        return -100.0
    inp = torch.tensor([full], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(inp)
    lp = F.log_softmax(logits[0], dim=-1)
    total = sum(lp[i-1, full[i]].item() for i in range(len(q_tokens), len(full)))
    return total / len(a_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lora", type=Path, default=None)
    parser.add_argument("--qa-dataset", type=Path, default=ROOT / "data" / "physics_qa_dataset.json")
    args = parser.parse_args()

    print("=" * 60)
    print("TinyLM v3 Physics QA Evaluation")
    print("=" * 60)

    with open(args.qa_dataset) as f:
        dataset = json.load(f)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tokenizer_model = Path(cfg["tokenizer_model"])
    st3._activate_tokenizer_model(tokenizer_model)
    vocab_size = st3._resolve_vocab_size(tokenizer_model)

    device = st3.DEVICE
    model = st3.TinyLMv3(vocab_size=vocab_size, d_model=cfg["d_model"],
                         n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                         max_seq_len=cfg["seq_len"]+1).to(device)
    model.load_state_dict(ckpt["model"])

    if args.lora and args.lora.exists():
        print(f"Loading LoRA: {args.lora}")
        sys.path.insert(0, str(ROOT / "scripts"))
        from phase2_lora_v3 import inject_lora, load_lora_state
        lora_ckpt = torch.load(args.lora, map_location="cpu", weights_only=False)
        lora_cfg = lora_ckpt["config"]
        inject_lora(model, r=lora_cfg["r"], alpha=lora_cfg["alpha"],
                    targets=tuple(lora_cfg.get("targets", ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"))))
        load_lora_state(model, lora_ckpt["lora_state"])
        model.to(device)
        print(f"  LoRA loaded (r={lora_cfg['r']})")

    model.eval()
    print(f"Model: {model.count_parameters():,} params on {device}\n")

    results = []
    for q in dataset["questions"]:
        q_tok = st3._tokenize_with_our_model(q["question"])
        all_opts = [q["expected_answer"]] + q["distractors"]
        scored = []
        for opt in all_opts:
            a_tok = st3._tokenize_with_our_model(f" {opt}")
            scored.append({"answer": opt, "lp": compute_avg_logprob(model, q_tok, a_tok, device)})
        scored.sort(key=lambda x: x["lp"], reverse=True)
        rank = next(i+1 for i,s in enumerate(scored) if s["answer"] == q["expected_answer"])
        score = 1.0 if rank == 1 else (0.5 if rank == 2 else 0.0)
        tag = "EXACT" if score == 1 else ("SEM" if score == 0.5 else "MISS")
        print(f"  [{tag}] {q['id']}: rank={rank}")
        results.append({"id": q["id"], "cat": q["category"], "score": score})

    scores = [r["score"] for r in results]
    avg = sum(scores)/len(scores)
    exact = sum(1 for s in scores if s == 1.0)/len(scores)
    sem_plus = sum(1 for s in scores if s >= 0.5)/len(scores)

    cats = {}
    for r in results:
        cats.setdefault(r["cat"], []).append(r["score"])

    print(f"\n{'='*60}")
    print(f"RESULTS — TinyLM v3 {'+ LoRA' if args.lora else '(base)'}")
    print(f"{'='*60}")
    print(f"  QA Score:           {avg:.3f}")
    print(f"  Exact Match:        {exact:.1%}")
    print(f"  Semantic-or-Better: {sem_plus:.1%}")
    for cat, cs in sorted(cats.items()):
        print(f"    {cat:30s} {sum(cs)/len(cs):.3f}")

    p_eos, top_p, top_id = st3.check_generation_health(model, device, vocab_size)
    print(f"\n  P(EOS)={p_eos:.4f} | Top={top_p:.4f}(id={top_id})")
    status = "HEALTHY" if p_eos < 0.1 else ("OK" if p_eos < 0.3 else "WARN")
    print(f"  Status: {status}")

    print(f"\n  COMPARISON:")
    print(f"    Random:        0.250")
    print(f"    v1 Phase 1:    0.325")
    print(f"    GPT-2 (124M):  0.325")
    print(f"    v1 Phase 2:    0.450")
    print(f"    v3 (this):     {avg:.3f}")

    out = {"avg_score": avg, "exact": exact, "sem_plus": sem_plus,
           "p_eos": p_eos, "by_cat": {c: sum(s)/len(s) for c,s in cats.items()}}
    out_path = ROOT / "results" / f"v3_qa_{'lora' if args.lora else 'base'}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")

if __name__ == "__main__":
    main()
