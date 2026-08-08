#!/usr/bin/env python3
"""
Phase 2: LoRA Fine-Tuning for TinyLM v3
========================================
Now with FULL LoRA support on all attention + FFN projections:
  - q_proj, k_proj, v_proj, o_proj (attention)
  - gate_proj, up_proj, down_proj (SwiGLU FFN)

Usage:
  python scripts/phase2_lora_v3.py --base-checkpoint checkpoints/v3_long50k_final.pt
"""
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import stream_train_v3 as st3


class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear. Compatible with all linear layers."""
    def __init__(self, original: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = original.in_features, original.out_features

        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.lora_dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = F.linear(x, self.original.weight, self.original.bias)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora


def inject_lora(model: nn.Module, r=8, alpha=16, dropout=0.05,
                targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
    """Inject LoRA into all matching Linear modules."""
    injected = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in targets):
            continue
        # Skip weight-tied head
        if name == "head":
            continue

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, r, alpha, dropout))
        injected.append(name)

    return injected


def extract_lora_state(model):
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            state[name] = {"lora_A": mod.lora_A.data.cpu(), "lora_B": mod.lora_B.data.cpu()}
    return state


def load_lora_state(model, state):
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear) and name in state:
            mod.lora_A.data = state[name]["lora_A"].to(mod.lora_A.device)
            mod.lora_B.data = state[name]["lora_B"].to(mod.lora_B.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-papers", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    print("=" * 70)
    print("TinyLM v3 — LoRA Fine-Tuning (Full Attention + FFN)")
    print("=" * 70)

    # Load base
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tokenizer_model = Path(cfg["tokenizer_model"])
    st3._activate_tokenizer_model(tokenizer_model)
    vocab_size = st3._resolve_vocab_size(tokenizer_model)

    device = st3.DEVICE
    model = st3.TinyLMv3(
        vocab_size=vocab_size, d_model=cfg["d_model"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        max_seq_len=cfg["seq_len"] + 1,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Base: {model.count_parameters():,} params on {device}")

    # Inject LoRA
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    injected = inject_lora(model, r=args.lora_r, alpha=args.lora_alpha, targets=targets)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, {len(injected)} layers")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    for n in injected[:6]:
        print(f"  - {n}")
    if len(injected) > 6:
        print(f"  ... and {len(injected)-6} more")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Data
    token_stream = st3.stream_arxiv(category="physics", max_results=args.max_papers)
    checkpoints_dir = ROOT / "checkpoints" / "phase2_lora_v3"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step, total_loss = 0, 0.0
    start = time.time()
    max_steps = args.max_steps
    warmup = 500

    print(f"\nTraining: {max_steps} steps, bs={args.batch_size}")

    for x, y in st3.make_packed_batches(token_stream, args.batch_size, cfg["seq_len"]):
        lr = st3.get_lr(step, warmup, max_steps, args.lr, args.lr * 0.1)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 20 == 0:
            avg = total_loss / step
            elapsed = time.time() - start
            eta = (max_steps - step) / (step / elapsed) / 60
            print(f"  Step {step:5d}/{max_steps} | Loss: {loss.item():.4f} | "
                  f"Avg: {avg:.4f} | LR: {lr:.2e} | ETA: {eta:.0f}min")

        if step % 1000 == 0:
            p_eos, _, _ = st3.check_generation_health(model, device, vocab_size)
            print(f"  [HEALTH] P(EOS)={p_eos:.4f}")
            lora_state = extract_lora_state(model)
            torch.save({"lora_state": lora_state, "step": step,
                        "config": {"r": args.lora_r, "alpha": args.lora_alpha,
                                   "targets": list(targets)},
                        "avg_loss": total_loss/step},
                       checkpoints_dir / f"lora_step{step}.pt")
            print(f"  [SAVE] lora_step{step}.pt")

        if step >= max_steps:
            break

    # Final
    elapsed = time.time() - start
    lora_state = extract_lora_state(model)
    final_path = checkpoints_dir / "best_lora_adapter.pt"
    torch.save({"lora_state": lora_state, "step": step,
                "config": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": list(targets)},
                "avg_loss": total_loss/step}, final_path)

    p_eos, _, _ = st3.check_generation_health(model, device, vocab_size)
    print(f"\n{'=' * 70}")
    print(f"LoRA Complete! Steps={step}, Loss={total_loss/step:.4f}, Time={elapsed/3600:.1f}h")
    print(f"P(EOS)={p_eos:.4f}, Adapter: {final_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
