#!/usr/bin/env python3
"""
stream_train_v3.py — TinyLM v3: Custom Transformer + Document Packing
======================================================================
Fixes from v2:
  1. DOCUMENT PACKING: Multiple documents packed end-to-end without EOS tokens
     in training targets. EOS never appears as a prediction target.
  2. CUSTOM TRANSFORMER: Hand-written multi-head attention with explicit
     Q/K/V/O projections that can be individually LoRA-injected.
  3. EOS-FREE LOSS: Cross-entropy computed only on content tokens.

Architecture:
  - Custom CausalSelfAttention with separate q_proj, k_proj, v_proj, o_proj
  - RMSNorm (more stable than LayerNorm for small models)
  - SwiGLU FFN (modern, higher capacity per parameter)
  - Rotary Position Embeddings (RoPE) — better length generalization

Usage:
  python stream_train_v3.py --preset long50k    # Full training (~6-8h on M3 Pro)
  python stream_train_v3.py --preset prototype  # Quick test (~5 min)
"""
import ast
import argparse
import json
import math
import random
import shutil
import time
import arxiv
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import subprocess
import requests
from typing import Optional

ROOT = Path(__file__).resolve().parent
TOKENIZER_PROJECT_DIR = ROOT / ".." / "subword_tokenizer"
if not TOKENIZER_PROJECT_DIR.exists():
    TOKENIZER_PROJECT_DIR = ROOT / "subword_tokenizer"
TOKENIZER_MODEL = TOKENIZER_PROJECT_DIR / "model_32k.json"
TOKENIZER_ACTIVE_MODEL = TOKENIZER_PROJECT_DIR / "model.json"
TOKENIZER_BIN = TOKENIZER_PROJECT_DIR / "target" / "release" / "bpe-tokenizer"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Special token IDs
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


# ============================================================================
# TOKENIZER UTILITIES
# ============================================================================

def _resolve_vocab_size(model_path: Path) -> int:
    with model_path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    vocab = model.get("vocab", {})
    return len(vocab) if isinstance(vocab, (dict, list)) else 32000


def _activate_tokenizer_model(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    shutil.copy2(model_path, TOKENIZER_ACTIVE_MODEL)


def _tokenize_with_our_model(text: str) -> list[int]:
    if TOKENIZER_BIN.exists():
        cmd = [str(TOKENIZER_BIN), "tokenize", text]
        cwd = str(TOKENIZER_PROJECT_DIR)
    else:
        cmd = ["cargo", "run", "--release", "--manifest-path",
               str(TOKENIZER_PROJECT_DIR / "Cargo.toml"), "--", "tokenize", text]
        cwd = str(ROOT)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Tokenizer failed: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        if line.startswith("IDs:"):
            return ast.literal_eval(line.replace("IDs:", "").strip())
    return []


# ============================================================================
# CUSTOM TRANSFORMER BLOCKS
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope(dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute Rotary Position Embedding frequencies."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to input tensor."""
    # x shape: (batch, n_heads, seq_len, head_dim)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with explicit Q/K/V/O projections.
    Each projection is a separate nn.Linear — fully LoRA-compatible.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Separate projections (LoRA-friendly)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (used in LLaMA, Mistral)."""
    def __init__(self, d_model: int, hidden_mult: float = 2.667):
        super().__init__()
        hidden_dim = int(d_model * hidden_mult)
        # Round to multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm, custom attention, SwiGLU FFN."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyLMv3(nn.Module):
    """
    Custom decoder-only transformer for domain SLM training.
    - Explicit Q/K/V/O projections (LoRA-friendly)
    - RoPE positional encoding
    - RMSNorm + SwiGLU FFN
    - Weight-tied embedding/head
    """
    def __init__(self, vocab_size=32000, d_model=384, n_layers=6, n_heads=6,
                 max_seq_len=512, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        # Precompute RoPE
        head_dim = d_model // n_heads
        rope_cos, rope_sin = precompute_rope(head_dim, max_seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.embed(x))
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)
        h = self.norm_f(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# DATA: DOCUMENT PACKING (NO EOS IN TARGETS)
# ============================================================================

def _build_arxiv_query(category: str) -> str:
    if any(op in category for op in ["cat:", "all:", "AND", "OR"]):
        return category
    return f"cat:{category}" if "." in category else f"all:{category}"


def stream_arxiv(category="physics", max_results=50000, delay_seconds=2,
                 min_tokens=12, max_retries=100, retry_backoff_seconds=5):
    """Stream tokenized papers, stripping EOS/BOS tokens from output."""
    query = _build_arxiv_query(category)
    retries = 0
    papers = 0
    while True:
        try:
            client = arxiv.Client(page_size=1000, delay_seconds=delay_seconds, num_retries=5)
            search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.SubmittedDate,
                                  max_results=max_results)
            for result in client.results(search):
                text = f"Title: {result.title}\nAbstract: {result.summary}\n\n"
                tokens = _tokenize_with_our_model(text)
                # CRITICAL: Strip EOS/BOS/PAD tokens — we pack documents continuously
                tokens = [t for t in tokens if t not in (PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID)]
                if len(tokens) >= min_tokens:
                    papers += 1
                    if papers % 1000 == 0:
                        print(f"  [stream] {papers} papers tokenized...")
                    yield tokens
            print(f"  [stream] Complete: {papers} papers")
            return
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"  [stream] Max retries. {papers} papers so far.")
                return
            sleep_s = retry_backoff_seconds * min(retries, 64)
            print(f"  [stream] Error ({type(e).__name__}). Retry {retries}/{max_retries} in {sleep_s}s...")
            time.sleep(sleep_s)


def make_packed_batches(token_stream, batch_size=4, seq_len=256, shuffle_buffer=4096):
    """
    Pack multiple documents into fixed-length sequences WITHOUT EOS separators.
    
    This is the GPT-style packing approach:
    - Documents are concatenated end-to-end in a flat token buffer
    - Fixed-length chunks are carved out (no padding, no EOS)
    - The model learns to predict the next content token at every position
    - This prevents the model from learning "predict EOS" as a dominant strategy
    """
    buffer = []
    sequences = []

    for tokens in token_stream:
        buffer.extend(tokens)

        # Carve out full sequences from the buffer
        while len(buffer) >= seq_len + 1:
            seq = buffer[:seq_len + 1]
            buffer = buffer[seq_len:]  # Shift by seq_len (not seq_len+1) for overlap
            sequences.append(seq)

            # Yield shuffled batches
            if len(sequences) >= shuffle_buffer:
                random.shuffle(sequences)
                while len(sequences) >= batch_size:
                    batch_seqs = sequences[:batch_size]
                    sequences = sequences[batch_size:]
                    batch = torch.tensor(batch_seqs, dtype=torch.long)
                    x = batch[:, :-1].to(DEVICE)  # Input: tokens 0..seq_len-1
                    y = batch[:, 1:].to(DEVICE)   # Target: tokens 1..seq_len
                    yield x, y

    # Flush remaining
    if sequences:
        random.shuffle(sequences)
        while len(sequences) >= batch_size:
            batch_seqs = sequences[:batch_size]
            sequences = sequences[batch_size:]
            batch = torch.tensor(batch_seqs, dtype=torch.long)
            x = batch[:, :-1].to(DEVICE)
            y = batch[:, 1:].to(DEVICE)
            yield x, y


# ============================================================================
# LR SCHEDULE & HEALTH MONITORING
# ============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def check_generation_health(model, device, vocab_size):
    """Check P(EOS) on random input to detect collapse."""
    model.eval()
    test_input = torch.randint(3, min(vocab_size, 1000), (1, 32), device=device)
    logits = model(test_input)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    p_eos = probs[EOS_TOKEN_ID].item()
    top_prob, top_idx = probs.max(dim=-1)
    model.train()
    return p_eos, top_prob.item(), top_idx.item()


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(config: dict):
    print("=" * 70)
    print("TinyLM v3 Training")
    print("  Custom Transformer + Document Packing (EOS-free)")
    print("=" * 70)

    tokenizer_model = Path(config["tokenizer_model"])
    _activate_tokenizer_model(tokenizer_model)
    vocab_size = _resolve_vocab_size(tokenizer_model)

    model = TinyLMv3(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_seq_len=config["seq_len"] + 1,
        dropout=config.get("dropout", 0.0),
    ).to(DEVICE)

    total_params = model.count_parameters()
    print(f"\nDevice: {DEVICE}")
    print(f"Model: d={config['d_model']}, L={config['n_layers']}, H={config['n_heads']}, "
          f"params={total_params:,}")
    print(f"Training: papers={config['max_papers']}, steps={config['max_steps']}, "
          f"bs={config['batch_size']}, seq={config['seq_len']}")
    print(f"LR: {config['lr']} → {config['lr']*0.1} (cosine, warmup={config['warmup_steps']})")
    print(f"Data: EOS-FREE document packing\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.1), betas=(0.9, 0.95))

    checkpoints_dir = ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    prefix = config["checkpoint_prefix"]

    max_steps = config["max_steps"]
    warmup_steps = config["warmup_steps"]
    max_lr = config["lr"]
    min_lr = max_lr * 0.1
    save_every = config.get("save_every_steps", 1000)
    health_every = config.get("health_check_every", 500)
    log_every = config.get("log_interval", 20)

    token_stream = stream_arxiv(
        category=config["category"],
        max_results=config["max_papers"],
        delay_seconds=config.get("delay_seconds", 2),
        min_tokens=config.get("min_tokens", 12),
    )

    model.train()
    step = 0
    total_loss = 0.0
    start_time = time.time()
    recent_losses = []

    for x, y in make_packed_batches(token_stream, config["batch_size"], config["seq_len"]):
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        recent_losses.append(loss_val)
        step += 1

        if step % log_every == 0:
            avg50 = sum(recent_losses[-50:]) / len(recent_losses[-50:])
            elapsed = time.time() - start_time
            eta = (max_steps - step) / (step / elapsed) / 60
            print(f"  Step {step:5d}/{max_steps} | Loss: {loss_val:.4f} | "
                  f"Avg50: {avg50:.4f} | LR: {lr:.2e} | ETA: {eta:.0f}min")

        if step % health_every == 0:
            p_eos, top_p, top_id = check_generation_health(model, DEVICE, vocab_size)
            status = "GOOD" if p_eos < 0.1 else ("OK" if p_eos < 0.5 else "WARN")
            print(f"  [HEALTH] P(EOS)={p_eos:.4f} | Top={top_p:.4f}(id={top_id}) | {status}")

        if save_every > 0 and step % save_every == 0:
            path = checkpoints_dir / f"{prefix}_step{step}.pt"
            torch.save({"model": model.state_dict(), "step": step,
                        "config": config, "loss": loss_val}, path)
            print(f"  [SAVE] {path.name}")

        if step >= max_steps:
            break

    elapsed = time.time() - start_time
    avg_loss = total_loss / step if step > 0 else 0

    final_path = checkpoints_dir / f"{prefix}_final.pt"
    torch.save({"model": model.state_dict(), "step": step,
                "config": config, "final_loss": avg_loss}, final_path)

    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"  Steps: {step}, Avg loss: {avg_loss:.4f}")
    print(f"  Time: {elapsed/3600:.1f}h ({step/elapsed:.1f} steps/sec)")
    print(f"  Checkpoint: {final_path}")

    p_eos, top_p, _ = check_generation_health(model, DEVICE, vocab_size)
    print(f"\n  [FINAL HEALTH] P(EOS)={p_eos:.4f}")
    if p_eos < 0.1:
        print("  *** EXCELLENT — No EOS collapse! ***")
    elif p_eos < 0.3:
        print("  *** GOOD — EOS is not dominant ***")
    else:
        print("  *** WARNING — EOS still elevated ***")

    summary = {**config, "device": DEVICE, "total_params": total_params,
               "total_steps": step, "final_avg_loss": avg_loss,
               "wall_time_hours": elapsed/3600, "p_eos_final": p_eos}
    (checkpoints_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved.")


# ============================================================================
# PRESETS
# ============================================================================

PRESETS = {
    "prototype": {
        "max_papers": 500, "batch_size": 4, "seq_len": 256,
        "d_model": 384, "n_layers": 6, "n_heads": 6,
        "lr": 3e-4, "max_steps": 100, "warmup_steps": 10,
        "save_every_steps": 50, "health_check_every": 25,
        "category": "physics", "checkpoint_prefix": "v3_proto",
        "tokenizer_model": str(TOKENIZER_MODEL),
    },
    "long50k": {
        "max_papers": 50000, "batch_size": 4, "seq_len": 256,
        "d_model": 384, "n_layers": 6, "n_heads": 6,
        "lr": 3e-4, "max_steps": 15000, "warmup_steps": 750,
        "save_every_steps": 1000, "health_check_every": 500,
        "log_interval": 20, "category": "physics",
        "dropout": 0.0, "weight_decay": 0.1,
        "delay_seconds": 2, "min_tokens": 12,
        "checkpoint_prefix": "v3_long50k",
        "tokenizer_model": str(TOKENIZER_MODEL),
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLM v3 Training")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="prototype")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-prefix", type=str, default=None)
    args = parser.parse_args()

    config = PRESETS[args.preset].copy()
    if args.max_papers: config["max_papers"] = args.max_papers
    if args.max_steps: config["max_steps"] = args.max_steps
    if args.lr: config["lr"] = args.lr
    if args.checkpoint_prefix: config["checkpoint_prefix"] = args.checkpoint_prefix

    train(config)
