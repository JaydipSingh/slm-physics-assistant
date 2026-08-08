# SLM Physics Research Assistant

[![arXiv](https://img.shields.io/badge/arXiv-2408.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2408.XXXXX)
[![GitHub](https://img.shields.io/badge/GitHub-JaydipSingh%2Fslm--physics--assistant-blue)](https://github.com/JaydipSingh/slm-physics-assistant)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, reproducible systems stack for building domain-specialized **Small Language Models (SLMs)** on resource-constrained hardware. This repository accompanies our paper:

> **A Reproducible Systems Stack for Domain-Specialized Small Language Models: Tokenizer, Training, and Evaluation on Constrained Hardware**
>
> Jaydip Singh, Linkan Kumbhar (2026)

## Key Results

| Model | Params | Physics QA Score | vs Random |
|-------|--------|-----------------|-----------|
| Random baseline | — | 0.250 | — |
| GPT-2 Small (general) | 124M | 0.325 | +30% |
| **TinyLM Phase 1 (ours)** | 35.2M | 0.325 | +30% |
| **TinyLM Phase 2 + LoRA (ours)** | 35.5M | **0.450** | **+80%** |

Our 35.5M domain-specialized model **outperforms GPT-2 Small (124M) by 38%** on physics QA despite being 3.5× smaller.

## Architecture

The system combines:
1. **Rust BPE Tokenizer** — 32K vocabulary optimized for scientific text
2. **Streaming Training Pipeline** — Live arXiv ingestion with retry/backoff
3. **LoRA Fine-Tuning** — 45% loss improvement with 0.77% additional parameters
4. **Multi-Metric Evaluation** — Loss, QA ranking, latency, retrieval (BM25/dense/hybrid)
5. **RAG Retrieval** — 0.713 MRR with reciprocal rank fusion

## Quick Start

### Prerequisites
- Rust (for tokenizer): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Python 3.10+ with PyTorch: `pip install -r requirements.txt`
- Hardware: Apple M1/M2/M3 (16GB+) or any CUDA GPU

### Build Tokenizer
```bash
cargo build --release
```

### Train Model (prototype, ~5 min)
```bash
python stream_train_v3.py --preset prototype
```

### Train Model (full, ~6-8 hours on M3 Pro)
```bash
python stream_train_v3.py --preset long50k
```

### Evaluate
```bash
python scripts/run_v3_eval.py --checkpoint checkpoints/v3_long50k_final.pt
```

### LoRA Fine-Tuning
```bash
python scripts/phase2_lora_v3.py --base-checkpoint checkpoints/v3_long50k_final.pt
```

## Repository Structure

```
├── Cargo.toml              # Rust project config
├── src/                    # Rust BPE tokenizer source
│   ├── lib.rs              # Core BPE algorithm
│   ├── main.rs             # CLI interface
│   ├── data.rs             # Dataset preparation
│   ├── tests.rs            # Unit tests
│   └── bin/prune.rs        # Vocabulary pruning tool
├── stream_train_v3.py      # Training script (custom transformer + doc packing)
├── scripts/
│   ├── phase2_lora_v3.py   # LoRA fine-tuning
│   ├── run_v3_eval.py      # Physics QA evaluation
│   └── gpt2_baseline.py    # GPT-2 baseline comparison
├── config/
│   └── phase2_lora_config.yaml
├── data/
│   └── physics_qa_dataset.json  # 20-question benchmark
├── doc/
│   ├── project_reference_arxiv.tex   # Paper LaTeX source
│   ├── project_reference_arxiv.pdf   # Compiled paper
│   └── figures/                      # Paper figures
└── results/                # Evaluation results (JSON)
```

## Tokenizer CLI

```bash
# Train tokenizer on corpus
cargo run --release -- train corpus.txt 32000

# Tokenize text
cargo run --release -- tokenize "quantum mechanics"

# Decode token IDs
cargo run --release -- decode "1,42,103,2"

# Prepare train/val/test splits
cargo run --release -- prepare input.txt --train train.bin --val val.bin --test test.bin
```

## Citation

```bibtex
@article{singh2026slmstack,
  title={A Reproducible Systems Stack for Domain-Specialized Small Language Models:
         Tokenizer, Training, and Evaluation on Constrained Hardware},
  author={Singh, Jaydip and Kumbhar, Linkan},
  journal={arXiv preprint arXiv:2408.XXXXX},
  url={https://github.com/JaydipSingh/slm-physics-assistant},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
