# SLM Physics Research Assistant

[![arXiv](https://img.shields.io/badge/arXiv-2408.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2408.XXXXX)
[![GitHub](https://img.shields.io/badge/GitHub-JaydipSingh%2Fslm--physics--assistant-blue)](https://github.com/JaydipSingh/slm-physics-assistant)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, reproducible systems stack for building domain-specialized **Small Language Models (SLMs)** on resource-constrained hardware. This repository accompanies the paper:

> **A Reproducible Systems Stack for Domain-Specialized Small Language Models: Tokenizer, Training, and Evaluation on Constrained Hardware**
>
> Jaydip Singh (2026)

## Key Results

| Model | Params | Physics QA Score | vs Random |
|-------|--------|-----------------|-----------|
| Random baseline | — | 0.250 | — |
| GPT-2 Small (general) | 124M | 0.325 | +30% |
| GPT-2 Medium (general) | 355M | 0.350 | +40% |
| GPT-2 Large (general) | 774M | 0.325 | +30% |
| **TinyLM v3 Base (ours)** | **23M** | **0.525** | **+110%** |
| **TinyLM v3 + LoRA (ours)** | **23M** | **0.550** | **+120%** |

Our 23M domain-specialized model **outperforms GPT-2 Large (774M) by 69%** on physics QA despite being **33× smaller**. Domain specialization decisively beats scale.

## Architecture (TinyLM v3)

Custom decoder-only transformer trained on 81K physics arXiv papers:
- **RoPE** positional encoding (rotary)
- **RMSNorm** (pre-norm, more stable)
- **SwiGLU** FFN (higher capacity per parameter)
- **Causal self-attention** with explicit Q/K/V/O projections
- **Document packing** (EOS-free training — prevents EOS collapse)
- **LoRA** fine-tuning on all attention + FFN projections

Pipeline:
1. **Rust BPE Tokenizer** — 32K vocabulary optimized for scientific text
2. **Streaming Training** — Live arXiv ingestion with 100-retry exponential backoff
3. **LoRA Fine-Tuning** — Full attention + FFN adaptation (42 layers)
4. **Multi-Metric Evaluation** — QA ranking, generation health monitoring

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

### Train Model (full, ~35 hours on M3 Pro)
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

### Evaluate with LoRA
```bash
python scripts/run_v3_eval.py --checkpoint checkpoints/v3_long50k_final.pt \
    --lora checkpoints/phase2_lora_v3/best_lora_adapter.pt
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
│   ├── phase2_lora_v3.py   # LoRA fine-tuning (full attention + FFN)
│   ├── run_v3_eval.py      # Physics QA evaluation
│   └── gpt2_baseline.py    # GPT-2 baseline comparison
├── config/
│   └── phase2_lora_config.yaml
├── data/
│   └── physics_qa_dataset.json  # 20-question benchmark (5 physics domains)
├── doc/
│   ├── project_reference_arxiv.tex   # Paper LaTeX source
│   ├── project_reference_arxiv.pdf   # Compiled paper
│   └── figures/                      # Paper figures
└── results/                # Evaluation results (JSON)
    ├── v3_qa_base.json     # TinyLM v3 base: 0.525
    ├── v3_qa_lora.json     # TinyLM v3 + LoRA: 0.550
    └── gpt2_all_sizes_results.json  # GPT-2 Small/Medium/Large
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

## Acknowledgements

Thanks to Linkan Kumbhar (Department of Physics, Rajendra University, Balangir, Odisha, India) for contributions to the initial Rust BPE tokenizer implementation.

## Citation

```bibtex
@article{singh2026slmstack,
  title={A Reproducible Systems Stack for Domain-Specialized Small Language Models:
         Tokenizer, Training, and Evaluation on Constrained Hardware},
  author={Singh, Jaydip},
  journal={arXiv preprint arXiv:2408.XXXXX},
  url={https://github.com/JaydipSingh/slm-physics-assistant},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
