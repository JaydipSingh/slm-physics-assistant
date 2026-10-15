---
title: 'TinyLM-Physics: A Reproducible Stack for Training Domain-Specialized Small Language Models on a Laptop'
tags:
  - Python
  - Rust
  - natural language processing
  - small language models
  - transformers
  - reproducible research
authors:
  - name: Jaydip Singh
    orcid: 0000-0003-3413-9548
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 1 October 2026
bibliography: paper.bib
---

# Summary

Large language models deliver strong performance but require data-center-scale
hardware to train, placing them out of reach for students, independent
researchers, and small labs. `TinyLM-Physics` is an open-source software stack
that makes the full lifecycle of a *small* language model (SLM)—tokenizer
training, model pre-training, parameter-efficient fine-tuning, and
evaluation—reproducible end-to-end on a single consumer laptop (an Apple M3 Pro
with 18 GB of unified memory). The stack combines a performance-oriented Byte
Pair Encoding (BPE) tokenizer written in Rust with a compact decoder-only
transformer implemented in PyTorch that uses modern components (rotary position
embeddings, RMS normalization, and SwiGLU feed-forward layers). It streams
training text directly from the arXiv API with automatic retry and
checkpointing, applies Low-Rank Adaptation (LoRA) [@hu2021lora] for cheap
domain fine-tuning, and ships a multi-metric evaluation harness. As a concrete
demonstration, a 23M-parameter model trained on physics papers scores 0.55 on a
multiple-choice physics question-answering benchmark, above several much larger
general-purpose GPT-2 baselines on the same protocol.

The software is intended both as a working research artifact and as a teaching
and prototyping tool: every stage runs without cluster access, the components
are small enough to read and modify, and all configuration, data, and results
needed to reproduce the accompanying study are included.

# Statement of need

There is a growing literature on small and efficient language models, but
published work usually reports *final model quality* while omitting a runnable,
end-to-end path from raw text to evaluated model—especially one that works
under the hardware constraints typical of individuals and classrooms. Newcomers
are often left to reassemble tokenization, streaming data ingestion, stable
training, fine-tuning, and evaluation from scattered fragments, and to
rediscover the practical failure modes that are rarely documented.

`TinyLM-Physics` addresses this gap by packaging the whole pipeline as a single,
transparent, laptop-scale reference implementation:

- **Rust BPE tokenizer** with a command-line and library interface, a 32K
  scientific-text vocabulary, and train/encode/decode/prepare operations, built
  on established subword tokenization methods [@sennrich2016bpe].
- **Streaming trainer** for a custom transformer [@vaswani2017attention] that
  ingests arXiv text live, with exponential-backoff retries and periodic
  checkpointing so that multi-hour runs survive transient network failures.
- **LoRA fine-tuning** [@hu2021lora] that adapts the base model to a domain by
  training well under 1% of the parameters.
- **Evaluation harness** covering question-answering ranking, inference
  latency, and—importantly—*generation-health* monitoring that detects a
  degenerate "early end-of-sequence collapse" failure mode we document and fix.

The target users are students and educators who need a readable, modifiable SLM
codebase; independent researchers who lack cluster access but want a credible
experimental baseline; and applied practitioners who need operational patterns
(robust data ingestion, checkpointing, reproducible evaluation) for building
domain-specific models. By making the entire process reproducible on commodity
hardware and documenting both the recipe and its pitfalls, the software lowers
the barrier to hands-on research with language models and provides a foundation
that extends to other scientific domains and to larger models on GPU hardware.

# Acknowledgements

We thank Linkan Kumbhar for contributions to the initial Rust BPE tokenizer
implementation.

# References
