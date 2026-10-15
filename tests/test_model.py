"""
Tests for the TinyLM v3 model architecture and training utilities.

These exercise the core research code in `stream_train_v3.py`:
  - the custom transformer (TinyLMv3) forward pass and parameter count
  - rotary position embedding (RoPE) shape behaviour
  - EOS-free document packing

The training script imports optional networking libraries (`arxiv`,
`requests`) at module load; if those are not installed the whole module is
skipped rather than failing, so the test suite stays lightweight.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the training module, skipping cleanly if optional deps are absent.
s3 = pytest.importorskip(
    "stream_train_v3",
    reason="stream_train_v3 requires optional deps (torch, arxiv, requests)",
)


def test_tinylmv3_forward_shapes():
    """A forward pass returns logits of shape (batch, seq, vocab)."""
    model = s3.TinyLMv3(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                        max_seq_len=32)
    model.eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(logits).all()


def test_tinylmv3_parameter_count_positive():
    """The model reports a sensible, positive parameter count."""
    model = s3.TinyLMv3(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                        max_seq_len=32)
    n = model.count_parameters()
    assert isinstance(n, int)
    assert n > 0


def test_weight_tying():
    """Embedding and output head share weights (weight tying)."""
    model = s3.TinyLMv3(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                        max_seq_len=32)
    assert model.head.weight is model.embed.weight


def test_rope_precompute_shapes():
    """RoPE cos/sin tables have shape (seq_len, head_dim/2)."""
    head_dim = 16
    seq = 32
    cos, sin = s3.precompute_rope(head_dim, seq)
    assert cos.shape == (seq, head_dim // 2)
    assert sin.shape == (seq, head_dim // 2)


def test_document_packing_is_eos_free():
    """
    Packed training batches must not contain the EOS token id in the targets.
    This guards the fix for the 'early-EOS collapse' failure mode: documents are
    concatenated without EOS separators.
    """
    eos = s3.EOS_TOKEN_ID
    # Two short 'documents' of plain content tokens (no special tokens).
    docs = [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19] for _ in range(20)]

    seq_len = 8
    batch_size = 2
    produced = 0
    for x, y in s3.make_packed_batches(iter(docs), batch_size=batch_size,
                                       seq_len=seq_len, shuffle_buffer=4):
        assert x.shape == (batch_size, seq_len)
        assert y.shape == (batch_size, seq_len)
        assert (y != eos).all(), "EOS token must never appear in packed targets"
        produced += 1
        if produced >= 2:
            break
    assert produced >= 1, "packing should yield at least one batch"
