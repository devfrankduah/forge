"""
Tests for Forge. The core guarantees:
  - the LoRA gradients are mathematically correct (gradient-checked),
  - fine-tuning changes ONLY the adapters (the base is provably frozen),
  - a LoRA fine-tune actually reduces loss on a new task,
  - the parameter accounting is honest.

Run:  python -m pytest tests/ -q     (or: python tests/test_forge.py)
Self-contained: no external dependency, just run it (NumPy only).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from forge.backbone.model import GPT
from forge.backbone.data import CharDataset
from forge.backbone.optimizer import Adam
from forge.backbone.gradcheck import numerical_grad, rel_error
from forge.backbone.attention import SelfAttentionHead

from forge.lora import LoRALinear
from forge.adapt import (LoRAHead, apply_lora, lora_params_and_grads, count_params)


# ---------------------------------------------------------------------------
# LoRALinear gradients (the core correctness guarantee)
# ---------------------------------------------------------------------------

def _mk_lora(seed=1):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    lora = LoRALinear(W, rank=2, alpha=4.0, seed=seed)
    lora.A = lora.A.astype(np.float64)
    lora.B = (rng.standard_normal((2, 5)) * 0.1)   # nonzero B for a real check
    return lora, W

def test_lora_input_gradient():
    lora, _ = _mk_lora()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 6)).astype(np.float64)
    w = rng.standard_normal((2, 4, 5))
    def f(xx): return float(np.sum(lora.forward(xx) * w))
    lora.forward(x); dx = lora.backward(w)
    assert rel_error(dx, numerical_grad(f, x.copy())) < 1e-4

def test_lora_adapter_gradients():
    lora, _ = _mk_lora()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4, 6)).astype(np.float64)
    w = rng.standard_normal((2, 4, 5))
    A0 = lora.A.copy(); B0 = lora.B.copy()
    def fA(AA):
        lora.A = AA; return float(np.sum(lora.forward(x) * w))
    lora.forward(x); lora.backward(w)
    assert rel_error(lora.dA, numerical_grad(fA, A0.copy())) < 1e-4
    lora.A = A0
    def fB(BB):
        lora.B = BB; return float(np.sum(lora.forward(x) * w))
    lora.forward(x); lora.backward(w)
    assert rel_error(lora.dB, numerical_grad(fB, B0.copy())) < 1e-4

def test_lora_zero_init_is_noop():
    # B initialized to zero => adapted layer equals the base layer at start
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    lora = LoRALinear(W, rank=2, alpha=4.0)
    x = rng.standard_normal((3, 6)).astype(np.float64)
    assert np.allclose(lora.forward(x), x @ W)

def test_lora_merged_weight_matches_forward():
    # merging the adapter into W should reproduce the adapter's forward exactly
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    lora = LoRALinear(W, rank=2, alpha=4.0, seed=2)
    lora.B = rng.standard_normal((2, 5)) * 0.1
    x = rng.standard_normal((3, 6)).astype(np.float64)
    merged = lora.merged_weight()
    assert np.allclose(lora.forward(x), x @ merged)


# ---------------------------------------------------------------------------
# LoRAHead (attention adapter) gradients
# ---------------------------------------------------------------------------

def test_lora_head_gradients():
    rng = np.random.default_rng(0)
    C, hd = 8, 8
    base = SelfAttentionHead(C, hd, rng)
    base.Wq = base.Wq.astype(np.float64); base.Wk = base.Wk.astype(np.float64); base.Wv = base.Wv.astype(np.float64)
    head = LoRAHead(base, rank=2, alpha=4.0, seed=1)
    for a in ("Aq", "Bq", "Av", "Bv"):
        setattr(head, a, getattr(head, a).astype(np.float64))
    head.Bq = rng.standard_normal(head.Bq.shape) * 0.1
    head.Bv = rng.standard_normal(head.Bv.shape) * 0.1
    x = rng.standard_normal((2, 5, C)).astype(np.float64)
    w = rng.standard_normal((2, 5, hd))
    def fx(xx): return float(np.sum(head.forward(xx) * w))
    head.forward(x); dx = head.backward(w)
    assert rel_error(dx, numerical_grad(fx, x.copy())) < 1e-4
    for a in ("Aq", "Bq", "Av", "Bv"):
        P0 = getattr(head, a).copy()
        def fP(PP, a=a):
            setattr(head, a, PP); return float(np.sum(head.forward(x) * w))
        head.forward(x); head.backward(w)
        assert rel_error(getattr(head, "d" + a), numerical_grad(fP, P0.copy())) < 1e-4
        setattr(head, a, P0)


# ---------------------------------------------------------------------------
# The fine-tuning workflow: frozen base, real adaptation, honest counts
# ---------------------------------------------------------------------------

def _tiny_base(vocab, seed=0, arch="gpt2"):
    return GPT(vocab_size=vocab, d_model=32, n_heads=2, n_layers=2,
               block_size=16, seed=seed, arch=arch)

def test_apply_lora_swaps_heads():
    ds = CharDataset("the cat sat on the mat " * 20, block_size=16)
    m = _tiny_base(ds.vocab_size)
    apply_lora(m, rank=2, alpha=4.0)
    from forge.adapt import LoRAHead as LH
    assert all(isinstance(h, LH) for blk in m.blocks for h in blk.attn.heads)

def test_finetune_leaves_base_frozen():
    ds = CharDataset("the cat sat on the mat and the dog ran " * 20, block_size=16)
    m = _tiny_base(ds.vocab_size)
    # pretrain briefly
    rng = np.random.default_rng(0); opt = Adam(lr=3e-3)
    for _ in range(20):
        x, y = ds.get_batch(16, rng); m.forward(x, y); m.backward()
        opt.step(m.params_and_grads())
    Wq0 = m.blocks[0].attn.heads[0].Wq.copy()
    tok0 = m.tok_emb.copy()
    apply_lora(m, rank=2, alpha=4.0)
    opt2 = Adam(lr=5e-3)
    for _ in range(30):
        x, y = ds.get_batch(16, rng); m.forward(x, y); m.backward()
        opt2.step(lora_params_and_grads(m))   # only adapters
    # base weights identical; adapters moved
    assert np.array_equal(Wq0, m.blocks[0].attn.heads[0].Wq)
    assert np.array_equal(tok0, m.tok_emb)
    assert not np.allclose(m.blocks[0].attn.heads[0].Bq, 0)

def test_lora_finetune_reduces_loss():
    ds = CharDataset("to be or not to be that is the question " * 30, block_size=16)
    m = _tiny_base(ds.vocab_size)
    apply_lora(m, rank=4, alpha=8.0)
    rng = np.random.default_rng(0); opt = Adam(lr=5e-3)
    losses = []
    for _ in range(120):
        x, y = ds.get_batch(16, rng); _, l = m.forward(x, y); m.backward()
        opt.step(lora_params_and_grads(m)); losses.append(float(l))
    assert losses[-1] < losses[0] * 0.7

def test_param_count_is_small_fraction():
    ds = CharDataset("the cat sat " * 20, block_size=16)
    m = _tiny_base(ds.vocab_size)
    apply_lora(m, rank=2, alpha=4.0)
    trainable, total = count_params(m)
    assert 0 < trainable < total          # adapters are a strict subset
    assert trainable == sum(h.n_trainable() for blk in m.blocks for h in blk.attn.heads)

def test_lora_works_on_llama_arch():
    # adapters must also work when the base uses RoPE/RMSNorm/SwiGLU
    ds = CharDataset("to be or not to be " * 30, block_size=16)
    m = _tiny_base(ds.vocab_size, arch="llama")
    apply_lora(m, rank=2, alpha=4.0)
    rng = np.random.default_rng(0); opt = Adam(lr=5e-3)
    losses = []
    for _ in range(80):
        x, y = ds.get_batch(16, rng); _, l = m.forward(x, y); m.backward()
        opt.step(lora_params_and_grads(m)); losses.append(float(l))
    assert np.isfinite(losses[-1]) and losses[-1] < losses[0]


def test_param_count_total_is_complete():
    # Regression: the total must include MLP + norm params, not just attention.
    # A wrong total silently misreports LoRA's efficiency.
    ds = CharDataset("the cat sat " * 20, block_size=16)
    for arch in ("gpt2", "llama"):
        m = _tiny_base(ds.vocab_size, arch=arch)
        apply_lora(m, rank=2, alpha=4.0)
        trainable, total = count_params(m)

        # independent full recount
        def nsz(n):
            return n.gamma.size + (n.beta.size if hasattr(n, "beta") else 0)
        true_total = m.tok_emb.size
        if m.pos_emb is not None:
            true_total += m.pos_emb.size
        if getattr(m, "Wout", None) is not None:
            true_total += m.Wout.size
        true_total += nsz(m.ln_f)
        for blk in m.blocks:
            for h in blk.attn.heads:
                true_total += h.Wq.size + h.Wk.size + h.Wv.size
            true_total += blk.attn.Wo.size + nsz(blk.ln1) + nsz(blk.ln2)
            true_total += sum(getattr(blk.ff, a).size for a in
                              ("W1", "b1", "W2", "b2", "Wg", "Wu", "Wd")
                              if hasattr(blk.ff, a))
        assert total == true_total, f"{arch}: total {total} != true {true_total}"



# ---------------------------------------------------------------------------
# DoRA (Weight-Decomposed LoRA, 2024)
# ---------------------------------------------------------------------------

def test_dora_gradients():
    from forge.dora import DoRALinear
    rng = np.random.default_rng(0)
    IN, OUT, R = 6, 5, 2
    W = rng.standard_normal((IN, OUT)).astype(np.float64)
    x = rng.standard_normal((2, 4, IN)).astype(np.float64)
    w = rng.standard_normal((2, 4, OUT))
    d = DoRALinear(W, rank=R, alpha=4.0, seed=1)
    d.A = d.A.astype(np.float64); d.B = rng.standard_normal((R, OUT)) * 0.1; d.m = d.m.astype(np.float64)
    def fx(xx): return float(np.sum(d.forward(xx) * w))
    d.forward(x); dx = d.backward(w)
    assert rel_error(dx, numerical_grad(fx, x.copy())) < 1e-4
    for name in ("A", "B", "m"):
        P0 = getattr(d, name).copy()
        def fP(PP, name=name):
            setattr(d, name, PP); return float(np.sum(d.forward(x) * w))
        d.forward(x); d.backward(w)
        assert rel_error(getattr(d, "d" + name), numerical_grad(fP, P0.copy())) < 1e-4
        setattr(d, name, P0)

def test_dora_zero_init_is_noop():
    from forge.dora import DoRALinear
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    d = DoRALinear(W, rank=2, alpha=4.0)
    x = rng.standard_normal((3, 6)).astype(np.float64)
    assert np.allclose(d.forward(x), x @ W)

def test_dora_merged_matches_forward():
    from forge.dora import DoRALinear
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    d = DoRALinear(W, rank=2, alpha=4.0, seed=2)
    d.B = rng.standard_normal((2, 5)) * 0.1
    x = rng.standard_normal((3, 6)).astype(np.float64)
    assert np.allclose(d.forward(x), x @ d.merged_weight())


# ---------------------------------------------------------------------------
# QLoRA-style int8 quantization of the frozen base
# ---------------------------------------------------------------------------

def test_int8_roundtrip_is_close():
    from forge.quant import quantize_int8, dequantize_int8, quantization_error
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 48)).astype(np.float32)
    assert quantization_error(W) < 0.02        # <2% reconstruction error
    q, s = quantize_int8(W)
    assert q.dtype == np.int8 and s.shape == (48,)

def test_quantized_lora_gradients_unaffected():
    # the adapters must train correctly even though the base is quantized
    from forge.quant import QuantizedLoRALinear
    rng = np.random.default_rng(0)
    IN, OUT, R = 6, 5, 2
    W = rng.standard_normal((IN, OUT)).astype(np.float64)
    ql = QuantizedLoRALinear(W, rank=R, alpha=4.0, seed=1)
    ql.A = ql.A.astype(np.float64); ql.B = rng.standard_normal((R, OUT)) * 0.1
    x = rng.standard_normal((2, 4, IN)).astype(np.float64)
    w = rng.standard_normal((2, 4, OUT))
    def fx(xx): return float(np.sum(ql.forward(xx) * w))
    ql.forward(x); dx = ql.backward(w)
    assert rel_error(dx, numerical_grad(fx, x.copy())) < 1e-4
    A0 = ql.A.copy()
    def fA(AA):
        ql.A = AA; return float(np.sum(ql.forward(x) * w))
    ql.forward(x); ql.backward(w)
    assert rel_error(ql.dA, numerical_grad(fA, A0.copy())) < 1e-4

def test_quantization_saves_memory():
    from forge.quant import QuantizedLoRALinear
    ql = QuantizedLoRALinear(np.zeros((256, 256), np.float32))
    qb, fb = ql.memory_report()
    assert fb / qb > 3.5     # ~4x smaller


# ---------------------------------------------------------------------------
# Adapter save / load / hot-swap and perplexity
# ---------------------------------------------------------------------------

def test_adapter_save_load_roundtrip(tmp_path=None):
    import tempfile, os
    ds = CharDataset("the cat sat on the mat " * 30, block_size=16)
    m = _tiny_base(ds.vocab_size)
    apply_lora(m, rank=2, alpha=4.0)
    # move adapters off zero
    for blk in m.blocks:
        for h in blk.attn.heads:
            h.Bq = np.random.default_rng(0).standard_normal(h.Bq.shape).astype(np.float32) * 0.1
    from forge.adapters import save_adapters, load_adapters
    d = tempfile.mkdtemp()
    p = os.path.join(d, "ad.npz")
    save_adapters(m, p)
    before = m.blocks[0].attn.heads[0].Bq.copy()
    # zero them, then load back
    for blk in m.blocks:
        for h in blk.attn.heads:
            h.Bq = np.zeros_like(h.Bq)
    load_adapters(m, p)
    assert np.allclose(m.blocks[0].attn.heads[0].Bq, before)

def test_perplexity_lower_after_adapting():
    from forge.adapters import perplexity
    ds = CharDataset("to be or not to be that is the question " * 40, block_size=16)
    m = _tiny_base(ds.vocab_size)
    apply_lora(m, rank=4, alpha=8.0)
    ppl_before = perplexity(m, ds, "to be or not to be that is the question " * 3)
    rng = np.random.default_rng(0); opt = Adam(lr=5e-3)
    for _ in range(120):
        x, y = ds.get_batch(16, rng); m.forward(x, y); m.backward()
        opt.step(lora_params_and_grads(m))
    ppl_after = perplexity(m, ds, "to be or not to be that is the question " * 3)
    assert ppl_after < ppl_before        # adaptation reduces perplexity



# ---------------------------------------------------------------------------
# DoRA integrated into the attention fine-tuning path (DoRAHead)
# ---------------------------------------------------------------------------

def test_dora_head_gradients():
    from forge.adapt import DoRAHead
    rng = np.random.default_rng(0)
    C, hd = 8, 8
    base = SelfAttentionHead(C, hd, rng)
    for a in ("Wq", "Wk", "Wv"):
        setattr(base, a, getattr(base, a).astype(np.float64))
    head = DoRAHead(base, rank=2, alpha=4.0, seed=1)
    for a in ("Aq", "Bq", "mq", "Av", "Bv", "mv"):
        setattr(head, a, getattr(head, a).astype(np.float64))
    head.Bq = rng.standard_normal(head.Bq.shape) * 0.1
    head.Bv = rng.standard_normal(head.Bv.shape) * 0.1
    x = rng.standard_normal((2, 5, C)).astype(np.float64)
    w = rng.standard_normal((2, 5, hd))
    def fx(xx): return float(np.sum(head.forward(xx) * w))
    head.forward(x); dx = head.backward(w)
    assert rel_error(dx, numerical_grad(fx, x.copy())) < 1e-4
    for a in ("Aq", "Bq", "mq", "Av", "Bv", "mv"):
        P0 = getattr(head, a).copy()
        def fP(PP, a=a):
            setattr(head, a, PP); return float(np.sum(head.forward(x) * w))
        head.forward(x); head.backward(w)
        assert rel_error(getattr(head, "d" + a), numerical_grad(fP, P0.copy())) < 1e-4
        setattr(head, a, P0)

def test_dora_finetune_leaves_base_frozen():
    ds = CharDataset("the cat sat on the mat and the dog ran " * 20, block_size=16)
    m = _tiny_base(ds.vocab_size)
    rng = np.random.default_rng(0); opt = Adam(lr=3e-3)
    for _ in range(20):
        x, y = ds.get_batch(16, rng); m.forward(x, y); m.backward()
        opt.step(m.params_and_grads())
    Wq0 = m.blocks[0].attn.heads[0].Wq.copy()
    tok0 = m.tok_emb.copy()
    apply_lora(m, rank=2, alpha=4.0, method="dora")
    opt2 = Adam(lr=5e-3)
    for _ in range(30):
        x, y = ds.get_batch(16, rng); m.forward(x, y); m.backward()
        opt2.step(lora_params_and_grads(m))
    assert np.array_equal(Wq0, m.blocks[0].attn.heads[0].Wq)
    assert np.array_equal(tok0, m.tok_emb)

def test_dora_finetune_reduces_loss():
    ds = CharDataset("to be or not to be that is the question " * 30, block_size=16)
    m = _tiny_base(ds.vocab_size)
    apply_lora(m, rank=4, alpha=8.0, method="dora")
    rng = np.random.default_rng(0); opt = Adam(lr=5e-3)
    losses = []
    for _ in range(120):
        x, y = ds.get_batch(16, rng); _, l = m.forward(x, y); m.backward()
        opt.step(lora_params_and_grads(m)); losses.append(float(l))
    assert losses[-1] < losses[0] * 0.85

def test_dora_works_on_llama_arch():
    ds = CharDataset("to be or not to be " * 30, block_size=16)
    m = _tiny_base(ds.vocab_size, arch="llama")
    apply_lora(m, rank=2, alpha=4.0, method="dora")
    rng = np.random.default_rng(0); opt = Adam(lr=5e-3)
    losses = []
    for _ in range(80):
        x, y = ds.get_batch(16, rng); _, l = m.forward(x, y); m.backward()
        opt.step(lora_params_and_grads(m)); losses.append(float(l))
    assert np.isfinite(losses[-1]) and losses[-1] < losses[0]

def test_dora_adapter_save_load_roundtrip():
    import tempfile, os
    ds = CharDataset("the cat sat on the mat " * 30, block_size=16)
    m = _tiny_base(ds.vocab_size)
    apply_lora(m, rank=2, alpha=4.0, method="dora")
    rng = np.random.default_rng(0)
    for blk in m.blocks:
        for h in blk.attn.heads:
            h.Bq = rng.standard_normal(h.Bq.shape).astype(np.float32) * 0.1
            h.mq = h.mq + 0.05
    from forge.adapters import save_adapters, load_adapters
    p = os.path.join(tempfile.mkdtemp(), "dora.npz")
    save_adapters(m, p)
    bq0 = m.blocks[0].attn.heads[0].Bq.copy()
    mq0 = m.blocks[0].attn.heads[0].mq.copy()
    for blk in m.blocks:
        for h in blk.attn.heads:
            h.Bq = np.zeros_like(h.Bq); h.mq = np.zeros_like(h.mq)
    load_adapters(m, p)
    assert np.allclose(m.blocks[0].attn.heads[0].Bq, bq0)
    assert np.allclose(m.blocks[0].attn.heads[0].mq, mq0)   # magnitude round-trips too


if __name__ == "__main__":


    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")
