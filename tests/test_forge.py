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


# ===========================================================================
# BACKBONE COVERAGE: gradcheck harness, attention internals, data pipeline
# ---------------------------------------------------------------------------
# The tests below exercise the frozen backbone that LoRA/DoRA sit on top of.
# They are behavior-focused: the gradient-check idiom (upcast to float64, set
# adapter B to a small nonzero value, compare analytic vs central-difference)
# is the same one the existing adapter tests use. A handful PIN current
# behavior where the production code has a genuine wart -- each is flagged with
# a "PIN:" comment and reported, per the no-fix-production-source constraint.
# ===========================================================================


# ---------------------------------------------------------------------------
# 1-2. gradcheck.py self-test: the checker itself must be trustworthy
# ---------------------------------------------------------------------------

def test_numerical_grad_matches_known_analytic():
    # f(x) = sum(x**2)  =>  df/dx = 2x, exactly. If the central-difference
    # helper can't reproduce a gradient we can compute by hand, every other
    # gradient check in this suite is meaningless.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3)).astype(np.float64)
    def f(xx): return float(np.sum(xx ** 2))
    num = numerical_grad(f, x.copy())
    assert rel_error(num, 2.0 * x) < 1e-6


def test_rel_error_bounds_and_zero_safety():
    # Identical arrays => ~0; exact opposites => ~1; all-zero inputs must NOT
    # divide by zero (the 1e-12 denom floor guards it).
    rng = np.random.default_rng(1)
    v = rng.standard_normal((5, 5)).astype(np.float64)
    assert rel_error(v, v) == 0.0
    assert abs(rel_error(v, -v) - 1.0) < 1e-12
    z = np.zeros((3, 3))
    err = rel_error(z, z)
    assert np.isfinite(err) and err == 0.0


def test_run_all_checks_passes():
    # The bundled gradient check over all five primitives (Matmul, Softmax,
    # LayerNorm, GELU, CrossEntropy) must return True.
    from forge.backbone.gradcheck import run_all_checks
    assert run_all_checks(seed=0) is True


# ---------------------------------------------------------------------------
# 3. Base (frozen) SelfAttentionHead backward -- the plain head, no adapters
# ---------------------------------------------------------------------------

def test_base_head_backward_gradcheck():
    rng = np.random.default_rng(0)
    C, hd = 8, 8
    head = SelfAttentionHead(C, hd, rng)
    for a in ("Wq", "Wk", "Wv"):
        setattr(head, a, getattr(head, a).astype(np.float64))
    x = rng.standard_normal((2, 5, C)).astype(np.float64)
    w = rng.standard_normal((2, 5, hd))
    def fx(xx): return float(np.sum(head.forward(xx) * w))
    head.forward(x); dx = head.backward(w)
    assert rel_error(dx, numerical_grad(fx, x.copy())) < 1e-4
    for a in ("Wq", "Wk", "Wv"):
        P0 = getattr(head, a).copy()
        def fP(PP, a=a):
            setattr(head, a, PP); return float(np.sum(head.forward(x) * w))
        head.forward(x); head.backward(w)
        assert rel_error(getattr(head, "d" + a), numerical_grad(fP, P0.copy())) < 1e-4
        setattr(head, a, P0)


# ---------------------------------------------------------------------------
# 4-5. Causal masking and attention-weight shape guarantees
# ---------------------------------------------------------------------------

def test_causal_mask_is_lower_triangular():
    from forge.backbone.attention import causal_mask
    T = 6
    m = causal_mask(T)
    assert m.shape == (T, T) and m.dtype == bool
    # row 0 may attend only to itself; the last row may attend to all T.
    assert m[0].sum() == 1
    assert m[T - 1].sum() == T
    # strictly lower-triangular-inclusive: no True above the diagonal.
    assert not m[np.triu_indices(T, k=1)].any()


def test_attention_weights_causal_and_row_normalized():
    rng = np.random.default_rng(2)
    C, hd, T = 8, 8, 5
    head = SelfAttentionHead(C, hd, rng)
    x = rng.standard_normal((2, T, C)).astype(np.float64)
    head.forward(x)
    att = head.att                                   # (B,T,T)
    # every query row is a probability distribution over keys
    assert np.allclose(att.sum(axis=-1), 1.0)
    # future positions get ~zero weight. The mask sentinel is -1e9 (not -inf),
    # so use atol rather than exact-zero.
    iu = np.triu_indices(T, k=1)
    assert np.max(np.abs(att[:, iu[0], iu[1]])) < 1e-6


# ---------------------------------------------------------------------------
# 6. KV-cache inference path matches the masked slow path
# ---------------------------------------------------------------------------

def test_kv_cache_matches_slow_path():
    rng = np.random.default_rng(0)
    C, hd, T = 8, 8, 4
    head = SelfAttentionHead(C, hd, rng)
    for a in ("Wq", "Wk", "Wv"):
        setattr(head, a, getattr(head, a).astype(np.float64))
    x = rng.standard_normal((1, T, C)).astype(np.float64)
    slow = head.forward(x)                            # full causal attention
    # feed the SAME prefix one token at a time through the cache
    head.reset_cache()
    last = None
    for pos in range(T):
        last = head.forward_cached(x[:, pos:pos + 1, :], pos)
    # the incremental final-token output must equal the slow path's last row
    assert np.allclose(last[:, 0, :], slow[:, -1, :], atol=1e-9)


def test_kv_cache_sliding_window_cap():
    rng = np.random.default_rng(0)
    C, hd = 8, 8
    head = SelfAttentionHead(C, hd, rng)
    x = rng.standard_normal((1, 1, C)).astype(np.float64)
    head.reset_cache()
    cap = 3
    for pos in range(10):
        head.forward_cached(x, pos, max_context=cap)
    # the cache is a sliding window: it never grows past max_context
    assert head._k_cache.shape[1] == cap
    assert head._v_cache.shape[1] == cap


def test_rope_at_clamps_past_table():
    from forge.backbone.rope import RoPE
    rng = np.random.default_rng(0)
    rope = RoPE(8, max_seq=4)                         # table covers positions 0..3
    head = SelfAttentionHead(8, 8, rng, rope=rope)
    x1 = rng.standard_normal((1, 1, 8)).astype(np.float64)
    past = head._rope_at(x1, 100)                     # way past the table
    clamped = head._rope_at(x1, 3)                    # last valid index
    assert np.all(np.isfinite(past))
    assert np.allclose(past, clamped)                 # clamped, not out-of-bounds


# ---------------------------------------------------------------------------
# 7. GroupedQueryAttention: repeat-then-sum-back KV grad + config guards
# ---------------------------------------------------------------------------

def test_gqa_gradcheck():
    from forge.backbone.attention import GroupedQueryAttention
    rng = np.random.default_rng(1)
    d_model, n_heads, n_kv = 8, 4, 2
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv, rng)
    for a in ("Wq", "Wk", "Wv", "Wo"):
        setattr(gqa, a, getattr(gqa, a).astype(np.float64))
    x = rng.standard_normal((2, 5, d_model)).astype(np.float64)
    w = rng.standard_normal((2, 5, d_model))
    def fx(xx): return float(np.sum(gqa.forward(xx) * w))
    gqa.forward(x); dx = gqa.backward(w)
    assert rel_error(dx, numerical_grad(fx, x.copy())) < 1e-4
    # Wk/Wv carry the repeat-then-sum-back KV gradient (each KV head is shared
    # by a group of query heads); Wq/Wo are ordinary. Check all four.
    for a in ("Wq", "Wk", "Wv", "Wo"):
        P0 = getattr(gqa, a).copy()
        def fP(PP, a=a):
            setattr(gqa, a, PP); return float(np.sum(gqa.forward(x) * w))
        gqa.forward(x); gqa.backward(w)
        assert rel_error(getattr(gqa, "d" + a), numerical_grad(fP, P0.copy())) < 1e-4
        setattr(gqa, a, P0)


def test_gqa_kv_cache_saving_ratio():
    from forge.backbone.attention import GroupedQueryAttention
    rng = np.random.default_rng(0)
    gqa = GroupedQueryAttention(8, 4, 2, rng)
    # 4 query heads sharing 2 KV heads => the KV-cache is 2x smaller
    assert gqa.kv_cache_saving() == 2.0
    mqa = GroupedQueryAttention(8, 4, 1, rng)         # multi-query extreme
    assert mqa.kv_cache_saving() == 4.0


def test_gqa_bad_config_raises():
    from forge.backbone.attention import GroupedQueryAttention
    rng = np.random.default_rng(0)
    # n_heads must be a multiple of n_kv_heads (4 % 3 != 0). d_model=8 is
    # divisible by n_heads=4, so the *n_kv_heads* assert is the one that fires.
    try:
        GroupedQueryAttention(8, 4, 3, rng)
        assert False, "expected AssertionError for n_heads % n_kv_heads != 0"
    except AssertionError as e:
        assert "multiple" in str(e)


# ---------------------------------------------------------------------------
# 8-9. CharDataset: batch window math and tokenization edge cases
# ---------------------------------------------------------------------------

def test_get_batch_shapes_and_shift():
    ds = CharDataset("the cat sat on the mat " * 20, block_size=8)
    rng = np.random.default_rng(0)
    batch = 5
    x, y = ds.get_batch(batch, rng)
    assert x.shape == (batch, ds.block_size)
    assert y.shape == (batch, ds.block_size)
    # target is the input shifted by one: y[b, j] == x[b, j+1] for all j.
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_get_batch_empty_range_raises():
    # A corpus of length exactly block_size+1 makes the sampling range empty
    # (n = len - block - 1 == 0). get_batch now raises a CLEAR ValueError naming
    # the block size, instead of letting rng.integers(0, 0) surface a raw
    # "high <= 0".
    ds = CharDataset("abcde", block_size=4)           # len 5 == block_size + 1
    assert len(ds.data) == ds.block_size + 1
    try:
        ds.get_batch(2, np.random.default_rng(0))
        assert False, "expected ValueError on empty sampling range"
    except ValueError as e:
        assert "corpus too short" in str(e) and "block_size=4" in str(e)


def test_chardataset_empty_text():
    ds = CharDataset("", block_size=4)
    assert ds.vocab_size == 0
    assert ds.stoi == {} and ds.itos == {}
    assert ds.data.shape == (0,)


def test_chardataset_encode_unseen_raises():
    ds = CharDataset("abc", block_size=2)
    try:
        ds.encode("z")                                # 'z' not in the vocab
        assert False, "expected KeyError for an unseen character"
    except KeyError:
        pass


def test_chardataset_decode_encode_roundtrip():
    ds = CharDataset("the quick brown fox", block_size=4)
    s = "brown fox"
    assert ds.decode(ds.encode(s)) == s


# ---------------------------------------------------------------------------
# 10-11. adapters.py save/load error paths and perplexity guards
# ---------------------------------------------------------------------------

def _adapted_base(seed=0, rank=2, method="lora"):
    ds = CharDataset("the cat sat on the mat " * 30, block_size=16)
    m = _tiny_base(ds.vocab_size, seed=seed)
    apply_lora(m, rank=rank, alpha=4.0, method=method)
    return ds, m


def test_load_adapters_missing_key_raises():
    from forge.adapters import load_adapters
    import tempfile, os
    _, m = _adapted_base()
    p = os.path.join(tempfile.mkdtemp(), "bad.npz")
    # a .npz that lacks every expected adapter key
    np.savez(p, __meta__=np.array([2, 2], dtype=np.int64), junk=np.zeros(3))
    try:
        load_adapters(m, p)
        assert False, "expected KeyError for a missing adapter key"
    except KeyError as e:
        assert "missing" in str(e)


def test_load_adapters_shape_mismatch_raises():
    from forge.adapters import save_adapters, load_adapters
    import tempfile, os
    ds = CharDataset("the cat sat on the mat " * 30, block_size=16)
    m2 = _tiny_base(ds.vocab_size); apply_lora(m2, rank=2, alpha=4.0)
    p = os.path.join(tempfile.mkdtemp(), "r2.npz")
    save_adapters(m2, p)
    # a model with a DIFFERENT rank has incompatible adapter shapes
    m4 = _tiny_base(ds.vocab_size); apply_lora(m4, rank=4, alpha=4.0)
    try:
        load_adapters(m4, p)
        assert False, "expected ValueError for a shape mismatch"
    except ValueError as e:
        assert "shape" in str(e)


def test_load_adapters_npz_suffix_append_branch():
    from forge.adapters import save_adapters, load_adapters
    import tempfile, os
    _, m = _adapted_base()
    rng = np.random.default_rng(0)
    for blk in m.blocks:
        for h in blk.attn.heads:
            h.Bq = rng.standard_normal(h.Bq.shape).astype(np.float32) * 0.1
    noext = os.path.join(tempfile.mkdtemp(), "ad")    # NO .npz suffix
    save_adapters(m, noext)                           # np.savez appends .npz
    assert os.path.exists(noext + ".npz")
    before = m.blocks[0].attn.heads[0].Bq.copy()
    for blk in m.blocks:
        for h in blk.attn.heads:
            h.Bq = np.zeros_like(h.Bq)
    # loading with the suffix-less path exercises the ".npz"-append branch
    loaded = load_adapters(m, noext)
    assert loaded > 0
    assert np.allclose(m.blocks[0].attn.heads[0].Bq, before)


def test_save_adapters_guard_when_no_adapters():
    from forge.adapters import save_adapters
    import tempfile, os
    ds = CharDataset("the cat sat " * 20, block_size=16)
    m = _tiny_base(ds.vocab_size)
    # an adapter-free model: empty the blocks so the adapter iterator yields
    # nothing and the explicit guard fires (rather than an AttributeError).
    m.blocks = []
    p = os.path.join(tempfile.mkdtemp(), "empty.npz")
    try:
        save_adapters(m, p)
        assert False, "expected ValueError when there are no adapters"
    except ValueError as e:
        assert "apply_lora" in str(e)


def test_save_adapters_unadapted_model_raises_valueerror():
    # On a NORMAL, un-adapted GPT, save_adapters must hit its friendly
    # "call apply_lora first" guard -- NOT an opaque AttributeError from reading
    # Aq/Bq on a plain SelfAttentionHead. The guard checks the `_lora` flag that
    # apply_lora sets, so it fires before any adapter attribute is touched.
    from forge.adapters import save_adapters
    import tempfile, os
    ds = CharDataset("the cat sat " * 20, block_size=16)
    m = _tiny_base(ds.vocab_size)                     # never adapted
    p = os.path.join(tempfile.mkdtemp(), "x.npz")
    try:
        save_adapters(m, p)
        assert False, "expected an error saving an un-adapted model"
    except ValueError as e:
        assert "apply_lora" in str(e)                 # the clear guard, not AttributeError


def test_perplexity_short_text_guard():
    from forge.adapters import perplexity
    _, m = _adapted_base()
    ds = CharDataset("the cat sat on the mat " * 30, block_size=16)
    # text shorter than block_size + 1 tokens must raise, not silently divide
    try:
        perplexity(m, ds, "the")
        assert False, "expected ValueError for too-short text"
    except ValueError as e:
        assert "short" in str(e)


def test_perplexity_stride_not_block_branch():
    from forge.adapters import perplexity
    _, m = _adapted_base()
    ds = CharDataset("the cat sat on the mat " * 30, block_size=16)
    text = "the cat sat on the mat " * 6
    # stride < block => overlapping windows; each window is weighted by `block`
    ppl = perplexity(m, ds, text, block_size=8, stride=4)
    assert np.isfinite(ppl) and ppl > 0.0


# ---------------------------------------------------------------------------
# 12. apply_lora dispatch: bad method + the MHA-only limitation
# ---------------------------------------------------------------------------

def test_apply_lora_bogus_method_raises():
    ds = CharDataset("the cat sat " * 20, block_size=16)
    m = _tiny_base(ds.vocab_size)
    try:
        apply_lora(m, rank=2, alpha=4.0, method="bogus")
        assert False, "expected ValueError for an unknown adapter method"
    except ValueError as e:
        assert "lora" in str(e) and "dora" in str(e)


def _gqa_base(vocab, seed=0, n_heads=4, n_kv_heads=2):
    # a GQA backbone: llama arch (RoPE/RMSNorm/SwiGLU) with fewer KV heads than
    # query heads, so blk.attn is a GroupedQueryAttention rather than MHA.
    return GPT(vocab_size=vocab, d_model=32, n_heads=n_heads, n_layers=2,
               block_size=16, seed=seed, arch="llama", n_kv_heads=n_kv_heads)


def test_apply_lora_gqa_swaps_attention():
    # apply_lora now SUPPORTS a GQA backbone: it replaces each
    # GroupedQueryAttention layer with a LoRAGroupedQueryAttention that shares
    # the frozen Q/K/V/O and adds Q,V adapters.
    from forge.backbone.attention import GroupedQueryAttention
    from forge.adapt import LoRAGroupedQueryAttention
    ds = CharDataset("the cat sat " * 20, block_size=16)
    m = _gqa_base(ds.vocab_size)
    assert isinstance(m.blocks[0].attn, GroupedQueryAttention)
    apply_lora(m, rank=2, alpha=4.0)
    assert all(isinstance(blk.attn, LoRAGroupedQueryAttention) for blk in m.blocks)
    # the LoRA layer shares the base's frozen weights and zero-inits B (no-op).
    for blk in m.blocks:
        assert np.allclose(blk.attn.Bq, 0) and np.allclose(blk.attn.Bv, 0)


def test_gqa_lora_head_gradients():
    # GRADIENT-CHECK the GQA LoRA adapter grads in float64 with nonzero B, the
    # same idiom as test_lora_head_gradients. No RoPE here (matches the base
    # test_gqa_gradcheck) -- the adapter math is independent of RoPE, whose own
    # backward is already gradient-checked.
    from forge.backbone.attention import GroupedQueryAttention
    from forge.adapt import LoRAGroupedQueryAttention
    rng = np.random.default_rng(1)
    d_model, n_heads, n_kv = 8, 4, 2
    base = GroupedQueryAttention(d_model, n_heads, n_kv, rng)
    for a in ("Wq", "Wk", "Wv", "Wo"):
        setattr(base, a, getattr(base, a).astype(np.float64))
    head = LoRAGroupedQueryAttention(base, rank=2, alpha=4.0, seed=1)
    for a in ("Aq", "Bq", "Av", "Bv"):
        setattr(head, a, getattr(head, a).astype(np.float64))
    head.Bq = rng.standard_normal(head.Bq.shape) * 0.1        # nonzero B
    head.Bv = rng.standard_normal(head.Bv.shape) * 0.1
    x = rng.standard_normal((2, 5, d_model)).astype(np.float64)
    w = rng.standard_normal((2, 5, d_model))                  # output is (B,T,d_model)
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


def test_gqa_lora_finetune_leaves_base_frozen():
    # The frozen/trainable split must hold on a GQA backbone too: after a real
    # fine-tune, the base Q/K/V/O and embeddings are byte-identical while the
    # adapters have moved.
    #
    # np.errstate: numpy 2.0.x on darwin emits SPURIOUS "divide by zero /
    # overflow / invalid encountered in matmul" RuntimeWarnings from the SIMD
    # float32 matmul kernel in the base GroupedQueryAttention path (a matmul has
    # no division, so "divide by zero" is provably not a real event -- it is a
    # leaked FPE status flag). The base GQA model reproduces it with NO LoRA
    # involved. We suppress that platform noise here, but the explicit isfinite
    # assertions below still turn any GENUINE divergence into a hard failure, so
    # nothing real is hidden.
    ds = CharDataset("the cat sat on the mat and the dog ran " * 20, block_size=16)
    m = _gqa_base(ds.vocab_size)
    losses = []
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # pretrain the FULL model briefly so the base demonstrably CAN move...
        rng = np.random.default_rng(0); opt = Adam(lr=3e-3)
        for _ in range(20):
            x, y = ds.get_batch(16, rng); _, l = m.forward(x, y); m.backward()
            opt.step(m.params_and_grads()); losses.append(float(l))
        Wq0 = m.blocks[0].attn.Wq.copy()
        Wk0 = m.blocks[0].attn.Wk.copy()
        Wo0 = m.blocks[0].attn.Wo.copy()
        tok0 = m.tok_emb.copy()
        # ...then LoRA-adapt and train ONLY the adapters.
        apply_lora(m, rank=2, alpha=4.0)
        opt2 = Adam(lr=5e-3)
        for _ in range(30):
            x, y = ds.get_batch(16, rng); _, l = m.forward(x, y); m.backward()
            opt2.step(lora_params_and_grads(m)); losses.append(float(l))   # only adapters
    # training stayed numerically healthy (guards against the suppression above
    # ever masking a real blow-up)
    assert np.isfinite(losses[-1])
    assert np.all(np.isfinite(m.blocks[0].attn.Bq))
    assert np.all(np.isfinite(m.blocks[0].attn.Av))
    # base is byte-identical; only the adapters moved
    assert np.array_equal(Wq0, m.blocks[0].attn.Wq)
    assert np.array_equal(Wk0, m.blocks[0].attn.Wk)
    assert np.array_equal(Wo0, m.blocks[0].attn.Wo)
    assert np.array_equal(tok0, m.tok_emb)
    assert not np.allclose(m.blocks[0].attn.Bq, 0)   # adapters moved off zero


def test_gqa_lora_param_count_is_small_fraction():
    # count_params must report the adapters as a small fraction of a GQA base.
    ds = CharDataset("the cat sat " * 20, block_size=16)
    m = _gqa_base(ds.vocab_size)
    apply_lora(m, rank=2, alpha=4.0)
    trainable, total = count_params(m)
    assert 0 < trainable < total
    assert trainable == sum(blk.attn.n_trainable() for blk in m.blocks)
    assert trainable / total < 0.1                   # a genuinely small adapter


def test_apply_lora_dora_on_gqa_is_not_implemented():
    # LoRA is implemented for GQA; DoRA is not. apply_lora must refuse the DoRA
    # method on a GQA backbone with a clear NotImplementedError rather than
    # silently installing LoRA (which would violate the requested method) or
    # crashing opaquely.
    ds = CharDataset("the cat sat " * 20, block_size=16)
    m = _gqa_base(ds.vocab_size)
    try:
        apply_lora(m, rank=2, alpha=4.0, method="dora")
        assert False, "expected NotImplementedError for DoRA on a GQA backbone"
    except NotImplementedError as e:
        assert "DoRA" in str(e) and "GroupedQueryAttention" in str(e)


# ---------------------------------------------------------------------------
# 13. LoRALinear scaling / rank edge cases + 2-D input gradient check
# ---------------------------------------------------------------------------

def test_lora_rank1_nonunit_scaling_merged_matches_forward():
    # rank=1 with alpha != rank => scaling != 1. The merged weight must still
    # reproduce forward exactly, proving the alpha/r factor lands in both paths.
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    lora = LoRALinear(W, rank=1, alpha=3.0, seed=2)   # scaling = 3.0
    assert lora.scaling == 3.0
    lora.A = lora.A.astype(np.float64)
    lora.B = rng.standard_normal((1, 5)) * 0.1
    x = rng.standard_normal((3, 6)).astype(np.float64)
    fwd = lora.forward(x)
    assert np.allclose(fwd, x @ lora.merged_weight())
    # and the update really is scaled by alpha/r (not absorbed elsewhere)
    manual = x @ W + (x @ lora.A @ lora.B) * lora.scaling
    assert np.allclose(fwd, manual)


def test_lora_2d_input_gradcheck():
    # the existing LoRA grad tests use a 3-D (B,T,in) input; this pins the
    # leading-dim flattening for a plain 2-D (N,in) matrix.
    rng = np.random.default_rng(2)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    lora = LoRALinear(W, rank=1, alpha=3.0, seed=2)
    lora.A = lora.A.astype(np.float64)
    lora.B = rng.standard_normal((1, 5)) * 0.1
    x = rng.standard_normal((4, 6)).astype(np.float64)     # (N, in)
    w = rng.standard_normal((4, 5))
    def fx(xx): return float(np.sum(lora.forward(xx) * w))
    lora.forward(x); dx = lora.backward(w)
    assert rel_error(dx, numerical_grad(fx, x.copy())) < 1e-4
    A0 = lora.A.copy(); B0 = lora.B.copy()
    def fA(AA): lora.A = AA; return float(np.sum(lora.forward(x) * w))
    lora.forward(x); lora.backward(w)
    assert rel_error(lora.dA, numerical_grad(fA, A0.copy())) < 1e-4
    lora.A = A0
    def fB(BB): lora.B = BB; return float(np.sum(lora.forward(x) * w))
    lora.forward(x); lora.backward(w)
    assert rel_error(lora.dB, numerical_grad(fB, B0.copy())) < 1e-4


# ---------------------------------------------------------------------------
# 14. DoRA numerical stability and merge equivalence
# ---------------------------------------------------------------------------

def test_dora_tiny_column_norm_stays_finite():
    # A near-zero weight column drives ||V||_col toward the 1e-8 eps floor.
    # dora_compose/dora_grads must stay finite (no 0/0), which is exactly what
    # the eps inside _col_norm buys us.
    from forge.dora import dora_compose, dora_grads
    rng = np.random.default_rng(3)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    W[:, 0] = 1e-13                                    # essentially a zero column
    A = rng.standard_normal((6, 2)) * 0.01
    B = np.zeros((2, 5))
    m = np.sqrt(np.sum(W * W, axis=0) + 1e-8)          # init magnitude = col norm
    Wp, V, n = dora_compose(W, A, B, m, scaling=2.0)
    assert np.all(np.isfinite(Wp)) and np.all(n > 0)
    dWp = rng.standard_normal((6, 5))
    dA, dB, dm = dora_grads(dWp, V, n, m, A, B, scaling=2.0)
    assert np.all(np.isfinite(dA))
    assert np.all(np.isfinite(dB))
    assert np.all(np.isfinite(dm))


def test_dora_merged_matches_forward_nondefault():
    # non-default rank AND alpha (=> non-default scaling): the folded weight
    # must still equal the forward pass.
    from forge.dora import DoRALinear
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 5)).astype(np.float64)
    d = DoRALinear(W, rank=3, alpha=5.0, seed=1)       # scaling = 5/3
    assert abs(d.scaling - 5.0 / 3.0) < 1e-12
    d.B = rng.standard_normal((3, 5)) * 0.1
    d.m = d.m.astype(np.float64) + 0.03                # move magnitude off init
    x = rng.standard_normal((4, 6)).astype(np.float64)
    assert np.allclose(d.forward(x), x @ d.merged_weight())


# ---------------------------------------------------------------------------
# 15. SMOKE: entrypoint glue imports and cheapest pure helpers run
# ---------------------------------------------------------------------------

def test_smoke_entrypoints_import_and_run():
    # Import the three CLI/web entrypoints (glue only -- never call serve() or a
    # full training main()) and run their cheapest pure helper for one step.
    import forge.demo as demo
    import forge.web as web            # noqa: F401  (import-only: don't serve)
    import forge.compare_dora as compare_dora

    # compare_dora._fit: one Adam step on a tiny LoRA adapter fitting x->target
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((6, 4)) / np.sqrt(6)).astype(np.float32)
    x = rng.standard_normal((8, 6)).astype(np.float32)
    target = rng.standard_normal((8, 4)).astype(np.float32)
    lora = LoRALinear(W, rank=2, alpha=4.0, seed=1)
    hist = compare_dora._fit(lora, x, target, steps=1, lr=1e-2)
    assert len(hist) == 1 and np.isfinite(hist[0])

    # demo._train_on: one training step on a tiny GPT (glue path, not main())
    ds = CharDataset("the cat sat on the mat " * 5, block_size=8)
    m = GPT(vocab_size=ds.vocab_size, d_model=16, n_heads=2, n_layers=1,
            block_size=8, seed=0)
    hist2 = demo._train_on(m, ds, "the cat sat on the mat " * 5, steps=1, lr=1e-3)
    assert len(hist2) == 1 and np.isfinite(hist2[0])


if __name__ == "__main__":


    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")
