# Forge

[![CI](https://github.com/devfrankduah/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/devfrankduah/forge/actions/workflows/ci.yml)

**LoRA fine-tuning and its modern successors, implemented from scratch — adapt a
frozen transformer by training a few percent of it, with the base provably
untouched (and optionally int8-quantized to cut its memory ~4x).**

No PyTorch, no PEFT library. LoRA, **DoRA** (Weight-Decomposed LoRA, 2024), and
**QLoRA-style quantization** are all hand-written — forward *and* backward — and
gradient-checked. The transformer being fine-tuned lives in `forge.backbone` —
the same from-scratch architecture as the companion Glassbox project, bundled
here so **Forge is completely self-contained**: clone it and demo it on its own,
nothing else to install. Adapter save/load/hot-swap and honest perplexity
evaluation are included.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-numpy%20only-brightgreen)
![Tests](https://img.shields.io/badge/tests-24%20passing-brightgreen)
![Gradients](https://img.shields.io/badge/gradients-checked%20to%201e--8-brightgreen)

---

## Why this exists

Full fine-tuning updates every weight — expensive in memory and compute. LoRA
(Hu et al. 2021) made adaptation cheap enough for one GPU and is now the default
way to specialize open models. Forge proves I understand it from the inside: I
derived the low-rank update and its gradients by hand, and I can show the base
model is genuinely frozen while a tiny adapter does the adapting.

It's the third of a set (each stands alone — no shared imports, so any one can
be demoed on its own):
- **Glassbox** — I understand what's *inside* a model (built one from scratch).
- **Forge** — I can *adapt* one efficiently (this project; it bundles its own
  copy of that transformer as `forge.backbone`).
- **Crucible** — I can *test* what I build (honest eval infrastructure).

## What it shows (live, in the sandbox)

Pretrain a base model on one style, freeze it, attach LoRA, fine-tune on a
different style — training ~5% of the parameters — and watch generation shift
while the base weights stay byte-for-byte identical:

```
base model (trained on style A):  'the mat the cat sat on the mat and the dog r'
LoRA-tuned (adapted to style B):  'the mind to be on the that is the question...'
training 5120 adapter params out of 104064  (4.9% of the model)
base weights unchanged by fine-tuning?  True
```

## Run it

```bash
python -m forge.demo               # LoRA fine-tune: pretrain -> adapt -> before/after
python -m forge.demo --method dora # same, but fine-tune with DoRA
python -m forge.compare_dora  # LoRA vs DoRA on a controlled task (2024 technique)
python -m forge.web           # dashboard: pretrain, fine-tune live, compare + perplexity
python -m pytest tests/ -q    # the test suite (or: python tests/test_forge.py)
```

---

## The concept, and where it lives

### `lora.py` — the LoRA math (`LoRALinear`)
Freeze a weight `W (in,out)`; learn a low-rank correction `A (in,r)`, `B (r,out)`:

```
y = x @ W                         (frozen base, unchanged)
  + (x @ A @ B) * (alpha / r)     (trainable low-rank update)
```

Points I can defend at a whiteboard:
- **Why low-rank works**: the *change* needed to adapt to a task has low
  intrinsic rank — it lives in a small subspace — even though the weights are
  full-rank. So a rank-8 adapter recovers most of a full fine-tune.
- **Why B starts at zero**: then `A@B = 0`, so the adapted model *is* the base
  model at step 0 — fine-tuning begins as a no-op and moves away smoothly
  instead of jolting the model with a random perturbation.
- **The alpha/r scaling** decouples the learning rate from the rank, so you can
  change `r` without re-tuning.
- **The backward pass**: since `W` is frozen we never need `dW` — only `dA`,
  `dB`, and the `dx` that carries gradients to earlier layers. All three are
  gradient-checked to 1e-8.
- **Merging**: `merged_weight() = W + scaling*A@B` folds the adapter back into
  the base, so inference has *zero* overhead once merged (a real deployment step).

### `adapt.py` — applying LoRA to the backbone model
`LoRAHead` subclasses the backbone's attention head, reuses the frozen Q/K/V
weights,
and adds adapters to the **query and value** projections (the LoRA paper's common
recipe). `apply_lora(model)` swaps every head; `lora_params_and_grads(model)`
collects *only* the adapters, so the optimizer can never touch the base —
guaranteed, and asserted in the tests. `count_params` reports the honest
trainable/total ratio (every base parameter counted, not just attention).

### `web.py` — the dashboard
Pretrain (streamed loss) → attach LoRA and fine-tune (streamed loss + live
parameter count) → generate from base vs adapted side by side, with a check that
the base stayed frozen. The "base" side works by temporarily zeroing the adapter
`B` matrices, which recovers the exact pre-LoRA output (verified).

## Modern extensions (the "latest field" pieces)

### DoRA — Weight-Decomposed LoRA (2024) · `dora.py`
LoRA changes a weight's magnitude and direction together. DoRA decomposes them:
it applies the low-rank update to the *direction* (unit-normalized columns),
re-normalizes, then rescales by a *separately trained magnitude* vector:
`W' = m · (W + s·A@B) / ||W + s·A@B||_col`. That decoupled magnitude knob is why
DoRA closes much of the gap to full fine-tuning. It's a first-class fine-tuning
option here: `apply_lora(model, method="dora")` swaps in weight-decomposed
adapters on attention Q/V, trained exactly like LoRA. On a controlled fit
(`compare_dora`) DoRA reaches ~half the error of LoRA at the same rank, for a
handful of extra parameters. Forward and the column-normalization backward are
hand-written and gradient-checked to 1e-8, both standalone and inside attention.

### QLoRA-style quantization · `quant.py`
The frozen base is the memory cost of fine-tuning a big model. QLoRA stores it in
low precision and keeps only the small adapters in full precision. We implement
symmetric per-column int8 quantization (4x smaller; real QLoRA uses 4-bit NF4 for
8x, same principle). The key correctness point: the base is frozen, so we never
backprop into it — we only need its *value*, which we dequantize on the fly. The
adapters train identically over a quantized base (gradient-checked), and can even
compensate for quantization error. Round-trip error is ~0.6%.

### Adapter save / load / hot-swap · `adapters.py`
The real production pattern: keep ONE frozen base in memory, swap tiny adapters
(here ~6 KB) per task/customer. `save_adapters`/`load_adapters` write and restore
just the adapter arrays; a saved adapter round-trips exactly, and swapping it in
restores the adapted behavior precisely.

### Honest evaluation — perplexity · `adapters.py`
`perplexity` reports `exp(mean cross-entropy)` on held-out text (lower = better),
in the Crucible spirit of measuring generalization rather than asserting it. In
the demo the adapted model's perplexity on the target style drops dramatically
below the base's.

## Honest limitations

- Tiny, character-level, CPU — it demonstrates the mechanism, not scale. On a
  real model the trainable fraction would be ~1%; here it's a few percent
  because attention Q/V is a larger slice of a small model. The code reports the
  real measured number, not a marketing figure.
- Adapting only attention Q/V (not the MLPs) limits how far a small model can
  move; the adaptation is visible but not dramatic at this size.
- Fine-tuning "style B" reuses the shared character vocabulary so token ids line
  up cleanly between the two styles.

## What I'd add next

- LoRA/DoRA on the MLP projections too (more adaptation capacity — currently
  the adapters target attention Q/V).
- True 4-bit NF4 quantization with bit-packing (this uses int8 for legibility).
- A larger held-out eval with per-style perplexity tables.
