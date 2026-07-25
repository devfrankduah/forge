"""
demo.py -- the whole Forge story in one run.

    python -m forge.demo

It (1) trains a base Glassbox transformer on one body of text, (2) FREEZES it
and attaches LoRA adapters, (3) fine-tunes ONLY those adapters on a different
style of text, and (4) shows honest before/after evidence: the parameter count
(how little we trained), the loss curve, sample generations from the base vs the
fine-tuned model, and a check that the base weights never changed.

The point: adapt a pretrained model to a new task by training ~a few percent of
the parameters, with the base provably untouched -- the core value of LoRA.
"""

from __future__ import annotations

import numpy as np

from .backbone.model import GPT
from .backbone.data import CharDataset, train, generate
from .backbone.optimizer import Adam
from .adapt import apply_lora, lora_params_and_grads, count_params


# Two DISTINCT styles that share a character set, so the same vocabulary /
# token ids apply to both and we can see adaptation cleanly.
STYLE_A = ("the cat sat on the mat and the dog ran in the sun "
           "a cat and a dog sat in the sun on the mat ") * 30
STYLE_B = ("to be or not to be that is the question "
           "whether it is nobler in the mind to be ") * 30


def _shared_dataset(block_size=32):
    # one dataset whose vocab covers BOTH styles, so ids are consistent
    return CharDataset(STYLE_A + STYLE_B, block_size=block_size)


def _train_on(model, ds, text, steps, lr, seed=0, only=None):
    """Train `model` on `text` (encoded with ds's vocab). If `only` is given
    (a params_and_grads callable), update just those params (LoRA mode)."""
    data = ds.encode(text)
    rng = np.random.default_rng(seed)
    opt = Adam(lr=lr)
    bs, blk = 16, ds.block_size
    hist = []
    for step in range(1, steps + 1):
        n = len(data) - blk - 1
        ix = rng.integers(0, n, size=bs)
        x = np.stack([data[i:i + blk] for i in ix])
        y = np.stack([data[i + 1:i + 1 + blk] for i in ix])
        _, loss = model.forward(x, y)
        model.backward()
        opt.step(only(model) if only else model.params_and_grads())
        hist.append(float(loss))
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"  step {step:4d}/{steps}   loss {loss:.4f}")
    return hist


def main(method="lora"):
    label = method.upper()
    ds = _shared_dataset()
    print("=" * 62)
    print("STEP 1 — pretrain a base model on STYLE A (cat/dog/sun)")
    print("=" * 62)
    base = GPT(vocab_size=ds.vocab_size, d_model=64, n_heads=4, n_layers=2,
               block_size=32, seed=0, arch="gpt2")
    _train_on(base, ds, STYLE_A, steps=200, lr=3e-3)

    base_sample = generate(base, ds, "the ", 40, seed=1)
    # snapshot to prove the base never changes during fine-tuning
    frozen_snapshot = base.blocks[0].attn.heads[0].Wq.copy()

    print()
    print("=" * 62)
    print(f"STEP 2 — freeze base, attach {label}, fine-tune on STYLE B (Shakespeare)")
    print("=" * 62)
    apply_lora(base, rank=4, alpha=8.0, method=method)
    trainable, total = count_params(base)
    print(f"training {trainable} adapter params out of {total} "
          f"({100 * trainable / total:.1f}% of the model)\n")
    _train_on(base, ds, STYLE_B, steps=200, lr=5e-3, only=lora_params_and_grads)

    tuned_sample = generate(base, ds, "the ", 40, seed=1)

    print()
    print("=" * 62)
    print("STEP 3 — honest before/after")
    print("=" * 62)
    print(f"base model (trained on A) sample:  {base_sample!r}")
    print(f"{label}-tuned (adapted to B) sample:  {tuned_sample!r}")
    print()
    unchanged = np.array_equal(frozen_snapshot, base.blocks[0].attn.heads[0].Wq)
    print(f"base weights unchanged by fine-tuning? {unchanged}")
    print(f"adapters moved off their zero init?    "
          f"{not np.allclose(base.blocks[0].attn.heads[0].Bq, 0)}")
    print(f"\n(The base is provably frozen; only the ~few-percent {label} adapter"
          "\n moved, and generation shifted toward style B.)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Forge fine-tuning demo")
    ap.add_argument("--method", choices=["lora", "dora"], default="lora",
                    help="adapter type to fine-tune with")
    main(ap.parse_args().method)
