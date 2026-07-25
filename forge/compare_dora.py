"""
compare_dora.py -- LoRA vs DoRA on a controlled task.

    python -m forge.compare_dora

Both adapt a frozen random base weight to match a target linear map, using the
SAME rank and step budget. DoRA additionally trains a magnitude vector, which
lets it move a weight's size and direction independently. This little
experiment makes the difference concrete: we watch both fit the target and
report the final error, and we peek at how DoRA's magnitude vector moved away
from the base's column norms (evidence the extra knob is being used).

This is a clean, honest demonstration of a current (2024) technique, separate
from the attention fine-tuning path so the comparison is apples-to-apples.
"""

from __future__ import annotations

import numpy as np

from .lora import LoRALinear
from .dora import DoRALinear, _col_norm
from .backbone.optimizer import Adam


def _fit(adapter, x, target_y, steps, lr):
    """Train an adapter's parameters to map x -> target_y (MSE). Returns the
    loss history. Works for both LoRALinear and DoRALinear via their shared
    forward/backward/params_and_grads interface."""
    opt = Adam(lr=lr)
    hist = []
    for _ in range(steps):
        y = adapter.forward(x)
        # MSE loss and its gradient
        diff = y - target_y
        loss = float(np.mean(diff * diff))
        dy = (2.0 / diff.size) * diff
        adapter.backward(dy)
        opt.step(adapter.params_and_grads())
        hist.append(loss)
    return hist


def main():
    rng = np.random.default_rng(0)
    IN, OUT, N, R = 16, 12, 64, 4

    # a frozen random base, an input batch, and a target we want to adapt toward
    W = (rng.standard_normal((IN, OUT)) / np.sqrt(IN)).astype(np.float32)
    x = rng.standard_normal((N, IN)).astype(np.float32)
    # target = a DIFFERENT linear map, so the adapter has to do real work
    W_target = (rng.standard_normal((IN, OUT)) / np.sqrt(IN)).astype(np.float32)
    target_y = x @ W_target

    print("Adapting a frozen base weight to a new target map (rank", R, ")\n")

    lora = LoRALinear(W, rank=R, alpha=2 * R, seed=1)
    lora_hist = _fit(lora, x, target_y, steps=400, lr=5e-2)

    dora = DoRALinear(W, rank=R, alpha=2 * R, seed=1)
    m_before = dora.m.copy()
    dora_hist = _fit(dora, x, target_y, steps=400, lr=5e-2)
    m_after = dora.m.copy()

    print(f"LoRA final MSE: {lora_hist[-1]:.5f}")
    print(f"DoRA final MSE: {dora_hist[-1]:.5f}")
    print()
    # how much did DoRA's magnitude knob move, relative to the base column norms?
    rel_move = np.linalg.norm(m_after - m_before) / (np.linalg.norm(m_before) + 1e-12)
    print(f"DoRA magnitude vector moved {rel_move*100:.1f}% from the base column norms")
    print("(the separately-trained magnitude is the knob LoRA doesn't have)")
    print()
    print(f"trainable params -- LoRA: {lora.n_trainable()}, "
          f"DoRA: {dora.n_trainable()} (+{OUT} for the magnitude vector)")


if __name__ == "__main__":
    main()
