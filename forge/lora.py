"""
lora.py -- Low-Rank Adaptation (LoRA), implemented from scratch.

WHAT LoRA IS, AND WHY IT'S EVERYWHERE
-------------------------------------
Fine-tuning a large model by updating ALL its weights is expensive: you need a
full copy of the (huge) weights' gradients and optimizer state. LoRA (Hu et al.
2021) is the trick that made fine-tuning cheap enough to do on one GPU, and it's
now the default for adapting open models.

The idea: freeze the original weight matrix W (in, out) and DON'T touch it.
Instead, learn a small correction, and constrain that correction to be
LOW-RANK -- the product of two thin matrices:

    A: (in, r)      B: (r, out)      with rank r much smaller than in/out

The adapted layer computes:

    y = x @ W          (frozen base, unchanged)
      + (x @ A @ B) * (alpha / r)     (the trainable low-rank update)

Only A and B are trained. If in=out=512 and r=8, the full weight has 262,144
numbers but the adapter has just 512*8 + 8*512 = 8,192 -- about 3%. That's the
whole win.

WHY LOW-RANK IS A REASONABLE CONSTRAINT
--------------------------------------------------------------------
The hypothesis behind LoRA is that the *change* a model needs to adapt to a new
task has low "intrinsic rank" -- it lives in a small subspace, even though the
weights themselves are full-rank. Empirically that holds surprisingly well, and
it's why a rank-8 adapter can recover most of the quality of a full fine-tune.

TWO INITIALIZATION DETAILS THAT MATTER
--------------------------------------
- B is initialized to ZERO, A to small random. So at the start A@B = 0, meaning
  the adapted model is EXACTLY the base model -- fine-tuning begins as a no-op
  and moves away smoothly. Initializing both randomly would jolt the model with
  a large random perturbation before training even starts.
- The alpha/r scaling decouples the learning rate from the choice of rank, so
  you can change r without re-tuning everything.

The backward pass is derived and gradient-checked (see gradcheck.py). Because
the base W is frozen, we never need dW -- only dA, dB, and the dx needed to
propagate gradients to earlier layers.
"""

from __future__ import annotations

import numpy as np


class LoRALinear:
    """A frozen linear layer y = x@W with a trainable low-rank adapter.

    W_frozen: the base weight (in, out). Treated as a constant -- never updated.
    rank r, alpha: adapter size and scaling.
    """

    def __init__(self, W_frozen, rank=4, alpha=8.0, seed=0):
        self.W = W_frozen                       # frozen base (in, out) -- NOT trained
        self.in_dim, self.out_dim = W_frozen.shape
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank             # the alpha/r factor

        rng = np.random.default_rng(seed)
        # A: small random (Kaiming-ish); B: zeros so the adapter starts at 0.
        self.A = (rng.standard_normal((self.in_dim, rank)) / np.sqrt(self.in_dim)).astype(np.float32)
        self.B = np.zeros((rank, self.out_dim), dtype=np.float32)

    def forward(self, x):
        # x: (..., in). Cache what backward needs.
        self.x = x
        self.xA = x @ self.A                    # (..., r)  -- the "down" projection
        base = x @ self.W                       # frozen path
        update = (self.xA @ self.B) * self.scaling   # low-rank "up" projection
        return base + update

    def backward(self, dy):
        """Gradients for the trainable adapter (dA, dB) and the input (dx).

        Forward (ignoring the frozen base for the trainable grads):
            update = scaling * (x @ A) @ B
        Let g = dy * scaling.
            dB  = (x @ A)^T @ g                          [ (r,out) ]
            d(xA) = g @ B^T                              [ (...,r) ]
            dA  = x^T @ d(xA)                            [ (in,r) ]
            dx  = dy @ W^T   (through frozen base)
                + d(xA) @ A^T (through the adapter)
        We collapse leading (batch/time) dims before the outer products, exactly
        like the base Matmul's backward.
        """
        g = dy * self.scaling

        # flatten leading dims -> (N, ...) for the matmul-style grads
        x2 = self.x.reshape(-1, self.in_dim)            # (N, in)
        xA2 = self.xA.reshape(-1, self.r)               # (N, r)
        g2 = g.reshape(-1, self.out_dim)                # (N, out)

        self.dB = xA2.T @ g2                            # (r, out)
        d_xA = g @ self.B.T                             # (..., r)
        d_xA2 = d_xA.reshape(-1, self.r)                # (N, r)
        self.dA = x2.T @ d_xA2                          # (in, r)

        # input gradient flows through BOTH paths
        dx = dy @ self.W.T + d_xA @ self.A.T
        return dx

    # --- parameter bookkeeping (only A and B are trainable) ---
    def params_and_grads(self):
        return [(self.A, self.dA), (self.B, self.dB)]

    def n_trainable(self):
        return self.A.size + self.B.size

    def n_frozen(self):
        return self.W.size

    def merged_weight(self):
        """The effective weight W + scaling*A@B. Merging LoRA back into the base
        (so inference has zero overhead) is a real deployment step -- being able
        to do it shows you understand that LoRA adds no inference cost once
        merged."""
        return self.W + (self.A @ self.B) * self.scaling
