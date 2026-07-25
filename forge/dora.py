"""
dora.py -- DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).

WHAT DoRA IS, AND WHY IT'S A REAL IMPROVEMENT ON LoRA
-----------------------------------------------------
LoRA adds a low-rank update: W' = W + (alpha/r) A@B. It works, but it changes a
weight's MAGNITUDE and DIRECTION together, coupled through one low-rank term.
DoRA's insight (2024, and now widely used) is to DECOMPOSE a weight into those
two parts and adapt them separately:

    - direction: the unit-normalized columns of the weight,
    - magnitude: a per-output-column scale (its own trainable vector).

DoRA applies the LoRA update to the DIRECTION, re-normalizes, then rescales by a
separately-trained magnitude:

    V   = W + (alpha/r) A@B                    # LoRA-updated weight
    W'  = m * V / ||V||_col                    # renormalize columns, rescale by m

where ||V||_col is the L2 norm of each column (length = out_dim), and m is a
trainable magnitude vector (length out_dim). This gives the optimizer an
explicit, decoupled knob for "how big" vs "which way", which empirically closes
much of the gap between LoRA and full fine-tuning. Being able to explain that
decoupling -- and derive its gradient -- is a strong "I follow current research"
signal.

INITIALIZATION (so training starts as a no-op, like LoRA)
---------------------------------------------------------
B starts at zero (so A@B = 0 => V = W), and m starts at the actual column norms
of W. Then W' = ||W||_col * W / ||W||_col = W exactly. Fine-tuning begins as the
base model and moves away smoothly.

The backward is the interesting part -- the column normalization couples every
element of a column -- so it's derived carefully below and gradient-checked in
the tests. Trainable params: A, B, and m. The base W is frozen.
"""

from __future__ import annotations

import numpy as np


def _col_norm(V, eps=1e-8):
    """L2 norm of each column of V (in,out) -> vector length out."""
    return np.sqrt(np.sum(V * V, axis=0) + eps)


def dora_compose(W, A, B, m, scaling):
    """Build the DoRA-composed weight from a frozen W and adapters A,B,m.

    Returns (Wp, V, n): the effective weight Wp = m * V / ||V||_col, plus the
    intermediate V = W + scaling*A@B and its column norms n (both needed for the
    backward). Shared by DoRALinear and the attention DoRAHead so the exact same
    gradient-checked math is used in both places."""
    V = W + scaling * (A @ B)
    n = _col_norm(V)
    Wp = m * (V / n)
    return Wp, V, n


def dora_grads(dWp, V, n, m, A, B, scaling):
    """Given the gradient w.r.t. the composed weight (dWp), return (dA, dB, dm).

    This is the column-normalization backward derived in DoRALinear.backward,
    factored out so the attention head reuses it verbatim:
        c_j = Σ_i dWp[i,j] V[i,j]
        dm  = c / n
        dV  = (m/n) (dWp - V c / n^2)
        dA  = (scaling dV) @ B^T,  dB = A^T @ (scaling dV)
    """
    c = np.sum(dWp * V, axis=0)                     # (out,)
    dm = c / n
    dV = (m / n) * (dWp - V * (c / (n * n)))        # (in,out)
    G = scaling * dV
    dA = G @ B.T
    dB = A.T @ G
    return dA, dB, dm


class DoRALinear:
    """A frozen linear layer adapted with DoRA (magnitude + direction).

    W_frozen: base weight (in, out), never updated.
    Trainable: A (in,r), B (r,out), and magnitude m (out,).
    """

    def __init__(self, W_frozen, rank=4, alpha=8.0, seed=0):
        self.W = W_frozen
        self.in_dim, self.out_dim = W_frozen.shape
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        rng = np.random.default_rng(seed)
        self.A = (rng.standard_normal((self.in_dim, rank)) / np.sqrt(self.in_dim)).astype(np.float32)
        self.B = np.zeros((rank, self.out_dim), dtype=np.float32)
        # magnitude initialized to the base weight's column norms -> starts as base
        self.m = _col_norm(W_frozen).astype(np.float32)

    def forward(self, x):
        self.x = x
        self.P = self.A @ self.B                       # low-rank update (in,out)
        self.V = self.W + self.scaling * self.P        # updated weight
        self.norm = _col_norm(self.V)                  # (out,)
        self.Wdir = self.V / self.norm                 # unit-norm columns (in,out)
        self.Wp = self.m * self.Wdir                   # rescale by magnitude
        return x @ self.Wp

    def backward(self, dy):
        """Gradients for A, B, m and the input x. Derivation:

        y = x @ W'      => dW' = x^T @ dy ,  dx = dy @ W'^T
        W'[i,j] = m_j * V[i,j] / n_j , with n_j = ||V[:,j]||

        dm_j = sum_i dW'[i,j] * V[i,j] / n_j = c_j / n_j   where c_j = Σ_i dW'[i,j] V[i,j]
        dV[k,j] = (m_j / n_j) * ( dW'[k,j] - V[k,j] * c_j / n_j^2 )
        Then V = W + s*A@B  =>  grad into (A@B) is  G = s * dV, and
            dA = G @ B^T ,  dB = A^T @ G .
        """
        # dW' and dx from the outer product
        x2 = self.x.reshape(-1, self.in_dim)                 # (N,in)
        dy2 = dy.reshape(-1, self.out_dim)                   # (N,out)
        dWp = x2.T @ dy2                                      # (in,out)
        dx = dy @ self.Wp.T

        n = self.norm                                        # (out,)
        # c_j = sum over the `in` axis of dW' * V   -> (out,)
        c = np.sum(dWp * self.V, axis=0)                     # (out,)

        # magnitude gradient
        self.dm = c / n                                      # (out,)

        # gradient into V (column-normalization backward)
        # dV = (m/n) * ( dW' - V * c / n^2 )
        dV = (self.m / n) * (dWp - self.V * (c / (n * n)))   # (in,out) via broadcasting

        # into the low-rank product, then into A and B
        G = self.scaling * dV                                # (in,out)
        self.dA = G @ self.B.T                               # (in,r)
        self.dB = self.A.T @ G                               # (r,out)
        return dx

    def params_and_grads(self):
        return [(self.A, self.dA), (self.B, self.dB), (self.m, self.dm)]

    def n_trainable(self):
        return self.A.size + self.B.size + self.m.size

    def merged_weight(self):
        """Effective weight m * (W + s*A@B) / ||W + s*A@B||_col, folded for
        zero-overhead inference."""
        V = self.W + self.scaling * (self.A @ self.B)
        return self.m * (V / _col_norm(V))
