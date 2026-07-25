"""
quant.py -- QLoRA-style quantization of the frozen base weights.

THE MEMORY PROBLEM QLoRA SOLVES
-------------------------------
LoRA already shrinks the *trainable* footprint (you train tiny adapters). But
you still have to hold the frozen base weights in memory, and for a large model
those weights ARE the memory. QLoRA (Dettmers et al., 2023) closes that: store
the frozen base in low precision (they use 4-bit; we use a clean int8 here to
keep the arithmetic legible), and keep only the small LoRA adapters in full
precision. The result is fine-tuning a model far larger than would otherwise fit.

The key correctness idea: the base is FROZEN, so we never backprop into it. We
only ever need its VALUE in the forward pass. So we can store it quantized and
dequantize on the fly for the matmul; the adapter gradients are unaffected by
how we happen to store a constant. And because the adapters are trained ON TOP
of the quantized base, they can partly compensate for the quantization error --
another reason QLoRA works as well as it does.

WHAT WE IMPLEMENT
-----------------
Symmetric per-column int8 quantization:

    scale_j = max(|W[:,j]|) / 127
    q[:,j]  = round(W[:,j] / scale_j)      (int8, range -127..127)
    dequant: W_hat[:,j] = q[:,j] * scale_j

Per-column (rather than one scale for the whole matrix) keeps error low when
columns have very different magnitudes -- a standard, cheap improvement. int8 is
4x smaller than float32, which we measure and report. (Real QLoRA uses 4-bit
NF4 for 8x; the principle is identical, the bit-packing is just fiddlier.)
"""

from __future__ import annotations

import numpy as np


def quantize_int8(W):
    """Symmetric per-column int8 quantization.

    Returns (q, scale) where q is int8 (in,out) and scale is (out,) float32.
    Dequantize with `q.astype(float) * scale`."""
    W = np.asarray(W, dtype=np.float32)
    # per-column absolute max; guard against an all-zero column
    amax = np.max(np.abs(W), axis=0)
    amax = np.where(amax == 0, 1.0, amax)
    scale = (amax / 127.0).astype(np.float32)        # (out,)
    q = np.round(W / scale).astype(np.int8)          # (in,out), -127..127
    return q, scale


def dequantize_int8(q, scale):
    """Reconstruct an approximate float weight from int8 + per-column scale."""
    return q.astype(np.float32) * scale              # broadcasting over columns


def quantization_error(W):
    """Relative reconstruction error of int8-quantizing W (for reporting)."""
    q, s = quantize_int8(W)
    W_hat = dequantize_int8(q, s)
    denom = np.linalg.norm(W) + 1e-12
    return float(np.linalg.norm(W - W_hat) / denom)


def bytes_of(W_float32):
    return int(np.asarray(W_float32).size * 4)       # float32 = 4 bytes


def bytes_int8(shape_in, shape_out):
    # int8 weights (1 byte each) + a float32 scale per column (4 bytes)
    return shape_in * shape_out + shape_out * 4


class QuantizedLoRALinear:
    """A LoRA-adapted linear layer whose FROZEN base is stored int8-quantized.

    Only the base storage differs from LoRALinear: we hold (q, scale) instead of
    a float W, and dequantize for the forward pass. Adapters (A, B) are full
    precision and train exactly as before -- their gradients don't depend on how
    the frozen base is stored. This is the essence of QLoRA.
    """

    def __init__(self, W_frozen, rank=4, alpha=8.0, seed=0):
        # quantize the base ONCE, up front, and keep only the int8 form.
        self.q, self.scale = quantize_int8(W_frozen)
        self.in_dim, self.out_dim = W_frozen.shape
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        rng = np.random.default_rng(seed)
        self.A = (rng.standard_normal((self.in_dim, rank)) / np.sqrt(self.in_dim)).astype(np.float32)
        self.B = np.zeros((rank, self.out_dim), dtype=np.float32)

    def _W(self):
        # dequantized base, reconstructed on demand for the matmul
        return dequantize_int8(self.q, self.scale)

    def forward(self, x):
        self.x = x
        self.xA = x @ self.A
        base = x @ self._W()                          # dequantized frozen base
        return base + (self.xA @ self.B) * self.scaling

    def backward(self, dy):
        # Identical adapter math to LoRALinear.backward -- the ONLY difference is
        # that the frozen base is dequantized (self._W()) instead of stored as a
        # float. Because the base is a constant either way, quantizing it does
        # not change the adapter gradients at all. That is the QLoRA insight.
        g = dy * self.scaling                            # fold in alpha/r
        x2 = self.x.reshape(-1, self.in_dim)             # flatten leading dims
        g2 = g.reshape(-1, self.out_dim)
        self.dB = self.xA.reshape(-1, self.r).T @ g2     # dB = (x@A)^T @ g
        d_xA = g @ self.B.T                              # grad w.r.t. (x@A)
        self.dA = x2.T @ d_xA.reshape(-1, self.r)        # dA = x^T @ d(x@A)
        # dx flows through the dequantized frozen base AND the adapter
        dx = dy @ self._W().T + d_xA @ self.A.T
        return dx

    def params_and_grads(self):
        return [(self.A, self.dA), (self.B, self.dB)]

    def memory_report(self):
        """(quantized_bytes, float_bytes) for the frozen base -- the headline
        memory saving from quantization."""
        return (bytes_int8(self.in_dim, self.out_dim),
                bytes_of(np.empty((self.in_dim, self.out_dim), np.float32)))
