# Vendored into Forge so the project is self-contained (no external
# dependency). Same code as the companion Glassbox project. NumPy only.
"""
rope.py -- Rotary Position Embeddings, the position encoding used by Llama,
Mistral, GPT-NeoX, and essentially every current open LLM.

THE IDEA (and why it beats learned absolute positions)
------------------------------------------------------
GPT-2 adds a learned "position embedding" vector to each token so the model
knows where each token sits. That works but has two weaknesses: it can't
extend past the trained length, and it encodes *absolute* position when what
attention really cares about is *relative* position (how far apart two tokens
are).

RoPE fixes both with a beautiful trick: instead of ADDING anything, it ROTATES
each query and key vector by an angle proportional to its position. Pairs of
feature dimensions are treated as 2D coordinates and spun:

    for a pair (x_a, x_b) at position m with frequency theta:
        x_a' = x_a cos(m*theta) - x_b sin(m*theta)
        x_b' = x_a sin(m*theta) + x_b cos(m*theta)

Different dimension-pairs use different frequencies (like the sinusoidal
encoding). The magic: when you later take the dot product q·k inside attention,
the result depends only on (m - n), the RELATIVE distance between the two
positions -- the absolute rotations cancel. So the model gets relative-position
awareness for free, and can generalize to longer sequences.

We apply RoPE to Q and K (not V): it shapes how positions match, not what
information is carried.

Backward: a rotation is linear, and the inverse of rotating by +angle is
rotating by -angle. So the gradient just rotates the incoming gradient the
other way. Clean, and gradient-checked.
"""

from __future__ import annotations

import numpy as np


class RoPE:
    """Precompute rotation angles once, then apply/unapply to (B, T, H, d)
    or (B, T, d) tensors. `dim` must be even (dimensions are rotated in pairs).
    """

    def __init__(self, dim, max_seq=512, base=10000.0):
        assert dim % 2 == 0, "RoPE dimension must be even (rotates dims in pairs)"
        self.dim = dim
        # frequency for each dimension-pair: theta_i = base^(-2i/dim)
        # low pairs rotate fast (fine position), high pairs rotate slow (coarse).
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        pos = np.arange(max_seq, dtype=np.float64)
        angles = np.outer(pos, inv_freq)          # (max_seq, dim/2)
        self.cos = np.cos(angles).astype(np.float32)   # (max_seq, dim/2)
        self.sin = np.sin(angles).astype(np.float32)

    def _cs(self, T):
        # cos/sin for the first T positions, shaped to broadcast over (..., dim/2)
        return self.cos[:T], self.sin[:T]

    def apply(self, x):
        """Rotate x by its position. x: (..., T, dim) with T on axis -2.
        Returns rotated x of the same shape. Splits the last dim into even/odd
        halves as the two coordinates of each rotated pair."""
        T = x.shape[-2]
        cos, sin = self._cs(T)
        # reshape cos/sin to broadcast against x's leading dims:
        # x is (..., T, dim); we want cos as (1.., T, dim/2)
        shape = [1] * (x.ndim - 2) + [T, self.dim // 2]
        cos = cos.reshape(shape)
        sin = sin.reshape(shape)

        x_even = x[..., 0::2]     # (..., T, dim/2)  first coord of each pair
        x_odd = x[..., 1::2]      # (..., T, dim/2)  second coord

        out = np.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def backward(self, dy):
        """Gradient of apply(). A rotation by +angle is orthogonal, so its
        transpose (the gradient) is rotation by -angle. We rotate dy backward."""
        T = dy.shape[-2]
        cos, sin = self._cs(T)
        shape = [1] * (dy.ndim - 2) + [T, self.dim // 2]
        cos = cos.reshape(shape)
        sin = sin.reshape(shape)

        d_even = dy[..., 0::2]
        d_odd = dy[..., 1::2]

        dx = np.empty_like(dy)
        # inverse rotation: [cos, sin; -sin, cos] applied to (d_even, d_odd)
        dx[..., 0::2] = d_even * cos + d_odd * sin
        dx[..., 1::2] = -d_even * sin + d_odd * cos
        return dx
