# Vendored into Forge so the project is self-contained (no external
# dependency). Same code as the companion Glassbox project. NumPy only.
"""
attention.py -- self-attention, the mechanism that makes a transformer a
transformer. Forward and backward, by hand.

THE IDEA IN ONE BREATH
----------------------
Every position in the sequence produces three vectors: a Query (what am I
looking for?), a Key (what do I offer?), and a Value (what I'll actually pass
on). Each position's output is a weighted average of all positions' Values,
where the weights come from how well its Query matches each Key. "Attention"
is literally: softmax(Q Kᵀ / sqrt(d)) V.

The sqrt(d) scaling stops the dot products from getting huge as dimension
grows (which would saturate the softmax into a one-hot and kill gradients) --
a small detail that reveals whether someone actually understands why the
formula is written the way it is.

CAUSAL MASK
-----------
For a language model, position t may only attend to positions <= t (you can't
peek at the future you're trying to predict). We enforce that by setting the
"future" scores to -inf before the softmax, so they get zero weight.

We keep the attention weights around after the forward pass -- that's what the
interpretability tooling visualizes: literally which past tokens each token
looked at.
"""

from __future__ import annotations

import numpy as np

from .tensor_ops import Matmul, Softmax


def causal_mask(T):
    """Lower-triangular (T,T) mask: True where attention is ALLOWED (j <= i)."""
    return np.tril(np.ones((T, T), dtype=bool))


class SelfAttentionHead:
    """One attention head. Operates on (B, T, head_dim).

    Holds its own Q/K/V projection weights. The forward computes attention;
    the backward pushes gradients back through the softmax, the score matrix,
    and the three projections -- all using the verified primitive ops.
    """

    def __init__(self, d_model, head_dim, rng, rope=None):
        # small random init scaled by 1/sqrt(d) (a standard, stable choice)
        scale = 1.0 / np.sqrt(d_model)
        self.Wq = (rng.standard_normal((d_model, head_dim)) * scale).astype(np.float32)
        self.Wk = (rng.standard_normal((d_model, head_dim)) * scale).astype(np.float32)
        self.Wv = (rng.standard_normal((d_model, head_dim)) * scale).astype(np.float32)
        self.head_dim = head_dim
        # optional Rotary Position Embedding: if given, Q and K are rotated by
        # position before computing attention scores (Llama-style). If None,
        # the model relies on the learned positional embedding instead (GPT-2).
        self.rope = rope

    def forward(self, x):
        B, T, C = x.shape
        self.x = x
        self.B, self.T = B, T

        # project to queries/keys/values (one Matmul op per projection so we
        # can reuse their verified backward)
        self.mq, self.mk, self.mv = Matmul(), Matmul(), Matmul()
        self.q = self.mq.forward(x, self.Wq)      # (B,T,hd)
        self.k = self.mk.forward(x, self.Wk)
        self.v = self.mv.forward(x, self.Wv)

        # RoPE: rotate q and k by position (applied to the last dim, T on -2)
        if self.rope is not None:
            self.q = self.rope.apply(self.q)
            self.k = self.rope.apply(self.k)

        # attention scores, scaled
        self.scale = 1.0 / np.sqrt(self.head_dim)
        scores = (self.q @ self.k.transpose(0, 2, 1)) * self.scale   # (B,T,T)

        # causal mask: future positions -> -inf so softmax zeroes them
        self.mask = causal_mask(T)
        scores = np.where(self.mask[None], scores, -1e9)

        # softmax over the last axis (per query position)
        self.sm = Softmax()
        self.att = self.sm.forward(scores)         # (B,T,T) attention weights
        out = self.att @ self.v                    # (B,T,hd)
        return out

    def backward(self, dout):
        # dout: (B,T,hd). Reverse of forward, step by step.
        # out = att @ v
        datt = dout @ self.v.transpose(0, 2, 1)    # (B,T,T)
        dv = self.att.transpose(0, 2, 1) @ dout    # (B,T,hd)

        # through softmax (per-row Jacobian handled by Softmax.backward)
        dscores = self.sm.backward(datt)           # (B,T,T)
        # masked positions contributed nothing; zero their grad for cleanliness
        dscores = np.where(self.mask[None], dscores, 0.0)
        dscores *= self.scale

        # scores = q @ k^T  ->  dq = dscores @ k ; dk = dscores^T @ q
        dq = dscores @ self.k                      # (B,T,hd)
        dk = dscores.transpose(0, 2, 1) @ self.q   # (B,T,hd)

        # back through RoPE (rotation is orthogonal: gradient rotates the other
        # way). Must happen BEFORE the projection backward, mirroring forward.
        if self.rope is not None:
            dq = self.rope.backward(dq)
            dk = self.rope.backward(dk)

        # back through the three projections; sum the dx contributions
        dx_q, dWq = self.mq.backward(dq)
        dx_k, dWk = self.mk.backward(dk)
        dx_v, dWv = self.mv.backward(dv)
        self.dWq, self.dWk, self.dWv = dWq, dWk, dWv
        return dx_q + dx_k + dx_v

    def params_and_grads(self):
        return [
            (self.Wq, self.dWq), (self.Wk, self.dWk), (self.Wv, self.dWv),
        ]

    # ------------------------------------------------------------------
    # KV-cache path (inference only)
    # ------------------------------------------------------------------
    def reset_cache(self):
        self._k_cache = None    # (B, t, hd) keys seen so far
        self._v_cache = None    # (B, t, hd) values seen so far

    def forward_cached(self, x_last, pos, max_context=None):
        """Process one new token. Appends its K,V to the cache and attends over
        all cached positions. Returns (B,1,hd).

        max_context, if given, caps the cache to a sliding window of the most
        recent positions -- matching the model's trained context window so
        generation past that window behaves like the (cropped) slow path rather
        than attending over an ever-growing history it never saw in training."""
        # project just the new token
        q = x_last @ self.Wq            # (B,1,hd)
        k = x_last @ self.Wk
        v = x_last @ self.Wv

        # RoPE: rotate q,k by THIS position's angle.
        if self.rope is not None:
            q = self._rope_at(q, pos)
            k = self._rope_at(k, pos)

        # append to cache
        if getattr(self, "_k_cache", None) is None:
            self._k_cache, self._v_cache = k, v
        else:
            self._k_cache = np.concatenate([self._k_cache, k], axis=1)
            self._v_cache = np.concatenate([self._v_cache, v], axis=1)

        # slide the window: keep only the most recent max_context entries
        if max_context is not None and self._k_cache.shape[1] > max_context:
            self._k_cache = self._k_cache[:, -max_context:]
            self._v_cache = self._v_cache[:, -max_context:]

        # attend: new query against all cached keys (no mask needed -- the cache
        # only contains past positions by construction)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ self._k_cache.transpose(0, 2, 1)) * scale   # (B,1,t)
        z = scores - scores.max(axis=-1, keepdims=True)
        att = np.exp(z); att /= att.sum(axis=-1, keepdims=True)
        return att @ self._v_cache                                # (B,1,hd)

    def _rope_at(self, x1, pos):
        """Apply RoPE to a single-position tensor x1 (B,1,hd) at absolute
        position `pos`, reusing the precomputed rotation tables.

        The tables cover positions 0..max_seq-1. If generation runs past that,
        we clamp to the last precomputed angle. (A production RoPE would extend
        or interpolate the tables; clamping keeps the demo robust and is a fine
        place to note that trade-off.)"""
        p = min(pos, self.rope.cos.shape[0] - 1)
        cos = self.rope.cos[p]          # (hd/2,)
        sin = self.rope.sin[p]
        x_even = x1[..., 0::2]
        x_odd = x1[..., 1::2]
        out = np.empty_like(x1)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


class MultiHeadAttention:
    """Several heads in parallel, their outputs concatenated and projected.

    Multiple heads let the model attend to different kinds of relationships at
    once (e.g. one head tracks the previous token, another tracks subject-verb
    agreement). We run each head, concatenate along the feature axis, then mix
    them with an output projection.
    """

    def __init__(self, d_model, n_heads, rng, rope=None):
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.head_dim = d_model // n_heads
        self.heads = [SelfAttentionHead(d_model, self.head_dim, rng, rope=rope)
                      for _ in range(n_heads)]
        scale = 1.0 / np.sqrt(d_model)
        self.Wo = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)
        self.n_heads = n_heads

    def forward(self, x):
        self.head_outs = [h.forward(x) for h in self.heads]   # each (B,T,hd)
        concat = np.concatenate(self.head_outs, axis=-1)      # (B,T,C)
        self.mo = Matmul()
        self.concat = concat
        return self.mo.forward(concat, self.Wo)

    def backward(self, dout):
        dconcat, self.dWo = self.mo.backward(dout)            # (B,T,C)
        # split the concat gradient back to each head and sum their input grads
        splits = np.split(dconcat, self.n_heads, axis=-1)
        dx = None
        for h, dseg in zip(self.heads, splits):
            dxi = h.backward(dseg)
            dx = dxi if dx is None else dx + dxi
        return dx

    def attention_maps(self):
        """The (B,T,T) attention weights for each head, for visualization."""
        return [h.att for h in self.heads]

    def params_and_grads(self):
        pg = [(self.Wo, self.dWo)]
        for h in self.heads:
            pg.extend(h.params_and_grads())
        return pg

    # ------------------------------------------------------------------
    # KV-cache path (inference only, no backward)
    # ------------------------------------------------------------------
    def reset_cache(self):
        for h in self.heads:
            h.reset_cache()

    def forward_cached(self, x_last, pos, max_context=None):
        """Incremental forward for ONE new position during generation.

        x_last: (B, 1, C) -- just the newest token's hidden state.
        pos:    integer index of this position (for RoPE).
        max_context: optional sliding-window cap on the KV cache.
        Each head appends this step's K and V to its cache and attends over the
        whole cached history, so we do O(T) work per new token instead of
        recomputing the full O(T^2) attention every step. This is exactly the
        optimization that makes real LLM generation fast."""
        outs = [h.forward_cached(x_last, pos, max_context=max_context) for h in self.heads]
        concat = np.concatenate(outs, axis=-1)          # (B,1,C)
        m = Matmul()
        return m.forward(concat, self.Wo)


class GroupedQueryAttention:
    """Grouped-Query Attention (GQA) -- the attention variant used by Llama-2/3
    and Mistral, and a genuinely current (2023+) architectural idea.

    THE PROBLEM IT SOLVES
    ---------------------
    In vanilla multi-head attention, every query head has its OWN key and value
    heads. During generation the KV-cache stores K and V for every head at every
    position, and that cache is the dominant memory cost of serving a long
    context. GQA shrinks it: you keep the full number of QUERY heads (for
    expressiveness) but only a SMALL number of KV heads, and each KV head is
    SHARED by a group of query heads.

        n_heads = 8 query heads, n_kv_heads = 2  ->  groups of 4 queries share
        one K/V head  ->  the KV-cache is 4x smaller.

    Two familiar points on the spectrum: n_kv_heads == n_heads is ordinary
    multi-head attention; n_kv_heads == 1 is Multi-Query Attention (MQA). GQA is
    the middle ground that keeps most of the quality with most of the memory
    saving -- which is why modern models use it. Being able to explain that
    quality/memory tradeoff is a strong signal of current understanding.

    This is a self-contained implementation (its own Q/K/V/O weights) with a
    hand-written backward, gradient-checked like everything else. Optional RoPE
    is applied to Q and each KV head.
    """

    def __init__(self, d_model, n_heads, n_kv_heads, rng, rope=None):
        assert d_model % n_heads == 0, "d_model must divide into query heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group = n_heads // n_kv_heads          # queries per KV head
        self.hd = d_model // n_heads
        self.rope = rope
        scale = 1.0 / np.sqrt(d_model)
        # Q projects to all query heads; K/V project to only the KV heads.
        self.Wq = (rng.standard_normal((d_model, n_heads * self.hd)) * scale).astype(np.float32)
        self.Wk = (rng.standard_normal((d_model, n_kv_heads * self.hd)) * scale).astype(np.float32)
        self.Wv = (rng.standard_normal((d_model, n_kv_heads * self.hd)) * scale).astype(np.float32)
        self.Wo = (rng.standard_normal((d_model, d_model)) * scale).astype(np.float32)

    def _split_heads(self, x, n):
        # (B,T,n*hd) -> (B,n,T,hd)
        B, T, _ = x.shape
        return x.reshape(B, T, n, self.hd).transpose(0, 2, 1, 3)

    def forward(self, x):
        B, T, C = x.shape
        self.x = x
        self.B, self.T = B, T
        self.mq, self.mk, self.mv, self.mo = Matmul(), Matmul(), Matmul(), Matmul()

        q = self.mq.forward(x, self.Wq)              # (B,T,n_heads*hd)
        k = self.mk.forward(x, self.Wk)              # (B,T,n_kv*hd)
        v = self.mv.forward(x, self.Wv)

        q = self._split_heads(q, self.n_heads)       # (B,H,T,hd)
        k = self._split_heads(k, self.n_kv_heads)    # (B,Hkv,T,hd)
        v = self._split_heads(v, self.n_kv_heads)

        if self.rope is not None:
            # RoPE expects (...,T,hd) with T on axis -2; our layout has T on -2.
            q = self.rope.apply(q)
            k = self.rope.apply(k)
        self.q, self.k, self.v = q, k, v

        # repeat each KV head `group` times so it lines up with its query group
        self.k_rep = np.repeat(k, self.group, axis=1)   # (B,H,T,hd)
        self.v_rep = np.repeat(v, self.group, axis=1)

        self.scale = 1.0 / np.sqrt(self.hd)
        scores = (q @ self.k_rep.transpose(0, 1, 3, 2)) * self.scale   # (B,H,T,T)
        self.mask = causal_mask(T)
        scores = np.where(self.mask[None, None], scores, -1e9)

        self.sm = Softmax()
        self.att = self.sm.forward(scores)           # (B,H,T,T)
        ctx = self.att @ self.v_rep                  # (B,H,T,hd)

        # merge heads back to (B,T,C)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.hd)
        self.ctx = ctx
        return self.mo.forward(ctx, self.Wo)

    def backward(self, dout):
        B, T = self.B, self.T
        dctx, self.dWo = self.mo.backward(dout)                      # (B,T,C)
        dctx = dctx.reshape(B, T, self.n_heads, self.hd).transpose(0, 2, 1, 3)  # (B,H,T,hd)

        # ctx = att @ v_rep
        datt = dctx @ self.v_rep.transpose(0, 1, 3, 2)              # (B,H,T,T)
        dv_rep = self.att.transpose(0, 1, 3, 2) @ dctx             # (B,H,T,hd)

        dscores = self.sm.backward(datt)
        dscores = np.where(self.mask[None, None], dscores, 0.0) * self.scale

        dq = dscores @ self.k_rep                                  # (B,H,T,hd)
        dk_rep = dscores.transpose(0, 1, 3, 2) @ self.q            # (B,H,T,hd)

        # undo the KV-head repeat: sum the `group` copies back onto each KV head
        dk = dk_rep.reshape(B, self.n_kv_heads, self.group, T, self.hd).sum(axis=2)
        dv = dv_rep.reshape(B, self.n_kv_heads, self.group, T, self.hd).sum(axis=2)

        if self.rope is not None:
            dq = self.rope.backward(dq)
            dk = self.rope.backward(dk)

        # merge heads back and go through the projections
        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.hd)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, self.n_kv_heads * self.hd)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, self.n_kv_heads * self.hd)

        dxq, self.dWq = self.mq.backward(dq)
        dxk, self.dWk = self.mk.backward(dk)
        dxv, self.dWv = self.mv.backward(dv)
        return dxq + dxk + dxv

    def attention_maps(self):
        # return per-query-head maps (B,T,T) to match the MHA interface
        return [self.att[:, h] for h in range(self.n_heads)]

    def params_and_grads(self):
        return [(self.Wq, self.dWq), (self.Wk, self.dWk),
                (self.Wv, self.dWv), (self.Wo, self.dWo)]

    def kv_cache_saving(self):
        """How much smaller the KV-cache is vs full multi-head (a headline
        selling point of GQA). Returns e.g. 4.0 for 8 query / 2 KV heads."""
        return self.n_heads / self.n_kv_heads
