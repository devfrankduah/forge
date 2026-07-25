# Vendored into Forge so the project is self-contained (no external
# dependency). Same code as the companion Glassbox project. NumPy only.
"""
model.py -- assemble the full transformer from verified parts.

ARCHITECTURE (a small GPT-style decoder):

    tokens --> token embedding + positional embedding
           --> [ Transformer Block ] x N
           --> final LayerNorm
           --> projection to vocabulary (logits)

Each Transformer Block is the classic residual sandwich:

    x = x + MultiHeadAttention(LayerNorm(x))     # attention sublayer
    x = x + FeedForward(LayerNorm(x))            # MLP sublayer

The residual connections (the `x +`) are what let gradients flow through many
layers without vanishing -- arguably the reason deep transformers train at all.
The "pre-norm" placement (LayerNorm INSIDE the residual, before each sublayer)
is what modern GPTs use because it trains more stably than the original
post-norm; knowing that distinction is a nice signal of current understanding.

Everything here is built from the gradient-checked ops in tensor_ops.py and
attention.py, so the whole model's backward is correct by construction.
"""

from __future__ import annotations

import numpy as np

from .tensor_ops import Matmul, Bias, LayerNorm, GELU, CrossEntropy, RMSNorm
from .attention import MultiHeadAttention, GroupedQueryAttention
from .rope import RoPE


def _make_norm(kind, dim):
    """Factory: pick the normalization layer by name."""
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return LayerNorm(dim)
    raise ValueError(f"unknown norm {kind!r}; use 'layernorm' or 'rmsnorm'")


class FeedForward:
    """Position-wise MLP: Linear -> GELU -> Linear, with a 4x hidden width
    (the standard expansion ratio). Applied identically to every position.

    This is the GPT-2-era feed-forward. See SwiGLU below for the modern one."""

    def __init__(self, d_model, rng, mult=4):
        hidden = d_model * mult
        s1 = 1.0 / np.sqrt(d_model)
        s2 = 1.0 / np.sqrt(hidden)
        self.W1 = (rng.standard_normal((d_model, hidden)) * s1).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, d_model)) * s2).astype(np.float32)
        self.b2 = np.zeros((d_model,), dtype=np.float32)

    def forward(self, x):
        self.m1, self.a1, self.g, self.m2, self.a2 = (
            Matmul(), Bias(), GELU(), Matmul(), Bias())
        h = self.a1.forward(self.m1.forward(x, self.W1), self.b1)
        h = self.g.forward(h)
        out = self.a2.forward(self.m2.forward(h, self.W2), self.b2)
        return out

    def backward(self, dout):
        dh, self.db2 = self.a2.backward(dout)
        dh, self.dW2 = self.m2.backward(dh)
        dh = self.g.backward(dh)
        dx, self.db1 = self.a1.backward(dh)
        dx, self.dW1 = self.m1.backward(dx)
        return dx

    def params_and_grads(self):
        return [(self.W1, self.dW1), (self.b1, self.db1),
                (self.W2, self.dW2), (self.b2, self.db2)]


class SwiGLU:
    """The modern feed-forward, used by Llama and PaLM. A *gated* MLP.

    Instead of one hidden projection through an activation, SwiGLU uses TWO
    projections and multiplies them, with one side passed through SiLU/Swish:

        h = silu(x @ W_gate) * (x @ W_up)      # elementwise gate
        out = h @ W_down

    where silu(z) = z * sigmoid(z). The intuition: the `silu(x @ W_gate)` branch
    acts as a learned, input-dependent gate deciding how much of each hidden
    unit of the `x @ W_up` branch to let through -- strictly more expressive
    than a fixed activation. Empirically it trains better, which is why it
    replaced GELU-MLPs in modern models.

    Because there are two input projections, the hidden width is usually scaled
    to ~2/3 * (4 * d_model) to keep the parameter count comparable. We expose
    that so the comparison to the GELU FF is fair.

    Backward is a good exercise in the product rule: out depends on h, and h is
    a product of two branches that BOTH depend on x, so dx gets a contribution
    from each. Derived below and gradient-checked.
    """

    def __init__(self, d_model, rng, mult=4):
        # 2/3 rule keeps params ~equal to a 4x GELU FF despite the extra matrix
        hidden = int((2 * (d_model * mult)) / 3)
        hidden = max(1, hidden)
        s_in = 1.0 / np.sqrt(d_model)
        s_out = 1.0 / np.sqrt(hidden)
        self.Wg = (rng.standard_normal((d_model, hidden)) * s_in).astype(np.float32)  # gate
        self.Wu = (rng.standard_normal((d_model, hidden)) * s_in).astype(np.float32)  # up
        self.Wd = (rng.standard_normal((hidden, d_model)) * s_out).astype(np.float32) # down
        self.d_model = d_model
        self.hidden = hidden

    @staticmethod
    def _sigmoid(z):
        # Numerically stable sigmoid. The naive 1/(1+exp(-z)) overflows for
        # large negative z (exp(-z) -> inf). Branch on sign so we only ever
        # exponentiate a non-positive number:
        #   z >= 0:  1 / (1 + exp(-z))
        #   z <  0:  exp(z) / (1 + exp(z))
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def forward(self, x):
        self.x = x
        self.mg, self.mu, self.md = Matmul(), Matmul(), Matmul()
        self.g_pre = self.mg.forward(x, self.Wg)        # x @ Wg
        self.u = self.mu.forward(x, self.Wu)            # x @ Wu
        self.sig = self._sigmoid(self.g_pre)
        self.silu = self.g_pre * self.sig               # silu(g_pre)
        self.h = self.silu * self.u                     # gated hidden
        return self.md.forward(self.h, self.Wd)         # h @ Wd

    def backward(self, dout):
        dh, self.dWd = self.md.backward(dout)           # grad into h

        # h = silu * u  ->  product rule
        d_silu = dh * self.u
        d_u = dh * self.silu

        # silu = g_pre * sigmoid(g_pre); d silu / d g_pre = sig + g_pre*sig*(1-sig)
        d_gpre = d_silu * (self.sig + self.g_pre * self.sig * (1.0 - self.sig))

        # back through the two input projections; x gets both contributions
        dx_g, self.dWg = self.mg.backward(d_gpre)
        dx_u, self.dWu = self.mu.backward(d_u)
        return dx_g + dx_u

    def params_and_grads(self):
        return [(self.Wg, self.dWg), (self.Wu, self.dWu), (self.Wd, self.dWd)]


class Block:
    """One transformer block: pre-norm attention + pre-norm feed-forward, each
    wrapped in a residual connection.

    Configurable so it can be either generation of transformer:
      - norm='layernorm' (GPT-2) or 'rmsnorm' (Llama)
      - ff='gelu' (GPT-2) or 'swiglu' (Llama)
      - rope: an optional RoPE applied inside attention (Llama) or None (GPT-2)

    The two norms have different backward signatures (LayerNorm has a beta/bias
    gradient, RMSNorm doesn't), so we store each norm's grads in a small dict
    keyed by parameter, which keeps params_and_grads() uniform regardless of
    which norm is in use. That's a deliberate design choice to avoid sprinkling
    `if isinstance(...)` all over the backward pass.
    """

    def __init__(self, d_model, n_heads, rng, norm="layernorm", ff="gelu",
                 rope=None, n_kv_heads=None):
        self.ln1 = _make_norm(norm, d_model)
        # Grouped-Query Attention when n_kv_heads is set and smaller than
        # n_heads; otherwise ordinary multi-head attention.
        if n_kv_heads is not None and n_kv_heads != n_heads:
            self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, rng, rope=rope)
        else:
            self.attn = MultiHeadAttention(d_model, n_heads, rng, rope=rope)
        self.ln2 = _make_norm(norm, d_model)
        self.ff = SwiGLU(d_model, rng) if ff == "swiglu" else FeedForward(d_model, rng)
        self._norm_grads = {}   # per-instance; filled during backward

    def forward(self, x):
        # residual 1: x + attn(norm1(x))
        a = self.attn.forward(self.ln1.forward(x))
        x = x + a
        # residual 2: x + ff(norm2(x))
        f = self.ff.forward(self.ln2.forward(x))
        return x + f

    def backward(self, dout):
        # Two residual connections, unwound in reverse. A skip y = x + f(x)
        # sends gradient to BOTH branches, so each d_x is (through sublayer) +
        # (through skip). _norm_backward hides the LayerNorm-vs-RMSNorm
        # signature difference and records the norm's param grads.

        # ---- second residual: out = x_mid + ff(norm2(x_mid)) ----
        df = self.ff.backward(dout)
        d_ln2_in = self._norm_backward(self.ln2, df, "ln2")
        d_xmid = dout + d_ln2_in

        # ---- first residual: x_mid = x + attn(norm1(x)) ----
        d_ln1_out = self.attn.backward(d_xmid)
        d_ln1_in = self._norm_backward(self.ln1, d_ln1_out, "ln1")
        dx = d_xmid + d_ln1_in
        return dx

    def _norm_backward(self, norm, dy, tag):
        """Run a norm's backward and stash its parameter gradients uniformly."""
        out = norm.backward(dy)
        if isinstance(norm, LayerNorm):
            dx, dgamma, dbeta = out
            self._norm_grads[tag] = [(norm.gamma, dgamma), (norm.beta, dbeta)]
        else:  # RMSNorm: no beta
            dx, dgamma = out
            self._norm_grads[tag] = [(norm.gamma, dgamma)]
        return dx

    def params_and_grads(self):
        pg = []
        pg.extend(self._norm_grads.get("ln1", []))
        pg.extend(self.attn.params_and_grads())
        pg.extend(self._norm_grads.get("ln2", []))
        pg.extend(self.ff.params_and_grads())
        return pg

    def attention_maps(self):
        return self.attn.attention_maps()

    # KV-cache path (inference only)
    def reset_cache(self):
        self.attn.reset_cache()

    def forward_cached(self, x_last, pos, max_context=None):
        # same structure as forward(), but attention uses its KV-cache and we
        # process only the newest position. Norms/FF are position-wise so they
        # work unchanged on a length-1 sequence.
        a = self.attn.forward_cached(self.ln1.forward(x_last), pos, max_context=max_context)
        x_last = x_last + a
        f = self.ff.forward(self.ln2.forward(x_last))
        return x_last + f


class GPT:
    """The full model, configurable between two generations of architecture.

    Presets via `arch`:
      arch='gpt2'  -> LayerNorm, learned positional embeddings, GELU MLP
      arch='llama' -> RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU MLP,
                      and weight tying (share input embedding with output head)

    Being able to instantiate BOTH and explain every difference -- why RMSNorm,
    why RoPE instead of learned positions, why SwiGLU, why tie weights -- is the
    point of this project: it shows understanding of how transformers actually
    evolved, not just one fixed recipe.

    Individual options can also be set explicitly to mix and match.
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 block_size=32, seed=0, arch="gpt2",
                 norm=None, ff=None, use_rope=None, tie_weights=None,
                 n_kv_heads=None):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        # resolve the preset, letting explicit kwargs override it
        if arch == "llama":
            norm = norm or "rmsnorm"
            ff = ff or "swiglu"
            use_rope = True if use_rope is None else use_rope
            tie_weights = True if tie_weights is None else tie_weights
        else:  # gpt2
            norm = norm or "layernorm"
            ff = ff or "gelu"
            use_rope = False if use_rope is None else use_rope
            tie_weights = False if tie_weights is None else tie_weights
        self.arch = arch
        self.norm_kind = norm
        self.ff_kind = ff
        self.use_rope = use_rope
        self.tie_weights = tie_weights

        # token embedding table (always present)
        self.tok_emb = (rng.standard_normal((vocab_size, d_model)) * 0.02).astype(np.float32)

        # positions: EITHER a learned positional embedding table (GPT-2) OR
        # RoPE applied inside attention (Llama). Not both.
        if use_rope:
            head_dim = d_model // n_heads
            self.rope = RoPE(head_dim, max_seq=block_size)
            self.pos_emb = None
        else:
            self.rope = None
            self.pos_emb = (rng.standard_normal((block_size, d_model)) * 0.02).astype(np.float32)

        self.blocks = [Block(d_model, n_heads, rng, norm=norm, ff=ff, rope=self.rope,
                             n_kv_heads=n_kv_heads)
                       for _ in range(n_layers)]
        self.n_kv_heads = n_kv_heads
        self.ln_f = _make_norm(norm, d_model)

        # final projection to vocabulary logits
        self.head = Matmul()
        if tie_weights:
            # weight tying: reuse the token embedding as the output projection
            # (transposed). Fewer parameters and a well-known regularizer --
            # the intuition is that "which token is this embedding near?" and
            # "predict this token" are the same geometry. The embedding is
            # (vocab, d_model); the head needs (d_model, vocab), so we treat the
            # embedding's transpose as the projection and add the grads together.
            self.Wout = None   # signals tied mode
        else:
            self.Wout = (rng.standard_normal((d_model, vocab_size))
                         * (1 / np.sqrt(d_model))).astype(np.float32)

    def _out_weight(self):
        # the effective (d_model, vocab) projection, tied or not
        return self.tok_emb.T if self.tie_weights else self.Wout

    def forward(self, idx, targets=None):
        # idx: (B, T) integer token ids
        B, T = idx.shape
        self.idx = idx
        x = self.tok_emb[idx]                          # (B,T,C)
        if self.pos_emb is not None:                   # GPT-2: add learned pos
            x = x + self.pos_emb[:T][None]
        # (Llama: positions are injected via RoPE inside attention, nothing here)
        for blk in self.blocks:
            x = blk.forward(x)
        x = self.ln_f.forward(x)
        self.x_final = x
        logits = self.head.forward(x, self._out_weight())   # (B,T,vocab)

        loss = None
        if targets is not None:
            self.ce = CrossEntropy()
            loss = self.ce.forward(logits.reshape(B * T, -1), targets.reshape(B * T))
            self._BT = (B, T)
        return logits, loss

    def backward(self):
        B, T = self._BT
        dlogits = self.ce.backward().reshape(B, T, -1)      # (B,T,vocab)
        dx, dWout = self.head.backward(dlogits)

        # final norm (uniform LayerNorm/RMSNorm handling)
        out = self.ln_f.backward(dx)
        if isinstance(self.ln_f, LayerNorm):
            dx, self.dgf, self.dbf = out
            self._lnf_grads = [(self.ln_f.gamma, self.dgf), (self.ln_f.beta, self.dbf)]
        else:
            dx, self.dgf = out
            self._lnf_grads = [(self.ln_f.gamma, self.dgf)]

        for blk in reversed(self.blocks):
            dx = blk.backward(dx)

        # token-embedding gradient: scatter-add from the input lookup...
        self.dtok = np.zeros_like(self.tok_emb)
        np.add.at(self.dtok, self.idx, dx)
        # ...plus, if weights are tied, the gradient flowing through the output
        # projection (which IS the embedding, transposed). dWout is
        # (d_model, vocab); its transpose matches tok_emb (vocab, d_model).
        if self.tie_weights:
            self.dtok = self.dtok + dWout.T
            self.dWout = None
        else:
            self.dWout = dWout

        if self.pos_emb is not None:
            self.dpos = np.zeros_like(self.pos_emb)
            self.dpos[:T] += dx.sum(axis=0)
        else:
            self.dpos = None
        return None

    def params_and_grads(self):
        pg = [(self.tok_emb, self.dtok)]
        if self.pos_emb is not None:
            pg.append((self.pos_emb, self.dpos))
        if not self.tie_weights:
            pg.append((self.Wout, self.dWout))
        pg.extend(self._lnf_grads)
        for blk in self.blocks:
            pg.extend(blk.params_and_grads())
        return pg

    def attention_maps(self):
        """List over layers, each a list over heads of (B,T,T) weights."""
        return [blk.attn.attention_maps() for blk in self.blocks]

    # ------------------------------------------------------------------
    # KV-cached generation (fast inference)
    # ------------------------------------------------------------------
    def reset_cache(self):
        for blk in self.blocks:
            blk.reset_cache()

    def forward_cached(self, idx_last, pos):
        """Forward ONE new token at absolute position `pos`. idx_last: (B,1)."""
        x = self.tok_emb[idx_last]                      # (B,1,C)
        if self.pos_emb is not None:                    # GPT-2 learned position
            # Learned positional embeddings only exist for positions
            # 0..block_size-1. Past that, clamp to the last slot (the GPT-2
            # design simply can't represent positions beyond its trained window;
            # RoPE, by contrast, extends naturally -- a concrete reason RoPE
            # replaced learned positions in modern models).
            p = min(pos, self.block_size - 1)
            x = x + self.pos_emb[p][None, None]
        for blk in self.blocks:
            x = blk.forward_cached(x, pos, max_context=self.block_size)
        x = self.ln_f.forward(x)
        logits = self.head.forward(x, self._out_weight())   # (B,1,vocab)
        return logits
