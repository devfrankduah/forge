"""
adapt.py -- apply LoRA to a trained Glassbox transformer and fine-tune it.

This is where Forge composes with Glassbox: we take a fully-trained Glassbox
GPT, FREEZE all of it, and inject small LoRA adapters into the attention
projections. Only the adapters train. This is exactly how you'd cheaply
specialize a pretrained model to a new task.

WHICH WEIGHTS GET ADAPTED
-------------------------
Following the LoRA paper's most common recipe, we adapt the QUERY and VALUE
projections of attention (Wq, Wv) in every head. The intuition: query/value
adaptation is enough to re-aim what the model attends to and what it passes on,
which is most of what task adaptation needs -- while leaving the bulk of the
network (embeddings, MLPs, key projections, output) frozen.

HOW THE FROZEN/TRAINABLE SPLIT WORKS IN BACKWARD
------------------------------------------------
Every layer still runs its normal backward so that gradients PROPAGATE back to
the adapters in earlier layers -- you can't skip the frozen layers, the signal
has to flow through them. But we only COLLECT and update the LoRA parameters;
the frozen weights' gradients are computed-and-discarded (or skipped). The
optimizer is only ever handed the adapter params, so the base is provably
untouched. We assert that at the end of fine-tuning.
"""

from __future__ import annotations

import numpy as np

from .backbone.attention import GroupedQueryAttention, SelfAttentionHead
from .backbone.model import GPT
from .lora import LoRALinear


class LoRAHead(SelfAttentionHead):
    """A Glassbox attention head with LoRA adapters on its Q and V projections.

    It inherits the frozen Wq/Wk/Wv from the base head (we copy the references)
    and adds a LoRALinear-style low-rank update to the q and v paths. Only the
    adapters are trainable.
    """

    def __init__(self, base_head: SelfAttentionHead, rank=4, alpha=8.0, seed=0):
        # reuse the trained (frozen) weights and config from the base head
        self.Wq = base_head.Wq
        self.Wk = base_head.Wk
        self.Wv = base_head.Wv
        self.head_dim = base_head.head_dim
        self.rope = base_head.rope
        # adapters for q and v (k is left un-adapted, per the common recipe)
        self.scaling = alpha / rank
        self.r = rank
        rng = np.random.default_rng(seed)
        d_model = self.Wq.shape[0]
        self.Aq = (rng.standard_normal((d_model, rank)) / np.sqrt(d_model)).astype(np.float32)
        self.Bq = np.zeros((rank, self.head_dim), dtype=np.float32)
        self.Av = (rng.standard_normal((d_model, rank)) / np.sqrt(d_model)).astype(np.float32)
        self.Bv = np.zeros((rank, self.head_dim), dtype=np.float32)

    def forward(self, x):
        from .backbone.tensor_ops import Matmul, Softmax
        B, T, C = x.shape
        self.x = x
        # NOTE ON NAMING: here mq/mk/mv are the Matmul *operators* for the Q/K/V
        # projections. In DoRAHead, `mq`/`mv` mean something completely different
        # (the trainable magnitude vectors). That name clash is a real trap --
        # code that needs to tell the two heads apart must check the head TYPE,
        # not `hasattr(head, "mq")`. (Getting this wrong caused two bugs once.)
        self.mq, self.mk, self.mv = Matmul(), Matmul(), Matmul()

        # base (frozen) projections
        q = self.mq.forward(x, self.Wq)
        k = self.mk.forward(x, self.Wk)
        v = self.mv.forward(x, self.Wv)

        # + LoRA low-rank updates on q and v
        self.xAq = x @ self.Aq
        self.xAv = x @ self.Av
        q = q + (self.xAq @ self.Bq) * self.scaling
        v = v + (self.xAv @ self.Bv) * self.scaling

        if self.rope is not None:
            q = self.rope.apply(q)
            k = self.rope.apply(k)
        self.q, self.k, self.v = q, k, v

        from .backbone.attention import causal_mask
        self.scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 2, 1)) * self.scale
        self.mask = causal_mask(T)
        scores = np.where(self.mask[None], scores, -1e9)
        self.sm = Softmax()
        self.att = self.sm.forward(scores)
        return self.att @ v

    def backward(self, dout):
        # Reverse of forward, step by step. We need dx (so gradients reach the
        # adapters in earlier layers) and the adapter grads dAq/dBq/dAv/dBv. The
        # frozen Wq/Wk/Wv get no grads. dout is (B,T,hd).
        #
        # --- back through attention (same chain rule as a plain head) ---
        # out = att @ v
        datt = dout @ self.v.transpose(0, 2, 1)          # (B,T,T)
        dv = self.att.transpose(0, 2, 1) @ dout          # (B,T,hd)
        # back through softmax (per-row Jacobian lives in Softmax.backward)
        dscores = self.sm.backward(datt)                 # (B,T,T)
        # masked (future) positions contributed nothing; also undo the 1/sqrt(d)
        dscores = np.where(self.mask[None], dscores, 0.0) * self.scale
        # scores = q @ k^T  ->  dq = dscores @ k ;  dk = dscores^T @ q
        dq = dscores @ self.k                            # (B,T,hd)
        dk = dscores.transpose(0, 2, 1) @ self.q         # (B,T,hd)

        # back through RoPE if used (rotation is orthogonal, so the gradient
        # rotates the other way); must mirror the forward's ordering. v is not
        # RoPE'd, so dv passes through untouched.
        if self.rope is not None:
            dq = self.rope.backward(dq)
            dk = self.rope.backward(dk)

        C = self.Wq.shape[0]

        # --- value path: v = x@Wv + scaling*(x@Av)@Bv ---
        # Only the adapter (Av,Bv) is trainable. g folds in the alpha/r scaling.
        # We flatten the (B,T,...) leading dims to (N,...) before the outer
        # products, exactly like a normal Matmul backward.
        gv = dv * self.scaling
        self.dBv = self.xAv.reshape(-1, self.r).T @ gv.reshape(-1, self.head_dim)
        d_xAv = gv @ self.Bv.T                             # grad w.r.t. (x@Av)
        self.dAv = self.x.reshape(-1, C).T @ d_xAv.reshape(-1, self.r)
        # dx gets contributions through BOTH the frozen base and the adapter
        dx_v = dv @ self.Wv.T + d_xAv @ self.Av.T          # frozen base + adapter

        # --- query path: q = x@Wq + scaling*(x@Aq)@Bq (same shape as value) ---
        gq = dq * self.scaling
        self.dBq = self.xAq.reshape(-1, self.r).T @ gq.reshape(-1, self.head_dim)
        d_xAq = gq @ self.Bq.T
        self.dAq = self.x.reshape(-1, C).T @ d_xAq.reshape(-1, self.r)
        dx_q = dq @ self.Wq.T + d_xAq @ self.Aq.T

        # --- key path: not adapted, so only the frozen base feeds dx ---
        dx_k = dk @ self.Wk.T

        return dx_q + dx_k + dx_v

    def params_and_grads(self):
        # ONLY the adapters -- the base Q/K/V/O stay frozen
        return [(self.Aq, self.dAq), (self.Bq, self.dBq),
                (self.Av, self.dAv), (self.Bv, self.dBv)]

    def n_trainable(self):
        return self.Aq.size + self.Bq.size + self.Av.size + self.Bv.size


class DoRAHead(SelfAttentionHead):
    """An attention head fine-tuned with DoRA (weight-decomposed) adapters on Q,V.

    Same idea as LoRAHead, but each adapted projection is composed the DoRA way:
    the frozen weight is split into direction and magnitude, the low-rank update
    is applied to the direction, the columns are renormalized, and a separately
    trained magnitude vector rescales them (see forge.dora). Trainable per head:
    Aq,Bq,mq and Av,Bv,mv. The key projection stays frozen, as with LoRAHead.
    """

    def __init__(self, base_head: SelfAttentionHead, rank=4, alpha=8.0, seed=0):
        from .dora import _col_norm
        self.Wq = base_head.Wq
        self.Wk = base_head.Wk
        self.Wv = base_head.Wv
        self.head_dim = base_head.head_dim
        self.rope = base_head.rope
        self.scaling = alpha / rank
        self.r = rank
        rng = np.random.default_rng(seed)
        d_model = self.Wq.shape[0]
        # low-rank factors (B zero) + magnitude vectors initialized to the base
        # column norms, so at init the composed weight equals the base exactly.
        # NOTE ON NAMING: here `mq`/`mv` are the trainable MAGNITUDE VECTORS. In
        # LoRAHead the same names mean the Matmul operators instead -- so tell
        # the head types apart by isinstance(...), never by attribute name.
        self.Aq = (rng.standard_normal((d_model, rank)) / np.sqrt(d_model)).astype(np.float32)
        self.Bq = np.zeros((rank, self.head_dim), dtype=np.float32)
        self.mq = _col_norm(self.Wq).astype(np.float32)
        self.Av = (rng.standard_normal((d_model, rank)) / np.sqrt(d_model)).astype(np.float32)
        self.Bv = np.zeros((rank, self.head_dim), dtype=np.float32)
        self.mv = _col_norm(self.Wv).astype(np.float32)

    def forward(self, x):
        from .backbone.attention import causal_mask
        from .backbone.tensor_ops import Softmax
        from .dora import dora_compose
        B, T, C = x.shape
        self.x = x

        # DoRA-composed Q and V projection weights (K stays frozen)
        self.Wp_q, self.Vq, self.nq = dora_compose(self.Wq, self.Aq, self.Bq, self.mq, self.scaling)
        self.Wp_v, self.Vv, self.nv = dora_compose(self.Wv, self.Av, self.Bv, self.mv, self.scaling)
        q = x @ self.Wp_q
        v = x @ self.Wp_v
        k = x @ self.Wk

        if self.rope is not None:
            q = self.rope.apply(q)
            k = self.rope.apply(k)
        self.q, self.k, self.v = q, k, v

        self.scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 2, 1)) * self.scale
        self.mask = causal_mask(T)
        scores = np.where(self.mask[None], scores, -1e9)
        self.sm = Softmax()
        self.att = self.sm.forward(scores)
        return self.att @ v

    def backward(self, dout):
        from .dora import dora_grads
        # --- back through attention (identical chain rule to a plain head) ---
        # out = att @ v ;  then softmax ;  then scores = q @ k^T. Shapes noted.
        datt = dout @ self.v.transpose(0, 2, 1)          # (B,T,T)
        dv = self.att.transpose(0, 2, 1) @ dout          # (B,T,hd)
        dscores = self.sm.backward(datt)                 # (B,T,T)
        dscores = np.where(self.mask[None], dscores, 0.0) * self.scale
        dq = dscores @ self.k                            # (B,T,hd)
        dk = dscores.transpose(0, 2, 1) @ self.q         # (B,T,hd)

        # back through RoPE if used (mirror forward; v is not RoPE'd)
        if self.rope is not None:
            dq = self.rope.backward(dq)
            dk = self.rope.backward(dk)

        C = self.Wq.shape[0]

        # --- query path: q = x @ Wp_q, where Wp_q is the DoRA-composed weight ---
        # First get the gradient w.r.t. that composed weight (dWp_q) and w.r.t.
        # the input (dx_q), treating Wp_q like an ordinary linear weight. Then
        # dora_grads() pushes dWp_q back through the magnitude/direction
        # decomposition to the actual trainables (Aq, Bq, and magnitude mq).
        dq2 = dq.reshape(-1, self.head_dim)              # flatten (B,T,hd)->(N,hd)
        x2 = self.x.reshape(-1, C)                       # (N, C)
        dWp_q = x2.T @ dq2                               # (C, head_dim)
        dx_q = dq @ self.Wp_q.T                          # input grad through Q
        self.dAq, self.dBq, self.dmq = dora_grads(dWp_q, self.Vq, self.nq, self.mq,
                                                  self.Aq, self.Bq, self.scaling)

        # --- value path: v = x @ Wp_v (same two-step structure as query) ---
        dv2 = dv.reshape(-1, self.head_dim)
        dWp_v = x2.T @ dv2
        dx_v = dv @ self.Wp_v.T
        self.dAv, self.dBv, self.dmv = dora_grads(dWp_v, self.Vv, self.nv, self.mv,
                                                  self.Av, self.Bv, self.scaling)

        # --- key path: not adapted, only the frozen base feeds dx ---
        dx_k = dk @ self.Wk.T

        return dx_q + dx_k + dx_v

    def params_and_grads(self):
        return [(self.Aq, self.dAq), (self.Bq, self.dBq), (self.mq, self.dmq),
                (self.Av, self.dAv), (self.Bv, self.dBv), (self.mv, self.dmv)]

    def n_trainable(self):
        return (self.Aq.size + self.Bq.size + self.mq.size +
                self.Av.size + self.Bv.size + self.mv.size)


class LoRAGroupedQueryAttention(GroupedQueryAttention):
    """A Grouped-Query Attention layer with LoRA adapters on its Q and V
    projections -- the GQA analogue of LoRAHead.

    Unlike MHA (whose Q/K/V weights live per-head), a GQA layer keeps ONE set of
    layer-level projections: Wq projects to ALL query heads, while Wk/Wv project
    to the smaller number of shared KV heads. So the adapters live on the
    layer's Wq (full query width) and Wv (KV width), following the same Q,V
    recipe LoRAHead uses. The base Wq/Wk/Wv/Wo stay frozen (their references are
    shared with the base layer); only Aq,Bq,Av,Bv train. B is zero-initialized,
    so apply_lora starts as an exact no-op, exactly like MHA.
    """

    def __init__(self, base_attn: GroupedQueryAttention, rank=4, alpha=8.0, seed=0):
        # reuse the trained (frozen) weights and config from the base layer
        self.d_model = base_attn.d_model
        self.n_heads = base_attn.n_heads
        self.n_kv_heads = base_attn.n_kv_heads
        self.group = base_attn.group
        self.hd = base_attn.hd
        self.rope = base_attn.rope
        self.Wq = base_attn.Wq
        self.Wk = base_attn.Wk
        self.Wv = base_attn.Wv
        self.Wo = base_attn.Wo
        # adapters on the Q (all query heads) and V (KV heads) projections; K and
        # O are left un-adapted, matching LoRAHead's common recipe.
        self.scaling = alpha / rank
        self.r = rank
        rng = np.random.default_rng(seed)
        d_model = self.d_model
        q_out = self.n_heads * self.hd
        v_out = self.n_kv_heads * self.hd
        self.Aq = (rng.standard_normal((d_model, rank)) / np.sqrt(d_model)).astype(np.float32)
        self.Bq = np.zeros((rank, q_out), dtype=np.float32)
        self.Av = (rng.standard_normal((d_model, rank)) / np.sqrt(d_model)).astype(np.float32)
        self.Bv = np.zeros((rank, v_out), dtype=np.float32)

    def forward(self, x):
        from .backbone.attention import causal_mask
        from .backbone.tensor_ops import Softmax
        B, T, C = x.shape
        self.x = x
        self.B, self.T = B, T

        # base (frozen) projections; K and O carry no adapter
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        # + LoRA low-rank updates on q and v, added in the pre-split projection
        # space, exactly like the base projection they correct.
        self.xAq = x @ self.Aq
        self.xAv = x @ self.Av
        q = q + (self.xAq @ self.Bq) * self.scaling
        v = v + (self.xAv @ self.Bv) * self.scaling

        q = self._split_heads(q, self.n_heads)       # (B,H,T,hd)
        k = self._split_heads(k, self.n_kv_heads)    # (B,Hkv,T,hd)
        v = self._split_heads(v, self.n_kv_heads)

        if self.rope is not None:
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
        return ctx @ self.Wo

    def backward(self, dout):
        # Mirror GroupedQueryAttention.backward, but the frozen Wq/Wk/Wv/Wo get
        # NO grads; we collect the adapter grads dAq/dBq/dAv/dBv plus the dx that
        # must flow to earlier layers (through BOTH the frozen base and adapters).
        B, T = self.B, self.T

        # out = ctx @ Wo (Wo frozen -> no dWo)
        dctx = dout @ self.Wo.T                                   # (B,T,C)
        dctx = dctx.reshape(B, T, self.n_heads, self.hd).transpose(0, 2, 1, 3)  # (B,H,T,hd)

        # ctx = att @ v_rep
        datt = dctx @ self.v_rep.transpose(0, 1, 3, 2)           # (B,H,T,T)
        dv_rep = self.att.transpose(0, 1, 3, 2) @ dctx           # (B,H,T,hd)

        dscores = self.sm.backward(datt)
        dscores = np.where(self.mask[None, None], dscores, 0.0) * self.scale

        dq = dscores @ self.k_rep                                # (B,H,T,hd)
        dk_rep = dscores.transpose(0, 1, 3, 2) @ self.q          # (B,H,T,hd)

        # undo the KV-head repeat: sum the `group` copies back onto each KV head
        dk = dk_rep.reshape(B, self.n_kv_heads, self.group, T, self.hd).sum(axis=2)
        dv = dv_rep.reshape(B, self.n_kv_heads, self.group, T, self.hd).sum(axis=2)

        if self.rope is not None:
            dq = self.rope.backward(dq)
            dk = self.rope.backward(dk)

        # merge heads back to the (B,T,*) projection layout
        dq = dq.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.hd)     # (B,T,q_out)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, T, self.n_kv_heads * self.hd)  # (B,T,v_out)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, T, self.n_kv_heads * self.hd)  # (B,T,v_out)

        C = self.d_model
        q_out = self.n_heads * self.hd
        v_out = self.n_kv_heads * self.hd

        # --- query path: q = x@Wq + scaling*(x@Aq)@Bq (base frozen; adapt Aq,Bq) ---
        gq = dq * self.scaling
        self.dBq = self.xAq.reshape(-1, self.r).T @ gq.reshape(-1, q_out)
        d_xAq = gq @ self.Bq.T                                    # grad w.r.t. (x@Aq)
        self.dAq = self.x.reshape(-1, C).T @ d_xAq.reshape(-1, self.r)
        dx_q = dq @ self.Wq.T + d_xAq @ self.Aq.T                 # frozen base + adapter

        # --- value path: v = x@Wv + scaling*(x@Av)@Bv (same shape logic) ---
        gv = dv * self.scaling
        self.dBv = self.xAv.reshape(-1, self.r).T @ gv.reshape(-1, v_out)
        d_xAv = gv @ self.Bv.T
        self.dAv = self.x.reshape(-1, C).T @ d_xAv.reshape(-1, self.r)
        dx_v = dv @ self.Wv.T + d_xAv @ self.Av.T

        # --- key path: not adapted, only the frozen base feeds dx ---
        dx_k = dk @ self.Wk.T

        return dx_q + dx_k + dx_v

    def params_and_grads(self):
        # ONLY the adapters -- the base Q/K/V/O stay frozen
        return [(self.Aq, self.dAq), (self.Bq, self.dBq),
                (self.Av, self.dAv), (self.Bv, self.dBv)]

    def n_trainable(self):
        return self.Aq.size + self.Bq.size + self.Av.size + self.Bv.size


def apply_lora(base_model: GPT, rank=4, alpha=8.0, seed=0, method="lora"):
    """Wrap a trained GPT with adapters on every attention layer's Q,V.

    method="lora" installs LoRA adapters; method="dora" installs DoRAHead
    (weight-decomposed). Returns the SAME model object, mutated in place. For an
    MHA backbone each per-head SelfAttentionHead is replaced by an adapter head;
    for a GroupedQueryAttention backbone the whole attention layer is replaced by
    a LoRAGroupedQueryAttention (GQA has layer-level, not per-head, projections).
    Either way the adapter shares the frozen weights, the model's
    forward/backward run unchanged, and only the adapters carry gradients we
    keep. Both adapter types are gradient-checked (see the tests).
    """
    method = method.lower()
    if method not in ("lora", "dora"):
        raise ValueError(f"method must be 'lora' or 'dora', got {method!r}")
    Head = DoRAHead if method == "dora" else LoRAHead
    h = 0
    for blk in base_model.blocks:
        attn = blk.attn
        if isinstance(attn, GroupedQueryAttention):
            # GQA keeps layer-level Q/K/V/O (no per-head `.heads`), so we adapt
            # the whole attention module rather than iterating heads.
            if method == "dora":
                raise NotImplementedError(
                    "DoRA adapters are not implemented for GroupedQueryAttention; "
                    "use method='lora' on a GQA backbone")
            blk.attn = LoRAGroupedQueryAttention(attn, rank=rank, alpha=alpha, seed=seed + h)
            h += 1
        else:
            new_heads = []
            for head in attn.heads:
                new_heads.append(Head(head, rank=rank, alpha=alpha, seed=seed + h))
                h += 1
            attn.heads = new_heads
    base_model._lora = True
    base_model._lora_cfg = {"rank": rank, "alpha": alpha, "method": method}
    return base_model


def lora_params_and_grads(model: GPT):
    """Collect ONLY the LoRA adapter parameters across the model. This is what
    we hand the optimizer, guaranteeing the frozen base is never updated."""
    pg = []
    for blk in model.blocks:
        attn = blk.attn
        if isinstance(attn, GroupedQueryAttention):
            pg.extend(attn.params_and_grads())
        else:
            for head in attn.heads:
                pg.extend(head.params_and_grads())
    return pg


def count_params(model: GPT):
    """Return (trainable_lora, total_base) parameter counts -- the headline
    efficiency number for a LoRA fine-tune.

    `total_base` counts EVERY base parameter (embeddings, all attention weights,
    the attention output projection, every MLP weight, every norm's scale/bias,
    and the output projection), so the trainable/total ratio is honest. An
    earlier version omitted the MLP and norm params, which understated the total
    and made LoRA look *less* efficient than it is -- fixed here."""
    trainable = 0
    for blk in model.blocks:
        attn = blk.attn
        if isinstance(attn, GroupedQueryAttention):
            trainable += attn.n_trainable()
        else:
            for head in attn.heads:
                trainable += head.n_trainable()

    def _norm_size(norm):
        n = norm.gamma.size
        if hasattr(norm, "beta"):          # LayerNorm has a bias; RMSNorm doesn't
            n += norm.beta.size
        return n

    def _ff_size(ff):
        return sum(getattr(ff, a).size for a in
                   ("W1", "b1", "W2", "b2", "Wg", "Wu", "Wd") if hasattr(ff, a))

    total = model.tok_emb.size
    if model.pos_emb is not None:
        total += model.pos_emb.size
    if getattr(model, "Wout", None) is not None:
        total += model.Wout.size
    total += _norm_size(model.ln_f)
    for blk in model.blocks:
        attn = blk.attn
        if isinstance(attn, GroupedQueryAttention):
            total += attn.Wq.size + attn.Wk.size + attn.Wv.size
        else:
            for head in attn.heads:
                total += head.Wq.size + head.Wk.size + head.Wv.size
        total += attn.Wo.size
        total += _norm_size(blk.ln1) + _norm_size(blk.ln2)
        total += _ff_size(blk.ff)
    return trainable, total
