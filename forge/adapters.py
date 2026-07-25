"""
adapters.py -- save, load, and hot-swap LoRA adapters, plus honest evaluation.

WHY THIS MATTERS (the production story)
---------------------------------------
The real operational payoff of LoRA is this: you keep ONE frozen base model in
memory and swap small adapters in and out to switch tasks/styles/customers.
Each adapter is a few hundred KB, so you can store thousands and load whichever
you need per request. This module implements that lifecycle:

    save_adapters(model, path)      -> write just the adapters to a .npz
    load_adapters(model, path)      -> load them onto a frozen base
    swap-in is just load_adapters with a different file

and, in the Crucible spirit of honest measurement:

    perplexity(model, dataset, text)   -> exp(mean cross-entropy) on held-out text

Perplexity is the standard language-model metric: loosely, "how many equally-
likely choices is the model deciding among per token" -- lower is better. We
report it on a HELD-OUT slice so it measures generalization, not memorization.
"""

from __future__ import annotations

import numpy as np


def _iter_head_adapters(model):
    """Yield (name, owner, attr) for every adapter parameter in the model, with
    a stable name so save/load line up regardless of iteration order.

    The attributes are the adapters that `apply_lora` installs on each attention
    head: the low-rank factors (Aq,Bq,Av,Bv), plus the DoRA magnitude vectors
    (mq,mv) when the model was adapted with method='dora'. We detect DoRA by the
    head type (not by attribute name -- a LoRA head reuses 'mq' internally as a
    Matmul op after a forward pass, so a name check would misfire)."""
    from .adapt import DoRAHead
    for li, blk in enumerate(model.blocks):
        for hi, head in enumerate(blk.attn.heads):
            names = ["Aq", "Bq", "Av", "Bv"]
            if isinstance(head, DoRAHead):
                names += ["mq", "mv"]
            for attr in names:
                yield f"L{li}_H{hi}_{attr}", head, attr


def save_adapters(model, path):
    """Write ONLY the adapter arrays (not the frozen base) to a .npz file."""
    # Guard BEFORE touching any adapter attribute. On an un-adapted model the
    # heads are plain SelfAttentionHead/GroupedQueryAttention with no
    # Aq/Bq/Av/Bv, so building `arrays` would raise an opaque AttributeError
    # ("no attribute 'Aq'") before the `if not arrays` check could ever fire.
    # `_lora` is set only by apply_lora, making it the authoritative
    # "adapters installed" signal -- detect its absence here.
    if not getattr(model, "_lora", False):
        raise ValueError("model has no LoRA adapters; call apply_lora first")
    arrays = {name: getattr(owner, attr) for name, owner, attr in _iter_head_adapters(model)}
    if not arrays:
        raise ValueError("model has no LoRA adapters; call apply_lora first")
    # store the config too, so a loader can sanity-check compatibility
    meta = np.array([len(model.blocks), len(model.blocks[0].attn.heads)], dtype=np.int64)
    np.savez(path, __meta__=meta, **arrays)
    return path


def load_adapters(model, path):
    """Load adapter arrays from a .npz onto a model that already has adapters
    of the matching shape (i.e. apply_lora was called with the same rank)."""
    data = np.load(path if str(path).endswith(".npz") else str(path) + ".npz")
    loaded = 0
    for name, owner, attr in _iter_head_adapters(model):
        if name not in data:
            raise KeyError(f"adapter file missing {name}; incompatible model?")
        arr = data[name]
        cur = getattr(owner, attr)
        if arr.shape != cur.shape:
            raise ValueError(f"{name}: shape {arr.shape} != expected {cur.shape}")
        setattr(owner, attr, arr.astype(cur.dtype))
        loaded += 1
    return loaded


def adapter_nbytes(model):
    """Total bytes of the adapter parameters -- how small a swappable adapter is."""
    return int(sum(getattr(owner, attr).nbytes
                   for _, owner, attr in _iter_head_adapters(model)))


def perplexity(model, dataset, text, block_size=None, stride=None):
    """Perplexity = exp(mean per-token cross-entropy) on `text`.

    We slide a window over the text and average the loss. Lower is better; a
    model that has adapted to this style should score lower here than one that
    hasn't. Uses the model's own forward (no grad needed)."""
    block = block_size or model.block_size
    stride = stride or block
    data = dataset.encode(text)
    if len(data) <= block + 1:
        raise ValueError("text too short for the block size")

    total_loss = 0.0
    total_tok = 0
    for start in range(0, len(data) - block - 1, stride):
        x = data[start:start + block][None]
        y = data[start + 1:start + 1 + block][None]
        _, loss = model.forward(x, y)          # mean CE over this window
        total_loss += float(loss) * block      # weight by tokens in the window
        total_tok += block
    mean_ce = total_loss / max(1, total_tok)
    return float(np.exp(mean_ce))
