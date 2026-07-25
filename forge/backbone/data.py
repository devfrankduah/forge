# Vendored into Forge so the project is self-contained (no external
# dependency). Same code as the companion Glassbox project. NumPy only.
"""
data.py + train.py combined -- a character-level dataset and the training loop.

We train on plain text at the CHARACTER level: the vocabulary is just the set
of distinct characters, each mapped to an integer. It's the simplest possible
setup that still exercises the entire transformer, and it learns fast enough to
show real progress on a CPU in minutes. The model's job: given a run of
characters, predict the next one.

`generate()` samples from the trained model one character at a time, feeding
each prediction back in -- this is exactly how GPT produces text, just tiny.
"""

from __future__ import annotations

import numpy as np


class CharDataset:
    """Turns a string into integer sequences and back."""

    def __init__(self, text, block_size):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = np.array([self.stoi[c] for c in text], dtype=np.int64)

    def get_batch(self, batch_size, rng):
        """Random (inputs, targets) where targets are inputs shifted by one."""
        n = len(self.data) - self.block_size - 1
        ix = rng.integers(0, n, size=batch_size)
        x = np.stack([self.data[i:i + self.block_size] for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.block_size] for i in ix])
        return x, y

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def train(model, dataset, steps=300, batch_size=16, lr=3e-3, seed=0, log_every=25):
    """Run the training loop. Returns the loss history (a real measured curve)."""
    from .optimizer import Adam
    rng = np.random.default_rng(seed)
    opt = Adam(lr=lr)
    history = []
    for step in range(1, steps + 1):
        x, y = dataset.get_batch(batch_size, rng)
        _, loss = model.forward(x, y)
        model.backward()
        opt.step(model.params_and_grads())
        history.append(float(loss))
        if step % log_every == 0 or step == 1:
            print(f"  step {step:4d}/{steps}   loss {loss:.4f}")
    return history


def generate(model, dataset, prompt, n_new, seed=0, temperature=1.0):
    """Autoregressively sample n_new characters continuing `prompt`."""
    rng = np.random.default_rng(seed)
    idx = list(dataset.encode(prompt))
    for _ in range(n_new):
        # feed the last block_size tokens (the model's context window)
        context = np.array(idx[-model.block_size:], dtype=np.int64)[None]
        logits, _ = model.forward(context)
        last = logits[0, -1] / temperature
        # softmax -> sample
        z = last - last.max()
        p = np.exp(z); p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        idx.append(nxt)
    return dataset.decode(idx)


def generate_cached(model, dataset, prompt, n_new, seed=0, temperature=1.0):
    """Same as generate(), but uses the KV-cache for O(T) per-token cost.

    We first "prime" the cache by feeding the prompt tokens one at a time, then
    generate new tokens, each attending over the cache instead of recomputing
    the whole sequence. With temperature and identical seeding, this produces
    the SAME text as generate() -- the cache is a speed optimization, not a
    behavior change (verified in the tests). This mirrors how real LLM serving
    works."""
    rng = np.random.default_rng(seed)
    idx = list(dataset.encode(prompt))
    model.reset_cache()

    # prime the cache with the prompt (positions 0..len(prompt)-1)
    logits = None
    for pos, tok in enumerate(idx):
        logits = model.forward_cached(np.array([[tok]], dtype=np.int64), pos)

    # now generate, continuing the position counter
    pos = len(idx) - 1
    for _ in range(n_new):
        last = logits[0, -1] / temperature
        z = last - last.max()
        p = np.exp(z); p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        idx.append(nxt)
        pos += 1
        logits = model.forward_cached(np.array([[nxt]], dtype=np.int64), pos)
    return dataset.decode(idx)
