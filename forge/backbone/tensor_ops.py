# Vendored into Forge so the project is self-contained (no external
# dependency). Same code as the companion Glassbox project. NumPy only.
"""
tensor_ops.py -- the primitive operations of a transformer, each with a
forward pass AND a hand-written backward pass (the gradient).

WHY THIS FILE IS THE HEART OF THE PROJECT
-----------------------------------------
In PyTorch you call `loss.backward()` and autograd computes every gradient for
you. That's convenient, and it also hides the one thing that actually
matters: HOW the gradients flow. So here we don't use
autograd. For every operation the model needs -- matmul, softmax, layer norm,
GELU, cross-entropy -- we implement:

    forward(x)      -> output, plus a "cache" of what backward will need
    backward(dout)  -> the gradient(s) with respect to the input(s)

The backward functions are just calculus: the chain rule applied to each op.
Every one is derived and commented so you can follow the math. And `gradcheck.py`
verifies each against a numerical gradient, so we KNOW the math is right, not
just that it looks right.

CONVENTION
----------
Each op is a small class with .forward() and .backward(). The forward stores
whatever the backward needs on `self`. This mirrors how a real autograd engine
works (it records a graph of these), but written out by hand so it's legible.
Shapes use (B, T, C): batch, time/sequence position, channels/features.
"""

from __future__ import annotations

import numpy as np


class Matmul:
    """y = x @ W  (a linear projection without bias).

    Backward (the classic result, worth knowing cold):
        dx = dy @ W.T
        dW = x.T @ dy   (summed over the batch/time dims)
    """

    def forward(self, x, W):
        self.x = x
        self.W = W
        return x @ W

    def backward(self, dy):
        # x: (..., in), W: (in, out), dy: (..., out)
        dx = dy @ self.W.T
        # collapse all leading dims into one before the outer product for dW
        x2 = self.x.reshape(-1, self.x.shape[-1])       # (N, in)
        dy2 = dy.reshape(-1, dy.shape[-1])              # (N, out)
        dW = x2.T @ dy2                                  # (in, out)
        return dx, dW


class Bias:
    """y = x + b, broadcasting b over all but the last dim.
    Backward: db is dy summed over every dim except the last."""

    def forward(self, x, b):
        self.x_shape = x.shape
        return x + b

    def backward(self, dy):
        dx = dy
        axes = tuple(range(dy.ndim - 1))
        db = dy.sum(axis=axes)
        return dx, db


class Softmax:
    """Row-wise softmax over the last axis.

    Forward uses the max-subtraction trick for numerical stability (exp of big
    numbers overflows). Backward uses the compact Jacobian-vector form:
        dx = s * (dy - sum(dy * s))
    where s is the softmax output. Deriving this from the full Jacobian and
    simplifying yields the compact one-liner above.
    """

    def forward(self, x):
        z = x - x.max(axis=-1, keepdims=True)
        e = np.exp(z)
        self.s = e / e.sum(axis=-1, keepdims=True)
        return self.s

    def backward(self, dy):
        s = self.s
        # sum over the last axis of (dy * s), keep dims for broadcasting
        dot = np.sum(dy * s, axis=-1, keepdims=True)
        return s * (dy - dot)


class LayerNorm:
    """Normalize the last dim to zero mean / unit variance, then scale+shift.

        y = gamma * (x - mean) / sqrt(var + eps) + beta

    LayerNorm's backward is famously fiddly because mean and variance both
    depend on every element, so the gradient has three terms. We derive it in
    the code. Getting this right by hand (and gradient-checking it) is exactly
    the kind of thing that convinces a reviewer you understand normalization,
    not just that you sprinkle it in because the paper did.
    """

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((dim,), dtype=np.float32)
        self.beta = np.zeros((dim,), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.xhat = (x - self.mean) / self.std        # normalized
        return self.gamma * self.xhat + self.beta

    def backward(self, dy):
        # params first (easy): they just scale/shift xhat
        axes = tuple(range(dy.ndim - 1))
        dgamma = np.sum(dy * self.xhat, axis=axes)
        dbeta = np.sum(dy, axis=axes)

        # input gradient: N is the size of the normalized (last) dim
        N = self.x.shape[-1]
        dxhat = dy * self.gamma
        # the three-term result of differentiating the normalization:
        #   dx = (1/std) * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
        dx = (1.0 / self.std) * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True)
        )
        return dx, dgamma, dbeta


class GELU:
    """Gaussian Error Linear Unit, the activation used in GPT-style models.

    We use the common tanh approximation:
        gelu(x) = 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))
    Backward differentiates that expression. Using GELU (not ReLU) is a small
    signal you know what modern transformers actually use.
    """

    _c = np.sqrt(2.0 / np.pi)

    def forward(self, x):
        self.x = x
        self.inner = self._c * (x + 0.044715 * x ** 3)
        self.tanh = np.tanh(self.inner)
        return 0.5 * x * (1.0 + self.tanh)

    def backward(self, dy):
        x = self.x
        # d/dx of the tanh-approx GELU
        dinner = self._c * (1.0 + 3 * 0.044715 * x ** 2)
        sech2 = 1.0 - self.tanh ** 2                 # derivative of tanh
        dgelu = 0.5 * (1.0 + self.tanh) + 0.5 * x * sech2 * dinner
        return dy * dgelu


class CrossEntropy:
    """Softmax + negative-log-likelihood in one numerically stable step.

    Combining them is standard because the gradient collapses to a beautifully
    simple form:
        dlogits = (softmax(logits) - onehot(targets)) / N
    That cancellation (no leftover softmax-Jacobian) is one of the most elegant
    results in ML.
    """

    def forward(self, logits, targets):
        # logits: (N, vocab), targets: (N,) integer class ids
        self.targets = targets
        self.N = logits.shape[0]
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        self.probs = e / e.sum(axis=-1, keepdims=True)
        # NLL of the correct class, averaged
        correct = self.probs[np.arange(self.N), targets]
        loss = -np.mean(np.log(correct + 1e-12))
        return loss

    def backward(self):
        d = self.probs.copy()
        d[np.arange(self.N), self.targets] -= 1.0     # subtract the one-hot
        return d / self.N


class RMSNorm:
    """Root-Mean-Square LayerNorm -- the normalization used by Llama, T5, and
    most current LLMs, and a genuinely modern replacement for LayerNorm.

    The insight (from Zhang & Sennrich 2019): LayerNorm's mean-subtraction
    (the "re-centering") turns out to matter little; what helps training is the
    "re-scaling" by the magnitude of the vector. So RMSNorm drops the mean and
    the bias entirely and just divides by the root-mean-square:

        rms(x) = sqrt(mean(x^2) + eps)
        y = (x / rms(x)) * gamma

    That's fewer operations and fewer parameters (no beta) than LayerNorm, which
    is why big models prefer it. Being able to say *why* it works -- that
    re-scaling is the part that matters, re-centering mostly isn't -- is a nice
    signal you understand normalization rather than cargo-culting it.

    Backward: simpler than LayerNorm's because there's no mean term. With
    r = rms(x) and N = feature dim, differentiating y = x/r * gamma gives:
        dx = (gamma / r) * ( dy - x * mean(dy * x * gamma) / r^2 )
    Derived and gradient-checked (see gradcheck.py).
    """

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((dim,), dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.ms = np.mean(x * x, axis=-1, keepdims=True)     # mean square
        self.r = np.sqrt(self.ms + self.eps)                 # RMS
        self.xhat = x / self.r                               # normalized (no gamma yet)
        return self.xhat * self.gamma

    def backward(self, dy):
        # gamma just scales the normalized value elementwise
        axes = tuple(range(dy.ndim - 1))
        dgamma = np.sum(dy * self.xhat, axis=axes)

        N = self.x.shape[-1]
        g = dy * self.gamma                                  # grad into (x / r)
        # d/dx of x / rms(x): the second term is the pull-back through rms,
        # which depends on every element of x.
        dx = (g - self.xhat * np.mean(g * self.xhat, axis=-1, keepdims=True)) / self.r
        return dx, dgamma
