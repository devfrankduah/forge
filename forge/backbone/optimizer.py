# Vendored into Forge so the project is self-contained (no external
# dependency). Same code as the companion Glassbox project. NumPy only.
"""
optimizer.py -- Adam, implemented by hand.

Adam is the default optimizer for transformers, and writing it out shows you
know what "the optimizer" actually does rather than treating it as magic. It
keeps two running averages per parameter:

    m = exp. moving avg of the gradient        (momentum: which way to go)
    v = exp. moving avg of the gradient^2      (per-parameter step scaling)

Then steps with a bias correction (m and v start at 0, so early steps are
corrected upward) :

    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    param -= lr * m_hat / (sqrt(v_hat) + eps)

The per-parameter scaling by 1/sqrt(v_hat) is the key idea: parameters with
consistently large gradients take smaller, steadier steps.
"""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}   # keyed by id(param)
        self.v = {}

    def step(self, params_and_grads):
        self.t += 1
        for param, grad in params_and_grads:
            key = id(param)
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * grad
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (grad * grad)
            m_hat = self.m[key] / (1 - self.b1 ** self.t)
            v_hat = self.v[key] / (1 - self.b2 ** self.t)
            # in-place update so the model's arrays are mutated directly
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
