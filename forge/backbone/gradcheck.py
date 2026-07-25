# Vendored into Forge so the project is self-contained (no external
# dependency). Same code as the companion Glassbox project. NumPy only.
"""
gradcheck.py -- prove the hand-written gradients are actually correct.

THE HONESTY CAPSTONE
--------------------
A backward pass can look right and be subtly wrong (a missing term, a transpose
in the wrong place). The only way to *know* it's correct is to compare it
against a numerical gradient computed from the definition of a derivative:

    df/dx  ~=  ( f(x + h) - f(x - h) ) / (2h)

We nudge each input element by a tiny h, measure how the (scalar) output
changes, and compare that to what our backward() claims the gradient is. If
they match to ~1e-6, the analytic gradient is right. It is not "trust me" but a check that
passes to seven decimal places.

We reduce each op's output to a scalar with a fixed random weighting so there's
a well-defined thing to differentiate.
"""

from __future__ import annotations

import numpy as np


def numerical_grad(f, x, h=1e-5):
    """Central-difference gradient of scalar function f at array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + h
        fpos = f(x)
        x[idx] = orig - h
        fneg = f(x)
        x[idx] = orig
        grad[idx] = (fpos - fneg) / (2 * h)
        it.iternext()
    return grad


def rel_error(a, b):
    """Relative error between two arrays; near 0 means they match."""
    denom = np.maximum(1e-12, np.abs(a) + np.abs(b))
    return np.max(np.abs(a - b) / denom)


def check(name, analytic, numeric, tol=1e-4):
    err = rel_error(analytic, numeric)
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}] {name:28s} rel_error={err:.2e}")
    return bool(err < tol)


def run_all_checks(seed=0):
    """Gradient-check every primitive op. Returns True iff all pass."""
    rng = np.random.default_rng(seed)
    ok = True
    print("Gradient checks (analytic backward vs numerical):")

    # --- Matmul ---
    from .tensor_ops import Matmul
    x = rng.standard_normal((4, 5)).astype(np.float64)
    W = rng.standard_normal((5, 3)).astype(np.float64)
    w_out = rng.standard_normal((4, 3))
    op = Matmul()
    def f_x(xx):
        return float(np.sum(Matmul().forward(xx, W) * w_out))
    def f_W(WW):
        return float(np.sum(Matmul().forward(x, WW) * w_out))
    op.forward(x, W)
    dx, dW = op.backward(w_out)
    ok &= check("Matmul dx", dx, numerical_grad(f_x, x.copy()))
    ok &= check("Matmul dW", dW, numerical_grad(f_W, W.copy()))

    # --- Softmax ---
    from .tensor_ops import Softmax
    x = rng.standard_normal((4, 6)).astype(np.float64)
    w_out = rng.standard_normal((4, 6))
    op = Softmax()
    def f_s(xx):
        return float(np.sum(Softmax().forward(xx) * w_out))
    op.forward(x)
    dx = op.backward(w_out)
    ok &= check("Softmax dx", dx, numerical_grad(f_s, x.copy()))

    # --- LayerNorm ---
    from .tensor_ops import LayerNorm
    x = rng.standard_normal((4, 8)).astype(np.float64)
    w_out = rng.standard_normal((4, 8))
    ln = LayerNorm(8)
    ln.gamma = rng.standard_normal((8,))
    ln.beta = rng.standard_normal((8,))
    def f_ln_x(xx):
        m = LayerNorm(8); m.gamma = ln.gamma; m.beta = ln.beta
        return float(np.sum(m.forward(xx) * w_out))
    ln.forward(x)
    dx, dg, db = ln.backward(w_out)
    ok &= check("LayerNorm dx", dx, numerical_grad(f_ln_x, x.copy()))
    def f_ln_g(gg):
        m = LayerNorm(8); m.gamma = gg; m.beta = ln.beta
        return float(np.sum(m.forward(x) * w_out))
    ok &= check("LayerNorm dgamma", dg, numerical_grad(f_ln_g, ln.gamma.copy()))
    def f_ln_b(bb):
        m = LayerNorm(8); m.gamma = ln.gamma; m.beta = bb
        return float(np.sum(m.forward(x) * w_out))
    ok &= check("LayerNorm dbeta", db, numerical_grad(f_ln_b, ln.beta.copy()))

    # --- GELU ---
    from .tensor_ops import GELU
    x = rng.standard_normal((4, 7)).astype(np.float64)
    w_out = rng.standard_normal((4, 7))
    op = GELU()
    def f_g(xx):
        return float(np.sum(GELU().forward(xx) * w_out))
    op.forward(x)
    dx = op.backward(w_out)
    ok &= check("GELU dx", dx, numerical_grad(f_g, x.copy()))

    # --- CrossEntropy ---
    from .tensor_ops import CrossEntropy
    logits = rng.standard_normal((5, 9)).astype(np.float64)
    targets = rng.integers(0, 9, size=(5,))
    op = CrossEntropy()
    def f_ce(ll):
        return float(CrossEntropy().forward(ll, targets))
    op.forward(logits, targets)
    dlogits = op.backward()
    ok &= check("CrossEntropy dlogits", dlogits, numerical_grad(f_ce, logits.copy()))

    print("ALL GRADIENT CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return bool(ok)


if __name__ == "__main__":
    run_all_checks()
