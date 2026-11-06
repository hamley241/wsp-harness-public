import itertools
import time

import numpy as np

from wsp import rnn, trained_baseline as tb
from wsp.dfa import parity_dfa


def test_gradient_matches_finite_differences():
    w = rnn.random_weights(3, seed=7)
    batch = [("0110", 1), ("10", 0)]
    g = tb.grads(w, batch)
    eps = 1e-6
    for k in rnn.KEYS:
        arr = np.array(w[k], dtype=np.float64)
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            for sign in (1, -1):
                wp = dict(w)
                a = arr.copy()
                a[idx] += sign * eps
                wp[k] = a
                num[idx] += sign * tb.loss(wp, batch)
            num[idx] /= 2 * eps
        rel = np.linalg.norm(num - g[k]) / max(np.linalg.norm(num) + np.linalg.norm(g[k]), 1e-12)
        assert rel < 1e-5, (k, rel)


def parity_samples():
    p = parity_dfa()
    strings = ["".join(b) for n in range(7) for b in itertools.product("01", repeat=n)]
    return [(s, p.label(s)) for s in strings[:64]]


def test_train_parity_reaches_accuracy_and_is_deterministic():
    samples = parity_samples()
    assert len(samples) == 64
    t0 = time.monotonic()
    w1, hist = tb.train(samples, h=8, seed=1, steps=600)
    assert time.monotonic() - t0 < 20
    assert [r["step"] for r in hist] == [100, 200, 300, 400, 500, 600]
    assert hist[-1]["acc"] == tb.accuracy(w1, samples) >= 0.9
    w2, _ = tb.train(samples, h=8, seed=1, steps=600)
    assert rnn.weights_equal(w1, w2)
