"""Gradient-trained baseline (capacity control): the wsp.rnn architecture
trained by backprop through time with Adam, numpy only.

Loss = mean binary cross-entropy on the logit. Deterministic given the seed.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from wsp import rnn

Sample = Tuple[str, int]
BETAS = (0.9, 0.999)
EPS = 1e-8


def _forward(weights: dict, s: str) -> Tuple[List[np.ndarray], float]:
    """Hidden states h_0..h_T (h_0 = 0) and the logit."""
    Wx, Wh, bh = weights["Wx"], weights["Wh"], weights["bh"]
    hs = [np.zeros(Wh.shape[0])]
    for ch in s:
        hs.append(np.tanh(Wx[int(ch)] + Wh @ hs[-1] + bh))
    return hs, float(weights["w"] @ hs[-1] + weights["c"])


def _bce(logit: float, y: int) -> float:
    # softplus(logit) - y*logit, numerically stable
    return float(np.logaddexp(0.0, logit) - y * logit)


def loss(weights: dict, batch: Sequence[Sample]) -> float:
    return sum(_bce(_forward(weights, s)[1], int(y)) for s, y in batch) / len(batch)


def grads(weights: dict, batch: Sequence[Sample]) -> Dict[str, np.ndarray]:
    """Gradient of the mean BCE over `batch` w.r.t. every weight, by BPTT."""
    g = {k: np.zeros_like(np.asarray(v, dtype=np.float64)) for k, v in weights.items()}
    Wh, w = weights["Wh"], weights["w"]
    for s, y in batch:
        hs, logit = _forward(weights, s)
        dlogit = 1.0 / (1.0 + np.exp(-logit)) - int(y)
        g["w"] += dlogit * hs[-1]
        g["c"] += dlogit
        dh = dlogit * w
        for t in range(len(s), 0, -1):
            dpre = dh * (1.0 - hs[t] ** 2)
            g["Wx"][int(s[t - 1])] += dpre
            g["Wh"] += np.outer(dpre, hs[t - 1])
            g["bh"] += dpre
            dh = Wh.T @ dpre
    n = len(batch)
    return {k: v / n for k, v in g.items()}


def accuracy(weights: dict, samples: Sequence[Sample]) -> float:
    return sum(rnn.predict(weights, s) == int(y) for s, y in samples) / len(samples)


def train(samples: Sequence[Sample], h: int, seed: int, steps: int = 2000, lr: float = 0.01,
          batch_size: int = 32, l2: float = 0.0) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """Adam on mean BCE. history: {"step", "loss", "acc"} every 100 steps
    (and at the last step)."""
    rng = np.random.default_rng(seed)
    weights = rnn.random_weights(h, seed)
    weights["c"] = np.asarray(weights["c"], dtype=np.float64)
    m = {k: np.zeros_like(v) for k, v in weights.items()}
    v = {k: np.zeros_like(x) for k, x in weights.items()}
    b1, b2 = BETAS
    samples = list(samples)
    bs = min(batch_size, len(samples))
    history = []
    for step in range(1, steps + 1):
        idx = rng.choice(len(samples), size=bs, replace=False)
        g = grads(weights, [samples[i] for i in idx])
        for k in weights:
            gk = g[k] + l2 * weights[k]
            m[k] = b1 * m[k] + (1 - b1) * gk
            v[k] = b2 * v[k] + (1 - b2) * gk * gk
            mhat = m[k] / (1 - b1 ** step)
            vhat = v[k] / (1 - b2 ** step)
            weights[k] = weights[k] - lr * mhat / (np.sqrt(vhat) + EPS)
        if step % 100 == 0 or step == steps:
            history.append({"step": step, "loss": loss(weights, samples),
                            "acc": accuracy(weights, samples)})
    return rnn.check_weights(weights, h), history
