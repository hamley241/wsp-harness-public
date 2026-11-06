"""D0: the shared starting discriminator, built by the mock constructor and stored as a file."""
from __future__ import annotations

import numpy as np

from wsp import rnn
from wsp.sandbox import mock_constructor_source, run_constructor


def make_d0(h: int, seed: int) -> dict[str, np.ndarray]:
    res = run_constructor(mock_constructor_source(h, seed), h)
    if res.weights is None:
        raise RuntimeError(f"mock constructor failed: {res.stderr.strip()}")
    return res.weights


def save_d0(path, weights: dict) -> None:
    np.savez(path, **weights)


def load_d0(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    return rnn.check_weights(raw, raw["Wh"].shape[0])
