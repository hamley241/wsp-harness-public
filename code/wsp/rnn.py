"""The discriminator: a fixed tanh RNN over {0,1} in pure numpy (float64).

one-hot(2) -> Wx (2,h), Wh (h,h), bh (h), h0 = 0, readout w (h), c ().
Accept iff logit > 0. Parameter count h*h + 4*h + 1.
"""
from __future__ import annotations

import numpy as np

KEYS = ("Wx", "Wh", "bh", "w", "c")


def shapes(h: int) -> dict[str, tuple]:
    return {"Wx": (2, h), "Wh": (h, h), "bh": (h,), "w": (h,), "c": ()}


def n_params(h: int) -> int:
    return h * h + 4 * h + 1


def check_weights(weights: dict, h: int) -> dict[str, np.ndarray]:
    """Validate keys, shapes, dtype and finiteness; return float64 read-only copies."""
    if not isinstance(weights, dict):
        raise ValueError(f"weights must be a dict, got {type(weights).__name__}")
    want = shapes(h)
    missing = set(want) - set(weights)
    extra = set(weights) - set(want)
    if missing or extra:
        raise ValueError(f"weight keys: missing={sorted(missing)} extra={sorted(map(str, extra))}")
    out = {}
    for k, shape in want.items():
        arr = np.asarray(weights[k])
        if arr.dtype.kind != "f":
            raise ValueError(f"{k}: dtype {arr.dtype} is not a float dtype")
        if arr.shape != shape:
            raise ValueError(f"{k}: shape {arr.shape} != {shape}")
        # Convert first: a finite wider float (e.g. float128) can overflow to inf in float64.
        arr = np.array(arr, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{k}: non-finite value")
        arr.flags.writeable = False
        out[k] = arr
    return out


def logit(weights: dict, s: str) -> float:
    Wx, Wh, bh = weights["Wx"], weights["Wh"], weights["bh"]
    h = np.zeros(Wh.shape[0])
    for ch in s:
        h = np.tanh(Wx[int(ch)] + Wh @ h + bh)
    return float(weights["w"] @ h + weights["c"])


def predict(weights: dict, s: str) -> int:
    return int(logit(weights, s) > 0)


def predict_many(weights: dict, strings) -> list[int]:
    return [predict(weights, s) for s in strings]


def random_weights(h: int, seed: int, scale: float = 0.5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: rng.normal(0.0, scale, size=shape) for k, shape in shapes(h).items()}


def weights_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
