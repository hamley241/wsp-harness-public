"""Evaluation side (hidden until the run ends). May import dfa / sat_dfa.

Module boundary: no arm imports this module (wsp/test_arms.py asserts it).

Every accuracy is reported with the size of the set it was measured on and the
binomial noise floor for that size (`<name>`, `<name>_n`, `<name>_noise_floor`).
"""
from __future__ import annotations

import math

import numpy as np

from wsp import rnn, sat_dfa, trained_baseline
from wsp.dfa import DFA
from wsp.oracle import SHORT_LENGTHS

CHANGE_EPS = 1e-12
PER_LENGTH_KEYS = tuple(SHORT_LENGTHS)  # 1..12, always all present


def noise_floor(n: int) -> float | None:
    """Binomial standard error at p = 0.5 for n items; None for an empty set."""
    return 0.5 / math.sqrt(n) if n else None


def accuracy(weights: dict, samples) -> float | None:
    samples = list(samples)
    if not samples:
        return None
    return sum(rnn.predict(weights, s) == int(y) for s, y in samples) / len(samples)


def dfa_accuracy(d: DFA, samples) -> float | None:
    samples = list(samples)
    if not samples:
        return None
    return sum(d.label(s) == int(y) for s, y in samples) / len(samples)


def with_noise(name: str, acc: float | None, n: int) -> dict:
    """An accuracy with its companions: {name, name_n, name_noise_floor}."""
    return {name: acc, f"{name}_n": n, f"{name}_noise_floor": noise_floor(n)}


def per_length_accuracy(weights: dict, samples) -> dict:
    """Fixed table over lengths 1..12: `per_length` (acc or None for an empty bucket),
    `per_length_n` (bucket sizes) and `per_length_noise_floor`."""
    by: dict[int, list] = {L: [] for L in PER_LENGTH_KEYS}
    for s, y in samples:
        by.setdefault(len(s), []).append((s, y))
    keys = sorted(by)
    return {"per_length": {L: accuracy(weights, by[L]) for L in keys},
            "per_length_n": {L: len(by[L]) for L in keys},
            "per_length_noise_floor": {L: noise_floor(len(by[L])) for L in keys}}


def correct_subset(weights: dict, samples) -> list[tuple[str, int]]:
    return [(s, y) for s, y in samples if rnn.predict(weights, s) == int(y)]


def _hidden_scores(weights: dict, sets: dict) -> dict:
    heldout, long_ = list(sets["heldout"]), list(sets["long"])
    return {**with_noise("heldout", accuracy(weights, heldout), len(heldout)),
            **with_noise("long", accuracy(weights, long_), len(long_)),
            **per_length_accuracy(weights, heldout)}


def evaluate_arm(weights: dict, sets: dict) -> dict:
    """sets: heldout, long, reported_failures (counterexamples shown so far, as (s, label)),
    initial_correct_d0 (initial examples D0 labelled correctly)."""
    failures, base = list(sets["reported_failures"]), list(sets["initial_correct_d0"])
    wrong = sum(rnn.predict(weights, s) != int(y) for s, y in base)
    return {
        **_hidden_scores(weights, sets),
        **with_noise("reported_failures", accuracy(weights, failures), len(failures)),
        "initial_regressions": wrong / len(base) if base else None, "initial_correct_d0_n": len(base),
        "initial_regressions_noise_floor": noise_floor(len(base)),
    }


def cross_play(weights_by_arm_round: dict, counterexample_sets: dict) -> dict:
    """matrix[weights key]["A/2"] = accuracy of those weights on that counterexample set (None if empty)."""
    return {wk: {ck: accuracy(w, ces) for ck, ces in counterexample_sets.items()}
            for wk, w in weights_by_arm_round.items()}


def cross_play_sizes(counterexample_sets: dict) -> dict:
    """Companions for every cross-play column: {"n": {key: n}, "noise_floor": {key: floor or None}}."""
    return {"n": {ck: len(ces) for ck, ces in counterexample_sets.items()},
            "noise_floor": {ck: noise_floor(len(ces)) for ck, ces in counterexample_sets.items()}}


def _delta_stats(delta: np.ndarray) -> dict:
    d = np.abs(np.ravel(delta))
    return {"n": int(d.size), "frac_changed": float(np.mean(d > CHANGE_EPS)),
            "mean_abs": float(d.mean()), "max_abs": float(d.max())}


def edit_stats(prev: dict, cur: dict) -> dict:
    deltas = {k: np.asarray(cur[k], dtype=np.float64) - np.asarray(prev[k], dtype=np.float64) for k in rnn.KEYS}
    out = _delta_stats(np.concatenate([np.ravel(d) for d in deltas.values()]))
    out["per_tensor"] = {k: _delta_stats(d) for k, d in deltas.items()}
    return out


def sat_baseline(samples, k: int, timeout_s: float) -> dict:
    """sat_dfa.identifiability on the accumulated labels. Safe to call during the run: it sees
    no hidden set. Score the found DFA afterwards with `sat_hidden_scores`."""
    res = sat_dfa.identifiability(list(samples), k, timeout_s)
    res["n_samples"] = len(samples)
    return res


def sat_hidden_scores(res: dict, sets: dict) -> dict:
    """`res` from sat_baseline plus the found DFA's held-out / long accuracy (None when no DFA)."""
    found = DFA.from_dict(res["dfa"]) if res.get("dfa") is not None else None
    heldout, long_ = list(sets["heldout"]), list(sets["long"])
    return {**res,
            **with_noise("heldout", dfa_accuracy(found, heldout) if found else None, len(heldout)),
            **with_noise("long", dfa_accuracy(found, long_) if found else None, len(long_))}


def trained_baseline_eval(initial_dataset, h: int, seed: int, steps: int, sets: dict) -> dict:
    weights, history = trained_baseline.train(initial_dataset, h, seed, steps)
    initial_dataset = list(initial_dataset)
    return {"h": h, "seed": seed, "steps": steps, "history": history,
            **with_noise("train_acc", trained_baseline.accuracy(weights, initial_dataset), len(initial_dataset)),
            **_hidden_scores(weights, sets)}
