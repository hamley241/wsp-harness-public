import math
import random

import numpy as np
import pytest

from wsp import evaluate, rnn
from wsp.dfa import random_dfa, random_string


def _labelled(dfa, rng, lengths, per_length):
    return [(s, dfa.label(s)) for L in lengths for s in (random_string(rng, L) for _ in range(per_length))]


def test_per_length_keys_and_evaluate_arm_fields():
    dfa, rng = random_dfa(4, seed=1), random.Random(0)
    heldout, long_ = _labelled(dfa, rng, range(1, 13), 4), _labelled(dfa, rng, range(13, 25), 2)
    d0, d1 = rnn.random_weights(8, 1), rnn.random_weights(8, 2)
    initial = _labelled(dfa, rng, range(1, 13), 2)
    correct = evaluate.correct_subset(d0, initial)
    ev = evaluate.evaluate_arm(d1, {"heldout": heldout, "long": long_, "reported_failures": heldout[:5],
                                    "initial_correct_d0": correct})
    assert list(ev["per_length"]) == list(range(1, 13))
    assert all(0.0 <= v <= 1.0 for v in ev["per_length"].values())
    assert ev["per_length_n"] == {L: 4 for L in range(1, 13)}
    assert ev["per_length_noise_floor"] == {L: pytest.approx(0.25) for L in range(1, 13)}
    assert 0.0 <= ev["heldout"] <= 1.0 and 0.0 <= ev["long"] <= 1.0
    assert ev["heldout_n"] == 48 and ev["long_n"] == 24 and ev["reported_failures_n"] == 5
    assert ev["heldout_noise_floor"] == pytest.approx(0.5 / 48 ** 0.5)
    assert ev["reported_failures_noise_floor"] == pytest.approx(0.5 / 5 ** 0.5)
    assert 0.0 <= ev["reported_failures"] <= 1.0 and 0.0 <= ev["initial_regressions"] <= 1.0
    assert ev["initial_correct_d0_n"] == len(correct)
    assert ev["initial_regressions_noise_floor"] == pytest.approx(0.5 / len(correct) ** 0.5)
    # D0 against its own correct subset never regresses; no failures shown -> None.
    ev0 = evaluate.evaluate_arm(d0, {"heldout": heldout, "long": long_, "reported_failures": [],
                                     "initial_correct_d0": correct})
    assert ev0["initial_regressions"] == 0.0 and ev0["reported_failures"] is None
    assert ev0["reported_failures_n"] == 0 and ev0["reported_failures_noise_floor"] is None


def test_per_length_always_has_all_twelve_keys():
    # A held-out set that lacks the short lengths (the real Oracle exhausts lengths 1-3 in the
    # initial dataset) still reports every length 1..12, with None and n = 0 for empty buckets.
    dfa, rng = random_dfa(4, seed=1), random.Random(0)
    w = rnn.random_weights(8, 1)
    pl = evaluate.per_length_accuracy(w, _labelled(dfa, rng, range(4, 13), 3))
    assert list(pl["per_length"]) == list(range(1, 13))
    assert all(pl["per_length"][L] is None and pl["per_length_n"][L] == 0 and pl["per_length_noise_floor"][L] is None
               for L in (1, 2, 3))
    assert all(0.0 <= pl["per_length"][L] <= 1.0 and pl["per_length_n"][L] == 3 for L in range(4, 13))
    assert evaluate.per_length_accuracy(w, [])["per_length"] == {L: None for L in range(1, 13)}


def test_cross_play_shape():
    arms, rounds = ("A", "B", "D"), 2
    keys = [f"{a}/{r}" for a in arms for r in range(rounds + 1)]
    weights = {k: rnn.random_weights(4, i) for i, k in enumerate(keys)}
    ces = {k: ([("01", 1), ("1", 0)] if k.endswith("/1") else []) for k in keys}
    m = evaluate.cross_play(weights, ces)
    assert list(m) == keys and all(list(row) == keys for row in m.values())
    assert sum(len(row) for row in m.values()) == (len(arms) * (rounds + 1)) ** 2
    for wk, row in m.items():
        for ck, acc in row.items():
            assert (acc is None) == (ces[ck] == []) and (acc is None or 0.0 <= acc <= 1.0)
    sizes = evaluate.cross_play_sizes(ces)
    assert list(sizes["n"]) == keys and list(sizes["noise_floor"]) == keys
    for ck in keys:
        assert sizes["n"][ck] == len(ces[ck])
        assert sizes["noise_floor"][ck] == (pytest.approx(0.5 / 2 ** 0.5) if ces[ck] else None)


def test_edit_stats_identical_is_all_zeros_and_detects_change():
    w = rnn.random_weights(6, 3)
    st = evaluate.edit_stats(w, {k: v.copy() for k, v in w.items()})
    assert st["frac_changed"] == 0.0 and st["mean_abs"] == 0.0 and st["max_abs"] == 0.0
    assert set(st["per_tensor"]) == set(rnn.KEYS) and st["n"] == rnn.n_params(6)
    assert all(t["frac_changed"] == 0.0 and t["max_abs"] == 0.0 for t in st["per_tensor"].values())
    w2 = {k: v.copy() for k, v in w.items()}
    w2["c"] = np.asarray(w2["c"] + 1.0)
    st2 = evaluate.edit_stats(w, w2)
    assert st2["frac_changed"] == pytest.approx(1 / rnn.n_params(6)) and st2["max_abs"] == pytest.approx(1.0)
    assert st2["per_tensor"]["c"]["frac_changed"] == 1.0 and st2["per_tensor"]["Wh"]["frac_changed"] == 0.0


def test_noise_floor():
    assert evaluate.noise_floor(500) == pytest.approx(0.0224, abs=5e-5)
    assert evaluate.noise_floor(100) == pytest.approx(0.05) and evaluate.noise_floor(0) is None
    assert math.isclose(evaluate.noise_floor(4), 0.25)


def test_sat_baseline_scores_hidden_sets_only_in_a_separate_step():
    dfa, rng = random_dfa(3, seed=2), random.Random(1)
    samples = _labelled(dfa, rng, range(1, 9), 6)
    sets = {"heldout": _labelled(dfa, rng, range(1, 13), 3), "long": _labelled(dfa, rng, range(13, 25), 2)}
    res = evaluate.sat_baseline(samples, 3, 5)  # run-time step: sees no hidden set
    assert res["status"] == "sat" and res["ambiguity"] in ("unique", "ambiguous", "no_signal")
    assert res["n_samples"] == len(samples) and "heldout" not in res and "long" not in res
    scored = evaluate.sat_hidden_scores(res, sets)  # end-of-run step
    assert 0.0 <= scored["heldout"] <= 1.0 and 0.0 <= scored["long"] <= 1.0
    assert scored["heldout_n"] == 36 and scored["long_n"] == 24
    assert scored["heldout_noise_floor"] == pytest.approx(0.5 / 6) and scored["long_noise_floor"] == pytest.approx(0.5 / 24 ** 0.5)
    assert "heldout" not in res  # the run-time record is not mutated
    none = evaluate.sat_hidden_scores({"status": "unsat", "dfa": None}, sets)
    assert none["heldout"] is None and none["long"] is None and none["heldout_n"] == 36


def test_trained_baseline_eval_fields():
    dfa, rng = random_dfa(3, seed=2), random.Random(1)
    initial = _labelled(dfa, rng, range(1, 13), 2)
    sets = {"heldout": _labelled(dfa, rng, range(1, 13), 2), "long": _labelled(dfa, rng, range(13, 25), 1)}
    tb = evaluate.trained_baseline_eval(initial, 4, 1, 20, sets)
    assert tb["steps"] == 20 and tb["history"][-1]["step"] == 20
    assert 0.0 <= tb["heldout"] <= 1.0 and 0.0 <= tb["long"] <= 1.0 and 0.0 <= tb["train_acc"] <= 1.0
    assert tb["train_acc_n"] == 24 and tb["train_acc_noise_floor"] == pytest.approx(0.5 / 24 ** 0.5)
    assert tb["heldout_n"] == 24 and tb["long_n"] == 12 and tb["long_noise_floor"] == pytest.approx(0.5 / 12 ** 0.5)
    assert list(tb["per_length"]) == list(range(1, 13)) and list(tb["per_length_n"]) == list(range(1, 13))
