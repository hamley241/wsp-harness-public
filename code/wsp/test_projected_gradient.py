import itertools

import numpy as np

from wsp import projected_gradient as pg
from wsp import rnn, trained_baseline as tb
from wsp.arms import target_histogram
from wsp.dfa import parity_dfa, random_dfa
from wsp.numeric_edit import EditBudget, _f32, flat_index
from wsp.oracle import Oracle

H = 8


def _d0(seed=0):
    return _f32(rnn.random_weights(H, seed))


def _dense(start, deltas: dict):
    """start (float64) plus `deltas` {(param, index): delta}."""
    d = {k: np.array(v, dtype=np.float64) for k, v in start.items()}
    for (k, i), delta in deltas.items():
        d[k].flat[i] += delta
    return d


def _assert_in_ball(edits, d0, max_norm):
    for p, i, v in edits:
        assert abs(float(np.float32(v)) - float(d0[p].flat[i])) <= max_norm


# ------------------------------------------------------------ project


def test_project_per_round_clipping_and_check():
    d0 = _d0()
    b = EditBudget(d0, per_round=5, distinct=8, max_norm=0.5)
    # every coordinate moved by 2.0 with magnitude decreasing in flat order: all selections are clipped
    coords = [(k, i) for k in rnn.KEYS for i in range(d0[k].size)]
    dense = _dense(d0, {c: 2.0 - n * 1e-3 for n, c in enumerate(coords)})
    edits, info = pg.project(d0, dense, b)
    assert info == {"n_selected": 5, "n_clipped": 5, "n_new_coords": 5}
    assert [(p, i) for p, i, _ in edits] == coords[:5]
    _assert_in_ball(edits, b.d0, 0.5)
    assert b.check(edits, d0) is None
    cur = b.apply(edits, d0)
    assert b.state()["n_touched"] == 5 and b.state()["displacement"] <= 0.5
    # a small move stays unclipped and is taken verbatim (as float32)
    e2, info2 = pg.project(cur, _dense(cur, {("w", 3): 0.25}), b)
    assert info2["n_clipped"] == 0 and ("w", 3, float(np.float32(cur["w"][3] + 0.25))) in e2


def test_project_ties_by_index_and_distinct_cap_across_rounds():
    d0 = {k: np.zeros(s, dtype=np.float32) for k, s in rnn.shapes(H).items()}  # exact deltas, exact ties
    b = EditBudget(d0, per_round=5, distinct=8, max_norm=0.5)
    # round 1: equal magnitudes everywhere -> the first five flat indices (Wx[0..4])
    dense = {k: np.full(s, 0.1) for k, s in rnn.shapes(H).items()}
    edits, info = pg.project(d0, dense, b)
    assert [(p, i) for p, i, _ in edits] == [("Wx", i) for i in range(5)]
    assert [flat_index(p, i, H) for p, i, _ in edits] == [0, 1, 2, 3, 4]
    cur = b.apply(edits, d0)
    # round 2: ten ranked new coordinates in Wh; only three fit under the distinct cap, the remaining
    # two selections fall back to already-touched coordinates (zero delta, ties by index)
    edits, info = pg.project(cur, _dense(cur, {("Wh", j): 1.0 - j * 0.05 for j in range(10)}), b)
    assert [(p, i) for p, i, _ in edits] == [("Wh", 0), ("Wh", 1), ("Wh", 2), ("Wx", 0), ("Wx", 1)]
    assert info == {"n_selected": 5, "n_clipped": 3, "n_new_coords": 3}
    assert b.check(edits, cur) is None
    cur = b.apply(edits, cur)
    assert b.state()["n_touched"] == 8 == b.distinct
    _assert_in_ball(edits, b.d0, 0.5)
    # round 3: nothing new may enter; selections are all from the touched set
    edits, info = pg.project(cur, _dense(cur, {("bh", j): 3.0 for j in range(H)}), b)
    assert info["n_new_coords"] == 0 and len(edits) == 5
    assert all((p, i) in b.touched for p, i, _ in edits)
    assert b.check(edits, cur) is None


def test_project_fewer_than_per_round_when_cap_binds_everything():
    d0 = _d0(2)
    b = EditBudget(d0, per_round=4, distinct=2, max_norm=1.0)
    edits, info = pg.project(d0, _dense(d0, {("w", 1): 0.3, ("w", 0): 0.2}), b)
    # only two coordinates may ever be touched: the walk ends with two selections, not four
    assert [(p, i) for p, i, _ in edits] == [("w", 1), ("w", 0)]
    assert info == {"n_selected": 2, "n_clipped": 0, "n_new_coords": 2} and b.check(edits, d0) is None


# ------------------------------------------------------------ query_batches


def test_query_batches_deterministic_histogram_no_duplicates():
    initial = [s for s, _ in Oracle(random_dfa(4, 1), 1).initial_dataset(32)]
    a = pg.query_batches(3, initial, rounds=3, n=32)
    assert a == pg.query_batches(3, initial, rounds=3, n=32)
    assert a != pg.query_batches(4, initial, rounds=3, n=32)
    target = target_histogram(initial, 32)
    for batch in a:
        assert len(batch) == 32 == len(set(batch))
        assert target_histogram(batch, 32) == target
        assert all(set(s) <= {"0", "1"} for s in batch)


# ------------------------------------------------------------ gradient_round


def test_gradient_round_does_not_worsen_training_loss_much():
    p = parity_dfa()
    samples = [(s, p.label(s)) for n in range(7) for s in ("".join(b) for b in itertools.product("01", repeat=n))][:64]
    w, _ = tb.train(samples, h=H, seed=1, steps=200)
    b = EditBudget(w, per_round=24, distinct=48, max_norm=1.0)
    before = tb.loss(w, samples)
    new, rec = pg.gradient_round(_f32(w), samples, b, steps=50)
    assert tb.loss(new, samples) <= before + 0.05
    assert rec["n_selected"] == 24 == len(rec["edits"]) == rec["n_touched"] == len(rec["touched"])
    assert 0.0 < rec["displacement"] <= 1.0
    assert new["Wh"].dtype == np.float32


# ------------------------------------------------------------ preflight


def test_preflight_small_returns_keys_and_bool():
    res = pg.preflight(1, k=4, h=H, n_initial=32, rounds=2, n=16, budget_args={"per_round": 24, "distinct": 48,
                       "max_norm": 1.0}, n_heldout=100, train_steps=50, adam_steps=50)
    for key in ("D0_heldout", "G_heldout", "N_heldout", "gain", "f", "feasible", "batches", "budget"):
        assert key in res
    assert isinstance(res["feasible"], bool) and res["f"] == 0.0224
    assert len(res["G_heldout"]) == 2 and len(res["batches"]) == 2 and all(len(bt) == 16 for bt in res["batches"])
    assert res["gain"] == res["G_heldout"][-1] - res["N_heldout"] and res["N_heldout"] == res["D0_heldout"]
    assert res["feasible"] == (res["gain"] > 0.0224)
    assert res["budget"]["n_touched"] <= 48 and res["budget"]["displacement"] <= 1.0
    assert res["ledger"]["G"]["total"] == 32
    new = [r["n_new_labels"] for r in res["G_rounds"]]  # re-queried strings (charged above) add no label
    assert all(0 <= n <= 16 for n in new) and res["G_rounds"][-1]["n_labels"] == 32 + sum(new)


# ------------------------------------------------------------ add_labels (P9)


def test_add_labels_first_occurrence_order_against_initial_and_within_batch():
    labels = [("0", 1), ("01", 0)]
    new = pg.add_labels(labels, ["01", "1", "1", "11"], [0, 1, 1, 0])
    assert new == [("1", 1), ("11", 0)]
    assert labels == [("0", 1), ("01", 0), ("1", 1), ("11", 0)]
    assert pg.add_labels(labels, ["11", "0", "1"], [0, 1, 1]) == [] and len(labels) == 4


def test_preflight_requeries_are_charged_but_add_nothing(monkeypatch):
    initial = [s for s, _ in Oracle(random_dfa(4, 1), 1).initial_dataset(32)]
    fresh = ["1" * 14 + format(i, "04b") for i in range(6)]
    assert not set(fresh) & set(initial)
    batch = initial[:2] + fresh
    monkeypatch.setattr(pg, "query_batches", lambda *a, **k: [batch, list(batch)])
    res = pg.preflight(1, k=4, h=H, n_initial=32, rounds=2, n=8, budget_args={"per_round": 24, "distinct": 48,
                       "max_norm": 1.0}, n_heldout=100, train_steps=50, adam_steps=50)
    assert res["ledger"]["G"]["total"] == 16  # every occurrence charged
    assert [r["n_new_labels"] for r in res["G_rounds"]] == [6, 0]
    assert [r["n_labels"] for r in res["G_rounds"]] == [38, 38]
