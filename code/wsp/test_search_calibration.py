import json
import random

import numpy as np
import pytest

from wsp import rnn, search_calibration as sc, trained_baseline as tb
from wsp.numeric_edit import EditBudget, flat_index

CFG = {"k": 4, "h": 8, "n_initial": 16, "n_heldout": 64, "n_long": 32, "train_steps": 100, "adam_steps": 50}
N = rnn.n_params(8)  # 97 coordinates at h=8; every coordinate is allowed in round 1
PROBE_BUDGET = 1 + N + 5


@pytest.fixture(scope="module")
def bundle():
    return sc._bundle(1, {**sc.DEFAULT_CFG, **CFG})


# ------------------------------------------------------------- flat view


def test_to_flat_from_flat_round_trip_and_flat_index():
    for h in (8, 16):
        w = rnn.random_weights(h, 5)
        flat = sc.to_flat(w)
        assert flat.dtype == np.float32 and flat.shape == (rnn.n_params(h),)
        assert h == 16 and flat.size == 321 or h == 8 and flat.size == 97
        back = sc.from_flat(flat, h)
        assert set(back) == set(rnn.KEYS) and all(back[k].shape == rnn.shapes(h)[k] for k in rnn.KEYS)
        assert all(np.array_equal(back[k], np.asarray(w[k], dtype=np.float32)) for k in rnn.KEYS)
        for p, i in sc.coords(h):
            assert flat[flat_index(p, i, h)] == np.float32(w[p].flat[i] if p != "c" else w[p])
        assert np.array_equal(sc.to_flat(back), flat)


def test_clip_box_keeps_every_coordinate_within_max_norm():
    d0 = (np.arange(97, dtype=np.float32) * 0.37 - 10).astype(np.float32)
    v = sc.clip_box(d0 + np.float32(2.0), d0, 1.0)
    assert np.all(np.abs(v.astype(np.float64) - d0) <= 1.0) and np.all(v > d0)
    inside = (d0 + np.float32(0.25)).astype(np.float32)
    assert np.array_equal(sc.clip_box(inside, d0, 1.0), inside)


# ------------------------------------------------------------ optimisers


def _box(vec, d0, max_norm):
    return bool(np.all(np.abs(vec.astype(np.float64) - d0.astype(np.float64)) <= max_norm))


@pytest.mark.parametrize("budget", [10, 30])
def test_hill_climb_exact_budget_monotone_and_in_box(bundle, budget):
    ev = sc.Evaluator(bundle.initial)
    support = sc.support_random(random.Random(7), range(N), 24)
    assert len(support) == 24 == len(set(support)) and support == sorted(support)
    vec, trace = sc.hill_climb(ev, bundle.d0_flat, support, bundle.d0_flat, 0.05, budget, np.random.default_rng(7), 0.1)
    assert ev.evaluations == budget == trace["evaluations"] and ev.forward_passes == budget * len(bundle.initial)
    objs = [trace["f0"]] + [f for _, f in trace["accepted"]]
    assert all(b < a for a, b in zip(objs, objs[1:])) and trace["final"] == objs[-1]
    assert trace["final"] == pytest.approx(tb.loss(sc.from_flat(vec, 8), bundle.initial))
    assert _box(vec, bundle.d0_flat, 0.05) and trace["n_accepted"] > 0
    changed = np.flatnonzero(vec != bundle.d0_flat)
    assert set(changed.tolist()) <= set(support)
    # with the incumbent value supplied, every evaluation is a candidate
    ev2 = sc.Evaluator(bundle.initial)
    _, t2 = sc.hill_climb(ev2, bundle.d0_flat, support, bundle.d0_flat, 1.0, 5, np.random.default_rng(1), 0.1,
                          f0=trace["f0"])
    assert ev2.evaluations == 5 == t2["evaluations"]


@pytest.mark.parametrize("budget", [10, 30])
def test_es_dense_exact_budget_monotone_sigma_rule_and_in_box(bundle, budget):
    ev = sc.Evaluator(bundle.initial)
    vec, trace = sc.es_dense(ev, bundle.d0_flat, bundle.d0_flat, 0.05, budget, np.random.default_rng(3), 0.05)
    assert ev.evaluations == budget == trace["evaluations"]
    objs = [trace["f0"]] + [f for _, f, _ in trace["accepted"]]
    assert all(b < a for a, b in zip(objs, objs[1:]))
    assert _box(vec, bundle.d0_flat, 0.05)
    # replay the one-fifth rule: ×1.5 after a success, ×1.5^(-1/4) after a failure, bounded
    sigma, accepted = 0.05, {e for e, _, _ in trace["accepted"]}
    for e in range(2, budget + 1):
        sigma = min(max(sigma * (1.5 if e in accepted else 1.5 ** -0.25), 1e-4), 1.0)
    assert trace["sigma"] == pytest.approx(sigma)
    assert 1e-4 <= trace["sigma"] <= 1.0


# ------------------------------------------------------------- supports


def test_support_probe_noop_below_cost_and_top_k_otherwise(bundle):
    ev = sc.Evaluator(bundle.initial)
    allowed = list(range(N))
    assert sc.support_probe(ev, bundle.d0_flat, allowed, 24, 0.1, 30, d0_flat=bundle.d0_flat) == ([], 0)
    assert sc.support_probe(ev, bundle.d0_flat, allowed, 24, 0.1, N, d0_flat=bundle.d0_flat) == ([], 0)
    assert ev.evaluations == 0
    result = sc.support_probe(ev, bundle.d0_flat, allowed, 24, 0.1, PROBE_BUDGET, d0_flat=bundle.d0_flat)
    assert isinstance(result, tuple) and len(result) == 2  # the frozen public contract: (support, evals_used)
    support, used = result
    assert used == 1 + N == ev.evaluations and len(support) == 24
    f0 = tb.loss(sc.from_flat(bundle.d0_flat, 8), bundle.initial)
    # the private helper used by run_cell also hands back the incumbent value it charged
    ev_p = sc.Evaluator(bundle.initial)
    assert sc._probe(ev_p, bundle.d0_flat, allowed, 24, 0.1, N, bundle.d0_flat, 1.0, (), None) == ([], 0, None)
    p_support, p_used, p_f0 = sc._probe(ev_p, bundle.d0_flat, allowed, 24, 0.1, PROBE_BUDGET, bundle.d0_flat, 1.0, (), None)
    assert (p_support, p_used) == (support, used) and p_f0 == pytest.approx(f0) and ev_p.evaluations == used
    # independent finite differences: the support is the k largest |Δobjective|
    deltas = []
    for j in allowed:
        cand = bundle.d0_flat.copy()
        cand[j] += np.float32(0.1)
        cand = sc.clip_box(cand, bundle.d0_flat, 1.0)
        deltas.append(abs(tb.loss(sc.from_flat(cand, 8), bundle.initial) - f0))
    top = sorted(sorted(allowed, key=lambda j: (-deltas[j], j))[:24])
    assert support == top
    # the climber receives exactly the remainder and never re-charges the incumbent
    _, trace = sc.hill_climb(ev, bundle.d0_flat, support, bundle.d0_flat, 1.0, PROBE_BUDGET - used,
                             np.random.default_rng(0), 0.1, f0=f0)
    assert trace["evaluations"] == 5 and ev.evaluations == PROBE_BUDGET


def test_support_fill_caps_new_coordinates_under_the_distinct_cap():
    touched = set(range(40))
    rng = random.Random(0)
    support = sc.support_random(rng, range(N), 24, touched, max_new=8)
    assert len(support) == 24 and len(set(support) - touched) <= 8
    only_touched = sc.support_random(rng, sorted(touched), 24, touched, max_new=0)
    assert set(only_touched) <= touched and len(only_touched) == 24
    assert sc._fill([5, 1, 50, 2, 60, 70], {1, 2}, 1, 4) == [5, 1, 2]  # at most one new, touched freely, short of k


def test_probe_cell_records_noop_rounds_and_charges_incumbent_once():
    noop = sc.run_cell(1, "rs-probe24", 30, 0, CFG)
    assert noop["status"] == "ok"
    for r in noop["rounds"]:
        assert r["noop"] and r["evaluations"] == 0 == r["forward_passes"] == r["probe_evaluations"]
        assert r["changed_from_round_start"] == 0 and r["support"] == [] and r["edits"] == []
        assert r["n_allowed"] == N and r["trace"] is None
    assert noop["totals"]["evaluations"] == 0 and noop["totals"]["oracle_labels"] == 96
    cell = sc.run_cell(1, "rs-probe24", PROBE_BUDGET, 0, CFG)
    assert cell["status"] == "ok"
    r1 = cell["rounds"][0]
    assert not r1["noop"] and r1["n_allowed"] == N and r1["probe_evaluations"] == 1 + N
    assert r1["climb_evaluations"] == 5 and r1["evaluations"] == PROBE_BUDGET
    assert r1["forward_passes"] == PROBE_BUDGET * r1["n_labels"] and r1["trace"]["evaluations"] == 5
    for r in cell["rounds"]:
        assert r["evaluations"] == (0 if r["noop"] else PROBE_BUDGET)
        assert r["probe_evaluations"] + r["climb_evaluations"] == r["evaluations"]


# -------------------------------------------------------- budgeted cells


def _check_edits(d0, rounds, h=8):
    """Replays the recorded edits through a fresh EditBudget from D0: every round's batch passes check."""
    eb, cur = EditBudget(d0, 24, 48, 1.0), sc._f32(d0)
    for r in rounds:
        edits = [(p, i, v) for p, i, v in r["edits"]]
        assert eb.check(edits, cur) is None
        cur = eb.apply(edits, cur)
        assert np.array_equal(sc.to_flat(cur), np.array(r["weights"], dtype=np.float32))
    return eb


@pytest.mark.parametrize("optimiser", ["rs-random24", "rs-probe24"])
def test_budgeted_optimisers_respect_24_per_round_and_48_distinct(bundle, optimiser):
    budget = 30 if optimiser == "rs-random24" else PROBE_BUDGET
    cell = sc.run_cell(1, optimiser, budget, 0, CFG)
    assert cell["status"] == "ok" and len(cell["rounds"]) == 3
    seen = set()
    for r in cell["rounds"]:
        assert r["evaluations"] == (0 if r["noop"] else budget)
        assert len(r["support"]) <= 24 and len(r["edits"]) <= 24 and r["changed_from_round_start"] <= 24
        assert r["budget"]["n_touched"] <= 48 and r["n_cumulative_touched"] <= 48
        assert r["max_displacement_from_D0"] <= 1.0 and r["budget"]["displacement"] <= 1.0
        assert len(r["support_names"]) == len(r["support"])
        new = set(r["support"]) - seen
        assert len(new) <= r["remaining_new"]
        seen |= {flat_index(p, i, 8) for p, i, _ in r["edits"]}
    eb = _check_edits(bundle.d0, cell["rounds"])
    assert eb.state() == cell["budget_state"] and len(eb.touched) <= 48
    assert cell["rounds"][-1]["cumulative_touched"] == sorted(flat_index(p, i, 8) for p, i in eb.touched)
    assert cell["rounds"][-1]["changed_from_round_start"] > 0 or cell["rounds"][0]["changed_from_round_start"] > 0


def test_es_dense_may_change_more_than_24(bundle):
    cell = sc.run_cell(1, "es-dense", 30, 0, CFG)
    assert cell["status"] == "ok" and cell["budget_state"] is None
    assert max(r["changed_from_round_start"] for r in cell["rounds"]) > 24
    for r in cell["rounds"]:
        assert r["evaluations"] == 30 and r["max_displacement_from_D0"] <= 1.0
        assert _box(np.array(r["weights"], dtype=np.float32), bundle.d0_flat, 1.0)
        assert "support" not in r and "edits" not in r


# ----------------------------------------------------------- accounting


def test_diagnostic_logits_count_every_uncharged_pass(bundle):
    """Item 13: per round, the round-start objective (labels) plus the endpoint passes (held-out, long,
    labelled objective and accuracy); D0 counts its training-history loss and accuracy passes too."""
    per_round = lambda n_labels: CFG["n_heldout"] + CFG["n_long"] + 3 * n_labels  # noqa: E731
    cell = sc.run_cell(1, "rs-random24", 10, 0, CFG)
    for r in cell["rounds"]:
        assert r["diagnostic_logits"] == per_round(r["n_labels"])
    assert cell["totals"]["diagnostic_logits"] == sum(per_round(r["n_labels"]) for r in cell["rounds"])
    assert cell["totals"]["search_logits"] == sum(10 * r["n_labels"] for r in cell["rounds"])
    ref = sc.run_reference(1, CFG)
    assert ref["status"] == "ok"
    for r in ref["rounds"]:
        assert r["diagnostic_logits"] == per_round(r["n_labels"])
        assert r["gradient_string_passes"] == CFG["adam_steps"] * r["n_labels"]
    t = ref["totals"]
    assert t["diagnostic_logits"] == sum(per_round(r["n_labels"]) for r in ref["rounds"])
    assert t["gradient_string_passes"] == sum(r["gradient_string_passes"] for r in ref["rounds"])
    assert t["oracle_labels"] == 96 and t["wall_s"] >= 0 and t["cpu_s"] >= 0 and t["search_logits"] == 0
    d0 = bundle.d0_endpoints()
    w = d0["work"]
    assert w["history_diagnostic_logits"] == 2 * CFG["n_initial"] * len(d0["train_history"]) == 2 * 16 * 1
    assert w["endpoint_diagnostic_logits"] == CFG["n_heldout"] + CFG["n_long"] + 2 * CFG["n_initial"]
    assert w["diagnostic_logits"] == w["history_diagnostic_logits"] + w["endpoint_diagnostic_logits"]
    assert w["gradient_string_passes"] == CFG["train_steps"] * CFG["n_initial"]
    assert w["wall_s"] >= 0 and w["cpu_s"] >= 0


# ----------------------------------------------------------- determinism


def test_determinism_same_cell_reproduces_and_replicates_differ():
    a = sc.run_cell(1, "rs-random24", 10, 0, CFG)
    b = sc.run_cell(1, "rs-random24", 10, 0, CFG)
    assert [r["weights"] for r in a["rounds"]] == [r["weights"] for r in b["rounds"]]
    assert [r["support"] for r in a["rounds"]] == [r["support"] for r in b["rounds"]]
    c = sc.run_cell(1, "rs-random24", 10, 1, CFG)
    assert c["S"] == 1001 and a["S"] == 1000
    assert [r["weights"] for r in a["rounds"]] != [r["weights"] for r in c["rounds"]]
    d, e = sc.run_cell(1, "es-dense", 10, 0, CFG), sc.run_cell(1, "es-dense", 10, 0, CFG)
    assert [r["weights"] for r in d["rounds"]] == [r["weights"] for r in e["rounds"]]
    assert d["rounds"][0]["weights"] != sc.run_cell(1, "es-dense", 10, 1, CFG)["rounds"][0]["weights"]


# --------------------------------------------------------------- outcome


def _cell(o, B, i, heldout, rounds=3, status="ok"):
    return {"optimiser": o, "budget": B, "replicate": i, "status": status,
            "rounds": [{"heldout": heldout - 0.1}] * (rounds - 1) + [{"heldout": heldout}] if rounds else []}


def _rec(seed, gains, d0=0.6, status="ok", reference=None):
    """gains: {(optimiser, B): [replicate gains]}; held-out after round 3 = d0 + gain; three rounds per cell
    and an ok three-round reference unless `reference` is given."""
    cells = [_cell(o, B, i, d0 + g) for (o, B), gs in gains.items() for i, g in enumerate(gs)]
    ref = {"status": "ok", "rounds": [{"heldout": d0}] * 3} if reference is None else reference
    return {"seed": seed, "status": status, "failure": None if status == "ok" else "boom",
            "D0": {"heldout": d0}, "cells": cells, "reference": ref}


def _grid(per_seed):
    return [_rec(s, {(o, B): per_seed(s, o, B) for o in sc.OPTIMISERS for B in sc.BUDGETS}) for s in sc.DEV_SEEDS]


def test_outcome_vocabulary_gate_and_precedence():
    f = sc.F
    good, bad = [0.05, 0.05, 0.05], [0.0, 0.0, 0.0]
    # 3 of 4 repaired at B=1000 only (seed 28 never) -> calibrated_at_1000 by rs-random24
    oc = sc.outcome(_grid(lambda s, o, B: good if (o == "rs-random24" and B == 1000 and s != 28) else bad))
    assert oc["outcome"] == "calibrated_at_1000" and oc["calibrated_optimiser"] == "rs-random24"
    assert oc["qualifying"] == ["rs-random24"] and oc["cells"]["rs-random24/B1000"]["n_repaired"] == 3
    assert oc["cells"]["rs-random24/B100"]["n_repaired"] == 0 and oc["missing"] == [] and oc["failed"] == []
    assert sc.outcome(_grid(lambda s, o, B: bad))["outcome"] == "not_calibrated"
    # precedence: the smallest B wins even when a larger B repairs more seeds
    oc = sc.outcome(_grid(lambda s, o, B: good if (o == "rs-probe24" and (B == 10000 or (B == 100 and s != 4))) else bad))
    assert oc["outcome"] == "calibrated_at_100" and oc["calibrated_optimiser"] == "rs-probe24"
    # es-dense repairing every seed never satisfies the gate
    oc = sc.outcome(_grid(lambda s, o, B: good if o == "es-dense" else bad))
    assert oc["outcome"] == "not_calibrated" and oc["cells"]["es-dense/B100"]["n_repaired"] == 4
    assert not oc["cells"]["es-dense/B100"]["eligible"] and oc["cells"]["rs-random24/B100"]["eligible"]
    # a 1-of-3 replicate set with a high mean is not repaired; 2 of 3 above f with mean above f is
    one_high = [0.5, 0.0, 0.0]
    oc = sc.outcome(_grid(lambda s, o, B: one_high if o == "rs-random24" and B == 100 else bad))
    assert oc["outcome"] == "not_calibrated" and oc["cells"]["rs-random24/B100"]["n_repaired"] == 0
    assert oc["cells"]["rs-random24/B100"]["per_seed"]["4"]["mean_gain"] > f
    two_up = [0.03, 0.03, 0.01]  # two gains above f and a mean of 0.0233 > f
    oc = sc.outcome(_grid(lambda s, o, B: two_up if o == "rs-random24" and B == 100 else bad))
    assert oc["cells"]["rs-random24/B100"]["n_repaired"] == 4 and oc["outcome"] == "calibrated_at_100"
    # ties at the smallest B: repaired count, then four-seed mean gain, then name
    oc = sc.outcome(_grid(lambda s, o, B: ([0.06] * 3 if o == "rs-probe24" else good) if B == 100 and o != "es-dense" else bad))
    assert oc["outcome"] == "calibrated_at_100" and oc["calibrated_optimiser"] == "rs-probe24"
    assert oc["qualifying"] == ["rs-probe24", "rs-random24"]
    oc = sc.outcome(_grid(lambda s, o, B: good if B == 100 and o != "es-dense" else bad))
    assert oc["calibrated_optimiser"] == "rs-probe24"  # lexical tie-break
    assert all(o in sc.OUTCOMES for o in ("calibrated_at_100", "calibrated_at_1000", "calibrated_at_10000",
                                           "not_calibrated", "no_result"))


def test_outcome_missing_or_failed_seed_is_no_result():
    recs = _grid(lambda s, o, B: [0.05] * 3)
    assert sc.outcome(recs)["outcome"] == "calibrated_at_100"
    oc = sc.outcome(recs[:3])
    assert oc["outcome"] == "no_result" and oc["missing"] == [28] and oc["calibrated_optimiser"] is None
    failed = recs[:3] + [_rec(28, {}, status="failed")]
    oc = sc.outcome(failed)
    assert oc["outcome"] == "no_result" and oc["failed"] == [{"seed": 28, "failure": "boom"}]
    assert oc["cells"]["rs-random24/B100"]["n_repaired"] == 3 and oc["cells"]["rs-random24/B100"]["mean_gain"] is None
    # seeds outside the roster are reported but never count
    assert sc.outcome(recs + [_rec(1, {("rs-random24", 100): [0.05] * 3})])["outcome"] == "calibrated_at_100"
    assert sc.outcome([_rec(1, {("rs-random24", 100): [0.05] * 3})])["outcome"] == "no_result"


def test_outcome_incomplete_development_seed_is_no_result():
    """Amendment 1 item 8: an ok seed missing any part of the frozen grid never qualifies, even when the
    cells it has would repair it."""
    good = [0.05] * 3
    complete = _grid(lambda s, o, B: good)
    assert sc.outcome(complete)["outcome"] == "calibrated_at_100" and sc.outcome(complete)["incomplete"] == []

    def with_seed4(rec):
        return [rec] + complete[1:]

    def problems(recs):
        oc = sc.outcome(recs)
        assert oc["outcome"] == "no_result" and oc["calibrated_optimiser"] is None and oc["qualifying"] == []
        assert oc["missing"] == [] and oc["failed"] == [] and [x["seed"] for x in oc["incomplete"]] == [4]
        return oc, oc["incomplete"][0]["problems"]

    full = {(o, B): good for o in sc.OPTIMISERS for B in sc.BUDGETS}
    # a missing optimiser × budget cell (an eligible one and a descriptive one alike)
    oc, p = problems(with_seed4(_rec(4, {k: v for k, v in full.items() if k != ("rs-probe24", 10000)})))
    assert any("rs-probe24/B10000" in x for x in p)
    oc, p = problems(with_seed4(_rec(4, {k: v for k, v in full.items() if k != ("es-dense", 100)})))
    assert any("es-dense/B100" in x for x in p)
    # only two replicates: two good gains never mark the seed repaired
    rec = _rec(4, {**full, ("rs-random24", 100): [0.05, 0.05]})
    oc, p = problems(with_seed4(rec))
    assert oc["cells"]["rs-random24/B100"]["per_seed"]["4"]["repaired"] is False
    assert oc["cells"]["rs-random24/B100"]["per_seed"]["4"]["n"] == 2 and oc["cells"]["rs-random24/B100"]["n_repaired"] == 3
    assert oc["cells"]["rs-random24/B100"]["mean_gain"] is None
    # four replicates, or a duplicated replicate index
    oc, p = problems(with_seed4(_rec(4, {**full, ("rs-random24", 100): [0.05] * 4})))
    assert any("rs-random24/B100 replicates [0, 1, 2, 3]" in x for x in p)
    rec = _rec(4, full)
    rec["cells"].append(_cell("rs-random24", 100, 1, 0.65))
    oc, p = problems(with_seed4(rec))
    assert any("replicates [0, 1, 1, 2]" in x for x in p)
    assert oc["cells"]["rs-random24/B100"]["per_seed"]["4"]["repaired"] is False
    # a failed cell inside an ok seed, or a cell with fewer than three rounds
    rec = _rec(4, full)
    rec["cells"][0] = _cell("rs-random24", 100, 0, 0.65, status="failed")
    oc, p = problems(with_seed4(rec))
    assert any("status 'failed'" in x for x in p)
    rec = _rec(4, full)
    rec["cells"][0] = _cell("rs-random24", 100, 0, 0.65, rounds=2)
    oc, p = problems(with_seed4(rec))
    assert any("has 2 rounds, not 3" in x for x in p)
    assert oc["cells"]["rs-random24/B100"]["per_seed"]["4"]["n"] == 2  # the short cell contributes no gain
    # a missing or failed reference, or one with too few rounds
    oc, p = problems(with_seed4(_rec(4, full, reference={"status": "failed", "rounds": []})))
    assert "reference status 'failed'" in p
    oc, p = problems(with_seed4(_rec(4, full, reference={"status": "ok", "rounds": [{"heldout": 0.6}] * 2})))
    assert "reference has 2 rounds, not 3" in p
    rec = _rec(4, full)
    del rec["reference"]
    oc, p = problems(with_seed4(rec))
    assert "reference status None" in p
    # incompleteness outside the roster never matters
    extra = _rec(1, {("rs-random24", 100): [0.05, 0.05]})
    assert sc.outcome(complete + [extra])["outcome"] == "calibrated_at_100"
    assert sc.outcome(complete + [extra])["incomplete"] == []


# ------------------------------------------------------------------- CLI


def _run(out, seeds="1,2"):
    return sc.main(["run", "--seeds", seeds, "--optimisers", "rs-random24,rs-probe24,es-dense", "--budgets", "10,30",
                    "--replicates", "3", "--k", "4", "--hidden", "8", "--n-initial", "16", "--n-heldout", "64",
                    "--n-long", "32", "--train-steps", "100", "--out", str(out)])


def test_cli_run_writes_cells_seed_files_meta_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_git_commit", lambda: "abc123")
    monkeypatch.setitem(sc.DEFAULT_CFG, "adam_steps", 50)
    assert _run(tmp_path) == 0
    cells = sorted(p.name for p in (tmp_path / "cells").glob("*.json"))
    assert len(cells) == 2 * (3 * 2 * 3 + 1) and "1_rs-probe24_B30_r2.json" in cells and "2_reference.json" in cells
    meta = json.loads((tmp_path / "study_meta.json").read_text())
    assert meta["git_commit"] == "abc123" and meta["config"]["hidden"] == 8 and meta["config"]["budgets"] == "10,30"
    for s in (1, 2):
        rec = json.loads((tmp_path / f"seed_{s}.json").read_text())
        assert rec["status"] == "ok" and rec["failure"] is None and len(rec["cells"]) == 18
        assert rec["reference"]["status"] == "ok" and len(rec["batches"]) == 3
        assert set(rec["ledger"]) == {c["arm"] for c in rec["cells"]} | {"reference"}
        assert all(e["total"] == 96 for e in rec["ledger"].values())
        d0 = np.array(rec["D0"]["weights"], dtype=np.float32)
        for c in rec["cells"]:
            assert c["status"] == "ok" and c["totals"]["oracle_labels"] == 96 and c["totals"]["wall_s"] >= 0
            for r in c["rounds"]:
                assert r["evaluations"] == (0 if r["noop"] else c["budget"])
                assert r["forward_passes"] == r["evaluations"] * r["n_labels"]
                assert _box(np.array(r["weights"], dtype=np.float32), d0, 1.0)
                if c["optimiser"] != "es-dense":
                    assert r["changed_from_round_start"] <= 24 and r["n_cumulative_touched"] <= 48
                    assert r["budget"]["n_touched"] <= 48
        for r in rec["reference"]["rounds"]:
            assert r["n_touched"] <= 48 and _box(np.array(r["weights"], dtype=np.float32), d0, 1.0)
    assert sc.main(["report", "--out", str(tmp_path)]) == 0
    study = json.loads((tmp_path / "study.json").read_text())
    md = (tmp_path / "study.md").read_text()
    assert study["outcome"]["outcome"] == "no_result" and "- outcome: no_result" in md
    assert study["outcome"]["missing"] == list(sc.DEV_SEEDS)
    assert {(r["optimiser"], r["budget"], r["seed"]) for r in study["cells"]} == {
        (o, B, s) for o in sc.OPTIMISERS for B in (10, 30) for s in (1, 2)}
    assert len(study["totals"]) == 36 and len(study["reference"]) == 2 and len(study["work"]) == 2
    assert all(t["search_logits"] >= 0 and t["diagnostic_logits"] > 0 for t in study["totals"])
    for w in study["work"]:
        assert w["reference_status"] == "ok" and w["reference_oracle_labels"] == 96
        for k in ("D0_gradient_string_passes", "D0_diagnostic_logits", "reference_gradient_string_passes",
                  "reference_diagnostic_logits"):
            assert w[k] > 0
        for k in ("D0_wall_s", "D0_cpu_s", "reference_wall_s", "reference_cpu_s"):
            assert w[k] >= 0
    assert "| optimiser | B | seed |" in md and "Totals per cell" in md and "## Study meta" in md
    assert "D0 diagnostic logits" in md and "reference cpu s" in md and "- incomplete seeds: []" in md
    # a second run skips every ok cell and leaves the cell files untouched
    before = {p: p.stat().st_mtime_ns for p in (tmp_path / "cells").glob("*.json")}
    assert _run(tmp_path) == 0
    assert {p: p.stat().st_mtime_ns for p in (tmp_path / "cells").glob("*.json")} == before


def test_run_seed_marks_a_failed_cell_and_keeps_the_rest(monkeypatch):
    real = sc.hill_climb

    def broken(ev, vec, support, *a, **k):
        if len(support) == 24 and ev.evaluations == 0:
            raise AssertionError("invariant violated")
        return real(ev, vec, support, *a, **k)

    monkeypatch.setattr(sc, "hill_climb", broken)
    cfg = {**CFG, "optimisers": ["rs-random24", "es-dense"], "budgets": [10], "replicates": 1}
    rec = sc.run_seed(1, cfg)
    assert rec["status"] == "failed" and "invariant violated" in rec["failure"]
    assert [c["status"] for c in rec["cells"]] == ["failed", "ok"] and rec["reference"]["status"] == "ok"
    assert sc.outcome([rec], seeds=(1,))["outcome"] == "no_result"
