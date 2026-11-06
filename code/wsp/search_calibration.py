"""Comparator calibration: black-box search under the rollout budget (rules/search-calibration-rule.md with
Amendment 1; the amendment's thirteen items override the body). No model calls anywhere.

    uv run python -m wsp.search_calibration run --seeds 4,6,25,28 --out DIR
    uv run python -m wsp.search_calibration report --out DIR

Construction per seed as wsp.repair_study.run_seed (same D0, labels, held-out sets and query batches). One
evaluation = mean BCE of one candidate on the full current labelled set (item 13: every uncached call counts).
Optimisers see flat float32 weight vectors in the wsp.numeric_edit coordinate order (Wx, Wh, bh, w, c).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wsp import projected_gradient as pg
from wsp import rnn, trained_baseline as tb
from wsp.dfa import random_dfa
from wsp.numeric_edit import EditBudget, _f32, coordinate_name, flat_index, param_sizes
from wsp.oracle import Oracle
from wsp.query_study import _git_commit

F = pg.F
DEV_SEEDS = (4, 6, 25, 28)
BUDGETS = (100, 1000, 10000)
OPTIMISERS = ("rs-random24", "rs-probe24", "es-dense")
GATE_ELIGIBLE = ("rs-random24", "rs-probe24")   # Amendment 1 item 1: es-dense is descriptive, never eligible
REPLICATES = 3                                  # item 6: exactly three replicates per (seed, optimiser, B)
ROUNDS = 3                                      # three rounds per cell
MIN_REPAIRED = 3                                # of the four development seeds
MIN_REPLICATES_ABOVE_F = 2                      # item 6: at least 2 of 3 gains > f and their mean > f
OUTCOMES = tuple(f"calibrated_at_{B}" for B in BUDGETS) + ("not_calibrated", "no_result")
RULE = "rules/search-calibration-rule.md"
DEFAULT_CFG = {"k": 6, "h": 16, "n_initial": 32, "rounds": 3, "n_queries": 32, "n_heldout": 500, "n_long": 200,
               "train_steps": 2000, "adam_steps": 200, "per_round": 24, "distinct": 48, "max_norm": 1.0,
               "optimisers": list(OPTIMISERS), "budgets": list(BUDGETS), "replicates": 3,
               "sigma": 0.1, "sigma0": 0.05, "eps": 0.1}
_BUNDLE_KEYS = ("k", "h", "n_initial", "rounds", "n_queries", "n_heldout", "n_long", "train_steps")


# ---------------------------------------------------------------- evaluator


class Evaluator:
    """The optimiser's only view of the labels: `objective` is one charged evaluation (mean BCE on the full
    current labelled set, len(labels) string logits); `accuracy` is diagnostic and never counted."""

    def __init__(self, labels):
        self.labels = list(labels)
        self.evaluations = 0
        self.forward_passes = 0

    def objective(self, weights) -> float:
        self.evaluations += 1
        self.forward_passes += len(self.labels)
        return tb.loss(weights, self.labels)

    def accuracy(self, weights) -> float:
        return tb.accuracy(weights, self.labels)


# -------------------------------------------------------------- flat view


def _h_of(n: int) -> int:
    h = int(round(-2 + (3 + n) ** 0.5))
    if rnn.n_params(h) != n:
        raise ValueError(f"{n} is not a parameter count of the wsp.rnn architecture")
    return h


def to_flat(weights) -> np.ndarray:
    """float32 vector in the numeric_edit order: to_flat(w)[flat_index(p, i, h)] == w[p].flat[i]."""
    w = _f32(weights)
    return np.concatenate([w[k].ravel() for k in rnn.KEYS]).astype(np.float32)


def from_flat(vec, h: int) -> dict[str, np.ndarray]:
    vec = np.asarray(vec, dtype=np.float32)
    out, pos = {}, 0
    for k, shape in rnn.shapes(h).items():
        n = int(np.prod(shape, dtype=int))
        out[k] = vec[pos:pos + n].reshape(shape).copy()
        pos += n
    return out


def coords(h: int) -> list[tuple[str, int]]:
    return [(k, i) for k in rnn.KEYS for i in range(param_sizes(h)[k])]


def clip_box(vec, d0_flat, max_norm: float) -> np.ndarray:
    """vec clipped to [d0 - max_norm, d0 + max_norm] coordinate-wise, as float32 values that EditBudget.check
    accepts (float32 rounding may land a ulp outside the ball; stepped back toward d0)."""
    d0 = np.asarray(d0_flat, dtype=np.float64)
    v = np.clip(np.asarray(vec, dtype=np.float64), d0 - max_norm, d0 + max_norm).astype(np.float32)
    while True:
        bad = np.abs(v.astype(np.float64) - d0) > max_norm
        if not bad.any():
            return v
        v[bad] = np.nextafter(v[bad], d0[bad].astype(np.float32))


# ---------------------------------------------------------------- supports


def _fill(ranked, touched, max_new, k: int) -> list[int]:
    """Amendment 1 item 3: walk `ranked`, take touched coordinates freely and at most `max_new` new ones, stop
    at k. max_new None = no cap."""
    out, n_new = [], 0
    for j in ranked:
        if len(out) == k:
            break
        if j in touched:
            out.append(j)
        elif max_new is None or n_new < max_new:
            out.append(j)
            n_new += 1
    return out


def support_random(rng: random.Random, allowed, k: int = 24, touched=(), max_new=None) -> list[int]:
    """Uniform draw of k coordinates from `allowed` (a uniform permutation filled under item 3)."""
    allowed = list(allowed)
    return sorted(_fill(rng.sample(allowed, len(allowed)), set(touched), max_new, k))


def _probe(ev: Evaluator, vec, allowed, k: int, eps: float, budget_left: int, d0_flat, max_norm: float,
           touched, max_new) -> tuple[list[int], int, float | None]:
    """support_probe plus the incumbent objective it charged, (support, evals_used, incumbent_objective), so
    run_cell can hand the value to the climber without re-charging it. ([], 0, None) on a no-op."""
    allowed = list(allowed)
    cost = 1 + len(allowed)
    if budget_left < cost:
        return [], 0, None
    h = _h_of(len(vec))
    f0 = ev.objective(from_flat(vec, h))
    deltas = []
    for j in allowed:
        cand = np.array(vec, dtype=np.float32)
        cand[j] += np.float32(eps)
        if d0_flat is not None:
            cand = clip_box(cand, d0_flat, max_norm)
        deltas.append(abs(ev.objective(from_flat(cand, h)) - f0))
    ranked = [j for _, j in sorted(zip(deltas, allowed), key=lambda t: (-t[0], t[1]))]
    return sorted(_fill(ranked, set(touched), max_new, k)), cost, f0


def support_probe(ev: Evaluator, vec, allowed, k: int = 24, eps: float = 0.1, budget_left: int = 0, *,
                  d0_flat=None, max_norm: float = 1.0, touched=(), max_new=None) -> tuple[list[int], int]:
    """Amendment 1 item 2: one incumbent evaluation plus one box-clipped +eps evaluation per allowed
    coordinate, ranked by |Δobjective| (ties by lower index), filled under item 3. Returns
    (support, evals_used); ([], 0) when budget_left < 1 + len(allowed), which makes the round a recorded
    no-op. The incumbent evaluation is charged here once; the climber receives exactly budget_left - evals_used
    (run_cell reuses the incumbent value through the private _probe)."""
    support, used, _ = _probe(ev, vec, allowed, k, eps, budget_left, d0_flat, max_norm, touched, max_new)
    return support, used


# -------------------------------------------------------------- optimisers


def hill_climb(ev: Evaluator, vec, support, d0_flat, max_norm: float, budget: int, rng: np.random.Generator,
               sigma: float = 0.1, f0: float | None = None):
    """(1+1) hill climbing with Gaussian steps on `support` only, clipped to the box, accepting iff strictly
    lower. Exactly `budget` evaluations: the incumbent is evaluated once and counted unless its value `f0` is
    supplied (the probe's incumbent evaluation), in which case every evaluation is a candidate."""
    h = _h_of(len(vec))
    cur, support = np.array(vec, dtype=np.float32), list(support)
    used = 0
    if f0 is None:
        if budget < 1:
            return cur, {"f0": None, "final": None, "evaluations": 0, "n_accepted": 0, "accepted": [], "sigma": sigma}
        f0 = ev.objective(from_flat(cur, h))
        used = 1
    f_cur, accepted = f0, []
    while used < budget:
        cand = cur.copy()
        cand[support] += rng.normal(0.0, sigma, len(support)).astype(np.float32)
        cand = clip_box(cand, d0_flat, max_norm)
        f = ev.objective(from_flat(cand, h))
        used += 1
        if f < f_cur:
            cur, f_cur = cand, f
            accepted.append([used, f])
    return cur, {"f0": f0, "final": f_cur, "evaluations": used, "n_accepted": len(accepted), "accepted": accepted,
                 "sigma": sigma}


def es_dense(ev: Evaluator, vec, d0_flat, max_norm: float, budget: int, rng: np.random.Generator,
             sigma0: float = 0.05):
    """(1+1)-ES with dense Gaussian steps; Amendment 1 item 4: σ × 1.5 after a success, σ × 1.5^(-1/4) after a
    failure, bounded to [1e-4, 1.0]; clipped to the box; exactly `budget` evaluations."""
    h = _h_of(len(vec))
    cur, sigma = np.array(vec, dtype=np.float32), float(sigma0)
    if budget < 1:
        return cur, {"f0": None, "final": None, "evaluations": 0, "n_accepted": 0, "accepted": [], "sigma": sigma}
    f0 = ev.objective(from_flat(cur, h))
    used, f_cur, accepted = 1, f0, []
    while used < budget:
        cand = clip_box(cur + rng.normal(0.0, sigma, cur.size).astype(np.float32), d0_flat, max_norm)
        f = ev.objective(from_flat(cand, h))
        used += 1
        if f < f_cur:
            cur, f_cur = cand, f
            sigma *= 1.5
            accepted.append([used, f, sigma])
        else:
            sigma *= 1.5 ** (-0.25)
        sigma = min(max(sigma, 1e-4), 1.0)
    return cur, {"f0": f0, "final": f_cur, "evaluations": used, "n_accepted": len(accepted), "accepted": accepted,
                 "sigma": sigma}


# ------------------------------------------------------------ construction


class _Bundle:
    """The per-seed construction, built once per process: DFA, oracle, labels, held-out sets, D0, batches."""

    def __init__(self, seed: int, cfg: dict):
        t0, c0 = time.monotonic(), time.process_time()
        self.seed, self.cfg = seed, cfg
        self.dfa = random_dfa(cfg["k"], seed)
        self.oracle = Oracle(self.dfa, seed)
        self.initial = self.oracle.initial_dataset(cfg["n_initial"])
        self.heldout = self.oracle.heldout(cfg["n_heldout"])
        self.long = self.oracle.long_set(cfg["n_long"])
        d0_raw, self.history = tb.train(self.initial, cfg["h"], seed, cfg["train_steps"])
        self.d0 = _f32(d0_raw)
        self.d0_flat = to_flat(self.d0)
        self.coords = coords(cfg["h"])
        self.batches = pg.query_batches(seed, [s for s, _ in self.initial], cfg["rounds"], cfg["n_queries"])
        # item 13: D0 work = the training gradient passes, the training-history loss and accuracy passes
        # (one of each over the initial set per history entry), and the endpoint diagnostics added below.
        self.d0_work = {"gradient_string_passes": cfg["train_steps"] * min(32, cfg["n_initial"]),
                        "history_diagnostic_logits": 2 * len(self.initial) * len(self.history),
                        "wall_s": time.monotonic() - t0, "cpu_s": time.process_time() - c0}

    def d0_endpoints(self) -> dict:
        t0, c0 = time.monotonic(), time.process_time()
        ends = _endpoints(self.d0_flat, self.cfg["h"], self.d0_flat, self, self.initial, set())
        work = {**self.d0_work, "endpoint_diagnostic_logits": ends["diagnostic_logits"],
                "wall_s": self.d0_work["wall_s"] + time.monotonic() - t0,
                "cpu_s": self.d0_work["cpu_s"] + time.process_time() - c0}
        work["diagnostic_logits"] = work["history_diagnostic_logits"] + work["endpoint_diagnostic_logits"]
        return {**ends, "train_history": self.history, "work": work}


_BUNDLES: dict = {}


def _bundle(seed: int, cfg: dict) -> _Bundle:
    key = (seed, json.dumps({k: cfg[k] for k in _BUNDLE_KEYS}, sort_keys=True))
    if key not in _BUNDLES:
        _BUNDLES[key] = _Bundle(seed, cfg)
    return _BUNDLES[key]


def _endpoints(flat: np.ndarray, h: int, start: np.ndarray, b: _Bundle, labels, touched_cum: set,
               extra_diagnostic_logits: int = 0) -> dict:
    """Diagnostics of round-end weights (never charged): held-out, long, labelled objective and accuracy,
    change counts and displacement. Updates `touched_cum` with the coordinates changed from round start.
    `diagnostic_logits` counts every string logit of these four passes plus `extra_diagnostic_logits`, the
    logits of diagnostic passes the caller made on the same round (the round-start objective, item 13)."""
    w = from_flat(flat, h)
    changed = np.flatnonzero(flat != start)
    touched_cum |= {int(j) for j in changed}
    return {"heldout": tb.accuracy(w, b.heldout), "long": tb.accuracy(w, b.long),
            "objective": tb.loss(w, labels), "accuracy": tb.accuracy(w, labels),
            "changed_from_round_start": int(changed.size),
            "n_cumulative_touched": len(touched_cum), "cumulative_touched": sorted(touched_cum),
            "max_displacement_from_D0": float(np.max(np.abs(flat.astype(np.float64) - b.d0_flat.astype(np.float64)))),
            "diagnostic_logits": len(b.heldout) + len(b.long) + 2 * len(labels) + int(extra_diagnostic_logits),
            "weights": flat.tolist()}


def _allowed(eb: EditBudget, n: int, h: int):
    """Coordinates still allowed under the distinct cap: every coordinate while new ones remain (the support
    may then include at most `remaining` new ones, item 3), else only the touched set."""
    touched = {flat_index(p, i, h) for p, i in eb.touched}
    remaining = eb.distinct - len(touched)
    return (list(range(n)) if remaining > 0 else sorted(touched)), touched, remaining


def _ledger_delta(before: dict | None, after: dict | None) -> dict:
    before, after = before or {"total": 0, "by_length": {}}, after or {"total": 0, "by_length": {}}
    lengths = set(before["by_length"]) | set(after["by_length"])
    by_length = {L: after["by_length"].get(L, 0) - before["by_length"].get(L, 0) for L in sorted(lengths)}
    return {"total": after["total"] - before["total"], "by_length": {k: v for k, v in by_length.items() if v}}


def _totals(rounds, ledger_entry, t0, c0) -> dict:
    return {"rounds": len(rounds), "evaluations": sum(r.get("evaluations", 0) for r in rounds),
            "search_logits": sum(r.get("forward_passes", 0) for r in rounds),
            "diagnostic_logits": sum(r.get("diagnostic_logits", 0) for r in rounds),
            "objective_decrease": sum(r["objective_start"] - r["objective"] for r in rounds),
            "oracle_labels": ledger_entry["total"], "wall_s": time.monotonic() - t0, "cpu_s": time.process_time() - c0}


# ------------------------------------------------------------------- cells


def run_cell(seed: int, optimiser: str, budget: int, replicate: int, cfg: dict | None = None) -> dict:
    """One cell = seed × optimiser × B × replicate, three rounds. S = 1000·seed + replicate seeds one persistent
    random.Random (support draws) and one persistent numpy Generator (Gaussian steps) for the cell (item 5).
    Batch r is charged under the arm name before round r's search. A failure marks the cell failed."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    if optimiser not in OPTIMISERS:
        raise ValueError(f"unknown optimiser {optimiser!r}; one of {OPTIMISERS}")
    b = _bundle(seed, cfg)
    S, h, n = 1000 * seed + replicate, cfg["h"], b.d0_flat.size
    arm = f"{optimiser}/B{budget}/r{replicate}"
    cell = {"seed": seed, "optimiser": optimiser, "budget": budget, "replicate": replicate, "arm": arm, "S": S,
            "status": "running", "failure": None, "rounds": [], "budget_state": None, "totals": None,
            "ledger_entry": None}
    rng_py, rng_np = random.Random(S), np.random.default_rng(S)
    budgeted = optimiser in GATE_ELIGIBLE
    eb = EditBudget(b.d0, cfg["per_round"], cfg["distinct"], cfg["max_norm"]) if budgeted else None
    cur, labels, touched_cum = b.d0_flat.copy(), list(b.initial), set()
    before = b.oracle.ledger().get(arm)
    t0, c0 = time.monotonic(), time.process_time()
    try:
        for r, batch in enumerate(b.batches, 1):
            ys = b.oracle.query(arm, batch)
            new = pg.add_labels(labels, batch, ys)
            ev, start = Evaluator(labels), cur.copy()
            rec = {"round": r, "n_new_labels": len(new), "n_labels": len(labels), "noop": False,
                   "probe_evaluations": 0, "climb_evaluations": 0,
                   "objective_start": tb.loss(from_flat(start, h), labels)}
            if budgeted:
                allowed, touched, remaining = _allowed(eb, n, h)
                if optimiser == "rs-random24":
                    support, used, f0 = support_random(rng_py, allowed, cfg["per_round"], touched, remaining), 0, None
                else:
                    support, used, f0 = _probe(ev, cur, allowed, cfg["per_round"], cfg["eps"], budget,
                                               b.d0_flat, cfg["max_norm"], touched, remaining)
                rec.update(n_allowed=len(allowed), remaining_new=remaining, support=support,
                           support_names=[coordinate_name(*b.coords[j], h) for j in support],
                           probe_evaluations=used)
                if optimiser == "rs-probe24" and used == 0:
                    rec["noop"], trace = True, None
                else:
                    cur, trace = hill_climb(ev, cur, support, b.d0_flat, cfg["max_norm"], budget - used, rng_np,
                                            cfg["sigma"], f0)
                    rec["climb_evaluations"] = trace["evaluations"]
                edits = [(*b.coords[j], float(cur[j])) for j in support if cur[j] != start[j]]
                reason = eb.check(edits, from_flat(start, h))
                assert reason is None, f"EditBudget rejected the round's net change: {reason}"
                applied = eb.apply(edits, from_flat(start, h))
                assert np.array_equal(to_flat(applied), cur), "applied edits differ from the climber's vector"
                rec.update(edits=[[p, i, v] for p, i, v in edits], budget=eb.state())
            else:
                cur, trace = es_dense(ev, cur, b.d0_flat, cfg["max_norm"], budget, rng_np, cfg["sigma0"])
            rec.update(trace=trace, evaluations=ev.evaluations, forward_passes=ev.forward_passes,
                       **_endpoints(cur, h, start, b, labels, touched_cum, extra_diagnostic_logits=len(labels)))
            assert rec["evaluations"] == (0 if rec["noop"] else budget), "evaluation count is not the budget"
            cell["rounds"].append(rec)
        cell["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - the cell is failed and reported; the seed then fails (item 8)
        cell.update(status="failed", failure=f"{type(e).__name__}: {e}"[:300])
    cell["budget_state"] = eb.state() if eb is not None else None
    cell["ledger_entry"] = _ledger_delta(before, b.oracle.ledger().get(arm))
    cell["totals"] = _totals(cell["rounds"], cell["ledger_entry"], t0, c0)
    return cell


def run_reference(seed: int, cfg: dict | None = None) -> dict:
    """Projected dense Adam under the edit budget (gradient_round, 200 steps) per round on the same labels.
    Reported, never a comparator: it uses gradients the optimisers do not have."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    b = _bundle(seed, cfg)
    h, arm = cfg["h"], "reference"
    ref = {"seed": seed, "arm": arm, "status": "running", "failure": None, "rounds": [], "budget_state": None,
           "totals": None, "ledger_entry": None}
    eb = EditBudget(b.d0, cfg["per_round"], cfg["distinct"], cfg["max_norm"])
    cur, labels, touched_cum = _f32(b.d0), list(b.initial), set()
    before = b.oracle.ledger().get(arm)
    t0, c0 = time.monotonic(), time.process_time()
    try:
        for r, batch in enumerate(b.batches, 1):
            ys = b.oracle.query(arm, batch)
            new = pg.add_labels(labels, batch, ys)
            start = to_flat(cur)
            rec = {"round": r, "n_new_labels": len(new), "n_labels": len(labels),
                   "objective_start": tb.loss(from_flat(start, h), labels)}
            cur, info = pg.gradient_round(cur, labels, eb, cfg["adam_steps"])
            rec.update(info, gradient_string_passes=cfg["adam_steps"] * len(labels),
                       **_endpoints(to_flat(cur), h, start, b, labels, touched_cum,
                                    extra_diagnostic_logits=len(labels)))
            ref["rounds"].append(rec)
        ref["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        ref.update(status="failed", failure=f"{type(e).__name__}: {e}"[:300])
    ref["budget_state"] = eb.state()
    ref["ledger_entry"] = _ledger_delta(before, b.oracle.ledger().get(arm))
    ref["totals"] = {**_totals(ref["rounds"], ref["ledger_entry"], t0, c0),
                     "gradient_string_passes": sum(r["gradient_string_passes"] for r in ref["rounds"])}
    return ref


def _cell_path(cell_dir: Path, seed: int, optimiser: str, budget: int, replicate: int) -> Path:
    return cell_dir / f"{seed}_{optimiser}_B{budget}_r{replicate}.json"


def _cached(path: Path | None) -> dict | None:
    if path is not None and path.is_file():
        rec = json.loads(path.read_text())
        if rec.get("status") == "ok":
            return rec
    return None


def _progress(seed: int, rec: dict, cached: bool) -> None:
    accs = [round(r["heldout"], 3) for r in rec["rounds"]]
    evals = [r.get("evaluations", 0) for r in rec["rounds"]]
    print(f"wsp.search_calibration: seed {seed} {rec['arm']} {rec['status']}{' (cached)' if cached else ''} "
          f"heldout={accs} evaluations={evals} wall={rec['totals']['wall_s']:.1f}s", file=sys.stderr)


def run_seed(seed: int, cfg: dict | None = None, cell_dir=None) -> dict:
    """Every optimiser × budget × replicate plus the reference on one construction. With `cell_dir`, each cell
    is written after it runs and existing ok cells are skipped. Status failed carries the first failure."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    cell_dir = Path(cell_dir) if cell_dir is not None else None
    if cell_dir is not None:
        cell_dir.mkdir(parents=True, exist_ok=True)
    b = _bundle(seed, cfg)
    out = {"seed": seed, "config": cfg, "status": "running", "failure": None,
           "initial": [list(t) for t in b.initial], "D0": b.d0_endpoints(), "batches": b.batches,
           "cells": [], "reference": None, "ledger": {}}
    for optimiser in cfg["optimisers"]:
        for budget in cfg["budgets"]:
            for rep in range(cfg["replicates"]):
                path = _cell_path(cell_dir, seed, optimiser, budget, rep) if cell_dir is not None else None
                cell = _cached(path)
                cached = cell is not None
                if not cached:
                    cell = run_cell(seed, optimiser, budget, rep, cfg)
                    if path is not None:
                        path.write_text(json.dumps(cell, indent=1))
                out["cells"].append(cell)
                _progress(seed, cell, cached)
    path = cell_dir / f"{seed}_reference.json" if cell_dir is not None else None
    ref = _cached(path)
    cached = ref is not None
    if not cached:
        ref = run_reference(seed, cfg)
        if path is not None:
            path.write_text(json.dumps(ref, indent=1))
    out["reference"] = ref
    _progress(seed, ref, cached)
    failures = [(c["arm"], c["failure"]) for c in out["cells"] + [ref] if c["status"] != "ok"]
    out["status"] = "failed" if failures else "ok"
    out["failure"] = f"{failures[0][0]}: {failures[0][1]}" if failures else None
    out["ledger"] = {c["arm"]: c["ledger_entry"] for c in out["cells"] + [ref]}
    return out


# ------------------------------------------------------------------ outcome


def _seed_problems(rec: dict) -> list[str]:
    """Amendment 1 item 8: why a seed record is incomplete against the frozen grid (empty when complete).
    Complete = D0 held-out present, a successful reference with ROUNDS rounds, and for every frozen
    optimiser × budget exactly REPLICATES ok cells with distinct replicates 0..REPLICATES-1 and ROUNDS rounds."""
    problems = []
    if (rec.get("D0") or {}).get("heldout") is None:
        problems.append("D0 held-out missing")
    ref = rec.get("reference") or {}
    if ref.get("status") != "ok":
        problems.append(f"reference status {ref.get('status')!r}")
    elif len(ref.get("rounds") or []) != ROUNDS:
        problems.append(f"reference has {len(ref.get('rounds') or [])} rounds, not {ROUNDS}")
    grid: dict[tuple, list] = {}
    for c in rec.get("cells", []):
        grid.setdefault((c.get("optimiser"), c.get("budget")), []).append(c)
    for o in OPTIMISERS:
        for B in BUDGETS:
            cs, key = grid.get((o, B), []), f"{o}/B{B}"
            reps = sorted(c.get("replicate") for c in cs)
            if reps != list(range(REPLICATES)):
                problems.append(f"{key} replicates {reps}, not {list(range(REPLICATES))}")
            for c in cs:
                if c.get("status") != "ok":
                    problems.append(f"{key}/r{c.get('replicate')} status {c.get('status')!r}")
                elif len(c.get("rounds") or []) != ROUNDS:
                    problems.append(f"{key}/r{c.get('replicate')} has {len(c.get('rounds') or [])} rounds, not {ROUNDS}")
    return problems


def outcome(records, f: float = F, seeds=DEV_SEEDS) -> dict:
    """Mechanical reading (rule + Amendment 1 items 1, 6, 8, 9). A cell (optimiser, B) repairs a seed iff at
    least two of its exactly three replicate gains (held-out after round 3 − D0 held-out) exceed f and their
    mean exceeds f. Any roster seed missing, not ok, or incomplete against the frozen grid (every optimiser ×
    budget, exactly three distinct replicates, three rounds each, a successful reference) -> no_result. Only
    rs-random24 and rs-probe24 may satisfy the gate; smallest B with ≥ 3 of 4 repaired wins; ties by repaired
    count, four-seed mean gain, then name."""
    roster = [int(s) for s in seeds]
    by_seed = {r["seed"]: r for r in records}
    missing = [s for s in roster if s not in by_seed]
    failed = [{"seed": s, "failure": by_seed[s].get("failure")} for s in roster
              if s in by_seed and by_seed[s].get("status") != "ok"]
    incomplete = []
    for s in roster:
        if s in by_seed and by_seed[s].get("status") == "ok":
            problems = _seed_problems(by_seed[s])
            if problems:
                incomplete.append({"seed": s, "problems": problems})
    cells: dict[str, dict] = {}
    for rec in sorted(records, key=lambda r: r["seed"]):
        if rec.get("status") != "ok" or (rec.get("D0") or {}).get("heldout") is None:
            continue
        d0h = rec["D0"]["heldout"]
        for c in rec.get("cells", []):
            if c.get("status") != "ok" or len(c.get("rounds") or []) != ROUNDS:
                continue
            key = f"{c['optimiser']}/B{c['budget']}"
            entry = cells.setdefault(key, {"optimiser": c["optimiser"], "budget": c["budget"],
                                           "eligible": c["optimiser"] in GATE_ELIGIBLE and c["budget"] in BUDGETS,
                                           "per_seed": {}})
            ps = entry["per_seed"].setdefault(str(rec["seed"]), {"gains": [], "replicates": []})
            ps["gains"].append(c["rounds"][-1]["heldout"] - d0h)
            ps["replicates"].append(c["replicate"])
    for entry in cells.values():
        for ps in entry["per_seed"].values():
            g = ps["gains"]
            complete = sorted(ps["replicates"]) == list(range(REPLICATES))
            ps.update(n=len(g), complete=complete, mean_gain=sum(g) / len(g), n_above_f=sum(x > f for x in g))
            # item 6: repair is read from exactly three replicate gains, never from fewer or from duplicates
            ps["repaired"] = bool(complete and ps["n_above_f"] >= MIN_REPLICATES_ABOVE_F and ps["mean_gain"] > f)
        on_roster = [entry["per_seed"].get(str(s)) for s in roster]
        entry["n_repaired"] = sum(bool(ps and ps["repaired"]) for ps in on_roster)
        entry["mean_gain"] = (sum(ps["mean_gain"] for ps in on_roster) / len(roster)
                              if roster and all(ps is not None and ps["complete"] for ps in on_roster) else None)
    label, chosen, qualifying = "no_result", None, []
    if not missing and not failed and not incomplete:
        label = "not_calibrated"
        for B in BUDGETS:
            q = [e for e in cells.values() if e["eligible"] and e["budget"] == B and e["n_repaired"] >= MIN_REPAIRED
                 and e["mean_gain"] is not None]
            if q:
                q.sort(key=lambda e: (-e["n_repaired"], -e["mean_gain"], e["optimiser"]))
                label, chosen, qualifying = f"calibrated_at_{B}", q[0]["optimiser"], [e["optimiser"] for e in q]
                break
    return {"f": f, "seeds": roster, "n_seeds": len(roster), "min_repaired": MIN_REPAIRED,
            "min_replicates_above_f": MIN_REPLICATES_ABOVE_F, "replicates": REPLICATES, "rounds": ROUNDS,
            "eligible": list(GATE_ELIGIBLE), "budgets": list(BUDGETS), "optimisers": list(OPTIMISERS),
            "missing": missing, "failed": failed, "incomplete": incomplete, "cells": cells, "outcome": label,
            "calibrated_optimiser": chosen, "qualifying": qualifying}


# ------------------------------------------------------------------- report


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    if isinstance(x, list):
        return ",".join(_fmt(v) for v in x)
    return str(x)


def _table(head, rows) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _descriptives(records) -> dict:
    """Per (optimiser, B, seed) over replicates: mean held-out gain, labelled accuracy, touched counts, forward
    passes; reference gains per seed; overfitting (labelled objective decrease vs held-out gain) per
    (optimiser, B); item 13 totals per cell and D0 / reference work per seed."""
    groups: dict[tuple, list] = {}
    totals, reference, work = [], [], []
    for rec in sorted(records, key=lambda r: r["seed"]):
        seed, d0 = rec["seed"], rec.get("D0") or {}
        d0w, rt = d0.get("work") or {}, (rec.get("reference") or {}).get("totals") or {}
        work.append({"seed": seed, "status": rec.get("status"), "D0_heldout": d0.get("heldout"),
                     "D0_gradient_string_passes": d0w.get("gradient_string_passes"),
                     "D0_diagnostic_logits": d0w.get("diagnostic_logits"),
                     "D0_wall_s": d0w.get("wall_s"), "D0_cpu_s": d0w.get("cpu_s"),
                     "reference_status": (rec.get("reference") or {}).get("status"),
                     "reference_gradient_string_passes": rt.get("gradient_string_passes"),
                     "reference_diagnostic_logits": rt.get("diagnostic_logits"),
                     "reference_oracle_labels": rt.get("oracle_labels"),
                     "reference_wall_s": rt.get("wall_s"), "reference_cpu_s": rt.get("cpu_s")})
        for c in rec.get("cells", []):
            t = c.get("totals") or {}
            totals.append({"seed": seed, "optimiser": c["optimiser"], "budget": c["budget"], "replicate": c["replicate"],
                           "status": c.get("status"), "rounds": len(c.get("rounds", [])),
                           **{k: t.get(k) for k in ("evaluations", "search_logits", "diagnostic_logits", "oracle_labels",
                                                    "wall_s", "cpu_s")}})
            if c.get("status") == "ok" and c.get("rounds") and "heldout" in d0:
                last = c["rounds"][-1]
                groups.setdefault((c["optimiser"], c["budget"], seed), []).append({
                    "gain": last["heldout"] - d0["heldout"], "long_gain": last["long"] - d0["long"],
                    "accuracy": last["accuracy"], "touched": last["n_cumulative_touched"],
                    "forward_passes": t.get("search_logits", 0), "objective_decrease": t.get("objective_decrease", 0.0),
                    "noop_rounds": sum(int(r.get("noop", False)) for r in c["rounds"])})
        ref = rec.get("reference") or {}
        if ref.get("status") == "ok" and ref.get("rounds") and "heldout" in d0:
            last = ref["rounds"][-1]
            reference.append({"seed": seed, "D0": d0["heldout"], "reference": last["heldout"],
                              "gain": last["heldout"] - d0["heldout"], "long_gain": last["long"] - d0["long"],
                              "accuracy": last["accuracy"], "touched": last["n_cumulative_touched"],
                              "gradient_string_passes": (ref.get("totals") or {}).get("gradient_string_passes")})
    cells = [{"optimiser": o, "budget": B, "seed": s, "n": len(g), "gains": [x["gain"] for x in g],
              "mean_gain": _mean(x["gain"] for x in g), "mean_long_gain": _mean(x["long_gain"] for x in g),
              "labelled_accuracy": _mean(x["accuracy"] for x in g), "touched": _mean(x["touched"] for x in g),
              "forward_passes": _mean(x["forward_passes"] for x in g),
              "objective_decrease": _mean(x["objective_decrease"] for x in g),
              "noop_rounds": sum(x["noop_rounds"] for x in g)}
             for (o, B, s), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))]
    overfit = {}
    for row in cells:
        agg = overfit.setdefault(f"{row['optimiser']}/B{row['budget']}", {"optimiser": row["optimiser"], "budget": row["budget"],
                                                                        "objective_decrease": [], "labelled_accuracy": [], "gain": []})
        for k in ("objective_decrease", "labelled_accuracy"):
            agg[k].append(row[k])
        agg["gain"].append(row["mean_gain"])
    overfit = {k: {**v, "objective_decrease": _mean(v["objective_decrease"]), "labelled_accuracy": _mean(v["labelled_accuracy"]),
                   "gain": _mean(v["gain"])} for k, v in overfit.items()}
    return {"cells": cells, "reference": reference, "overfitting": overfit, "totals": totals, "work": work}


def _markdown(study: dict) -> str:
    oc = study["outcome"]
    lines = [f"# Comparator calibration ({RULE}, Amendment 1)", "", "## Outcome (mechanical reading)", "",
             f"- f: {oc['f']}", f"- development seeds: {oc['seeds']} (gate: at least {oc['min_repaired']} of {oc['n_seeds']}; "
             f"a seed is repaired iff at least {oc['min_replicates_above_f']} of 3 replicate gains > f and their mean > f)",
             f"- gate-eligible optimisers: {oc['eligible']} (es-dense is descriptive, never eligible)",
             f"- frozen grid per seed: optimisers {oc['optimisers']} × budgets {oc['budgets']} × exactly "
             f"{oc['replicates']} replicates, {oc['rounds']} rounds each, plus the reference (item 8: any incomplete "
             f"development seed -> no_result)",
             f"- missing seeds: {oc['missing']}", f"- failed seeds: {[x['seed'] for x in oc['failed']]}",
             f"- incomplete seeds: {[x['seed'] for x in oc['incomplete']]}"]
    lines += [f"  - seed {x['seed']}: {'; '.join(x['problems'])}" for x in oc["incomplete"]]
    lines += [f"- {key}: repaired {e['n_repaired']} of {oc['n_seeds']}, four-seed mean gain {_fmt(e['mean_gain'])}, "
              f"eligible {e['eligible']}" for key, e in sorted(oc["cells"].items(), key=lambda kv: (kv[1]["budget"], kv[0]))]
    lines += [f"- outcome: {oc['outcome']}", f"- calibrated_optimiser: {oc['calibrated_optimiser']}",
              f"- qualifying: {oc['qualifying']}", "",
              "## Per optimiser × B × seed (mean over replicates; gain = held-out after round 3 − D0)", "",
              *_table(["optimiser", "B", "seed", "n", "gains", "mean gain", "long gain", "labelled acc", "touched",
                       "forward passes", "noop rounds"],
                      [[r["optimiser"], r["budget"], r["seed"], r["n"], r["gains"], r["mean_gain"], r["mean_long_gain"],
                        r["labelled_accuracy"], r["touched"], r["forward_passes"], r["noop_rounds"]] for r in study["cells"]]),
              "", "## Reference (projected dense Adam under the edit budget; reported, never a comparator)", "",
              *_table(["seed", "D0", "reference", "gain", "long gain", "labelled acc", "touched", "gradient string passes"],
                      [[r["seed"], r["D0"], r["reference"], r["gain"], r["long_gain"], r["accuracy"], r["touched"],
                        r["gradient_string_passes"]] for r in study["reference"]]),
              "", "## Labelled objective versus held-out (descriptive; never changes the outcome)", "",
              "objective decrease = sum over rounds of (labelled BCE at round start − at round end). A large decrease with a",
              "held-out gain at or below f means the search fits the labelled set without repairing the discriminator",
              "(overfitting); the labelled accuracy column shows the fit the optimiser could see.", "",
              *_table(["optimiser", "B", "mean objective decrease", "mean labelled acc", "mean held-out gain"],
                      [[v["optimiser"], v["budget"], v["objective_decrease"], v["labelled_accuracy"], v["gain"]]
                       for v in sorted(study["overfitting"].values(), key=lambda v: (v["optimiser"], v["budget"]))]),
              "", "## Totals per cell (Amendment 1 item 13: string logits by purpose, oracle labels, CPU wall time)", "",
              *_table(["seed", "optimiser", "B", "rep", "status", "rounds", "evaluations", "search logits",
                       "diagnostic logits", "oracle labels", "wall s", "cpu s"],
                      [[t["seed"], t["optimiser"], t["budget"], t["replicate"], t["status"], t["rounds"], t["evaluations"],
                        t["search_logits"], t["diagnostic_logits"], t["oracle_labels"], t["wall_s"], t["cpu_s"]]
                       for t in study["totals"]]),
              "", "## D0 and reference work per seed (item 13; D0 diagnostic logits = training-history loss and "
              "accuracy passes plus the D0 endpoint passes)", "",
              *_table(["seed", "status", "D0 held-out", "D0 gradient string passes", "D0 diagnostic logits",
                       "D0 wall s", "D0 cpu s", "reference status", "reference gradient string passes",
                       "reference diagnostic logits", "reference oracle labels", "reference wall s", "reference cpu s"],
                      [[w["seed"], w["status"], w["D0_heldout"], w["D0_gradient_string_passes"], w["D0_diagnostic_logits"],
                        w["D0_wall_s"], w["D0_cpu_s"], w["reference_status"], w["reference_gradient_string_passes"],
                        w["reference_diagnostic_logits"], w["reference_oracle_labels"], w["reference_wall_s"],
                        w["reference_cpu_s"]] for w in study["work"]]),
              "", "## Study meta", ""]
    meta = study["study_meta"] or {}
    lines += [f"- {k}: {json.dumps(v) if isinstance(v, dict) else v}" for k, v in meta.items()]
    return "\n".join(lines) + "\n"


def _records(out: Path) -> list[dict]:
    paths = sorted(out.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return [json.loads(p.read_text()) for p in paths]


def report(out_dir) -> tuple[dict, str]:
    out = Path(out_dir)
    records = _records(out)
    meta_path = out / "study_meta.json"
    study = {"outcome": outcome(records), **_descriptives(records),
             "study_meta": json.loads(meta_path.read_text()) if meta_path.is_file() else None}
    md = _markdown(study)
    (out / "study.json").write_text(json.dumps(study, indent=1))
    (out / "study.md").write_text(md)
    return study, md


# ----------------------------------------------------------------------- CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="wsp.search_calibration")
    sub = p.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run", help="run the grid, one JSON per cell (resumable) and one per seed")
    rn.add_argument("--seeds", default=",".join(map(str, DEV_SEEDS)), help="comma-separated")
    rn.add_argument("--optimisers", default=",".join(OPTIMISERS), help="comma-separated")
    rn.add_argument("--budgets", default=",".join(map(str, BUDGETS)), help="comma-separated evaluations per round")
    rn.add_argument("--replicates", type=int, default=DEFAULT_CFG["replicates"])
    rn.add_argument("--k", type=int, default=DEFAULT_CFG["k"])
    rn.add_argument("--hidden", type=int, default=DEFAULT_CFG["h"])
    rn.add_argument("--n-initial", type=int, default=DEFAULT_CFG["n_initial"])
    rn.add_argument("--n-heldout", type=int, default=DEFAULT_CFG["n_heldout"])
    rn.add_argument("--n-long", type=int, default=DEFAULT_CFG["n_long"])
    rn.add_argument("--train-steps", type=int, default=DEFAULT_CFG["train_steps"])
    rn.add_argument("--jobs", type=int, default=1, help="seeds in separate processes")
    rn.add_argument("--out", required=True)
    rp = sub.add_parser("report", help="study.json and study.md from the seed records")
    rp.add_argument("--out", required=True)
    return p.parse_args(argv)


def _seed_job(job) -> str:
    seed, cfg, out = job
    out = Path(out)
    rec = run_seed(seed, cfg, out / "cells")
    (out / f"seed_{seed}.json").write_text(json.dumps(rec, indent=1))
    print(f"wsp.search_calibration: wrote seed_{seed}.json status={rec['status']} failure={rec['failure']}",
          file=sys.stderr)
    return rec["status"]


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {key: v for key, v in vars(args).items() if key not in ("cmd", "out")}
    (out / "study_meta.json").write_text(json.dumps({
        "rule": RULE, "git_commit": _git_commit(), "config": config,
        "written": datetime.now(timezone.utc).isoformat()}, indent=1))
    cfg = {**DEFAULT_CFG, "k": args.k, "h": args.hidden, "n_initial": args.n_initial, "n_heldout": args.n_heldout,
           "n_long": args.n_long, "train_steps": args.train_steps, "optimisers": args.optimisers.split(","),
           "budgets": [int(b) for b in args.budgets.split(",")], "replicates": args.replicates}
    jobs = [(int(s), cfg, str(out)) for s in args.seeds.split(",")]
    if args.jobs > 1:
        with multiprocessing.get_context("spawn").Pool(args.jobs) as pool:
            pool.map(_seed_job, jobs)
    else:
        for job in jobs:
            _seed_job(job)
    return 0


def cmd_report(args) -> int:
    study, _ = report(args.out)
    print(f"wsp.search_calibration: outcome={study['outcome']['outcome']} -> {Path(args.out) / 'study.md'}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return cmd_run(args) if args.cmd == "run" else cmd_report(args)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"wsp.search_calibration: failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
