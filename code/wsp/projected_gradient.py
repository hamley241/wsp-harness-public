"""Arm G (projected dense Adam), the shared random query batches and the feasibility preflight
(redesign decision record P5, P7, P9). No LLM anywhere in this module.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from wsp import rnn, trained_baseline as tb
from wsp.arms import _uniform_fill, target_histogram
from wsp.dfa import random_dfa
from wsp.numeric_edit import EditBudget, _f32
from wsp.oracle import Oracle

F = 0.0224  # held-out noise floor for 500 strings at p = 0.5 (P7)


def adam_dense(weights: dict, samples, steps: int = 200, lr: float = 0.01) -> dict[str, np.ndarray]:
    """Full-batch Adam with fresh state on mean BCE over `samples`, float64, from the given weights."""
    w = {k: np.array(v, dtype=np.float64) for k, v in weights.items()}
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: np.zeros_like(x) for k, x in w.items()}
    b1, b2 = tb.BETAS
    samples = list(samples)
    for step in range(1, steps + 1):
        g = tb.grads(w, samples)
        for k in w:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k]
            w[k] = w[k] - lr * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + tb.EPS)
    return w


def _clip(d0_val: float, dense_val: float, max_norm: float) -> tuple[float, bool]:
    """dense_val clipped to [d0 - max_norm, d0 + max_norm], as a float32 value that EditBudget.check accepts."""
    lo, hi = d0_val - max_norm, d0_val + max_norm
    clipped = not lo <= dense_val <= hi
    v = np.float32(min(max(dense_val, lo), hi))
    while abs(float(v) - d0_val) > max_norm:  # float32 rounding may land a ulp outside the ball
        v = np.nextafter(v, np.float32(d0_val))
    return float(v), clipped


def project(start: dict, dense: dict, budget: EditBudget):
    """P5 projection: rank |dense - start| descending (ties by lower flat index), select in rank order
    coordinates already touched or new ones while the touched set has fewer than `distinct` members,
    stop at `per_round`; values are dense's clipped to the max-norm ball around D0."""
    s, d = _f32(start), {k: np.asarray(v, dtype=np.float64) for k, v in dense.items()}
    coords = [(k, i) for k in rnn.KEYS for i in range(s[k].size)]
    mags = [abs(float(d[k].flat[i]) - float(s[k].flat[i])) for k, i in coords]
    order = sorted(range(len(coords)), key=lambda n: (-mags[n], n))
    touched = set(budget.touched)
    edits, n_clipped, n_new = [], 0, 0
    for n in order:
        if len(edits) == budget.per_round:
            break
        k, i = coords[n]
        if (k, i) not in touched:
            if len(touched) >= budget.distinct:
                continue
            touched.add((k, i))
            n_new += 1
        value, clipped = _clip(float(budget.d0[k].flat[i]), float(d[k].flat[i]), budget.max_norm)
        n_clipped += clipped
        edits.append((k, i, value))
    return edits, {"n_selected": len(edits), "n_clipped": n_clipped, "n_new_coords": n_new}


def gradient_round(weights: dict, samples, budget: EditBudget, steps: int = 200, lr: float = 0.01):
    dense = adam_dense(weights, samples, steps, lr)
    edits, info = project(weights, dense, budget)
    new = budget.apply(edits, weights)
    state = budget.state()
    record = {**info, "edits": [[p, i, v] for p, i, v in edits], "touched": state["touched"],
              "n_touched": state["n_touched"], "displacement": state["displacement"]}
    return new, record


def add_labels(labels: list, batch, ys) -> list[tuple[str, int]]:
    """P9 accumulation: appends to `labels` (first-occurrence ledger order, initialized with the initial set) the
    pairs whose string is not yet labeled, and returns them. A re-query is charged by the oracle but adds nothing:
    it neither reweights G's mean BCE nor repeats a line in E's prompt."""
    seen = {s for s, _ in labels}
    new = []
    for s, y in zip(batch, ys):
        if s not in seen:
            seen.add(s)
            new.append((s, y))
    labels += new
    return new


def query_batches(seed: int, initial_strings, rounds: int = 3, n: int = 32) -> list[list[str]]:
    """P9: `rounds` batches of n uniform strings with the initial histogram; distinct within a batch."""
    rng = random.Random(seed + 1000)
    target = target_histogram(initial_strings, n)
    out = []
    for _ in range(rounds):
        taken: set[str] = set()
        out.append([s for L, k in target.items() for s in _uniform_fill(rng, L, k, taken)])
    return out


def preflight(seed: int, k: int = 6, h: int = 16, n_initial: int = 32, rounds: int = 3, n: int = 32,
              budget_args: dict | None = None, n_heldout: int = 500, train_steps: int = 2000,
              adam_steps: int = 200) -> dict:
    """P7 feasibility check: G under the budget on an offline random-query run against N (= D0)."""
    dfa = random_dfa(k, seed)
    oracle = Oracle(dfa, seed)
    initial = oracle.initial_dataset(n_initial)
    heldout = oracle.heldout(n_heldout)
    d0, _ = tb.train(initial, h, seed, train_steps)
    budget = EditBudget(d0, **(budget_args or {}))
    batches = query_batches(seed, [s for s, _ in initial], rounds, n)
    d0_acc = tb.accuracy(d0, heldout)
    cur, labels, g_acc, records = _f32(d0), list(initial), [], []
    for batch in batches:
        new = add_labels(labels, batch, oracle.query("G", batch))  # same semantics as repair_study._run_arm
        cur, rec = gradient_round(cur, labels, budget, adam_steps)
        rec.update(n_new_labels=len(new), n_labels=len(labels))
        g_acc.append(tb.accuracy(cur, heldout))
        records.append(rec)
    gain = g_acc[-1] - d0_acc
    return {"seed": seed, "k": k, "h": h, "n_initial": n_initial, "rounds": rounds, "n": n,
            "n_heldout": n_heldout, "train_steps": train_steps, "adam_steps": adam_steps,
            "D0_heldout": d0_acc, "G_heldout": g_acc, "N_heldout": d0_acc, "gain": gain, "f": F,
            "feasible": bool(gain > F), "batches": batches, "budget": budget.state(),
            "G_rounds": records, "ledger": oracle.ledger()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wsp.projected_gradient")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("preflight")
    q.add_argument("--seeds", required=True, help="comma-separated")
    q.add_argument("--per-round", type=int, default=24)
    q.add_argument("--distinct", type=int, default=48)
    q.add_argument("--max-norm", type=float, default=1.0)
    q.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget_args = {"per_round": args.per_round, "distinct": args.distinct, "max_norm": args.max_norm}
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        res = preflight(seed, budget_args=budget_args)
        (out / f"preflight_{seed}.json").write_text(json.dumps(res, indent=1))
        rows.append(res)
        print(f"seed {seed}: D0 {res['D0_heldout']:.3f} G {[round(a, 3) for a in res['G_heldout']]} "
              f"gain {res['gain']:+.3f} feasible {res['feasible']}")
    rounds = rows[0]["rounds"]
    head = ["seed", "D0"] + [f"G r{r}" for r in range(1, rounds + 1)] + ["N", "gain", "feasible"]
    lines = [f"# Preflight (P7): budget {budget_args}, f = {F}", "", "| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r["seed"]), f"{r['D0_heldout']:.3f}"] + [f"{a:.3f}" for a in r["G_heldout"]] + [
            f"{r['N_heldout']:.3f}", f"{r['gain']:+.4f}", str(r["feasible"])]
        lines.append("| " + " | ".join(cells) + " |")
    (out / "preflight.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
