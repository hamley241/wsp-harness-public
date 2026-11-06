"""Repair redesign runner and report (rules/redesign-rule.md, P5–P9).

    uv run python -m wsp.repair_study run --production --seeds 15,20,21,24,30 --backend openrouter --out DIR
    uv run python -m wsp.repair_study report --out DIR

Per seed, in the frozen call order: E realization 1 (rounds 1..3), E realization 2, G, N, all on the same D0,
labels, held-out sets and byte-identical query batches (P9). The outcome vocabulary is P7's as amended by
rules/redesign-rule.md Amendment 2. Reserve seed 38 (Amendment 2 item 4) is a manual decision logged by hand:
this runner never substitutes a seed; a failed seed stays failed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wsp import projected_gradient as pg
from wsp import rnn, trained_baseline as tb
from wsp.dfa import random_dfa
from wsp.evaluate import accuracy, with_noise
from wsp.llm import CallLog, MockLLM, OpenRouterClient
from wsp.numeric_edit import EditBudget, _f32, editor_round
from wsp.oracle import Oracle
from wsp.query_study import RATES, rates_for, Budget, BudgetStop, _accrued_usd, _append_usage, _BudgetedLLM, _git_commit

F = pg.F
OUTCOMES = ("no_repair", "E_beats_gradient", "E_below_gradient", "E_matches_gradient", "mixed", "no_result")
N_SEEDS, MIN_ANALYZABLE = 5, 4   # P7: five frozen seeds; "4 of 5" and "all 5" are counts over those five
ARMS = ("E1", "E2", "G", "N")
EDITED_ARMS = ("E1", "E2", "G")
READOUT = {"w", "c"}
DEFAULT_CFG = {"k": 6, "h": 16, "n_initial": 32, "rounds": 3, "n_queries": 32, "n_heldout": 500, "n_long": 200,
               "train_steps": 2000, "adam_steps": 200, "per_round": 24, "distinct": 48, "max_norm": 1.0}
RULE = "rules/redesign-rule.md"
# Amendment 2 item 3: what `--production` requires, by argparse destination.
PRODUCTION = {"seeds": "15,20,21,24,30", "model": "anthropic/claude-opus-5.5", "reasoning_tokens": 16000,
              "max_output_tokens": 24000, "per_round": 24, "distinct": 48, "max_norm": 1.0, "spend_cap_usd": 50.0,
              "backend": "openrouter"}
ATTEMPT_OUTCOMES = ("ok", "refusal", "llm_error", "client_error", "parse", "budget")


# ------------------------------------------------------------------ run_seed


def blocks_touched(touched) -> list[str]:
    present = {p for p, _ in touched}
    return [k for k in rnn.KEYS if k in present]


def _endpoints(weights, d0, heldout, long_) -> dict:
    changed = sum(int(np.sum(np.asarray(weights[k], dtype=np.float32) != d0[k])) for k in rnn.KEYS)
    return {**with_noise("heldout", accuracy(weights, heldout), len(heldout)),
            **with_noise("long", accuracy(weights, long_), len(long_)), "coordinates_changed": changed}


def _run_arm(name, arm, oracle, batches, d0, initial, sets, cfg, llm, calls) -> None:
    """Fills arm["rounds"] one round at a time, so a failure keeps the rounds completed before it."""
    budget = EditBudget(d0, cfg["per_round"], cfg["distinct"], cfg["max_norm"])
    cur, labels = _f32(d0), list(initial)
    for r, batch in enumerate(batches, 1):
        start = cur
        ys = oracle.query(name, batch)  # charged under the arm's own ledger name before edit r, every arm
        new = pg.add_labels(labels, batch, ys)  # P9: a re-queried string adds nothing to labels or feedback
        rec = {"round": r, "queried": list(batch), "labels": ys, "n_new_labels": len(new), "n_labels": len(labels)}
        if name.startswith("E"):
            cex = [(s, y, p) for s, y in new if (p := rnn.predict(start, s)) != y]  # round-start weights
            rec.update(counterexamples=[list(c) for c in cex], n_counterexamples=len(cex))
            cur, info = editor_round(llm, start, labels, cex, budget, calls, fence=cfg.get("fence", "strict"))
            # Amendment 2 item 6: a successful, non-empty round whose edits are all in w or c
            info["readout_only"] = (not info["noop"] and bool(info["edits"])
                                    and all(p in READOUT for p, _, _ in info["edits"]))
        elif name == "G":
            cur, info = pg.gradient_round(start, labels, budget, cfg["adam_steps"])
        else:  # N: no edit
            state = budget.state()
            info = {"edits": [], **{k: state[k] for k in ("touched", "n_touched", "displacement")}}
        changed = sum(int(np.sum(np.asarray(cur[k], dtype=np.float32) != start[k])) for k in rnn.KEYS)
        rec.update(info, blocks_touched=blocks_touched(budget.touched), changed_float32=changed,
                   **_endpoints(cur, budget.d0, *sets))
        arm["rounds"].append(rec)
    arm["budget"] = budget.state()


def run_seed(seed: int, cfg: dict | None, editor_llm, spend: Budget | None = None, record_path=None) -> dict:
    """One seed, every arm. A BudgetStop or any other error marks the seed `failed`; nothing is imputed.
    With `record_path`, the record (batches included) is written before the first call (P9)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    spend = spend if spend is not None else Budget(math.inf)
    before = spend.snapshot()
    dfa = random_dfa(cfg["k"], seed)
    oracle = Oracle(dfa, seed)
    initial = oracle.initial_dataset(cfg["n_initial"])
    heldout, long_ = oracle.heldout(cfg["n_heldout"]), oracle.long_set(cfg["n_long"])
    d0_raw, history = tb.train(initial, cfg["h"], seed, cfg["train_steps"])
    d0 = _f32(d0_raw)
    batches = pg.query_batches(seed, [s for s, _ in initial], cfg["rounds"], cfg["n_queries"])
    out = {"seed": seed, "config": cfg, "status": "running", "failure": None, "stopped_at_cap": False,
           "initial": [list(t) for t in initial],
           "D0": {**_endpoints(d0, d0, heldout, long_), "train_history": history},
           "batches": batches, "arms": {name: {"rounds": []} for name in ARMS}}
    if record_path is not None:
        Path(record_path).write_text(json.dumps(out, indent=1))
    llm = _BudgetedLLM(editor_llm, spend)
    try:
        for name in ARMS:
            calls = CallLog() if name.startswith("E") else None
            if calls is not None:
                out["arms"][name]["calls"] = calls.calls
            _run_arm(name, out["arms"][name], oracle, batches, d0, initial, (heldout, long_), cfg, llm, calls)
        out["status"] = "ok"
    except BudgetStop as e:
        out.update(status="failed", failure=f"BudgetStop: {e}", stopped_at_cap=True)
    except Exception as e:  # noqa: BLE001 - unrecoverable: the seed is failed, reported, never imputed
        out.update(status="failed", failure=f"{type(e).__name__}: {e}"[:300])
    out["ledger"] = oracle.ledger()
    after = spend.snapshot()
    out["usage"] = {key: after[key] - before[key] for key in after}
    return out


# ------------------------------------------------------------------- outcome


def outcome(seed_records, f: float = F) -> dict:
    """P7 precedence as amended (Amendment 2 item 6). Counts are over the five frozen seeds: a failed seed
    satisfies no condition, so with exactly four analyzable seeds E_beats_gradient and E_below_gradient are
    false and no_repair / E_matches_gradient need all four; fewer than four analyzable gives no_result."""
    rows, failed = [], []
    for rec in sorted(seed_records, key=lambda r: r["seed"]):
        if rec.get("status") != "ok":
            failed.append({"seed": rec["seed"], "failure": rec.get("failure")})
            rows.append({"seed": rec["seed"], "status": rec.get("status")})
            continue
        final = {a: rec["arms"][a]["rounds"][-1]["heldout"] for a in ARMS}
        gE = (final["E1"] + final["E2"]) / 2 - final["N"]
        gG = final["G"] - final["N"]
        rows.append({"seed": rec["seed"], "status": "ok", "D0": rec["D0"]["heldout"], **final,
                     "gE": gE, "gG": gG, "d": gE - gG})
    ok = [r for r in rows if r["status"] == "ok"]
    counts = {"gE_at_most_f": sum(r["gE"] <= f for r in ok), "d_above_f": sum(r["d"] > f for r in ok),
              "d_below_minus_f": sum(r["d"] < -f for r in ok), "abs_d_at_most_f": sum(abs(r["d"]) <= f for r in ok)}
    if len(ok) < MIN_ANALYZABLE:
        label = "no_result"
    elif counts["gE_at_most_f"] >= MIN_ANALYZABLE:
        label = "no_repair"
    elif counts["d_above_f"] == N_SEEDS:
        label = "E_beats_gradient"
    elif counts["d_below_minus_f"] == N_SEEDS:
        label = "E_below_gradient"
    elif counts["abs_d_at_most_f"] >= MIN_ANALYZABLE:
        label = "E_matches_gradient"
    else:
        label = "mixed"
    return {"f": f, "n_seeds": N_SEEDS, "min_analyzable": MIN_ANALYZABLE, "n_analyzable": len(ok),
            "failed": failed, "seeds": rows, "counts": counts, "outcome": label}


# -------------------------------------------------------------------- report


OVERLAPS = (("E1", "G"), ("E2", "G"), ("E1", "E2"))


def _touched_at(rec: dict, arm: str, r: int) -> set | None:
    """The arm's cumulative touched set after round r, or None when that round was not completed."""
    rounds = rec.get("arms", {}).get(arm, {}).get("rounds", [])
    return {tuple(c) for c in rounds[r - 1]["touched"]} if r <= len(rounds) else None


def _descriptives(records) -> dict:
    """P6, numeric and without exclusions. `rounds`: one row per seed and round (every seed, failed or partial
    included) with each edited arm's availability, block-touch indicators and touched-set size after that round,
    and the E1–G, E2–G, E1–E2 coordinate overlaps of that round (None where an arm lacks the round).
    `blocks_touched`: per arm and round, seeds with the round available and, per block, how many touched it.
    `seeds`: status and rounds completed per arm. `editor`: rounds and attempts by outcome, every completed round,
    and `failures`: every non-ok attempt with its category and reason."""
    rows, seeds = [], []
    blocks = {a: {} for a in EDITED_ARMS}
    editor = {a: {"rounds": 0, "noop_rounds": 0, "readout_only_rounds": 0, "attempts": 0,
                  **{k: 0 for k in ATTEMPT_OUTCOMES}, "failures": []} for a in ("E1", "E2")}
    for rec in records:
        arms = rec.get("arms", {})
        completed = {a: len(arms.get(a, {}).get("rounds", [])) for a in ARMS}
        n_rounds = rec.get("config", {}).get("rounds") or max(completed.values(), default=0)
        seeds.append({"seed": rec["seed"], "status": rec.get("status"), "failure": rec.get("failure"),
                      "n_rounds": n_rounds, "rounds_completed": completed})
        for r in range(1, n_rounds + 1):
            sets = {a: _touched_at(rec, a, r) for a in EDITED_ARMS}
            rows.append({
                "seed": rec["seed"], "round": r, "status": rec.get("status"),
                "available": {a: sets[a] is not None for a in EDITED_ARMS},
                "blocks": {a: None if sets[a] is None else {k: int(k in blocks_touched(sets[a])) for k in rnn.KEYS}
                           for a in EDITED_ARMS},
                "n_touched": {a: None if sets[a] is None else len(sets[a]) for a in EDITED_ARMS},
                "overlap": {f"{x}_{y}": None if sets[x] is None or sets[y] is None else len(sets[x] & sets[y])
                            for x, y in OVERLAPS}})
            for a in EDITED_ARMS:
                agg = blocks[a].setdefault(str(r), {"n_available": 0, **{k: 0 for k in rnn.KEYS}})
                if sets[a] is not None:
                    agg["n_available"] += 1
                    for k in blocks_touched(sets[a]):
                        agg[k] += 1
        for a in ("E1", "E2"):
            for rd in arms.get(a, {}).get("rounds", []):
                ed = editor[a]
                ed["rounds"] += 1
                ed["noop_rounds"] += int(rd["noop"])
                ed["readout_only_rounds"] += int(rd["readout_only"])
                for at in rd["attempts"]:
                    ed["attempts"] += 1
                    ed[at["outcome"]] = ed.get(at["outcome"], 0) + 1
                    if at["outcome"] != "ok":
                        ed["failures"].append({"seed": rec["seed"], "round": rd.get("round"),
                                               "attempt": at.get("attempt"), "outcome": at["outcome"],
                                               "finish_reason": at.get("finish_reason"), "reason": at.get("reason")})
    return {"seeds": seeds, "rounds": rows, "blocks_touched": blocks, "editor": editor}


def _spend(path: Path) -> dict:
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    return {"entries": len(entries), "aborted": sum(int(e["aborted"]) for e in entries),
            "prompt_tokens": sum(e["prompt_tokens"] for e in entries),
            "completion_tokens": sum(e["completion_tokens"] for e in entries), "usd": sum(e["usd"] for e in entries)}


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _table(head, rows) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]


def _blocks_cell(ind: dict | None) -> str:
    if ind is None:
        return "n/a"
    return ",".join(k for k in rnn.KEYS if ind[k]) or "-"


def _markdown(study: dict) -> str:
    oc = study["outcome"]
    cols = ["seed", "status", "D0", "N", "G", "E1", "E2", "gE", "gG", "d"]
    round_rows = [[rw["seed"], rw["round"], rw["status"], *[_blocks_cell(rw["blocks"][a]) for a in EDITED_ARMS],
                   *[rw["n_touched"][a] for a in EDITED_ARMS], *[rw["overlap"][f"{x}_{y}"] for x, y in OVERLAPS]]
                  for rw in study["rounds"]]
    block_rows = [[a, r, agg["n_available"], *[agg[k] for k in rnn.KEYS]]
                  for a, per_round in study["blocks_touched"].items() for r, agg in per_round.items()]
    seed_rows = [[s["seed"], s["status"], s["failure"], s["n_rounds"], *[s["rounds_completed"].get(a) for a in ARMS]]
                 for s in study["seeds"]]
    lines = ["# Repair study (redesign decision record P5–P9)", "",
             "## Held-out accuracy after the last round, per seed", "",
             *_table(cols, [[r.get(c) for c in cols] for r in oc["seeds"]]), "",
             "## Outcome", "", f"- f: {oc['f']}",
             f"- analyzable seeds: {oc['n_analyzable']} of {oc['n_seeds']} (minimum {oc['min_analyzable']})",
             f"- failed seeds: {[x['seed'] for x in oc['failed']]}",
             *[f"- {k}: {v}" for k, v in oc["counts"].items()],
             f"- outcome: {oc['outcome']}", "",
             "## Seeds (status and rounds completed per arm; no seed is excluded)", "",
             *_table(["seed", "status", "failure", "n_rounds", *[f"{a} done" for a in ARMS]], seed_rows), "",
             "## Per seed and round (P6): blocks touched, touched-set size, coordinate overlap", "",
             "n/a: the arm did not complete the round. Overlaps are coordinate counts for that round.", "",
             *_table(["seed", "round", "status", "E1 blocks", "E2 blocks", "G blocks", "n_E1", "n_E2", "n_G",
                      "E1∩G", "E2∩G", "E1∩E2"], round_rows), "",
             "## Blocks touched per arm and round (seeds with the round available; per block, seeds touching it)",
             "", *_table(["arm", "round", "n_available", *rnn.KEYS], block_rows),
             "", "## Editor rounds and attempts", ""]
    ekeys = ["rounds", "noop_rounds", "readout_only_rounds", "attempts", *ATTEMPT_OUTCOMES]
    lines += _table(["realization", *ekeys], [[a, *[e[k] for k in ekeys]] for a, e in study["editor"].items()])
    fkeys = ["seed", "round", "attempt", "outcome", "finish_reason", "reason"]
    lines += ["", "## Editor attempt failures (every non-ok attempt, with its reason)", "",
              *_table(["realization", *fkeys], [[a, *[f[k] for k in fkeys]]
                                                for a, e in study["editor"].items() for f in e["failures"]])]
    sp = study["spend"]
    lines += ["", "## Spend (usage.jsonl)", "", *_table(list(sp), [list(sp.values())]), "", "## Study meta", ""]
    meta = study["study_meta"] or {}
    lines += [f"- {k}: {json.dumps(v) if isinstance(v, dict) else v}" for k, v in meta.items()]
    return "\n".join(lines) + "\n"


def _seed_path(out: Path, seed: int) -> Path:
    return out / f"seed_{seed}.json"


def _records(out: Path) -> list[dict]:
    paths = sorted(out.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return [json.loads(p.read_text()) for p in paths]


def report(out_dir) -> tuple[dict, str]:
    out = Path(out_dir)
    meta_path = out / "study_meta.json"
    study = {"outcome": outcome(_records(out)), **_descriptives(_records(out)), "spend": _spend(out / "usage.jsonl"),
             "study_meta": json.loads(meta_path.read_text()) if meta_path.is_file() else None}
    md = _markdown(study)
    (out / "study.json").write_text(json.dumps(study, indent=1))
    (out / "study.md").write_text(md)
    return study, md


# ----------------------------------------------------------------------- CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="wsp.repair_study")
    sub = p.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run", help="run the seeds, one JSON per seed, resumable")
    rn.add_argument("--seeds", default=PRODUCTION["seeds"], help="comma-separated, run in this order")
    rn.add_argument("--production", action="store_true",
                    help=f"refuse any run argument that differs from {RULE} Amendment 2 item 3")
    rn.add_argument("--backend", choices=("mock", "openrouter"), default="mock")
    rn.add_argument("--model", default="anthropic/claude-opus-5.5")
    rn.add_argument("--reasoning-tokens", type=int, default=16000)
    rn.add_argument("--max-output-tokens", type=int, default=24000)
    rn.add_argument("--k", type=int, default=DEFAULT_CFG["k"])
    rn.add_argument("--hidden", type=int, default=DEFAULT_CFG["h"])
    rn.add_argument("--n-initial", type=int, default=DEFAULT_CFG["n_initial"])
    rn.add_argument("--rounds", type=int, default=DEFAULT_CFG["rounds"])
    rn.add_argument("--n-candidates", type=int, default=DEFAULT_CFG["n_queries"], help="queries per batch")
    rn.add_argument("--n-heldout", type=int, default=DEFAULT_CFG["n_heldout"])
    rn.add_argument("--n-long", type=int, default=DEFAULT_CFG["n_long"])
    rn.add_argument("--train-steps", type=int, default=DEFAULT_CFG["train_steps"])
    rn.add_argument("--fence", choices=("strict", "relaxed"), default="strict",
                    help="strict = rule P3; relaxed = rule Amendment 3 rerun (prose around the single json fence ignored)")
    rn.add_argument("--per-round", type=int, default=DEFAULT_CFG["per_round"])
    rn.add_argument("--distinct", type=int, default=DEFAULT_CFG["distinct"])
    rn.add_argument("--max-norm", type=float, default=DEFAULT_CFG["max_norm"])
    rn.add_argument("--spend-cap-usd", type=float, default=50.0)
    rn.add_argument("--out", required=True)
    rp = sub.add_parser("report", help="study.json and study.md from the seed records")
    rp.add_argument("--out", required=True)
    return p.parse_args(argv)


RESUME_FREE = ("seeds", "spend_cap_usd")  # the only run arguments a resume may change; everything else defines the study


def study_meta(out: Path, args) -> dict:
    """New study: writes study_meta.json (provenance timestamp, git commit, full config). Resume: loads it and
    requires every experiment-defining argument to match the stored config, so skipped seed records are never
    described by a configuration that did not produce them; the original provenance is kept and the resume
    (its seeds, spend cap, commit, time) is appended under "resumes". A mismatch raises; use a new --out."""
    path = out / "study_meta.json"
    config = {key: v for key, v in vars(args).items() if key not in ("cmd", "out")}
    now = datetime.now(timezone.utc).isoformat()
    if not path.is_file():
        if any(out.glob("seed_*.json")):
            raise RuntimeError(f"{out} holds seed records but no study_meta.json; refusing to resume. Use a new --out")
        meta = {"git_commit": _git_commit(), "config": config, "written": now, "resumes": []}
    else:
        meta = json.loads(path.read_text())
        stored = meta.get("config", {})
        diff = {key: {"stored": stored.get(key), "given": v} for key, v in config.items()
                if key not in RESUME_FREE and stored.get(key) != v}
        if diff:
            raise RuntimeError(f"{path} was written for a different experiment; differing arguments: "
                               f"{json.dumps(diff, sort_keys=True)}. Use a new --out")
        commit = _git_commit()
        if commit != meta.get("git_commit"):
            print(f"wsp.repair_study: resuming at commit {commit}, study written at {meta.get('git_commit')}")
        meta.setdefault("resumes", []).append({"time": now, "git_commit": commit,
                                               **{key: config[key] for key in RESUME_FREE}})
    path.write_text(json.dumps(meta, indent=1))
    return meta


def production_violations(args) -> dict:
    """{argument: {"given", "required"}} for every run argument that differs from the rule's production
    configuration (Amendment 2 item 3); empty when the run may proceed."""
    return {key: {"given": getattr(args, key), "required": v} for key, v in PRODUCTION.items()
            if getattr(args, key) != v}


def cmd_run(args) -> int:
    if args.production:
        bad = production_violations(args)
        if bad:
            raise RuntimeError(f"--production refuses this run: {RULE} Amendment 2 item 3 fixes "
                               f"{json.dumps(bad, sort_keys=True)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.backend == "openrouter":  # fail fast on a missing key
        client = OpenRouterClient(model=args.model, max_tokens=args.max_output_tokens,
                                  reasoning_max_tokens=args.reasoning_tokens)
    study_meta(out, args)
    cfg = {"k": args.k, "h": args.hidden, "n_initial": args.n_initial, "rounds": args.rounds,
           "n_queries": args.n_candidates, "n_heldout": args.n_heldout, "n_long": args.n_long,
           "train_steps": args.train_steps, "adam_steps": DEFAULT_CFG["adam_steps"], "fence": args.fence, "per_round": args.per_round,
           "distinct": args.distinct, "max_norm": args.max_norm}
    prompt_rate, completion_rate = rates_for(args.backend, getattr(args, "model", None))
    usage_path = out / "usage.jsonl"
    budget = Budget(args.spend_cap_usd, prompt_rate, completion_rate, args.max_output_tokens,
                    spent_usd=_accrued_usd(usage_path))
    for seed in [int(s) for s in args.seeds.split(",")]:
        path = _seed_path(out, seed)
        if path.is_file() and json.loads(path.read_text()).get("status") != "running":
            print(f"wsp.repair_study: {path} exists, skipping")
            continue
        if args.backend == "mock":
            client = MockLLM(seed=seed * 100 + 11, h=args.hidden, n_candidates=args.n_candidates)
        before = budget.snapshot()
        rec = run_seed(seed, cfg, client, budget, record_path=path)
        path.write_text(json.dumps(rec, indent=1))
        _append_usage(usage_path, "repair", seed, budget, before, aborted=rec["status"] != "ok")
        print(f"wsp.repair_study: wrote {path} status={rec['status']} failure={rec['failure']}")
        if rec["stopped_at_cap"]:
            stop = {"seed": seed, "spent_usd": budget.spent_usd, "cap_usd": args.spend_cap_usd,
                    "detail": rec["failure"], "time": datetime.now(timezone.utc).isoformat()}
            (out / "stopped_at_cap.json").write_text(json.dumps(stop, indent=1))
            print(f"wsp.repair_study: stopped at cap on seed {seed}")
            return 0
    return 0


def cmd_report(args) -> int:
    study, _ = report(args.out)
    print(f"wsp.repair_study: outcome={study['outcome']['outcome']} -> {Path(args.out) / 'study.md'}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return cmd_run(args) if args.cmd == "run" else cmd_report(args)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"wsp.repair_study: failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
