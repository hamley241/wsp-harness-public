"""Released analyzers for the three derived redesign records (released analyzer): recomputed from the seed records and
the frozen construction, deterministically and without a model call.

    uv run python -m wsp.redesign_extras --main DIR --seed1 DIR --out DIR [--relaxed DIR]

1. outcomes-both-sets.json: `repair_study.outcome` over the as-run seed set and the production-order set.
2. replay-first-attempts.md: every editor round's first stored response parsed with the relaxed fence.
3. offline-scoring.md: every parse- or budget-rejected proposal applied to its reconstructed round-start
   weights and scored as held-out gain, after a budget check identical to the run's.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

from wsp import trained_baseline as tb
from wsp.dfa import random_dfa
from wsp.llm import _FENCE
from wsp.numeric_edit import EditBudget, _f32, parse_edits
from wsp.oracle import Oracle
from wsp.repair_study import F, _records, _table, outcome

AS_RUN = (15, 20, 21, 24, 30)            # the seeds as run in `run/`
PRODUCTION_ORDER = (1, 15, 20, 21, 24)   # the rule's production order, seed 1 from `run-seed1/`
EDITOR_ARMS = ("E1", "E2")
REJECTED = ("parse", "budget")           # attempt outcomes that carry a stored, rejected proposal
OPENING_MAX = 160
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class ReconstructionError(AssertionError):
    """The reconstructed construction or round-start weights do not reproduce the record."""


def _placeholder(seed: int) -> dict:
    return {"seed": seed, "status": "missing", "failure": "no seed record"}


def both_sets(main_dir, seed1_dir, f: float = F) -> dict:
    """Outcome over both seed sets from the records present in the two directories (main first)."""
    recs: dict[int, dict] = {}
    for d in (main_dir, seed1_dir):
        for rec in _records(Path(d)):
            recs.setdefault(rec["seed"], rec)
    sets = {"as_run": AS_RUN, "production_order": PRODUCTION_ORDER}
    out = {name: outcome([recs.get(s, _placeholder(s)) for s in seeds], f) for name, seeds in sets.items()}
    out["sets"] = {name: list(seeds) for name, seeds in sets.items()}
    return out


def _load(seed_dirs) -> list[tuple[str, dict]]:
    """[(run label, record)] over {label: dir} (or [(label, dir), ...]), every seed record in each."""
    items = seed_dirs.items() if isinstance(seed_dirs, dict) else seed_dirs
    return [(run, rec) for run, d in items for rec in _records(Path(d))]


def _by_round(arm: dict) -> list[tuple[dict, list[tuple[dict, dict]]]]:
    """[(round record, [(attempt record, call entry), ...])]: the k-th editor call of the arm is attempt k
    counting across rounds, so calls are sliced by each round's n_attempts."""
    calls, rounds = arm.get("calls", []), arm.get("rounds", [])
    if len(calls) != sum(rd["n_attempts"] for rd in rounds):
        raise ValueError(f"{len(calls)} calls do not match {[rd['n_attempts'] for rd in rounds]} attempts")
    out, i = [], 0
    for rd in rounds:
        n = rd["n_attempts"]
        out.append((rd, list(zip(rd["attempts"], calls[i:i + n]))))
        i += n
    return out


def _opening(prose: str) -> str:
    first = _SENTENCE_END.split(prose, 1)[0] if prose else ""
    return " ".join(first.split())[:OPENING_MAX]


def replay_first_attempts(seed_dirs) -> list[dict]:
    """One row per editor round: the round's first attempt with a stored response (an llm_error or
    client_error attempt stores none), parsed with the relaxed fence."""
    rows = []
    for run, rec in _load(seed_dirs):
        for arm in EDITOR_ARMS:
            for rd, pairs in _by_round(rec["arms"][arm]):
                row = {"run": run, "seed": rec["seed"], "arm": arm, "round": rd["round"], "attempt": None,
                       "strict_outcome": None, "relaxed_parse": "no stored response", "n_edits": None,
                       "prose_chars": None, "opening_sentence": ""}
                first = next(((at, c) for at, c in pairs if c.get("response") is not None), None)
                if first is not None:
                    at, call = first
                    edits = parse_edits(call["response"], fence="relaxed")
                    prose = _FENCE.sub("", call["response"]).strip()
                    row.update(attempt=at["attempt"], strict_outcome=at["outcome"],
                               relaxed_parse="ok" if isinstance(edits, list) else edits,
                               n_edits=len(edits) if isinstance(edits, list) else None,
                               prose_chars=len(prose), opening_sentence=_opening(prose))
                rows.append(row)
    return rows


def _construction(rec: dict, cfg: dict, cache: dict):
    """(D0 float32, held-out set) of the record's seed from the frozen construction, checked against the
    record's initial dataset and D0 held-out accuracy."""
    seed = rec["seed"]
    key = (seed, cfg["k"], cfg["h"], cfg["n_initial"], cfg["n_heldout"], cfg["train_steps"])
    if key not in cache:
        oracle = Oracle(random_dfa(cfg["k"], seed), seed)
        initial = oracle.initial_dataset(cfg["n_initial"])
        heldout = oracle.heldout(cfg["n_heldout"])
        if [list(t) for t in initial] != [list(t) for t in rec["initial"]]:
            raise ReconstructionError(f"seed {seed}: reconstructed initial dataset differs from the record")
        d0 = _f32(tb.train(initial, cfg["h"], seed, cfg["train_steps"])[0])
        if tb.accuracy(d0, heldout) != rec["D0"]["heldout"]:
            raise ReconstructionError(f"seed {seed}: D0 held-out {tb.accuracy(d0, heldout)} != "
                                      f"record {rec['D0']['heldout']}")
        cache[key] = (d0, heldout)
    return cache[key]


def score_rejected(seed_dirs, cfg: dict | None = None, f: float = F) -> list[dict]:
    """One row per parse- or budget-rejected attempt, scored at its round-start weights. `cfg` overrides the
    record's config. The reconstructed weights must reproduce every recorded held-out accuracy exactly."""
    rows, cache = [], {}
    for run, rec in _load(seed_dirs):
        if not rec.get("arms"):
            continue
        c = {**rec["config"], **(cfg or {})}
        d0, heldout = _construction(rec, c, cache)
        for arm in EDITOR_ARMS:
            budget = EditBudget(d0, c.get("per_round", 24), c.get("distinct", 48), c.get("max_norm", 1.0))
            cur, expected = _f32(d0), rec["D0"]["heldout"]
            for rd, pairs in _by_round(rec["arms"][arm]):
                start = tb.accuracy(cur, heldout)
                if start != expected:
                    raise ReconstructionError(f"{run} seed {rec['seed']} {arm} round {rd['round']}: "
                                              f"reconstructed round-start held-out {start} != record {expected}")
                for at, call in pairs:
                    if at["outcome"] not in REJECTED:
                        continue
                    row = {"run": run, "seed": rec["seed"], "arm": arm, "round": rd["round"],
                           "attempt": at["attempt"], "n_edits": None, "budget_ok": False, "reason": "",
                           "start_heldout": start, "after_heldout": None, "gain": None, "above_f": None}
                    text = call.get("response")
                    edits = parse_edits(text, fence="relaxed") if text is not None else "no stored response"
                    if isinstance(edits, str):
                        row["reason"] = edits
                    else:
                        row["n_edits"] = len(edits)
                        trial = copy.deepcopy(budget)
                        reason = trial.check(edits, cur)
                        if reason is not None:
                            row["reason"] = reason
                        else:
                            after = tb.accuracy(trial.apply(edits, cur), heldout)
                            row.update(budget_ok=True, after_heldout=after, gain=after - start,
                                       above_f=after - start > f)
                    rows.append(row)
                if rd["edits"]:
                    cur = budget.apply([(p, i, v) for p, i, v in rd["edits"]], cur)
                expected = rd["heldout"]
            if tb.accuracy(cur, heldout) != expected:
                raise ReconstructionError(f"{run} seed {rec['seed']} {arm}: reconstructed final held-out "
                                          f"{tb.accuracy(cur, heldout)} != record {expected}")
    return rows


# ------------------------------------------------------------------- report


def _acc(x) -> str:
    return "" if x is None else f"{x:.3f}"


def _gain(x) -> str:
    return "" if x is None else f"{x:+.3f}"


def _yes(x) -> str:
    return "" if x is None else ("yes" if x else "no")


def _cell(s) -> str:
    return str(s).replace("|", "\\|")


def replay_markdown(rows: list[dict]) -> str:
    head = ["run", "seed", "arm", "round", "attempt", "strict outcome", "relaxed parse", "edits proposed",
            "prose chars", "opening sentence of the prose"]
    body = [[r["run"], r["seed"], r["arm"], r["round"], r["attempt"], r["strict_outcome"], _cell(r["relaxed_parse"]),
             r["n_edits"], r["prose_chars"], _cell(r["opening_sentence"])] for r in rows]
    ok = sum(r["relaxed_parse"] == "ok" for r in rows)
    hist = dict(sorted(Counter(r["n_edits"] for r in rows if r["n_edits"] is not None).items()))
    return "\n".join([
        "# Offline replay of the editor's first attempts (relaxed fence)", "",
        "One row per editor round: the round's first attempt with a stored response, parsed with the relaxed "
        "rule (take the single json fence, ignore prose). Recomputed by `wsp.redesign_extras`; no model calls.", "",
        *_table(head, body), "",
        f"Summary: {ok} of {len(rows)} first attempts parse under the relaxed rule. Edits proposed: {hist}.", ""])


def scoring_summary(rows: list[dict], f: float = F) -> str:
    scored = [r for r in rows if r["budget_ok"]]
    gains = [r["gain"] for r in scored]
    above = sum(g > f for g in gains)
    below = sum(g <= -f for g in gains)
    extreme = f"max gain {_gain(max(gains))}, min {_gain(min(gains))}" if gains else "no gains"
    return (f"Summary: {len(rows)} proposals scored, {len(scored)} within budget; gains above f = {above}; "
            f"gains at or below −f = {below}; within ±f = {len(scored) - above - below}; {extreme}; "
            f"zero-edit proposals {sum(r['n_edits'] == 0 for r in rows)}; "
            f"proposals with zero held-out gain {sum(g == 0 for g in gains)}.")


def scoring_markdown(rows: list[dict], f: float = F) -> str:
    head = ["run", "seed", "arm", "round", "attempt", "n_edits", "budget ok", "round-start held-out",
            "after proposal", "gain", "> f", "reason"]
    body = [[r["run"], r["seed"], r["arm"], r["round"], r["attempt"], r["n_edits"], _yes(r["budget_ok"]),
             _acc(r["start_heldout"]), _acc(r["after_heldout"]), _gain(r["gain"]), _yes(r["above_f"]),
             _cell(r["reason"])] for r in rows]
    return "\n".join([
        "# Offline scoring of the stored rejected editor proposals", "",
        "Each parse- or budget-rejected proposal is applied to its recorded round-start weights (D0 reproduced by "
        "the frozen trainer, plus the edits the record shows applied in earlier rounds of the same realization) "
        "and scored as held-out gain over that round-start accuracy, after a budget check identical to the run's. "
        "The reconstruction reproduces every recorded held-out accuracy exactly. Recomputed by "
        "`wsp.redesign_extras`; no model calls.", "",
        *_table(head, body), "", scoring_summary(rows, f), ""])


def report(main_dir, seed1_dir, out_dir, relaxed_dir=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = {"main": main_dir, "seed1": seed1_dir}
    if relaxed_dir is not None:
        dirs["relaxed"] = relaxed_dir
    both = both_sets(main_dir, seed1_dir)
    replay, scoring = replay_first_attempts(dirs), score_rejected(dirs)
    (out / "outcomes-both-sets.json").write_text(json.dumps(both, indent=1))
    (out / "replay-first-attempts.md").write_text(replay_markdown(replay))
    (out / "offline-scoring.md").write_text(scoring_markdown(scoring))
    return {"both_sets": both, "replay": replay, "scoring": scoring}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wsp.redesign_extras")
    p.add_argument("--main", required=True)
    p.add_argument("--seed1", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--relaxed", default=None)
    args = p.parse_args(argv)
    res = report(args.main, args.seed1, args.out, args.relaxed)
    print(f"wsp.redesign_extras: as_run={res['both_sets']['as_run']['outcome']} "
          f"production_order={res['both_sets']['production_order']['outcome']}; "
          f"{len(res['replay'])} replay rows; {scoring_summary(res['scoring'])} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
