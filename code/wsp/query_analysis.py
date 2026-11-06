"""Query-policy study analysis (rules/query-study-rule.md, Amendment 1 items 4, 5, 7): pairing, exact
one-sided sign test, thresholds and outcomes, sensitivity, reliability tally, and the study report.

    uv run python -m wsp.query_analysis --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ALPHA = 0.05
MIN_READ = 12        # 12..19 informative pairs: a rejection counts, a non-rejection is underpowered
MIN_FULL = 20        # 20 or more: the test is read in full
SENSITIVITY_SEEDS = 5
OUTCOMES = ("A_beats_random", "no_advantage_shown", "underpowered_no_result", "no_result")


# ------------------------------------------------------------------ pairing


def event_values(arm: dict):
    """unique -> {event_time}; censored -> "censored"; indeterminate -> set of consistent round totals
    (empty = unknown); failed -> None."""
    status = arm["final_status"]
    if status == "unique":
        return {int(arm["event_time"])}
    if status == "censored":
        return "censored"
    if status == "indeterminate":
        return {int(t) for t in (arm.get("event_set") or [])}
    return None


def _event_json(values):
    if isinstance(values, set):
        return sorted(values)[0] if len(values) == 1 else sorted(values)
    return values


def _compare(a, b):
    """(winner, reason): winner "A"/"B" when every value consistent with both gives the same strict ordering."""
    if a is None or b is None:
        return None, "failed"
    if a == "censored" and b == "censored":
        return None, "both_censored"
    if (isinstance(a, set) and not a) or (isinstance(b, set) and not b):
        return None, "indeterminate_unknown"
    if a == "censored":
        return "B", None
    if b == "censored":
        return "A", None
    if max(a) < min(b):
        return "A", None
    if max(b) < min(a):
        return "B", None
    return None, "equal" if a == b and len(a) == 1 else "overlap"


def pair(seed_record: dict) -> dict:
    arm_a, arm_b = seed_record["arms"]["A"], seed_record["arms"]["B"]
    a, b = event_values(arm_a), event_values(arm_b)
    winner, reason = _compare(a, b)
    return {"seed": seed_record["seed"], "informative": winner is not None, "winner": winner, "reason": reason,
            "event_A": _event_json(a), "status_A": arm_a["final_status"],
            "event_B": _event_json(b), "status_B": arm_b["final_status"]}


# ------------------------------------------------------------------- test


def sign_test(wins_A: int, n: int) -> float:
    """Exact one-sided binomial upper tail P(X >= wins_A) at p = 0.5."""
    if n <= 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins_A, n + 1)) / 2 ** n


def decide(pairs: list[dict], extension_done: bool, stopped_at_cap: bool = False, main_done: bool = True) -> dict:
    """The test is read only once the required roster is complete (spec item 4); an incomplete main roster,
    like a cap stop, is `no_result`."""
    informative = [p for p in pairs if p["informative"]]
    n = len(informative)
    wins_a = sum(1 for p in informative if p["winner"] == "A")
    wins_b = n - wins_a
    p = sign_test(wins_a, n)
    margin_ok = wins_a > n / 2
    rejected = main_done and n >= MIN_READ and p < ALPHA and margin_ok
    if stopped_at_cap or not main_done or n < MIN_READ:
        outcome = "no_result"
    elif rejected:
        outcome = "A_beats_random"
    elif n >= MIN_FULL:
        outcome = "no_advantage_shown"
    else:
        outcome = "underpowered_no_result"
    return {"n_pairs": len(pairs), "n_informative": n, "wins_A": wins_a, "wins_B": wins_b, "p": p,
            "alpha": ALPHA, "margin_ok": margin_ok, "rejected": rejected, "outcome": outcome,
            "stopped_at_cap": stopped_at_cap, "main_done": main_done, "extension_done": extension_done,
            "needs_extension": main_done and not extension_done and n < MIN_FULL}


# ------------------------------------------------------------ sensitivity


def discordant(a: dict | None, b: dict | None) -> bool:
    """Both assessable and: both unique with different times, or one unique and the other censored."""
    if a is None or b is None:
        return False
    sa, sb = a["final_status"], b["final_status"]
    if sa not in ("unique", "censored") or sb not in ("unique", "censored"):
        return False
    if sa == "unique" and sb == "unique":
        return a["event_time"] != b["event_time"]
    return sa != sb


def _load_seeds(d: Path) -> dict[int, dict]:
    if not d.is_dir():
        return {}
    out = {}
    for path in d.glob("seed_*.json"):
        rec = json.loads(path.read_text())
        out[int(rec["seed"])] = rec
    return out


def _read_roster(out: Path) -> dict:
    path = out / "roster.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def roster_scope(ros: dict) -> dict[str, list[int]]:
    """Directory name -> the seeds frozen for it in roster.json. Sensitivity reruns cover the first
    SENSITIVITY_SEEDS main seeds (spec item 2)."""
    main = [int(s) for s in ros.get("main", [])]
    ext = [int(s) for s in ros.get("extension", [])]
    sens = main[:SENSITIVITY_SEEDS]
    return {"main": main, "extension": ext, "sensitivity-replicate": sens, "sensitivity-inverted": sens}


def _scoped(records: dict[int, dict], seeds: list[int]) -> tuple[dict[int, dict], list[int], list[int]]:
    """(rostered records present, rostered seeds missing, unrostered seeds found on disk)."""
    present = {s: records[s] for s in seeds if s in records}
    missing = [s for s in seeds if s not in records]
    unrostered = sorted(s for s in records if s not in set(seeds))
    return present, missing, unrostered


def _arm_a(seeds: dict[int, dict], seed: int) -> dict | None:
    rec = seeds.get(seed)
    return rec["arms"]["A"] if rec is not None else None


def sensitivity(main_dir, replicate_dir, inverted_dir, seeds: list[int] | None = None) -> dict:
    main, rep, inv = (_load_seeds(Path(d)) for d in (main_dir, replicate_dir, inverted_dir))
    if seeds is None:
        seeds = sorted(main)[:SENSITIVITY_SEEDS]
    rows, d_inv, d_rep, inconclusive = [], 0, 0, False
    for seed in seeds:
        runs = {name: _arm_a(s, seed) for name, s in (("main", main), ("replicate", rep), ("inverted", inv))}
        row = {"seed": seed}
        for name, arm in runs.items():
            row[f"status_{name}"] = arm["final_status"] if arm is not None else "missing"
            row[f"event_{name}"] = arm["event_time"] if arm is not None else None
            if arm is None or arm["final_status"] in ("failed", "indeterminate"):
                inconclusive = True
        row["discordant_inverted"] = discordant(runs["main"], runs["inverted"])
        row["discordant_replicate"] = discordant(runs["main"], runs["replicate"])
        d_inv += row["discordant_inverted"]
        d_rep += row["discordant_replicate"]
        rows.append(row)
    sensitive = d_inv >= 2 and d_inv > d_rep
    return {"seeds": list(seeds), "d_inv": d_inv, "d_rep": d_rep, "inconclusive": inconclusive,
            "tiebreak_sensitive": sensitive, "bars_adversary_arm": inconclusive or sensitive, "rows": rows}


# ------------------------------------------------------------ reliability


def reliability(out_dir) -> dict:
    out = Path(out_dir)
    tally = {"ineligible": {}, "refusals": 0, "errors": 0, "attempts": 0, "fallback_random_rounds": 0,
             "filled_strings": 0, "standin_timeouts": 0, "endpoint_timeouts": 0, "indeterminate_arms": 0,
             "failed_arms": 0, "reruns": 0, "arms": 0, "rounds": 0,
             "usage": {"prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0, "entries": 0, "aborted": 0},
             "stopped_at_cap": None}
    ros = _read_roster(out)
    for e in ros.get("ineligible", []):
        tally["ineligible"][e["reason"]] = tally["ineligible"].get(e["reason"], 0) + 1
    # Only the rostered seeds of the rostered directories count; stale or unrelated seed files do not.
    for name, seeds in roster_scope(ros).items():
        rostered, _missing, _unrostered = _scoped(_load_seeds(out / name), seeds)
        for rec in rostered.values():
            for arm in rec["arms"].values():
                tally["arms"] += 1
                tally["indeterminate_arms"] += arm["final_status"] == "indeterminate"
                tally["failed_arms"] += arm["final_status"] == "failed"
                for r in arm["rounds"]:
                    tally["rounds"] += 1
                    tally["refusals"] += r.get("n_refusals", 0)
                    tally["errors"] += r.get("n_errors", 0)
                    tally["attempts"] += r.get("attempts", 0)
                    tally["fallback_random_rounds"] += bool(r.get("fallback_random"))
                    tally["filled_strings"] += r.get("n_filled_random", 0)
                    tally["standin_timeouts"] += bool(r.get("standin_timeout"))
                    tally["endpoint_timeouts"] += bool(r.get("sat") and r["sat"].get("status") == "timeout")
                    tally["reruns"] += bool(r.get("n_errors", 0) and r.get("queried"))
    usage_path = out / "usage.jsonl"
    if usage_path.is_file():
        for line in usage_path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                u = tally["usage"]
                u["prompt_tokens"] += int(e.get("prompt_tokens") or 0)
                u["completion_tokens"] += int(e.get("completion_tokens") or 0)
                u["usd"] += float(e.get("usd") or 0.0)
                u["entries"] += 1
                u["aborted"] += bool(e.get("aborted"))
    cap_path = out / "stopped_at_cap.json"
    if cap_path.is_file():
        tally["stopped_at_cap"] = json.loads(cap_path.read_text())
    return tally


# ----------------------------------------------------------------- report


def _table(pairs: list[dict]) -> str:
    lines = ["| seed | event_A | status_A | event_B | status_B | informative | winner/reason |",
             "|---|---|---|---|---|---|---|"]
    for p in pairs:
        lines.append(f"| {p['seed']} | {p['event_A']} | {p['status_A']} | {p['event_B']} | {p['status_B']} | "
                     f"{p['informative']} | {p['winner'] or p['reason']} |")
    return "\n".join(lines)


def _kv(d: dict) -> str:
    return "\n".join(f"- {k}: {json.dumps(v)}" for k, v in d.items())


def report(out_dir) -> tuple[dict, str]:
    out = Path(out_dir)
    ros = _read_roster(out)
    scope = roster_scope(ros)
    # The analysis is scoped to the seeds frozen in roster.json, in roster order: a missing rostered record
    # leaves the roster incomplete, and an unrostered seed file on disk is reported but never analysed.
    main_recs, main_missing, main_unrostered = _scoped(_load_seeds(out / "main"), scope["main"])
    ext_recs, ext_missing, ext_unrostered = _scoped(_load_seeds(out / "extension"), scope["extension"])
    main_pairs = [pair(main_recs[s]) for s in scope["main"] if s in main_recs]
    ext_pairs = [pair(ext_recs[s]) for s in scope["extension"] if s in ext_recs]
    main_done = bool(scope["main"]) and not main_missing
    extension_done = bool(scope["extension"]) and not ext_missing
    rel = reliability(out)
    decision = decide(main_pairs + ext_pairs, extension_done, stopped_at_cap=rel["stopped_at_cap"] is not None,
                      main_done=main_done)
    roster_check = {"main_missing": main_missing, "main_unrostered": main_unrostered,
                    "extension_missing": ext_missing, "extension_unrostered": ext_unrostered}
    study = {"roster": ros, "roster_check": roster_check, "main": main_pairs, "extension": ext_pairs,
             "decision": decision, "reliability": rel, "spend": rel["usage"]}
    rep_dir, inv_dir = out / "sensitivity-replicate", out / "sensitivity-inverted"
    if rep_dir.is_dir() or inv_dir.is_dir():
        study["sensitivity"] = sensitivity(out / "main", rep_dir, inv_dir, scope["sensitivity-replicate"] or None)
    meta_path = out / "study_meta.json"
    study["study_meta"] = json.loads(meta_path.read_text()) if meta_path.is_file() else None

    md = ["# Query-policy study", "", "## Main roster", "", _table(main_pairs), ""]
    if ext_pairs or scope["extension"]:
        md += ["## Extension roster", "", _table(ext_pairs), ""]
    md += ["## Roster check", "", _kv(roster_check), "", "## Decision", "", _kv(decision), ""]
    if "sensitivity" in study:
        sens = study["sensitivity"]
        md += ["## Sensitivity (arm A, seeds " + ", ".join(str(s) for s in sens["seeds"]) + ")", "",
               "| seed | event_main | status_main | event_replicate | status_replicate | event_inverted | "
               "status_inverted | discordant_inverted | discordant_replicate |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {r['seed']} | {r['event_main']} | {r['status_main']} | {r['event_replicate']} | "
               f"{r['status_replicate']} | {r['event_inverted']} | {r['status_inverted']} | "
               f"{r['discordant_inverted']} | {r['discordant_replicate']} |" for r in sens["rows"]]
        md += ["", _kv({k: v for k, v in sens.items() if k not in ("rows", "seeds")}), ""]
    md += ["## Reliability", "", _kv({k: v for k, v in rel.items() if k != "usage"}), "",
           "## Spend", "", _kv(rel["usage"]), "",
           "## Study meta", "", "```json", json.dumps(study["study_meta"], indent=1), "```", ""]
    text = "\n".join(md)
    (out / "study.json").write_text(json.dumps(study, indent=1))
    (out / "study.md").write_text(text)
    return study, text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wsp.query_analysis")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    study, _ = report(args.out)
    d = study["decision"]
    print(f"wsp.query_analysis: outcome={d['outcome']} n_informative={d['n_informative']} wins_A={d['wins_A']} "
          f"p={d['p']:.4f} -> {Path(args.out) / 'study.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
