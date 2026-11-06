"""Query-policy study (rules/query-study-rule.md, Amendment 1): arm A's LLM query policy against
random queries (arm B), each judged by a SAT stand-in instead of the pilot's edited network.

    uv run python -m wsp.query_study replay --pilot records/pilot/run --k 6 --out DIR
    uv run python -m wsp.query_study run --roster main --backend mock --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from wsp import prompts
from wsp.arms import PROBE_LIMIT, _uniform_fill, parse_strings, target_histogram
from wsp.dfa import DFA, minimize, random_dfa, reachable_states, trim
from wsp.llm import BudgetStop, CallLog, LLMError, LLMRefusal, MockLLM, OpenRouterClient
from wsp.oracle import Oracle
from wsp.sat_dfa import identifiability, identify

REPLAY_ROUNDS = (1, 2, 3)
MAX_REFUSALS = 3     # refused attempts per round before the round is random-filled
MAX_ERRORS = 2       # non-refusal failures before charging, within one round, that mark the arm failed
SENSITIVITY_SEEDS = 5
ROSTERS = ("main", "extension", "sensitivity-replicate", "sensitivity-inverted")
# $ per token (rule: $4 / $20 per million for anthropic/claude-opus-5.5); the mock bills nothing.
RATES = {"mock": (0.0, 0.0), "openrouter": (4e-6, 20e-6)}
# USD per token (prompt, completion incl. reasoning) by OpenRouter model id, verified against
# https://openrouter.ai/api/v1/models on 2026-11-05; the spend guard refuses an unknown model.
MODEL_RATES = {
    "anthropic/claude-opus-5.5": (4e-6, 20e-6),
    "openai/gpt-6-astra": (10e-6, 50e-6),
    "openai/gpt-6-sol": (2e-6, 10e-6),
    "openai/gpt-5.6-sol": (4e-6, 20e-6),
}


def rates_for(backend: str, model: str | None):
    """(prompt_rate, completion_rate): mock is free; openrouter needs a model in MODEL_RATES."""
    if backend == "mock":
        return RATES["mock"]
    if model not in MODEL_RATES:
        raise SystemExit(f"no verified price for model {model!r}; add it to MODEL_RATES before spending")
    return MODEL_RATES[model]


# ------------------------------------------------------------ SAT samples


def sat_samples(initial, charged) -> list[tuple[str, int]]:
    """Initial pairs in order, then each charged query's first occurrence in ledger order."""
    out = [(s, int(y)) for s, y in initial]
    seen: set[str] = set()
    for s, y in charged:
        if s not in seen:
            seen.add(s)
            out.append((s, int(y)))
    return out


def eligible(dfa: DFA, k: int) -> str | None:
    """None when the DFA is minimal at k states with every state reachable, else the reason."""
    if len(reachable_states(dfa)) != dfa.n_states:
        return "unreachable"
    if minimize(trim(dfa)).n_states != k:
        return "not_minimal"
    return None


def roster(k: int, n_initial: int, n_main: int, n_ext: int, sat_timeout: float = 30) -> dict:
    """Scan seeds upward from 1: the first n_main eligible seeds, then the next n_ext."""
    main: list[int] = []
    ext: list[int] = []
    ineligible: list[dict] = []
    seed = 0
    while len(main) + len(ext) < n_main + n_ext:
        seed += 1
        dfa = random_dfa(k, seed)
        reason = eligible(dfa, k)
        if reason is None:
            initial = Oracle(dfa, seed).initial_dataset(n_initial)
            res = identifiability(initial, k, sat_timeout)
            if res["status"] == "timeout" or res["ambiguity"] == "no_signal":
                raise RuntimeError(f"preflight: identifiability timed out on seed {seed}")
            if res["ambiguity"] == "unique":
                reason = "initial_unique"
        if reason is not None:
            ineligible.append({"seed": seed, "reason": reason})
        elif len(main) < n_main:
            main.append(seed)
        else:
            ext.append(seed)
    return {"main": main, "extension": ext, "ineligible": ineligible}


# ---------------------------------------------------------------- stand-in


class StandIn:
    """The SAT DFA at k states consistent with `samples`, decoded from the solver's first model."""

    def __init__(self, samples, k: int, timeout_s: float = 30, phases: str = "default"):
        res = identify(samples, k, timeout_s, phases)
        self.status = res.status
        self.dfa = res.dfa

    def predict(self, strings) -> list[int] | None:
        if self.dfa is None:
            return None
        return [self.dfa.label(s) for s in strings]


def determined(samples, k: int, s: str, y: int, timeout_s: float = 30) -> bool | None:
    """True iff every k-state DFA consistent with `samples` labels s as y (adding (s, 1-y) is UNSAT)."""
    try:
        res = identify(list(samples) + [(s, 1 - int(y))], k, timeout_s)
    except ValueError:  # s is already in the samples with label y
        return True
    if res.status == "timeout":
        return None
    return res.status == "unsat"


def replay(pilot_dir, k: int = 6, timeout_s: float = 30, phases: str = "default") -> dict:
    """Machinery check: the stand-in over the pilot's information set against A's recorded predictions."""
    pilot_dir = Path(pilot_dir)
    results = json.loads((pilot_dir / "results.json").read_text())
    info = [(s, int(y)) for s, y in results["initial_dataset"]]
    rounds = []
    for r in REPLAY_ROUNDS:
        rec = json.loads((pilot_dir / "A" / f"round_{r}" / "record.json").read_text())
        probes = [(s, int(p)) for s, p in rec["probe_predictions"]]
        standin = StandIn(info, k, timeout_s, phases)
        preds = standin.predict([s for s, _ in probes])
        row = {"round": r, "n_probes": len(probes), "n_determined": 0, "agree_determined": 0,
               "agree_undetermined": 0, "n_determination_timeouts": 0, "standin_status": standin.status}
        for (s, p), q in zip(probes, preds or []):
            d = determined(info, k, s, q, timeout_s)
            if d is None:
                row["n_determination_timeouts"] += 1
            elif d:
                row["n_determined"] += 1
                row["agree_determined"] += int(p == q)
            else:
                row["agree_undetermined"] += int(p == q)
        rounds.append(row)
        info += [(s, int(y)) for s, y, _p in rec["counterexamples"]]
    n_det = sum(r["n_determined"] for r in rounds)
    ok = (all(r["standin_status"] == "sat" for r in rounds)
          and not any(r["n_determination_timeouts"] for r in rounds)
          and all(r["agree_determined"] == r["n_determined"] for r in rounds))
    return {"pilot_dir": str(pilot_dir), "k": k, "phases": phases, "timeout_s": timeout_s, "rounds": rounds,
            "n_determined": n_det, "pass": ok, "vacuous": n_det == 0}


# ------------------------------------------------------------------ budget


class Budget:
    """Spend guard (rule Amendment 1 item 7). Rates are $ per token. A cap of 0 permits no call."""

    def __init__(self, cap_usd: float, prompt_rate: float = 0.0, completion_rate: float = 0.0,
                 max_output_tokens: int = 0, spent_usd: float = 0.0):
        self.cap_usd = cap_usd
        self.prompt_rate = prompt_rate
        self.completion_rate = completion_rate
        self.max_output_tokens = max_output_tokens
        self.spent_usd = spent_usd
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def reserve(self, prompt_text: str) -> float:
        return len(prompt_text) / 3 * self.prompt_rate + self.max_output_tokens * self.completion_rate

    def can(self, prompt_text: str) -> bool:
        return self.cap_usd > 0 and self.spent_usd + self.reserve(prompt_text) <= self.cap_usd

    def record(self, usage: dict) -> None:
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.spent_usd += pt * self.prompt_rate + ct * self.completion_rate

    def snapshot(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "usd": self.spent_usd}


class _BudgetedLLM:
    """Checks the budget before every call and records provider usage after it, refusals and errors included.
    Every other attribute (last_usage with its finish reason, usage_log, model, ...) is the wrapped client's."""

    def __init__(self, llm, budget: Budget):
        self.llm = llm
        self.budget = budget
        self._recorded = None

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def complete(self, system: str, user: str) -> str:
        if not self.budget.can(system + user):
            raise BudgetStop(f"spend cap {self.budget.cap_usd} reached: spent {self.budget.spent_usd:.4f} USD")
        try:
            return self.llm.complete(system, user)
        finally:
            usage = getattr(self.llm, "last_usage", None)
            if usage is not None and usage is not self._recorded:
                self._recorded = usage
                self.budget.record(usage)


# ------------------------------------------------------------------- arms


def _gather_a(initial, samples, target, k, gen_llm, calls, phases, sat_timeout, rng):
    """One uncharged attempt of A's round: probe call, stand-in predictions, candidate call, re-prompt, fill."""
    initial_strings = [s for s, _ in initial]
    gen_sys = prompts.generator_system()
    probe_text = calls.complete(gen_llm, gen_sys, prompts.generator_user(
        initial, [], target_histogram(initial_strings, PROBE_LIMIT)))
    probes = parse_strings(probe_text)[:PROBE_LIMIT]
    standin = StandIn(samples, k, sat_timeout, phases)
    preds = standin.predict(probes)
    probe_preds = list(zip(probes, preds)) if preds is not None else []
    chosen: dict[int, list[str]] = {L: [] for L in target}
    taken: set[str] = set()

    def take(strings):
        for s in strings:
            L = len(s)
            if L in target and len(chosen[L]) < target[L] and s not in taken:
                chosen[L].append(s)
                taken.add(s)

    def missing():
        return {L: target[L] - len(chosen[L]) for L in target if len(chosen[L]) < target[L]}

    take(parse_strings(calls.complete(gen_llm, gen_sys, prompts.generator_user(initial, probe_preds, target))))
    if missing():  # re-prompt once for the missing buckets
        take(parse_strings(calls.complete(gen_llm, gen_sys, prompts.generator_user(initial, probe_preds, missing()))))
    n_filled = 0
    for L, c in missing().items():
        chosen[L].extend(_uniform_fill(rng, L, c, taken))
        n_filled += c
    extra = {"standin_status": standin.status, "standin_timeout": standin.status != "sat",
             "standin_predictions": [list(t) for t in probe_preds], "n_filled_random": n_filled}
    return [s for L in target for s in chosen[L]], standin, extra


def _random_batch(target, rng) -> list[str]:
    taken: set[str] = set()
    return [s for L, c in target.items() for s in _uniform_fill(rng, L, c, taken)]


def run_arm(name, seed, dfa, initial, k, rounds, n_candidates, gen_llm, phases, sat_timeout, rng, budget) -> dict:
    """One arm on one seed; returns the per-round records and the endpoint. B ignores gen_llm."""
    oracle = Oracle(dfa, seed)
    target = target_histogram([s for s, _ in initial], n_candidates)
    calls = CallLog()
    charged: list[tuple[str, int]] = []
    records: list[dict] = []
    final_status, event_time, failure = None, None, None
    for r in range(1, rounds + 1):
        samples = sat_samples(initial, charged)
        rec = {"round": r, "attempts": 0, "n_refusals": 0, "n_errors": 0, "fallback_random": False,
               "n_filled_random": 0}
        if name == "A":  # overwritten by a completed gather; a fallback_random round never showed predictions
            rec.update(standin_status=None, standin_timeout=False, standin_predictions=[], n_counterexamples=None)
        candidates, standin = None, None
        while candidates is None:
            rec["attempts"] += 1
            try:
                if name == "A":
                    candidates, standin, extra = _gather_a(initial, samples, target, k, gen_llm, calls, phases,
                                                           sat_timeout, rng)
                    rec.update(extra)
                else:
                    candidates = _random_batch(target, rng)
            except BudgetStop:
                raise
            except LLMRefusal:
                rec["n_refusals"] += 1
                if rec["n_refusals"] >= MAX_REFUSALS:
                    candidates = _random_batch(target, rng)
                    rec.update(fallback_random=True, n_filled_random=len(candidates))
            except Exception as e:  # noqa: BLE001 - a non-refusal failure before any label is charged
                rec["n_errors"] += 1  # per round: each charged round gets one whole-round rerun
                rec["error_detail"] = f"{type(e).__name__}: {e}"[:300]
                if rec["n_errors"] >= MAX_ERRORS:
                    failure = f"round {r}: {rec['error_detail']}"
                    break
        if failure is not None:
            rec.update(queried=[], labels=[], ledger_total=len(charged), sat=None)
            records.append(rec)
            final_status = "failed"
            break
        try:
            labels = oracle.query(name, candidates)  # charged
            charged += list(zip(candidates, labels))
            total = oracle.ledger()[name]["total"]
            rec.update(queried=list(candidates), labels=list(labels), ledger_total=total)
            if standin is not None and standin.dfa is not None:
                rec["n_counterexamples"] = sum(1 for s, y in zip(candidates, labels) if standin.dfa.label(s) != y)
            sat = identifiability(sat_samples(initial, charged), k, sat_timeout, phases)
            rec["sat"] = sat
        except Exception as e:  # noqa: BLE001 - after charging: never rerun
            rec.setdefault("ledger_total", len(charged))
            rec.setdefault("sat", None)
            rec["error_detail"] = f"{type(e).__name__}: {e}"[:300]
            records.append(rec)
            failure = f"round {r} after charging: {rec['error_detail']}"
            final_status = "failed"
            break
        records.append(rec)
        if sat["status"] == "unsat":
            failure = f"round {r}: unsat"
            final_status = "failed"
            break
        if sat["ambiguity"] == "unique":
            final_status, event_time = "unique", total
            break
    if final_status is None:
        last = records[-1]["sat"]
        final_status = "indeterminate" if last["ambiguity"] == "no_signal" else "censored"
    # Round totals the event time can take, given the solved rounds.
    solved = [rec for rec in records if rec.get("sat")]
    if any(rec["sat"]["ambiguity"] == "no_signal" for rec in solved):
        event_set = []
        for rec in solved:
            if rec["sat"]["ambiguity"] == "ambiguous":
                event_set = []
            elif rec["sat"]["ambiguity"] in ("no_signal", "unique"):
                event_set.append(rec["ledger_total"])
    else:
        event_set = [event_time] if event_time is not None else []
    return {"name": name, "final_status": final_status, "event_time": event_time, "event_set": event_set,
            "ledger_total": len(charged), "failure": failure, "rounds": records, "calls": calls.calls}


def run_seed(seed, k, n_initial, rounds, n_candidates, gen_llm, arms=("A", "B"), phases="default",
             sat_timeout=30, budget: Budget | None = None) -> dict:
    budget = budget if budget is not None else Budget(math.inf)
    gen = _BudgetedLLM(gen_llm, budget)
    before = budget.snapshot()
    dfa = random_dfa(k, seed)
    oracle = Oracle(dfa, seed)
    initial = oracle.initial_dataset(n_initial)
    out = {"seed": seed, "k": k, "phases": phases, "n_initial": n_initial, "rounds": rounds,
           "n_candidates": n_candidates, "arms": {}}
    for name in arms:
        out["arms"][name] = run_arm(name, seed, dfa, initial, k, rounds, n_candidates, gen, phases, sat_timeout,
                                    random.Random(seed), budget)
    after = budget.snapshot()
    out["usage"] = {key: after[key] - before[key] for key in after}
    return out


# -------------------------------------------------------------------- CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="wsp.query_study")
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="stand-in replay of the pilot's A rounds")
    rp.add_argument("--pilot", required=True)
    rp.add_argument("--k", type=int, default=6)
    rp.add_argument("--sat-timeout", type=float, default=30)
    rp.add_argument("--phases", choices=("default", "inverted"), default="default")
    rp.add_argument("--out", required=True)
    rn = sub.add_parser("run", help="run one roster, one JSON per seed")
    rn.add_argument("--roster", choices=ROSTERS, required=True)
    rn.add_argument("--n-main", type=int, default=20)
    rn.add_argument("--n-ext", type=int, default=10)
    rn.add_argument("--k", type=int, default=6)
    rn.add_argument("--n-initial", type=int, default=32)
    rn.add_argument("--rounds", type=int, default=3)
    rn.add_argument("--n-candidates", type=int, default=32)
    rn.add_argument("--backend", choices=("mock", "openrouter"), default="mock")
    rn.add_argument("--model", default="anthropic/claude-opus-5.5")
    rn.add_argument("--reasoning-tokens", type=int, default=16000)
    rn.add_argument("--max-output-tokens", type=int, default=32000)
    rn.add_argument("--sat-timeout", type=float, default=30)
    rn.add_argument("--spend-cap-usd", type=float, default=10.0)
    rn.add_argument("--out", required=True)
    return p.parse_args(argv)


def _git_commit() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _accrued_usd(path: Path) -> float:
    """Every persisted entry counts, aborted seeds included: a partial seed's spend is not refunded."""
    if not path.is_file():
        return 0.0
    return sum(json.loads(line)["usd"] for line in path.read_text().splitlines() if line.strip())


def _append_usage(path: Path, roster_name: str, seed: int, budget: Budget, before: dict,
                  aborted: bool = False) -> None:
    """Append the provider usage this seed accrued since `before`, which is the budget snapshot taken before it."""
    after = budget.snapshot()
    entry = {"roster": roster_name, "seed": seed, **{key: after[key] - before[key] for key in after},
             "aborted": aborted, "time": datetime.now(timezone.utc).isoformat()}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def cmd_replay(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = replay(args.pilot, args.k, args.sat_timeout, args.phases)
    (out / "replay.json").write_text(json.dumps(rep, indent=1))
    print(f"wsp.query_study: replay pass={rep['pass']} vacuous={rep['vacuous']} -> {out / 'replay.json'}")
    return 0


def cmd_run(args) -> int:
    import pysat

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.backend == "openrouter":  # fail fast on a missing key, exactly as wsp.run builds it
        client = OpenRouterClient(model=args.model, max_tokens=args.max_output_tokens,
                                  reasoning_max_tokens=args.reasoning_tokens)
    roster_path = out / "roster.json"
    if roster_path.is_file():
        ros = json.loads(roster_path.read_text())
    else:
        ros = roster(args.k, args.n_initial, args.n_main, args.n_ext, args.sat_timeout)
        roster_path.write_text(json.dumps(ros, indent=1))
    (out / "study_meta.json").write_text(json.dumps({
        "git_commit": _git_commit(), "python_sat_version": pysat.__version__,
        "config": {key: v for key, v in vars(args).items() if key not in ("cmd", "out")},
        "written": datetime.now(timezone.utc).isoformat()}, indent=1))
    if args.roster == "main":
        seeds, arms, phases = ros["main"], ("A", "B"), "default"
    elif args.roster == "extension":
        seeds, arms, phases = ros["extension"], ("A", "B"), "default"
    else:
        seeds, arms = ros["main"][:SENSITIVITY_SEEDS], ("A",)
        phases = "inverted" if args.roster == "sensitivity-inverted" else "default"
    prompt_rate, completion_rate = rates_for(args.backend, getattr(args, "model", None))
    usage_path = out / "usage.jsonl"
    budget = Budget(args.spend_cap_usd, prompt_rate, completion_rate, args.max_output_tokens,
                    spent_usd=_accrued_usd(usage_path))
    roster_dir = out / args.roster
    roster_dir.mkdir(exist_ok=True)
    for seed in seeds:
        path = roster_dir / f"seed_{seed}.json"
        if path.is_file():
            print(f"wsp.query_study: {path} exists, skipping")
            continue
        if args.backend == "mock":
            client = MockLLM(seed=seed * 100 + 11, h=16, n_candidates=args.n_candidates)
        before = budget.snapshot()
        try:
            result = run_seed(seed, args.k, args.n_initial, args.rounds, args.n_candidates, client, arms, phases,
                              args.sat_timeout, budget)
        except BudgetStop as e:
            # The discarded partial seed still cost money: persist it so a later invocation counts it.
            _append_usage(usage_path, args.roster, seed, budget, before, aborted=True)
            stop = {"roster": args.roster, "seed": seed, "spent_usd": budget.spent_usd,
                    "cap_usd": args.spend_cap_usd, "detail": str(e), "time": datetime.now(timezone.utc).isoformat()}
            (out / "stopped_at_cap.json").write_text(json.dumps(stop, indent=1))
            print(f"wsp.query_study: stopped at cap before seed {seed} ({e})")
            return 0
        path.write_text(json.dumps(result, indent=1))
        _append_usage(usage_path, args.roster, seed, budget, before)
        print(f"wsp.query_study: wrote {path} "
              + " ".join(f"{n}={a['final_status']}@{a['event_time']}" for n, a in result["arms"].items()))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return cmd_replay(args) if args.cmd == "replay" else cmd_run(args)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"wsp.query_study: failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
