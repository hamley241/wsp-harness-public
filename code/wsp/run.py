"""CLI: the end-to-end run. `uv run python -m wsp.run --mock --rounds 3`."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wsp import evaluate, prompts, rnn, sandbox
from wsp.arms import ArmA, ArmB, ArmD, counterexamples_per_query_by_length, run_editor_attempts
from wsp.d0 import load_d0, make_d0, save_d0
from wsp.dfa import parity_dfa, random_dfa, random_string
from wsp.llm import AnthropicClient, CallLog, MockLLM, OpenRouterClient, extract_fenced
from wsp.oracle import Oracle

FIXTURE = Path(__file__).parent / "fixtures" / "d0_h16_seed1.npz"
ARMS = (("A", ArmA), ("B", ArmB), ("D", ArmD))
NO_PREVIOUS = "# (no previous constructor: this is the first one)\n"
PARITY_TRAIN, PARITY_TEST = 32, 200
DEFAULT_MODEL = {"mock": None, "anthropic": "claude-fable-5-1", "openrouter": "anthropic/claude-fable-5.1"}
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="wsp.run", description="Counterexample-guided weight-space repair: one run.")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--backend", choices=("mock", "anthropic", "openrouter"), default="mock")
    p.add_argument("--mock", action="store_true", help="alias for --backend mock (offline MockLLM)")
    p.add_argument("--model", default=None, help="per-backend default: see DEFAULT_MODEL")
    p.add_argument("--reasoning-tokens", type=int, default=16000, help="openrouter only; 0 disables")
    p.add_argument("--max-output-tokens", type=int, default=20000)
    p.add_argument("--out", default=None, help="output directory (default runs/<timestamp>)")
    p.add_argument("--d0-from", default=None, metavar="DIR",
                   help="reuse DIR/d0.npz and DIR/d0_constructor.py from a previous run (same k, seed, hidden)")
    p.add_argument("--n-initial", type=int, default=96)
    p.add_argument("--n-candidates", type=int, default=32, help="per round; one value for A and B")
    p.add_argument("--n-heldout", type=int, default=500)
    p.add_argument("--n-long", type=int, default=200)
    p.add_argument("--sat-timeout", type=float, default=30)
    p.add_argument("--constructor-timeout", type=int, default=60,
                   help="seconds per constructor run in the sandbox (wall and CPU), arms and D0")
    p.add_argument("--max-attempts", type=int, default=3, help="editor attempts per round (and for D0)")
    p.add_argument("--train-steps", type=int, default=2000)
    args = p.parse_args(argv)
    if args.mock:
        args.backend = "mock"
    args.mock = args.backend == "mock"
    if args.model is None:
        args.model = DEFAULT_MODEL[args.backend]
    return args


def make_clients(args) -> dict:
    """Every LLM client the run needs, keyed by arm_or_role label. Resolves the OpenRouter key up front."""
    seed = args.seed
    labels = [("d0", 0)] + [(f"{name}/{role}", i + off) for i, (name, _) in enumerate(ARMS)
                            for role, off in (("editor", 1), ("generator", 11))] + [("parity", 30)]

    def build(i: int):
        if args.backend == "mock":
            return MockLLM(seed=seed * 100 + i, h=args.hidden, n_candidates=args.n_candidates)
        if args.backend == "anthropic":
            return AnthropicClient(model=args.model, max_tokens=args.max_output_tokens)
        return OpenRouterClient(model=args.model, max_tokens=args.max_output_tokens,
                                reasoning_max_tokens=args.reasoning_tokens)

    return {label: build(i) for label, i in labels}


def llm_usage(clients: dict) -> dict:
    calls = [{"arm_or_role": label, **u} for label, c in clients.items() for u in getattr(c, "usage_log", [])]
    return {"calls": calls, "totals": {f: sum(u[f] for u in calls) for f in USAGE_FIELDS}}


ATTEMPT_KEYS = {"refusal_attempts": "refusal", "error_attempts": "llm_error",
                "audit_attempts": "audit", "sandbox_attempts": "sandbox"}
FAILURE_KEYS = [*ATTEMPT_KEYS, "failed_rounds"]


def _attempt_counts(attempts) -> dict:
    kinds = Counter(a["outcome"] for a in attempts)
    return {key: kinds[kind] for key, kind in ATTEMPT_KEYS.items()}


def llm_failures(arms: dict, d0_attempts: list | None = None) -> dict:
    """Failed editor attempts by outcome (every attempt, retried or not) and rounds that ended unaccepted.

    `refusals` / `errors` are aliases of the attempt totals (D0 attempts included).
    """
    by_arm = {}
    for name, arm in arms.items():
        hist = arm.state.history
        by_arm[name] = {**_attempt_counts(a for rec in hist for a in rec["editor_attempts"]),
                        "failed_rounds": sum(1 for rec in hist if not rec["accepted"])}
    d0 = _attempt_counts(d0_attempts or [])
    total = {key: sum(v[key] for v in by_arm.values()) + d0.get(key, 0) for key in FAILURE_KEYS}
    return {**total, "refusals": total["refusal_attempts"], "errors": total["error_attempts"],
            "by_arm": by_arm, "d0": d0}


def write_mock_d0(h: int, seed: int, path) -> str:
    """The fixture when (h, seed) == (16, 1), else make_d0; returns the constructor source."""
    if (h, seed) == (16, 1):
        shutil.copyfile(FIXTURE, path)
    else:
        save_d0(path, make_d0(h, seed))
    return sandbox.mock_constructor_source(h, seed)


def write_llm_d0(editor, initial, h, path, out, max_attempts: int = 3, timeout_s: int = 60) -> str:
    """Round-0 editor call with no previous constructor and no feedback, through the arms' attempt loop.

    Every attempt is listed in d0_attempts.json, which is the only file written under `out`
    when every attempt fails.
    """
    calls = CallLog()
    result, attempts = run_editor_attempts(calls, editor, NO_PREVIOUS, initial, h, None,
                                           [s for s, _ in initial], max_attempts, timeout_s)
    with open(out / "d0_attempts.json", "w") as f:
        json.dump(attempts, f, indent=1)
    last = attempts[-1]
    if last["outcome"] != "ok":
        raise RuntimeError(f"D0 editor call failed after {len(attempts)} attempts "
                           f"(final outcome: {last['outcome']}): {last['reason']}")
    src, res = result
    calls.dump(out / "d0_calls.json")
    save_d0(path, res.weights)
    return src


def check_reused_d0(src_dir: Path, h: int, k: int, seed: int) -> None:
    """The reused D0 must come from the same hidden DFA (k, seed) and have hidden size h."""
    results_path = src_dir / "results.json"
    if not results_path.is_file():
        raise RuntimeError(f"--d0-from: {results_path} not found")
    cfg = json.loads(results_path.read_text())["config"]
    if (cfg["k"], cfg["seed"]) != (k, seed):
        raise RuntimeError(f"--d0-from: source run has k={cfg['k']} seed={cfg['seed']}, this run has k={k} seed={seed}")
    src_h = load_d0(src_dir / "d0.npz")["Wh"].shape[0]
    if src_h != h:
        raise RuntimeError(f"--d0-from: source D0 has hidden size {src_h}, this run has --hidden {h}")


def parity_check(editor, h, seed, out, sets, timeout_s: int = 60) -> dict:
    """One-shot synthesis sanity check on the parity DFA (not an arm)."""
    rng, pd = random.Random(seed), parity_dfa()
    train = [(s, pd.label(s)) for s in (random_string(rng, rng.randint(1, 12)) for _ in range(PARITY_TRAIN))]
    test = [(s, pd.label(s)) for s in (random_string(rng, rng.randint(1, 12)) for _ in range(PARITY_TEST))]
    calls = CallLog()
    user = prompts.editor_user(NO_PREVIOUS, train, h, None)
    src = extract_fenced(calls.complete(editor, prompts.editor_system(), user), "python")
    calls.dump(out / "parity_calls.json")
    (out / "parity_constructor.py").write_text(src)
    res = sandbox.run_constructor(src, h, timeout_s=timeout_s, observed_strings=[s for s, _ in train])
    return {"ran": True, "accepted": res.weights is not None,
            "accuracy": evaluate.accuracy(res.weights, test) if res.weights is not None else None,
            "n_train": len(train), "n_test": len(test), "noise_floor": evaluate.noise_floor(len(test))}


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _table(header, rows) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows), ""]


def write_markdown(results: dict, path) -> None:
    cfg = results["config"]
    nf = results["noise_floor"]
    lines = [f"# wsp run: seed {cfg['seed']}, k={cfg['k']}, h={cfg['hidden']}, rounds={cfg['rounds']}", ""]
    lines += ["## Config", ""] + _table(["flag", "value"], [(k, v) for k, v in cfg.items()])
    lines += ["## Hidden sets and noise floor (SE at p = 0.5)", ""] + _table(
        ["set", "n", "noise floor"],
        [(name, cfg[f"n_{name}"], nf[name]) for name in ("initial", "heldout", "long")])
    ds = results["d0_source"]
    records = [e["record"] for rounds in results["arms"].values() for e in rounds if e["record"] is not None]
    lines += [f"d0_shared: {_fmt(results['d0_shared'])}",
              f"d0_source: kind={ds['kind']} from={_fmt(ds['from'])} sha256={ds['sha256']}",
              f"editor attempts: {sum(rec['n_attempts'] for rec in records)} / "
              f"failed rounds: {sum(1 for rec in records if not rec['accepted'])}",
              f"hidden DFA states: {results['hidden_dfa']['n_states']}",
              f"held-out / long / SAT held-out / SAT long noise floor: {_fmt(nf['heldout'])} / {_fmt(nf['long'])}"
              f" / {_fmt(nf['heldout'])} / {_fmt(nf['long'])}", ""]
    rows = []
    for arm, rounds in results["arms"].items():
        for e in rounds:
            sat, es, rec = e["sat"], e["edit_stats"], e["record"]
            last = rec["editor_attempts"][-1] if rec and rec["editor_attempts"] else None
            rows.append((arm, e["round"], e["heldout"], e["long"], e["reported_failures"], e["reported_failures_n"],
                         e["reported_failures_noise_floor"], e["initial_regressions"], e["initial_correct_d0_n"],
                         e["initial_regressions_noise_floor"], e["ledger_total"], e["n_counterexamples"],
                         e["accepted"], rec["n_attempts"] if rec else None,
                         last["src_lines"] if last else None, last["literal_matches"] if last else None,
                         sat["status"], sat["n_states_found"], sat["ambiguity"], sat["heldout"],
                         sat["long"], es["frac_changed"] if es else None, es["max_abs"] if es else None))
    lines += ["## Arms, per round", ""] + _table(
        ["arm", "round", "held-out", "long", "reported failures", "n failures", "failures noise floor",
         "regressions", "n D0-correct", "regressions noise floor", "ledger total", "n counterexamples", "accepted",
         "attempts", "src lines", "literal matches",
         "SAT status", "SAT states", "ambiguity", "SAT held-out", "SAT long", "frac weights changed",
         "max |delta|"], rows)
    tb, pc = results["trained_baseline"], results["parity_check"]
    final = {arm: rounds[-1] for arm, rounds in results["arms"].items()}
    first = next(iter(final.values()))
    lengths = list(first["per_length"])
    lines += ["## Per-length held-out accuracy (final round; n and noise floor per length)", ""] + _table(
        ["length", "n", "noise floor", *(f"{arm} held-out" for arm in final), "trained RNN"],
        [(L, first["per_length_n"][L], first["per_length_noise_floor"][L],
          *(final[arm]["per_length"][L] for arm in final), tb["per_length"][L]) for L in lengths])
    lines += ["## Baselines", ""] + _table(
        ["baseline", "held-out", "held-out noise floor", "long", "long noise floor", "train acc", "n train",
         "train noise floor", "steps"],
        [(f"trained RNN h={tb['h']}", tb["heldout"], tb["heldout_noise_floor"], tb["long"], tb["long_noise_floor"],
          tb["train_acc"], tb["train_acc_n"], tb["train_acc_noise_floor"], tb["steps"])])
    lines += _table(["parity one-shot", "ran", "accepted", "accuracy", "n test", "noise floor"],
                    [("", pc["ran"], pc["accepted"], pc["accuracy"], pc["n_test"], pc["noise_floor"])])
    keys = list(results["cross_play"])
    sizes = results["cross_play_sets"]
    lines += ["## Cross-play (row: weights arm/round; column: counterexample set arm/round)", ""] + _table(
        ["weights \\ set", *keys],
        [("n", *(sizes["n"][ck] for ck in keys)), ("noise floor", *(sizes["noise_floor"][ck] for ck in keys)),
         *((wk, *(results["cross_play"][wk][ck] for ck in keys)) for wk in keys)])
    lines += ["## Oracle ledger", ""] + _table(
        ["arm", "total", "by length"],
        [(arm, e["total"], json.dumps(e["by_length"])) for arm, e in results["oracle_ledger"].items()])
    usage, lf = results["llm_usage"], results["llm_failures"]
    lines += ["## LLM usage", ""] + _table(["calls", *USAGE_FIELDS],
                                           [(len(usage["calls"]), *(usage["totals"][f] for f in USAGE_FIELDS))])
    arms_ = list(lf["by_arm"])
    lines += _table(["failures", *arms_, "D0", "total"],
                    [(key, *(lf["by_arm"][a][key] for a in arms_), lf["d0"].get(key, 0), lf[key])
                     for key in FAILURE_KEYS])
    lines += ["## Timing", ""] + _table(["field", "seconds"], [("total_s", results["timing"]["total_s"]),
                                                               *((f"round_{i + 1}_s", s) for i, s in
                                                                 enumerate(results["timing"]["rounds_s"]))])
    Path(path).write_text("\n".join(lines))


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def run(args) -> dict:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    clients = make_clients(args)  # fail fast on a missing key: nothing drawn or written yet
    if args.d0_from is not None:
        check_reused_d0(Path(args.d0_from), args.hidden, args.k, args.seed)
    out = Path(args.out or f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    args.out = str(out)
    seed, h, k = args.seed, args.hidden, args.k
    random.seed(seed)
    np.random.seed(seed)

    dfa = random_dfa(k, seed)
    oracle = Oracle(dfa, seed)
    initial = oracle.initial_dataset(args.n_initial)
    heldout = oracle.heldout(args.n_heldout)
    long_ = oracle.long_set(args.n_long)
    hidden = {"heldout": heldout, "long": long_}

    d0_path = out / "d0.npz"
    if args.d0_from is not None:
        shutil.copyfile(Path(args.d0_from) / "d0.npz", d0_path)
        d0_src = (Path(args.d0_from) / "d0_constructor.py").read_text()
        d0_kind, d0_origin = "reused", "reused"
    elif args.mock:
        d0_src = write_mock_d0(h, seed, d0_path)
        d0_kind, d0_origin = "mock", ("fixture" if (h, seed) == (16, 1) else "make_d0")
    else:
        d0_src = write_llm_d0(clients["d0"], initial, h, d0_path, out, args.max_attempts, args.constructor_timeout)
        d0_kind, d0_origin = "editor", "editor_round0"
    (out / "d0_constructor.py").write_text(d0_src)
    d0 = load_d0(d0_path)
    # d0_source is the provenance record; d0.source keeps the pre-existing per-run origin label.
    d0_source = {"kind": d0_kind, "from": args.d0_from,
                 "sha256": hashlib.sha256(d0_path.read_bytes()).hexdigest()}

    arms = {name: cls(name, oracle, clients[f"{name}/editor"], clients[f"{name}/generator"], d0_path, initial,
                      args.n_candidates, h, random.Random(seed * 100 + 20 + i), out, d0_src,
                      args.max_attempts, args.constructor_timeout)
            for i, (name, cls) in enumerate(ARMS)}
    d0_shared = all(rnn.weights_equal(arm.weights, d0) for arm in arms.values())
    if not d0_shared:
        raise RuntimeError("arms do not share D0 at round 0")

    # During the rounds the SAT baseline sees only the arm's own labels; the hidden sets
    # are scored at the end, together with everything else.
    sat0 = evaluate.sat_baseline(initial, k, args.sat_timeout)
    weights_hist = {name: [arm.weights] for name, arm in arms.items()}
    labels = {name: list(initial) for name in arms}
    ce_sets = {f"{name}/0": [] for name in arms}
    entries = {name: [{"round": 0, "ledger_total": 0, "ledger_by_length": {}, "sat": sat0, "edit_stats": None,
                       "accepted": None, "n_queried": 0, "n_counterexamples": 0, "n_filled_random": 0,
                       "counterexamples_per_query_by_length": {}, "record": None}] for name in arms}
    rounds_s = []
    for r in range(1, args.rounds + 1):
        t0 = time.monotonic()
        for name, arm in arms.items():
            prev = arm.weights
            rec = arm.run_round()
            labels[name] += list(zip(rec["queried"], rec["labels"]))
            ces = [(s, y) for s, y, _p in rec["counterexamples"]]
            ce_sets[f"{name}/{r}"] = ces
            weights_hist[name].append(arm.weights)
            sat = (evaluate.sat_baseline(labels[name], k, args.sat_timeout) if rec["queried"]
                   else entries[name][-1]["sat"])
            ledger = oracle.ledger().get(name, {"total": 0, "by_length": {}})
            entries[name].append({
                "round": r, "ledger_total": ledger["total"], "ledger_by_length": ledger["by_length"], "sat": sat,
                "edit_stats": evaluate.edit_stats(prev, arm.weights), "accepted": rec["accepted"],
                "n_queried": len(rec["queried"]), "n_counterexamples": len(ces),
                "n_filled_random": rec["n_filled_random"],
                "counterexamples_per_query_by_length": counterexamples_per_query_by_length(rec) if rec["queried"] else {},
                "record": {k_: v for k_, v in rec.items() if k_ != "constructor_src"}})
        rounds_s.append(time.monotonic() - t0)

    # ---- evaluation: only now -------------------------------------------
    d0_correct = evaluate.correct_subset(d0, initial)
    arms_out = {}
    for name in arms:
        shown: list = []
        rounds_out = []
        for r, entry in enumerate(entries[name]):
            shown = shown + ce_sets[f"{name}/{r}"]
            sets = {**hidden, "reported_failures": shown, "initial_correct_d0": d0_correct}
            rounds_out.append({**evaluate.evaluate_arm(weights_hist[name][r], sets), **entry,
                               "sat": evaluate.sat_hidden_scores(entry["sat"], hidden)})
        arms_out[name] = rounds_out
    weights_by = {f"{name}/{r}": w for name in arms for r, w in enumerate(weights_hist[name])}
    cross_sets = {key: ce_sets[key] for key in weights_by}
    cross = evaluate.cross_play(weights_by, cross_sets)
    trained = evaluate.trained_baseline_eval(initial, h, seed, args.train_steps, hidden)
    parity = parity_check(clients["parity"], h, seed, out, hidden, args.constructor_timeout)

    results = {
        # The output directory is not part of the payload: same-seed runs written to different
        # directories must produce identical results.json (modulo timing).
        "config": {key: v for key, v in vars(args).items() if key != "out"},
        "d0_shared": d0_shared,
        "d0": {"path": "d0.npz", "source": d0_origin, "constructor": "d0_constructor.py"},
        "d0_source": d0_source,
        "noise_floor": {"initial": evaluate.noise_floor(len(initial)), "heldout": evaluate.noise_floor(len(heldout)),
                        "long": evaluate.noise_floor(len(long_))},
        "arms": arms_out,
        "cross_play": cross,
        "cross_play_sets": evaluate.cross_play_sizes(cross_sets),
        "trained_baseline": trained,
        "parity_check": parity,
        "oracle_ledger": oracle.ledger(),
        "llm_usage": llm_usage(clients),
        "llm_failures": llm_failures(arms, json.loads((out / "d0_attempts.json").read_text())
                                     if (out / "d0_attempts.json").is_file() else None),
        "hidden_dfa": dfa.to_dict(),
        "initial_dataset": initial,
        "heldout": heldout,
        "long": long_,
        "timing": {"started": started, "finished": datetime.now(timezone.utc).isoformat(),
                   "total_s": time.monotonic() - t_start, "rounds_s": rounds_s},
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=1, default=_json_default)
    write_markdown(results, out / "results.md")
    return results


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        results = run(args)
    except Exception as e:
        traceback.print_exc()
        print(f"wsp.run: failed: {e}", file=sys.stderr)
        return 1
    print(f"wsp.run: wrote {args.out}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
