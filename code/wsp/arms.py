"""The three arms (A: LLM adversary, B: random, D: self-refinement) on the gradient-free path.

Module boundary: nothing here imports wsp.dfa, wsp.sat_dfa or wsp.evaluate; the
oracle is used only through `query(arm, strings)`.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wsp import prompts, rnn, sandbox
from wsp.d0 import load_d0
from wsp.llm import CallLog, LLMError, LLMRefusal, extract_fenced

PROBE_LIMIT = 64
STDERR_TAIL_LINES = 20


def _failure(res: sandbox.SandboxResult) -> tuple[str, str]:
    """(kind, reason) for a rejected constructor.

    Audit reasons verbatim; otherwise the stderr tail, which for a wall timeout already ends with
    the canonical `wall timeout after Ns` line. A return-code line is appended only when the
    process ran to completion (a timeout or a failed launch has none).
    """
    if not res.audit.ok:
        return "audit", "\n".join(res.audit.reasons)
    tail = "\n".join(res.stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
    if res.timed_out or res.returncode is None:
        return "sandbox", tail
    return "sandbox", f"{tail}\nreturn code: {res.returncode}"


def run_editor_attempts(calls: CallLog, editor_llm, previous_src: str, initial_dataset, h: int, feedback,
                        observed_strings, max_attempts: int = 3, timeout_s: int = 60):
    """Up to `max_attempts` editor calls; attempt n+1 re-sends the prompt plus why attempt n failed.

    Returns ((src, SandboxResult) of the last attempt that produced a response, or None when it was a
    refusal / LLM error, attempts). Stops at the first accepted constructor.
    """
    attempts: list[dict] = []
    result = None
    failure = None
    for n in range(1, max_attempts + 1):
        t0 = time.monotonic()
        user = prompts.editor_user(previous_src, initial_dataset, h, feedback, failure)
        rec = {"attempt": n, "src_lines": 0, "literal_matches": 0, "finish_reason": None}
        try:
            src = extract_fenced(calls.complete(editor_llm, prompts.editor_system(), user), "python")
        except LLMError as e:
            kind, reason = ("refusal" if isinstance(e, LLMRefusal) else "llm_error"), str(e)
            rec["finish_reason"] = e.finish_reason
            result = None
        else:
            res = sandbox.run_constructor(src, h, timeout_s=timeout_s, observed_strings=observed_strings)
            result = (src, res)
            kind, reason = ("ok", "") if res.weights is not None else _failure(res)
            rec.update(src_lines=len(src.splitlines()), literal_matches=len(res.audit.literal_matches))
        rec.update(outcome=kind, reason=reason[:300], elapsed_s=time.monotonic() - t0)
        attempts.append(rec)
        if kind == "ok":
            break
        failure = prompts.failure_block(kind, reason)
    return result, attempts


@dataclass
class ArmState:
    name: str
    weights: dict
    constructor_src: str
    round: int = 0
    history: list[dict] = field(default_factory=list)


def length_histogram(strings) -> dict[int, int]:
    return dict(sorted(Counter(len(s) for s in strings).items()))


def target_histogram(initial_dataset_strings, n: int) -> dict[int, int]:
    """Initial length distribution scaled to n (largest-remainder rounding); zero buckets dropped."""
    hist = length_histogram(initial_dataset_strings)
    total = sum(hist.values())
    raw = {L: c * n / total for L, c in hist.items()}
    out = {L: math.floor(v) for L, v in raw.items()}
    short = n - sum(out.values())
    for L in sorted(raw, key=lambda L: (-(raw[L] - out[L]), L))[:short]:
        out[L] += 1
    return {L: c for L, c in out.items() if c > 0}


def parse_strings(text: str) -> list[str]:
    """Binary strings from the ```strings fence, one per line, deduped in order."""
    out: list[str] = []
    for line in extract_fenced(text, "strings").splitlines():
        s = line.strip()
        if s and set(s) <= {"0", "1"} and s not in out:
            out.append(s)
    return out


ENUMERATE_BELOW = 1 << 12  # bucket sizes up to this are enumerated and sampled without replacement


def _uniform_fill(rng: random.Random, length: int, count: int, avoid: set) -> list[str]:
    """`count` distinct uniform random strings of this length, none in `avoid`; adds them to `avoid`.

    Raises ValueError if the bucket does not have `count` unused strings left, so a caller can
    never receive a duplicate or a string it has already used.
    """
    space = 1 << length
    free = space - sum(1 for s in avoid if len(s) == length)
    if count > free:
        raise ValueError(f"length {length}: {count} strings requested but only {free} unused of {space}")
    if space <= ENUMERATE_BELOW:
        pool = ["".join(bits) for bits in itertools.product("01", repeat=length)]
        out = rng.sample([s for s in pool if s not in avoid], count)
    else:  # collisions are rare here (|avoid| << 2**length); retry until unused
        out = []
        while len(out) < count:
            s = "".join(rng.choice("01") for _ in range(length))
            if s not in avoid and s not in out:
                out.append(s)
    avoid.update(out)
    return out


def counterexamples_per_query_by_length(record: dict) -> dict[int, float]:
    queried = Counter(len(s) for s in record["queried"])
    ce = Counter(len(s) for s, _label, _pred in record["counterexamples"])
    return {L: ce[L] / q for L, q in sorted(queried.items())}


class Arm:
    def __init__(self, name, oracle, editor_llm, generator_llm, d0_path, initial_dataset,
                 n_candidates, h, rng: random.Random, log_dir, d0_src: str | None = None,
                 max_attempts: int = 3, constructor_timeout_s: int = 60):
        self.name = name
        self.oracle = oracle
        self.editor_llm = editor_llm
        self.generator_llm = generator_llm
        self.initial_dataset = list(initial_dataset)
        self.n_candidates = n_candidates
        self.h = h
        self.max_attempts = max_attempts
        self.constructor_timeout_s = constructor_timeout_s
        self.rng = rng
        self.log_dir = Path(log_dir)
        self.calls = CallLog()
        self.seen: list[str] = []  # every string this arm has queried, in order
        # Round-0 weights always come from the file; the fixture D0 is mock seed 1.
        self.state = ArmState(name, load_d0(d0_path), d0_src or sandbox.mock_constructor_source(h, 1))

    @property
    def weights(self) -> dict:
        return self.state.weights

    @property
    def initial_strings(self) -> list[str]:
        return [s for s, _ in self.initial_dataset]

    def _gather(self, record: dict):
        """Arm-specific: fill record['queried'/'labels'/'counterexamples'], return editor feedback."""
        return None

    def _label(self, record: dict, candidates: list[str]) -> list[tuple[str, int, int]]:
        labels = self.oracle.query(self.name, candidates)  # charged
        preds = rnn.predict_many(self.weights, candidates)
        feedback = [(s, y, p) for s, y, p in zip(candidates, labels, preds) if y != p]
        record.update(queried=list(candidates), labels=list(labels),
                      counterexamples=[list(t) for t in feedback])
        self.seen.extend(candidates)
        return feedback

    def editor_attempts(self, previous_src: str, feedback):
        """The round's editor calls (oracle queries already made): see run_editor_attempts."""
        return run_editor_attempts(self.calls, self.editor_llm, previous_src, self.initial_dataset, self.h,
                                   feedback, self.initial_strings + self.seen, self.max_attempts,
                                   self.constructor_timeout_s)

    def run_round(self) -> dict:
        r = self.state.round + 1
        round_dir = self.log_dir / self.name / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        record: dict = {"round": r, "queried": [], "labels": [], "counterexamples": [],
                        "n_filled_random": 0, "failed": False, "editor_attempts": [], "n_attempts": 0}
        try:
            feedback = self._gather(record)
        except LLMError as e:
            # The generator failed: no editor attempt was made, and only queries already made
            # (none, if the generator refused) were charged.
            kind = "refusal" if isinstance(e, LLMRefusal) else "llm_error"
            record.update(failed=True, error_kind=kind, final_outcome=kind, finish_reason=e.finish_reason,
                          detail=str(e)[:300], n_queried=len(record["queried"]), accepted=False,
                          constructor_src=self.state.constructor_src)
        else:
            result, attempts = self.editor_attempts(self.state.constructor_src, feedback)
            last = attempts[-1]
            record.update(editor_attempts=attempts, n_attempts=len(attempts), final_outcome=last["outcome"],
                          n_queried=len(record["queried"]))
            if result is None:  # the last attempt got no usable response: weights and constructor stay
                record.update(failed=True, error_kind=last["outcome"], finish_reason=last["finish_reason"],
                              detail=last["reason"], accepted=False, constructor_src=self.state.constructor_src)
            else:
                src, res = result
                accepted = res.weights is not None
                if accepted:
                    self.state.weights = res.weights
                    self.state.constructor_src = src
                (round_dir / "constructor.py").write_text(src)
                record.update(
                    constructor_src=src, accepted=accepted,
                    audit={"ok": res.audit.ok, "reasons": res.audit.reasons,
                           "literal_matches": res.audit.literal_matches,
                           "literal_match_fraction": res.audit.literal_match_fraction},
                    sandbox={"ok": accepted, "returncode": res.returncode, "timed_out": res.timed_out,
                             "elapsed_s": res.elapsed_s, "confined": res.confined, "stderr": res.stderr[-2000:]})
        weights_path = round_dir / "weights.npz"
        np.savez(weights_path, **self.state.weights)
        # Relative to the run directory, so records do not depend on where the run was written.
        record["weights_path"] = weights_path.relative_to(self.log_dir).as_posix()
        with open(round_dir / "record.json", "w") as f:
            json.dump(record, f, indent=1)
        self.calls.dump(self.log_dir / self.name / "calls.json")
        self.state.round = r
        self.state.history.append(record)
        return record


class ArmA(Arm):
    """LLM adversary: probes D (uncharged), proposes N candidates matching the target histogram."""

    def _gather(self, record):
        target = target_histogram(self.initial_strings, self.n_candidates)
        gen_sys = prompts.generator_system()
        probe_text = self.calls.complete(self.generator_llm, gen_sys, prompts.generator_user(
            self.initial_dataset, [], target_histogram(self.initial_strings, PROBE_LIMIT)))
        probes = parse_strings(probe_text)[:PROBE_LIMIT]
        probe_preds = list(zip(probes, rnn.predict_many(self.weights, probes)))
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

        take(parse_strings(self.calls.complete(
            self.generator_llm, gen_sys, prompts.generator_user(self.initial_dataset, probe_preds, target))))
        if missing():  # re-prompt once for the missing buckets
            take(parse_strings(self.calls.complete(
                self.generator_llm, gen_sys, prompts.generator_user(self.initial_dataset, probe_preds, missing()))))
        n_filled = 0
        for L, k in missing().items():
            chosen[L].extend(_uniform_fill(self.rng, L, k, taken))
            n_filled += k
        record.update(n_filled_random=n_filled, probe_predictions=[list(t) for t in probe_preds])
        return self._label(record, [s for L in target for s in chosen[L]])


class ArmB(Arm):
    """Random: N uniform random strings with the target histogram."""

    def _gather(self, record):
        target = target_histogram(self.initial_strings, self.n_candidates)
        taken: set[str] = set()
        return self._label(record, [s for L, k in target.items() for s in _uniform_fill(self.rng, L, k, taken)])


class ArmD(Arm):
    """Self-refinement: no oracle queries, no feedback block."""
