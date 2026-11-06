"""Deterministic finite automata over a binary alphabet.

Pure, side-effect free. Used by three parties that must stay separated at
runtime: the Oracle (hidden target, charges queries), the SAT identifier
(hypotheses consistent with observed labels), and the evaluation-only
equivalence checker. Isolation is enforced by who holds a DFA object.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

ALPHABET = ("0", "1")


@dataclass(frozen=True)
class DFA:
    n_states: int
    delta: Tuple[Tuple[int, ...], ...]  # delta[state][symbol] -> state
    accepting: Tuple[bool, ...]
    start: int = 0

    def accepts(self, s: str) -> bool:
        q = self.start
        for ch in s:
            q = self.delta[q][int(ch)]
        return self.accepting[q]

    def label(self, s: str) -> int:
        return int(self.accepts(s))

    def to_dict(self) -> dict:
        return {"n_states": self.n_states, "delta": [list(r) for r in self.delta],
                "accepting": list(self.accepting), "start": self.start}

    @staticmethod
    def from_dict(d: dict) -> "DFA":
        return DFA(d["n_states"], tuple(tuple(r) for r in d["delta"]),
                   tuple(bool(a) for a in d["accepting"]), d.get("start", 0))


def reachable_states(d: DFA) -> List[int]:
    seen = {d.start}
    dq = deque([d.start])
    while dq:
        q = dq.popleft()
        for a in range(2):
            r = d.delta[q][a]
            if r not in seen:
                seen.add(r)
                dq.append(r)
    return sorted(seen)


def trim(d: DFA) -> DFA:
    reach = reachable_states(d)
    order = [d.start] + [q for q in reach if q != d.start]
    idx = {q: i for i, q in enumerate(order)}
    delta = tuple(tuple(idx[d.delta[q][a]] for a in range(2)) for q in order)
    return DFA(len(order), delta, tuple(d.accepting[q] for q in order), 0)


def minimize(d: DFA) -> DFA:
    """Moore partition refinement on the trimmed automaton."""
    d = trim(d)
    n = d.n_states
    block = [int(d.accepting[q]) for q in range(n)]
    while True:
        sig, new_block = {}, [0] * n
        for q in range(n):
            key = (block[q], block[d.delta[q][0]], block[d.delta[q][1]])
            sig.setdefault(key, len(sig))
            new_block[q] = sig[key]
        if len(sig) == len(set(block)):
            block = new_block
            break
        block = new_block
    k = len(set(block))
    order = [block[d.start]] + [b for b in range(k) if b != block[d.start]]
    remap = {b: i for i, b in enumerate(order)}
    delta, acc = [None] * k, [False] * k
    for q in range(n):
        b = remap[block[q]]
        delta[b] = tuple(remap[block[d.delta[q][a]]] for a in range(2))
        acc[b] = d.accepting[q]
    return DFA(k, tuple(delta), tuple(acc), 0)


def distinguishing_string(a: DFA, b: DFA) -> Optional[str]:
    """Shortest string on which a and b disagree; None if equivalent.
    EVALUATION-ONLY: this output must never reach an experimental arm."""
    if a.accepting[a.start] != b.accepting[b.start]:
        return ""
    seen = {(a.start, b.start)}
    dq = deque([((a.start, b.start), "")])
    while dq:
        (p, q), s = dq.popleft()
        for sym in range(2):
            p2, q2 = a.delta[p][sym], b.delta[q][sym]
            if (p2, q2) in seen:
                continue
            s2 = s + str(sym)
            if a.accepting[p2] != b.accepting[q2]:
                return s2
            seen.add((p2, q2))
            dq.append(((p2, q2), s2))
    return None


def equivalent(a: DFA, b: DFA) -> bool:
    return distinguishing_string(a, b) is None


def random_string(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def acceptance_rate(d: DFA, rng: random.Random, lengths: Iterable[int], n: int = 2000) -> float:
    lengths = list(lengths)
    return sum(d.accepts(random_string(rng, rng.choice(lengths))) for _ in range(n)) / n


def random_dfa(k: int, seed: int, lengths: Iterable[int] = range(1, 13),
               rate_bounds: Tuple[float, float] = (0.3, 0.7), max_tries: int = 20000) -> DFA:
    """Random MINIMAL k-state DFA with acceptance rate in rate_bounds on uniform
    strings of the given lengths. Construction-time sampling: not charged, and
    the strings are discarded."""
    rng = random.Random(seed)
    lengths = list(lengths)
    for _ in range(max_tries):
        delta = tuple(tuple(rng.randrange(k) for _ in range(2)) for _ in range(k))
        acc = tuple(rng.random() < 0.5 for _ in range(k))
        if not any(acc) or all(acc):
            continue
        m = minimize(DFA(k, delta, acc, 0))
        if m.n_states != k:
            continue
        if rate_bounds[0] <= acceptance_rate(m, rng, lengths) <= rate_bounds[1]:
            return m
    raise RuntimeError(f"no {k}-state DFA met rate bounds in {max_tries} tries")


def parity_dfa() -> DFA:
    """Accept iff the number of 1s is odd."""
    return DFA(2, ((0, 1), (1, 0)), (False, True), 0)
