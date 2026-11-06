"""Minimum consistent DFA by SAT (Heule & Verwer 2010, direct encoding, pysat).

Evaluation-side symbolic baseline: the smallest DFA consistent with a labeled
sample, plus an ambiguity re-solve (block the found DFA, look for a
non-equivalent one under the same state bound). A timeout is reported as
"timeout" / "no_signal", never as "unique".
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from pysat.solvers import Solver

from wsp.dfa import DFA, equivalent, minimize, reachable_states

SOLVER_NAME = "g4"  # Glucose4; supports solve_limited(expect_interrupt=True)

Sample = Tuple[str, int]


@dataclass(frozen=True)
class IdentifyResult:
    status: str  # "sat" | "unsat" | "timeout"
    dfa: Optional[DFA]
    n_states: int
    elapsed_s: float


# ---------------------------------------------------------------- APTA


def build_apta(samples: Sequence[Sample]) -> Tuple[List[List[Optional[int]]], List[Optional[int]]]:
    """Prefix tree acceptor. Returns (children, label): children[v][a] is the
    child node on symbol a or None; label[v] in {None, 0, 1}. Root is 0."""
    children: List[List[Optional[int]]] = [[None, None]]
    label: List[Optional[int]] = [None]
    for s, lab in samples:
        lab = int(lab)
        v = 0
        for ch in s:
            a = int(ch)
            if children[v][a] is None:
                children[v][a] = len(children)
                children.append([None, None])
                label.append(None)
            v = children[v][a]
        if label[v] is not None and label[v] != lab:
            raise ValueError(f"conflicting labels for string {s!r}")
        label[v] = lab
    return children, label


# ------------------------------------------------------------- encoding


class _Encoding:
    """Variable numbering and clause generation for a state bound n."""

    def __init__(self, samples: Sequence[Sample], n: int):
        if n < 1:
            raise ValueError("n_states must be >= 1")
        self.children, self.label = build_apta(samples)
        self.n = n
        self.n_nodes = len(self.children)
        nv = self.n_nodes
        self._x0 = 1
        self._y0 = self._x0 + nv * n
        self._z0 = self._y0 + 2 * n * n
        self.n_vars = self._z0 + n - 1

    def x(self, v: int, i: int) -> int:
        return self._x0 + v * self.n + i

    def y(self, a: int, i: int, j: int) -> int:
        return self._y0 + (a * self.n + i) * self.n + j

    def z(self, i: int) -> int:
        return self._z0 + i

    def clauses(self) -> List[List[int]]:
        n, cl = self.n, []
        rng = range(n)
        # every node has at least one color
        for v in range(self.n_nodes):
            cl.append([self.x(v, i) for i in rng])
        # accepting / rejecting nodes agree with z
        for v, lab in enumerate(self.label):
            if lab is None:
                continue
            for i in rng:
                cl.append([-self.x(v, i), self.z(i) if lab else -self.z(i)])
        for a in range(2):
            for i in rng:
                # at least one target
                cl.append([self.y(a, i, j) for j in rng])
                # determinism: at most one target
                for j in rng:
                    for k in range(j + 1, n):
                        cl.append([-self.y(a, i, j), -self.y(a, i, k)])
        # consistency with the APTA edges
        for v, ch in enumerate(self.children):
            for a in range(2):
                w = ch[a]
                if w is None:
                    continue
                for i in rng:
                    for j in rng:
                        cl.append([-self.x(v, i), -self.x(w, j), self.y(a, i, j)])
                        cl.append([-self.y(a, i, j), -self.x(v, i), self.x(w, j)])
        # symmetry breaking: root has color 0
        cl.append([self.x(0, 0)])
        return cl

    def decode(self, model: Sequence[int]) -> Tuple[List[List[int]], List[bool]]:
        true = {lit for lit in model if lit > 0}
        delta = [[0, 0] for _ in range(self.n)]
        for a in range(2):
            for i in range(self.n):
                for j in range(self.n):
                    if self.y(a, i, j) in true:
                        delta[i][a] = j
        acc = [self.z(i) in true for i in range(self.n)]
        return delta, acc

    def block(self, solver: Solver, delta: Sequence[Sequence[int]], acc: Sequence[bool],
              states: Sequence[int]) -> None:
        """Forbid any model that repeats this y/z assignment on `states`."""
        clause = []
        for i in states:
            for a in range(2):
                clause.append(-self.y(a, i, delta[i][a]))
            clause.append(-self.z(i) if acc[i] else self.z(i))
        solver.add_clause(clause)


PHASES = ("default", "inverted")


def _set_phases(solver: Solver, enc: _Encoding, phases: str) -> None:
    """"default" leaves Glucose's preference (negative) alone; "inverted" prefers every variable positive."""
    if phases == "inverted":
        solver.set_phases([+v for v in range(1, enc.n_vars + 1)])
    elif phases != "default":
        raise ValueError(f"phases must be one of {PHASES}, got {phases!r}")


def _solve(solver: Solver, deadline: float) -> Optional[bool]:
    """True/False, or None when the wall budget ran out."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    timer = threading.Timer(remaining, solver.interrupt)
    timer.start()
    try:
        res = solver.solve_limited(expect_interrupt=True)
    finally:
        timer.cancel()
    # The timer may fire late (or the solver may finish just after the
    # deadline without being interrupted): an answer delivered after the
    # wall budget elapsed is not a signal, so re-check the clock.
    if res is None or time.monotonic() >= deadline:
        return None
    return res


# ------------------------------------------------------------------ API


def identify(samples: Sequence[Sample], n_states: int, timeout_s: float = 30,
             phases: str = "default") -> IdentifyResult:
    t0 = time.monotonic()
    enc = _Encoding(samples, n_states)
    with Solver(name=SOLVER_NAME, bootstrap_with=enc.clauses()) as solver:
        _set_phases(solver, enc, phases)
        res = _solve(solver, t0 + timeout_s)
        if res is None:
            return IdentifyResult("timeout", None, n_states, time.monotonic() - t0)
        if res is False:
            return IdentifyResult("unsat", None, n_states, time.monotonic() - t0)
        delta, acc = enc.decode(solver.get_model())
    dfa = minimize(DFA(n_states, tuple(tuple(r) for r in delta), tuple(acc), 0))
    return IdentifyResult("sat", dfa, n_states, time.monotonic() - t0)


def identify_min(samples: Sequence[Sample], max_states: int, timeout_s: float = 30) -> IdentifyResult:
    last = None
    for n in range(1, max_states + 1):
        last = identify(samples, n, timeout_s)
        if last.status != "unsat":
            return last
    if last is None:
        raise ValueError("max_states must be >= 1")
    return last


def ambiguity(samples: Sequence[Sample], n_states: int, found: DFA, timeout_s: float = 30) -> str:
    """"unique" if no non-equivalent DFA with <= n_states states fits the sample,
    "ambiguous" if one exists, "no_signal" if the budget ran out."""
    if found.n_states > n_states:
        raise ValueError("found DFA has more states than the bound")
    deadline = time.monotonic() + timeout_s
    enc = _Encoding(samples, n_states)
    with Solver(name=SOLVER_NAME, bootstrap_with=enc.clauses()) as solver:
        enc.block(solver, found.delta, found.accepting, range(found.n_states))
        while True:
            res = _solve(solver, deadline)
            if res is None:
                return "no_signal"
            if res is False:
                return "unique"
            delta, acc = enc.decode(solver.get_model())
            cand = DFA(n_states, tuple(tuple(r) for r in delta), tuple(acc), 0)
            if not equivalent(minimize(cand), found):
                return "ambiguous"
            # same language: block its reachable part and keep looking
            enc.block(solver, delta, acc, reachable_states(cand))


def identifiability(samples: Sequence[Sample], n_states: int, timeout_s: float = 30,
                    phases: str = "default") -> Dict:
    """`phases` picks which consistent DFA `identify` decodes; the ambiguity answer does not depend on it."""
    t0 = time.monotonic()
    res = identify(samples, n_states, timeout_s, phases)
    if res.status == "sat":
        amb = ambiguity(samples, n_states, res.dfa, timeout_s)
    else:
        amb = "no_signal" if res.status == "timeout" else None
    return {
        "status": res.status,
        "n_states_found": res.dfa.n_states if res.dfa is not None else None,
        "ambiguity": amb,
        "dfa": res.dfa.to_dict() if res.dfa is not None else None,
        "elapsed_s": time.monotonic() - t0,
    }
