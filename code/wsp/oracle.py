"""The Oracle: the only runtime holder of the hidden DFA.

Construction-time sets (initial dataset, held-out, long) are drawn by EXACT
sampling and are not charged. Every `query` is charged to the calling arm's
ledger, repeats and all, counted per string length.
"""
from __future__ import annotations

import json
import random
from collections import Counter

from wsp.dfa import DFA

SHORT_LENGTHS = range(1, 13)
LONG_LENGTHS = range(13, 25)
MAX_LEN = LONG_LENGTHS.stop - 1


class Oracle:
    def __init__(self, dfa: DFA, seed: int):
        self._dfa = dfa
        self._rng = random.Random(seed)
        # _acc[r][q] = number of strings of length r from state q that end accepting.
        acc = [[int(a) for a in dfa.accepting]]
        for r in range(1, MAX_LEN + 1):
            prev = acc[r - 1]
            acc.append([prev[dfa.delta[q][0]] + prev[dfa.delta[q][1]] for q in range(dfa.n_states)])
        self._acc = acc
        self._taken: set[str] = set()          # every string in any drawn set
        self._taken_by: Counter = Counter()     # (length, label) -> count drawn
        self._initial_drawn = False
        self._ledger: dict[str, dict] = {}

    # ---- exact sampling -------------------------------------------------
    def _count(self, q: int, r: int, label: int) -> int:
        a = self._acc[r][q]
        return a if label == 1 else (1 << r) - a

    def _available(self, length: int, label: int) -> int:
        return self._count(self._dfa.start, length, label) - self._taken_by[(length, label)]

    def _draw(self, length: int, label: int) -> str | None:
        """Uniformly random not-yet-taken string of this length and label, or None."""
        if self._available(length, label) <= 0:
            return None
        delta = self._dfa.delta
        while True:
            q, chars = self._dfa.start, []
            for r in range(length, 0, -1):
                w0 = self._count(delta[q][0], r - 1, label)
                w1 = self._count(delta[q][1], r - 1, label)
                sym = 0 if self._rng.randrange(w0 + w1) < w0 else 1
                chars.append("01"[sym])
                q = delta[q][sym]
            s = "".join(chars)
            if s not in self._taken:
                self._taken.add(s)
                self._taken_by[(length, label)] += 1
                return s

    def _sample(self, n: int, lengths) -> list[tuple[str, int]]:
        lengths = list(lengths)
        k = len(lengths)
        quota = [n // k] * k
        for i in range(n % k):
            quota[k - 1 - i] += 1
        out: list[tuple[str, int]] = []
        carry = {1: 0, 0: 0}
        for i, (length, q) in enumerate(zip(lengths, quota)):
            n_acc = q // 2 + (q % 2 if i % 2 == 0 else 0)
            want = {1: n_acc + carry[1], 0: q - n_acc + carry[0]}
            for label in (1, 0):
                got = 0
                while got < want[label]:
                    s = self._draw(length, label)
                    if s is None:
                        break
                    out.append((s, label))
                    got += 1
                carry[label] = want[label] - got
        # Deficit after the last length: fill the shorter class first, any length.
        for label in sorted(carry, key=lambda c: -carry[c]):
            for length in reversed(lengths):
                while len(out) < n:
                    s = self._draw(length, label)
                    if s is None:
                        break
                    out.append((s, label))
        if len(out) != n:
            raise ValueError(f"only {len(out)} distinct strings available for a set of {n}")
        return out

    def initial_dataset(self, n: int) -> list[tuple[str, int]]:
        data = self._sample(n, SHORT_LENGTHS)
        self._initial_drawn = True
        return data

    def heldout(self, n: int) -> list[tuple[str, int]]:
        if not self._initial_drawn:
            raise RuntimeError("draw initial_dataset before heldout")
        return self._sample(n, SHORT_LENGTHS)

    def long_set(self, n: int) -> list[tuple[str, int]]:
        if not self._initial_drawn:
            raise RuntimeError("draw initial_dataset before long_set")
        return self._sample(n, LONG_LENGTHS)

    # ---- charged queries ------------------------------------------------
    def query(self, arm: str, strings: list[str]) -> list[int]:
        entry = self._ledger.setdefault(arm, {"total": 0, "by_length": Counter()})
        labels = []
        for s in strings:
            if any(ch not in "01" for ch in s):
                raise ValueError(f"not a binary string: {s!r}")
            labels.append(self._dfa.label(s))
            entry["total"] += 1
            entry["by_length"][len(s)] += 1
        return labels

    def ledger(self) -> dict[str, dict]:
        return {arm: {"total": e["total"], "by_length": dict(sorted(e["by_length"].items()))}
                for arm, e in self._ledger.items()}

    def ledger_json(self) -> str:
        return json.dumps(self.ledger(), sort_keys=True)
