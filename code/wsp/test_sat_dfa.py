import itertools

import pytest

from wsp import sat_dfa
from wsp.dfa import DFA, equivalent, random_dfa


def all_strings(max_len):
    for n in range(max_len + 1):
        for bits in itertools.product("01", repeat=n):
            yield "".join(bits)


@pytest.fixture(scope="module")
def hidden4():
    return random_dfa(4, seed=3)


@pytest.fixture(scope="module")
def sample4(hidden4):
    return [(s, hidden4.label(s)) for s in all_strings(6)]


def test_identify_min_recovers_hidden_dfa_and_is_unique(hidden4, sample4):
    res = sat_dfa.identify_min(sample4, 4)
    assert res.status == "sat"
    assert res.dfa.n_states == 4
    assert equivalent(res.dfa, hidden4)
    assert all(res.dfa.label(s) == y for s, y in sample4)
    assert sat_dfa.ambiguity(sample4, 4, res.dfa) == "unique"


def test_identify_too_few_states_is_unsat(sample4):
    assert sat_dfa.identify(sample4, 2).status == "unsat"
    assert sat_dfa.identify_min(sample4, 3).status == "unsat"


def test_ambiguous_sample():
    samples = [("", 0), ("1", 1)]
    res = sat_dfa.identify(samples, 2)
    assert res.status == "sat"
    assert sat_dfa.ambiguity(samples, 2, res.dfa) == "ambiguous"


def test_conflicting_sample_raises():
    with pytest.raises(ValueError):
        sat_dfa.identify([("01", 0), ("01", 1)], 2)


def test_identifiability_dict(sample4, hidden4):
    d = sat_dfa.identifiability(sample4, 4, timeout_s=30)
    assert d["status"] == "sat" and d["ambiguity"] == "unique" and d["n_states_found"] == 4
    assert equivalent(DFA.from_dict(d["dfa"]), hidden4)
    assert d["elapsed_s"] >= 0


def test_timeout_is_never_reported_as_unique(monkeypatch):
    hidden = random_dfa(6, seed=1)
    samples = [(s, hidden.label(s)) for s in all_strings(8)]
    res = sat_dfa.identify(samples, 6, timeout_s=0.0001)
    assert res.status in {"timeout", "sat"}
    if res.status == "timeout":
        assert res.dfa is None
    # exhausted budget: the ambiguity re-solve must say no_signal, not unique
    assert sat_dfa.ambiguity(samples, 6, hidden, timeout_s=0.0) == "no_signal"
    # monkeypatched clock: the budget elapses between solver calls
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(sat_dfa.time, "monotonic", lambda: float(next(clock)))
    assert sat_dfa.ambiguity(samples, 6, hidden, timeout_s=10.0) == "no_signal"
    assert sat_dfa.identify(samples, 6, timeout_s=10.0).status == "timeout"
    assert sat_dfa.identifiability(samples, 6, timeout_s=10.0)["ambiguity"] == "no_signal"


class _LateUnsatSolver:
    """Stand-in for pysat's Solver: solve_limited returns UNSAT, but only after
    advancing the (mocked) clock past the deadline, as if the interrupt timer
    fired late or the solver finished just after the budget elapsed."""

    instances = []

    def __init__(self, clock, *args, **kwargs):
        self._clock = clock
        self.interrupted = False
        self.solve_calls = 0
        _LateUnsatSolver.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add_clause(self, clause):
        pass

    def interrupt(self):
        self.interrupted = True

    def solve_limited(self, expect_interrupt=False):
        self.solve_calls += 1
        self._clock["t"] += 100.0  # budget is 10s: the deadline passes here
        return False

    def get_model(self):
        raise AssertionError("get_model must not be called on an UNSAT/timeout result")


def test_unsat_delivered_after_deadline_is_no_signal(monkeypatch):
    """The race from the review: a late UNSAT must never become 'unique'."""
    hidden = random_dfa(4, seed=3)
    samples = [(s, hidden.label(s)) for s in all_strings(6)]
    clock = {"t": 0.0}
    monkeypatch.setattr(sat_dfa.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(sat_dfa, "Solver",
                        lambda *a, **kw: _LateUnsatSolver(clock, *a, **kw))
    _LateUnsatSolver.instances.clear()

    assert sat_dfa.ambiguity(samples, 4, hidden, timeout_s=10.0) == "no_signal"
    assert sat_dfa.identify(samples, 4, timeout_s=10.0).status == "timeout"
    assert sat_dfa.identifiability(samples, 4, timeout_s=10.0)["ambiguity"] == "no_signal"
    # each call went through the solver exactly once and bailed on the clock
    assert [s.solve_calls for s in _LateUnsatSolver.instances] == [1, 1, 1]


def test_unsat_within_deadline_is_still_unique(monkeypatch):
    """Control for the test above: the same fake solver, but the budget is
    large enough that its UNSAT lands before the deadline."""
    hidden = random_dfa(4, seed=3)
    samples = [(s, hidden.label(s)) for s in all_strings(6)]
    clock = {"t": 0.0}
    monkeypatch.setattr(sat_dfa.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(sat_dfa, "Solver",
                        lambda *a, **kw: _LateUnsatSolver(clock, *a, **kw))
    assert sat_dfa.ambiguity(samples, 4, hidden, timeout_s=1000.0) == "unique"
    assert sat_dfa.identify(samples, 4, timeout_s=1000.0).status == "unsat"
