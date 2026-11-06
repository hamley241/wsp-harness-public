import json

import pytest

from wsp.dfa import DFA, random_dfa
from wsp.oracle import Oracle

ENDS_IN_11 = DFA(3, ((0, 1), (0, 2), (0, 2)), (False, False, True), 0)
ALL_ACCEPT = DFA(1, ((0, 0),), (True,), 0)


def _check_set(data, n, lengths, dfa):
    assert len(data) == n
    strings = [s for s, _ in data]
    assert len(set(strings)) == n
    assert all(len(s) in lengths and set(s) <= {"0", "1"} for s in strings)
    assert all(label == dfa.label(s) for s, label in data)


@pytest.mark.parametrize("dfa", [random_dfa(6, seed=1), ENDS_IN_11])
def test_exact_sizes_ranges_disjointness_balance(dfa):
    o = Oracle(dfa, seed=7)
    init, held, long = o.initial_dataset(96), o.heldout(500), o.long_set(200)
    _check_set(init, 96, range(1, 13), dfa)
    _check_set(held, 500, range(1, 13), dfa)
    _check_set(long, 200, range(13, 25), dfa)
    assert not {s for s, _ in init} & {s for s, _ in held}
    assert not {s for s, _ in init} & {s for s, _ in long}
    for data in (init, held, long):
        assert sum(label for _, label in data) == len(data) // 2
    # Length stratification: every length is represented (the initial set
    # exhausts all 14 strings of lengths 1..3, so held-out starts at 4).
    assert {len(s) for s, _ in init} == set(range(1, 13))
    assert {len(s) for s, _ in held} == set(range(4, 13))
    assert {len(s) for s, _ in long} == set(range(13, 25))
    assert o.ledger() == {}  # construction-time sets are not charged


def test_sparse_class_fills_exactly_at_short_lengths():
    # Only 1 + 2 + 4 = 7 accepting strings of length <= 4; the quota carries forward.
    o = Oracle(ENDS_IN_11, seed=1)
    data = o.initial_dataset(96)
    assert sum(1 for s, label in data if label == 1 and len(s) <= 4) == 7
    assert sum(label for _, label in data) == 48


def test_one_state_all_accepting():
    o = Oracle(ALL_ACCEPT, seed=3)
    data = o.initial_dataset(96)
    _check_set(data, 96, range(1, 13), ALL_ACCEPT)
    assert all(label == 1 for _, label in data)


def test_sets_are_seeded_and_deterministic():
    a, b = Oracle(random_dfa(6, seed=1), seed=5), Oracle(random_dfa(6, seed=1), seed=5)
    assert a.initial_dataset(96) == b.initial_dataset(96)
    assert a.heldout(100) == b.heldout(100)


def test_heldout_and_long_require_initial_first():
    o = Oracle(random_dfa(6, seed=1), seed=0)
    with pytest.raises(RuntimeError):
        o.heldout(10)
    with pytest.raises(RuntimeError):
        o.long_set(10)


def test_every_query_is_charged_per_length():
    dfa = random_dfa(6, seed=1)
    o = Oracle(dfa, seed=0)
    assert o.query("A", ["0", "0", "0"]) == [dfa.label("0")] * 3  # duplicates in one call
    assert o.ledger()["A"] == {"total": 3, "by_length": {1: 3}}
    o.query("A", ["0"])  # the same string again
    o.query("A", ["0101", "11"])
    # Strings whose labels are already known (no counterexample) are charged too.
    o.query("A", ["0101", "11"])
    o.query("B", ["1" * 12])
    led = o.ledger()
    assert led["A"] == {"total": 8, "by_length": {1: 4, 2: 2, 4: 2}}
    assert led["B"] == {"total": 1, "by_length": {12: 1}}
    for e in led.values():
        assert sum(e["by_length"].values()) == e["total"]
    assert json.loads(o.ledger_json()) == {"A": {"total": 8, "by_length": {"1": 4, "2": 2, "4": 2}},
                                           "B": {"total": 1, "by_length": {"12": 1}}}
    assert o.query("A", []) == [] and o.ledger()["A"]["total"] == 8


def _contains_dfa(x) -> bool:
    if isinstance(x, DFA):
        return True
    if isinstance(x, dict):
        return any(_contains_dfa(v) for v in x.values())
    if isinstance(x, (list, tuple, set)):
        return any(_contains_dfa(v) for v in x)
    return False


def test_public_api_exposes_no_dfa():
    o = Oracle(random_dfa(6, seed=1), seed=0)
    public = [n for n in dir(o) if not n.startswith("_")]
    assert public  # sanity
    for name in public:
        assert not _contains_dfa(getattr(o, name)), name
    for name in vars(o):
        assert name.startswith("_") or not _contains_dfa(getattr(o, name)), name
    returns = [o.initial_dataset(24), o.heldout(24), o.long_set(24),
               o.query("A", ["01"]), o.ledger(), o.ledger_json()]
    assert not any(_contains_dfa(r) for r in returns)
