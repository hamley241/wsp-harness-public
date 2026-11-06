import random
from wsp.dfa import DFA, random_dfa, minimize, equivalent, distinguishing_string, parity_dfa

def test_parity():
    p = parity_dfa()
    assert p.accepts("1") and not p.accepts("11") and p.accepts("0100")

def test_random_dfa_is_minimal_and_rated():
    d = random_dfa(6, seed=1)
    assert minimize(d).n_states == 6
    rng = random.Random(0)
    acc = sum(d.accepts("".join(rng.choice("01") for _ in range(8))) for _ in range(500)) / 500
    assert 0.2 < acc < 0.8

def test_equivalence_and_distinguisher():
    a = random_dfa(5, seed=2)
    assert equivalent(a, minimize(a))
    b = DFA(a.n_states, a.delta, tuple(not x if i == 0 else x for i, x in enumerate(a.accepting)), 0)
    s = distinguishing_string(a, b)
    assert s is not None and a.accepts(s) != b.accepts(s)
