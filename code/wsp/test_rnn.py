import math

import numpy as np
import pytest

from wsp import rnn


def test_shapes_and_n_params():
    s = rnn.shapes(16)
    assert s == {"Wx": (2, 16), "Wh": (16, 16), "bh": (16,), "w": (16,), "c": ()}
    assert sum(math.prod(v) for v in s.values()) == rnn.n_params(16) == 16 * 16 + 4 * 16 + 1


def test_check_weights_accepts_and_freezes():
    w = rnn.check_weights(rnn.random_weights(8, seed=3), 8)
    assert set(w) == set(rnn.KEYS)
    for arr in w.values():
        assert arr.dtype == np.float64 and not arr.flags.writeable


@pytest.mark.parametrize("mutate", [
    lambda w: w.pop("c"),
    lambda w: w.__setitem__("extra", np.zeros(1)),
    lambda w: w.__setitem__("Wh", np.zeros((8, 7))),
    lambda w: w.__setitem__("bh", np.zeros(8, dtype=np.int64)),
    lambda w: w.__setitem__("w", np.full(8, np.nan)),
])
def test_check_weights_rejects(mutate):
    w = rnn.random_weights(8, seed=3)
    mutate(w)
    with pytest.raises(ValueError):
        rnn.check_weights(w, 8)


def test_check_weights_rejects_overflow_on_float64_conversion():
    w = rnn.random_weights(8, seed=3)
    w["w"] = np.full(8, np.finfo(np.longdouble).max, dtype=np.longdouble)
    if np.isfinite(np.float64(w["w"][0])):
        pytest.skip("longdouble is not wider than float64 on this platform")
    with pytest.raises(ValueError, match="non-finite"):
        rnn.check_weights(w, 8)


def test_logit_empty_string_is_bias():
    w = rnn.random_weights(5, seed=0)
    assert rnn.logit(w, "") == pytest.approx(float(w["c"]))
    assert rnn.predict(w, "") == int(w["c"] > 0)


def test_predict_hand_set_h2_matches_reference():
    # Parity-like h=2 cell: unit 0 is driven by the symbol, unit 1 echoes unit 0.
    w = {"Wx": np.array([[-3.0, 0.0], [3.0, 0.0]]), "Wh": np.array([[0.0, 0.0], [4.0, -4.0]]),
         "bh": np.array([0.0, 0.5]), "w": np.array([1.0, -2.0]), "c": np.array(0.1)}
    w = rnn.check_weights(w, 2)
    strings = ["0", "1", "01", "10", "11", "0110", "101010", "1" * 12]

    def ref(s):
        h0 = h1 = 0.0
        for ch in s:
            x = -3.0 if ch == "0" else 3.0
            h0, h1 = math.tanh(x + 0.0), math.tanh(4.0 * h0 - 4.0 * h1 + 0.5)
        return h0 - 2.0 * h1 + 0.1

    for s in strings:
        assert rnn.logit(w, s) == pytest.approx(ref(s))
        assert rnn.predict(w, s) == int(ref(s) > 0)
    assert rnn.predict_many(w, strings) == [rnn.predict(w, s) for s in strings]
    assert rnn.predict_many(w, strings) == rnn.predict_many(w, strings)
    assert len(set(rnn.predict_many(w, strings))) == 2


def test_weights_equal():
    a, b = rnn.random_weights(4, seed=1), rnn.random_weights(4, seed=1)
    assert rnn.weights_equal(a, b)
    b["c"] = b["c"] + 1.0
    assert not rnn.weights_equal(a, b)
    assert not rnn.weights_equal(a, {k: v for k, v in a.items() if k != "c"})
