from pathlib import Path

from wsp import rnn
from wsp.d0 import load_d0, make_d0, save_d0

FIXTURE = Path(__file__).parent / "fixtures" / "d0_h16_seed1.npz"


def test_fixture_is_reproducible():
    assert FIXTURE.is_file()
    assert rnn.weights_equal(make_d0(16, 1), load_d0(FIXTURE))


def test_loaded_fixture_has_h16():
    w = load_d0(FIXTURE)
    assert w["Wh"].shape == (16, 16) and set(w) == set(rnn.KEYS)
    assert all(not a.flags.writeable for a in w.values())


def test_save_load_roundtrip(tmp_path):
    w = make_d0(4, 2)
    save_d0(tmp_path / "d0.npz", w)
    assert rnn.weights_equal(w, load_d0(tmp_path / "d0.npz"))
