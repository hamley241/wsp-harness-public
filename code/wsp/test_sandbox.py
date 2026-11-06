import os
import sys
from pathlib import Path

import numpy as np

from wsp import rnn
from wsp.sandbox import (FAIL_CLOSED_REASON, audit, confinement_command, mock_constructor_source,
                         run_constructor)

REPO = Path(__file__).resolve().parents[1]
VALID_H4 = """
def build_weights():
    return {"Wx": np.zeros((2, 4)), "Wh": np.zeros((4, 4)), "bh": np.zeros(4),
            "w": np.zeros(4), "c": np.zeros(())}
"""
# Audit-free chain from numpy to the interpreter's module table.
MODS = "np._core.numeric.sys.modules"


def _assert_os_denied(res, needle="PermissionError"):
    assert res.weights is None and res.confined and res.returncode != 0
    assert needle in res.stderr, res.stderr


def test_audit_rejects_forbidden_import_statically():
    rep = audit("import numpy as np\nimport torch\ndef build_weights():\n    return {}\n", [])
    assert not rep.ok and any("forbidden optimizer" in r and "torch" in r for r in rep.reasons)
    for stmt in ("from scipy.optimize import minimize", "from scipy import optimize",
                 "import scipy.optimize._minimize", "from scipy.optimize._minimize import minimize",
                 "import torch.nn"):
        rep = audit(f"{stmt}\ndef build_weights():\n    return {{}}\n", [])
        assert not rep.ok and any("forbidden optimizer" in r for r in rep.reasons), stmt
    rep = audit("from scipy import linalg\ndef build_weights():\n    return {}\n", [])
    assert not rep.ok and not any("forbidden optimizer" in r for r in rep.reasons)
    rep = audit("import os\ndef build_weights():\n    return {}\n", [])
    assert not rep.ok
    assert not audit("import numpy as np\n", []).ok  # no build_weights
    res = run_constructor("import torch\ndef build_weights():\n    return {}\n", 4)
    assert res.weights is None and res.returncode is None and not res.audit.ok


def test_audit_rejects_line_cap():
    src = "import numpy as np\n" + "x = 1\n" * 300 + VALID_H4
    assert not audit(src, []).ok
    assert audit("import numpy as np\n" + VALID_H4, [], line_cap=10).ok
    res = run_constructor(src, 4)
    assert res.weights is None and res.returncode is None


def test_audit_reports_literal_matches_without_blocking():
    src = 'import numpy as np\nSEEN = ["0110", "111"]\n' + VALID_H4
    rep = audit(src, ["0110", "0"])
    assert rep.ok and rep.literal_matches == ["0110"] and rep.literal_match_fraction == 0.5
    assert run_constructor(src, 4, observed_strings=["0110"]).weights is not None
    assert audit("import numpy as np\n" + VALID_H4, []).literal_match_fraction == 0.0


def test_audit_finds_observed_strings_inside_a_multiline_literal():
    observed = ["0", "10", "011", "1101", "000111"]
    data = "\n".join(f"{s}\t{i % 2}" for i, s in enumerate(observed))  # dataset as one tab-separated literal
    src = f'import numpy as np\nDATA = """\n{data}\n"""\nPAIRS = "(0,1), [10,0] {{011}}"\n' + VALID_H4
    rep = audit(src, observed)
    assert rep.ok and rep.literal_matches == sorted(observed) and rep.literal_match_fraction == 1.0
    # Substrings do not count: only whole tokens.
    rep = audit('import numpy as np\nX = "0110111"\n' + VALID_H4, ["11", "0110111"])
    assert rep.literal_matches == ["0110111"] and rep.literal_match_fraction == 0.5
    res = run_constructor(src, 4, observed_strings=observed)
    assert res.weights is not None and res.audit.literal_match_fraction == 1.0  # reported, never blocking


def test_mock_constructor_runs_confined():
    res = run_constructor(mock_constructor_source(16, 1), 16)
    assert res.weights is not None and res.confined and res.returncode == 0
    w = rnn.check_weights(res.weights, 16)
    assert rnn.weights_equal(w, rnn.random_weights(16, 1))
    assert res.elapsed_s > 0


def test_wrong_shape_yields_none():
    src = "import numpy as np\n" + VALID_H4.replace("np.zeros((4, 4))", "np.zeros((4, 3))")
    res = run_constructor(src, 4)
    assert res.weights is None and res.audit.ok and "invalid weights" in res.stderr


def test_escape_write_outside_tmp_via_numpy(tmp_path):
    out = tmp_path / "escaped.npy"
    assert not str(out).startswith("/Users")  # the write is denied by the tmp-only rule, not by /Users
    res = run_constructor(f"import numpy as np\ndef build_weights():\n    np.save({str(out)!r}, np.zeros(3))\n", 4)
    assert res.weights is None and not out.exists()
    # Through a real file handle that bypasses the nulled builtins.open: the OS layer must deny it.
    src = (f"import numpy as np\ndef build_weights():\n"
           f"    np.save({MODS}['io'].open({str(out)!r}, 'wb'), np.zeros(3))\n")
    _assert_os_denied(run_constructor(src, 4))
    assert not out.exists()


def test_escape_open_called_directly():
    res = run_constructor("import numpy as np\ndef build_weights():\n    open('/etc/hosts').read()\n", 4)
    assert res.weights is None and not res.audit.ok
    # Aliased so the audit passes; the Python layer has open=None and the run fails.
    res = run_constructor("import numpy as np\ndef build_weights():\n    o = open\n    o('/etc/hosts').read()\n", 4)
    assert res.weights is None and res.audit.ok and res.returncode != 0


def test_escape_socket_via_recovered_importer():
    src = f"""import numpy as np
def build_weights():
    il = {MODS}["importlib"]
    try:
        sock = il.import_module("socket")
        sock.create_connection(("127.0.0.1", 9), timeout=1)
    except TypeError:  # loading socket.py needs exec; the extension module does not
        s = il.import_module("_socket").socket()
        s.settimeout(1)
        s.connect(("127.0.0.1", 9))
"""
    _assert_os_denied(run_constructor(src, 4))


def test_escape_read_under_home():
    target = REPO / "pyproject.toml"
    assert target.is_file() and target.is_absolute()
    src = f"import numpy as np\ndef build_weights():\n    {MODS}['io'].open({str(target)!r}).read()\n"
    _assert_os_denied(run_constructor(src, 4))


def test_wall_timeout_is_explicit_with_no_return_code():
    res = run_constructor("import numpy as np\ndef build_weights():\n    while True:\n        pass\n", 4, timeout_s=1)
    assert res.weights is None and res.audit.ok and res.confined and res.timed_out and res.returncode is None
    assert res.stderr.rstrip("\n").endswith("wall timeout after 1s")
    # Every other outcome, accepted or not, is not a timeout.
    assert run_constructor(mock_constructor_source(4, 1), 4).timed_out is False
    res = run_constructor("import numpy as np\ndef build_weights():\n    return 1 / 0\n", 4)
    assert res.weights is None and res.returncode not in (None, 0) and res.timed_out is False


def test_fails_closed_without_confinement(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "unknown-os")
    assert confinement_command(sys.executable, str(tmp_path)) is None
    res = run_constructor(mock_constructor_source(4, 1), 4)
    assert res.weights is None and not res.confined and res.returncode is None
    assert FAIL_CLOSED_REASON in res.stderr and res.audit.ok


def test_launch_failure_returns_none_with_reason(monkeypatch):
    import wsp.sandbox as sb
    monkeypatch.setattr(sb, "confinement_command", lambda py, tmp: ["/nonexistent/sandbox-bin", py])
    res = run_constructor(mock_constructor_source(4, 1), 4)
    assert res.weights is None and not res.confined and res.returncode is None
    assert "sandbox launch failed" in res.stderr and "FileNotFoundError" in res.stderr

    def boom(py, tmp):
        raise OSError("cannot write profile")
    monkeypatch.setattr(sb, "confinement_command", boom)
    res = run_constructor(mock_constructor_source(4, 1), 4)
    assert res.weights is None and not res.confined and "sandbox setup failed" in res.stderr


def test_confinement_command_darwin_profile(tmp_path):
    if sys.platform != "darwin":
        return
    cmd = confinement_command(sys.executable, str(tmp_path))
    assert cmd[0] == "sandbox-exec" and cmd[-1] == sys.executable
    profile = Path(cmd[2]).read_text()
    assert os.path.realpath(tmp_path) in profile and sys.prefix in profile and "(deny default)" in profile
