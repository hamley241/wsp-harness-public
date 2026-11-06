import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from wsp import rnn
from wsp import run as run_mod
from wsp.d0 import load_d0, make_d0
from wsp.llm import LLMRefusal
from wsp.run import write_llm_d0, write_mock_d0

ROOT = Path(__file__).parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "d0_h16_seed1.npz"
ARGS = ["--mock", "--rounds", "2", "--k", "4", "--hidden", "8", "--n-initial", "48", "--n-heldout", "96",
        "--n-long", "48", "--sat-timeout", "5", "--train-steps", "100", "--constructor-timeout", "1"]
TIMING_KEYS = {"timing", "elapsed_s"}
ROUND_KEYS = {"heldout", "long", "per_length", "reported_failures", "initial_regressions", "ledger_total", "sat"}
ZERO_ATTEMPTS = {"refusal_attempts": 0, "error_attempts": 0, "audit_attempts": 0, "sandbox_attempts": 0}
ZERO_FAILURES = {**ZERO_ATTEMPTS, "failed_rounds": 0}


def _proc(out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "wsp.run", *ARGS, *extra, "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def _run(out: Path, *extra: str) -> str:
    proc = _proc(out, *extra)
    assert proc.returncode == 0, proc.stderr
    return (out / "results.json").read_text()


def _strip_timing(obj):
    if isinstance(obj, dict):
        return {k: _strip_timing(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [_strip_timing(v) for v in obj]
    return obj


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    # Two distinct output directories: the payload must not depend on where it was written.
    out = tmp_path_factory.mktemp("run_first")
    first = _run(out)
    second = _run(tmp_path_factory.mktemp("run_second"))
    return out, first, second


def test_results_json_structure(runs):
    out, first, _ = runs
    res = json.loads(first)
    assert {"config", "d0_shared", "arms", "cross_play", "cross_play_sets", "trained_baseline", "noise_floor"} <= set(res)
    assert res["d0_shared"] is True and res["config"]["rounds"] == 2 and res["config"]["n_candidates"] == 32
    assert "out" not in res["config"] and str(out) not in first
    assert set(res["arms"]) == {"A", "B", "D"}
    lengths = [str(L) for L in range(1, 13)]
    for arm, rounds in res["arms"].items():
        assert len(rounds) == 3 and [e["round"] for e in rounds] == [0, 1, 2]
        for e in rounds:
            assert ROUND_KEYS <= set(e) and 0.0 <= e["heldout"] <= 1.0 and 0.0 <= e["long"] <= 1.0
            assert e["heldout_n"] == 96 and e["heldout_noise_floor"] == pytest.approx(0.5 / 96 ** 0.5)
            assert e["long_n"] == 48 and e["reported_failures_noise_floor"] == (
                None if e["reported_failures_n"] == 0 else pytest.approx(0.5 / e["reported_failures_n"] ** 0.5))
            assert e["initial_regressions_noise_floor"] == pytest.approx(0.5 / e["initial_correct_d0_n"] ** 0.5)
            # Fixed 1..12 table even though the Oracle exhausts lengths 1-3 in the initial dataset.
            assert list(e["per_length"]) == lengths and list(e["per_length_n"]) == lengths
            assert list(e["per_length_noise_floor"]) == lengths
            for L in lengths:
                n, acc = e["per_length_n"][L], e["per_length"][L]
                assert (acc is None) == (n == 0) and (acc is None or 0.0 <= acc <= 1.0)
                assert e["per_length_noise_floor"][L] == (None if n == 0 else pytest.approx(0.5 / n ** 0.5))
            assert sum(e["per_length_n"].values()) == 96
            sat = e["sat"]
            assert {"status", "ambiguity", "heldout", "heldout_n", "heldout_noise_floor", "long", "long_n",
                    "long_noise_floor"} <= set(sat)
            assert sat["heldout_n"] == 96 and sat["long_n"] == 48
            assert (sat["heldout"] is None) == (sat["dfa"] is None)
        assert rounds[0]["ledger_total"] == 0
        assert rounds[2]["ledger_total"] == (0 if arm == "D" else 64)
        assert rounds[1]["edit_stats"]["frac_changed"] > 0.0
        for e in rounds[1:]:  # artifact paths are relative to the run directory
            assert e["record"]["weights_path"] == f"{arm}/round_{e['round']}/weights.npz"
            assert (out / e["record"]["weights_path"]).is_file()
    keys = [f"{a}/{r}" for a in "ABD" for r in range(3)]
    assert list(res["cross_play"]) == keys and all(list(row) == keys for row in res["cross_play"].values())
    assert res["cross_play"]["A/0"]["D/1"] is None  # D is never shown counterexamples
    sizes = res["cross_play_sets"]
    assert list(sizes["n"]) == keys and list(sizes["noise_floor"]) == keys
    for ck in keys:
        n = sizes["n"][ck]
        assert n == res["arms"][ck[0]][int(ck[2])]["n_counterexamples"]
        assert sizes["noise_floor"][ck] == (None if n == 0 else pytest.approx(0.5 / n ** 0.5))
        assert all((row[ck] is None) == (n == 0) for row in res["cross_play"].values())
    tb = res["trained_baseline"]
    assert tb["steps"] == 100 and 0.0 <= tb["heldout"] <= 1.0
    assert tb["heldout_n"] == 96 and tb["train_acc_n"] == 48 and list(tb["per_length"]) == lengths
    assert res["noise_floor"]["heldout"] == pytest.approx(0.5 / 96 ** 0.5)
    assert res["parity_check"]["ran"] and res["hidden_dfa"]["n_states"] == 4
    assert res["config"]["backend"] == "mock" and res["config"]["mock"] is True
    assert res["d0"] == {"path": "d0.npz", "source": "make_d0", "constructor": "d0_constructor.py"}
    assert res["d0_source"] == {"kind": "mock", "from": None,
                                "sha256": hashlib.sha256((out / "d0.npz").read_bytes()).hexdigest()}
    assert res["llm_usage"] == {"calls": [], "totals": {"prompt_tokens": 0, "completion_tokens": 0,
                                                        "reasoning_tokens": 0}}
    assert res["llm_failures"] == {**ZERO_FAILURES, "refusals": 0, "errors": 0,
                                   "by_arm": {a: dict(ZERO_FAILURES) for a in "ABD"}, "d0": dict(ZERO_ATTEMPTS)}
    assert res["config"]["constructor_timeout"] == 1 and res["config"]["max_attempts"] == 3
    for rounds in res["arms"].values():
        assert all(e["record"]["failed"] is False for e in rounds[1:])
        for e in rounds[1:]:  # the mock editor is accepted on attempt 1
            rec = e["record"]
            assert rec["n_attempts"] == 1 and rec["final_outcome"] == "ok" and rec["accepted"]
            (attempt,) = rec["editor_attempts"]
            assert attempt["outcome"] == "ok" and attempt["attempt"] == 1 and attempt["src_lines"] > 0
            assert rec["audit"]["literal_match_fraction"] == 0.0
    assert (out / "d0.npz").is_file() and rnn.weights_equal(load_d0(out / "d0.npz"), make_d0(8, 1))
    for arm in "ABD":
        assert (out / arm / "round_2" / "weights.npz").is_file() and (out / arm / "calls.json").is_file()


def test_results_md_is_numbers_only(runs):
    out, _, _ = runs
    md = (out / "results.md").read_text()
    assert "| arm | round | held-out |" in md and "Cross-play" in md
    assert "failures noise floor" in md and "regressions noise floor" in md
    assert "| noise floor |" in md and "Per-length held-out accuracy" in md
    assert "| length | n | noise floor | A held-out | B held-out | D held-out | trained RNN |" in md
    assert "| held-out noise floor |" in md and str(out) not in md
    assert "## LLM usage" in md and "| calls | prompt_tokens | completion_tokens | reasoning_tokens |" in md
    assert "| failures | A | B | D | D0 | total |" in md
    for key in ("refusal_attempts", "error_attempts", "audit_attempts", "sandbox_attempts"):
        assert f"| {key} | 0 | 0 | 0 | 0 | 0 |" in md
    assert "| failed_rounds | 0 | 0 | 0 | 0 | 0 |" in md
    assert "| accepted | attempts | src lines | literal matches | SAT status |" in md
    assert "editor attempts: 6 / failed rounds: 0" in md  # 3 arms x 2 rounds, one attempt each
    assert not re.search(r"\b(suggests|indicates|because)\b", md, re.IGNORECASE)


def test_same_seed_identical_modulo_timing(runs):
    _, first, second = runs  # written to two different --out directories
    a, b = json.loads(first), json.loads(second)
    assert a["timing"] != b["timing"] or a == b
    assert json.dumps(_strip_timing(a)) == json.dumps(_strip_timing(b))


def test_openrouter_without_key_fails_before_writing(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env["HOME"] = str(home)
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, "-m", "wsp.run", "--backend", "openrouter", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode != 0 and "no OpenRouter key" in proc.stderr
    assert not out.exists()


def test_d0_from_reuses_calibration_d0(tmp_path):
    a_out, b_out = tmp_path / "a", tmp_path / "b"
    a = json.loads(_run(a_out, "--rounds", "0"))
    b = json.loads(_run(b_out, "--rounds", "1", "--d0-from", str(a_out)))
    a_bytes = (a_out / "d0.npz").read_bytes()
    assert (b_out / "d0.npz").read_bytes() == a_bytes
    assert (b_out / "d0_constructor.py").read_text() == (a_out / "d0_constructor.py").read_text()
    assert b["d0_source"] == {"kind": "reused", "from": str(a_out), "sha256": hashlib.sha256(a_bytes).hexdigest()}
    assert a["d0_source"]["kind"] == "mock" and a["d0_source"]["sha256"] == b["d0_source"]["sha256"]
    assert a["d0"]["source"] == "make_d0" and b["d0"]["source"] == "reused"
    assert b["d0_shared"] is True and b["config"]["rounds"] == 1
    for arm in "ABD":
        assert len(a["arms"][arm]) == 1 and len(b["arms"][arm]) == 2
        assert b["arms"][arm][0]["heldout"] == a["arms"][arm][0]["heldout"]
    assert f"d0_source: kind=reused from={a_out} sha256={b['d0_source']['sha256']}" in (b_out / "results.md").read_text()


def test_d0_from_mismatch_refused_before_writing(tmp_path):
    a_out = tmp_path / "a"
    _run(a_out, "--rounds", "0")
    for extra, msg in ((("--k", "5"), "k=4 seed=1"), (("--hidden", "16"), "hidden size 8")):
        out = tmp_path / "-".join(extra)
        proc = _proc(out, "--rounds", "0", "--d0-from", str(a_out), *extra)
        assert proc.returncode != 0 and msg in proc.stderr
        assert not out.exists()
    proc = _proc(tmp_path / "missing", "--rounds", "0", "--d0-from", str(tmp_path / "nope"))
    assert proc.returncode != 0 and "results.json not found" in proc.stderr


def test_backend_flags():
    from wsp.run import parse_args
    a = parse_args(["--mock"])
    assert a.backend == "mock" and a.mock and a.model is None
    a = parse_args(["--backend", "openrouter"])
    assert a.model == "anthropic/claude-fable-5.1" and a.reasoning_tokens == 16000 and a.max_output_tokens == 20000
    assert not a.mock
    assert parse_args(["--backend", "anthropic", "--model", "x"]).model == "x"
    assert parse_args(["--backend", "anthropic"]).model == "claude-fable-5-1"
    assert a.constructor_timeout == 60 and a.max_attempts == 3
    a = parse_args(["--mock", "--constructor-timeout", "1", "--max-attempts", "5"])
    assert a.constructor_timeout == 1 and a.max_attempts == 5


class _AlwaysRefusing:
    def __init__(self):
        self.n = 0

    def complete(self, system, user):
        self.n += 1
        raise LLMRefusal("blocked", finish_reason="content_filter", native_finish_reason="refusal",
                         refusal_text="policy", id=f"gen-{self.n}")


def test_d0_editor_refusal_retries_three_times_then_fails(tmp_path):
    editor = _AlwaysRefusing()
    with pytest.raises(RuntimeError) as ei:
        write_llm_d0(editor, [("01", 1), ("1", 0)], 4, tmp_path / "d0.npz", tmp_path)
    assert str(ei.value).startswith("D0 editor call failed after 3 attempts (final outcome: refusal)")
    assert editor.n == 3
    attempts = json.loads((tmp_path / "d0_attempts.json").read_text())
    assert [a["attempt"] for a in attempts] == [1, 2, 3] and [a["outcome"] for a in attempts] == ["refusal"] * 3
    assert all(a["finish_reason"] == "content_filter" and a["reason"].startswith("blocked") for a in attempts)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d0_attempts.json"]  # nothing else written under --out
    # A single attempt when asked for one.
    out1 = tmp_path / "one"
    out1.mkdir()
    editor = _AlwaysRefusing()
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        write_llm_d0(editor, [("01", 1), ("1", 0)], 4, out1 / "d0.npz", out1, max_attempts=1)
    assert editor.n == 1


class _RefusesFirstAttempt:
    """Wraps a mock editor: attempt 1 of every round is refused, the retry gets the mock constructor."""

    def __init__(self, inner):
        self.inner, self.refuse_next = inner, True

    def complete(self, system, user):
        if self.refuse_next:
            self.refuse_next = False
            raise LLMRefusal("blocked", finish_reason="content_filter", native_finish_reason="refusal",
                             refusal_text="policy", id="gen-x")
        self.refuse_next = True
        return self.inner.complete(system, user)


def test_retried_refusals_are_counted_per_attempt(tmp_path, monkeypatch):
    real = run_mod.make_clients
    monkeypatch.setattr(run_mod, "make_clients", lambda args: {
        label: _RefusesFirstAttempt(c) if label.endswith("/editor") else c for label, c in real(args).items()})
    res = run_mod.run(run_mod.parse_args([*ARGS, "--rounds", "2", "--out", str(tmp_path)]))
    lf = res["llm_failures"]
    assert lf["refusal_attempts"] == 6 and lf["refusals"] == 6  # 3 arms x 2 rounds, each retried once
    assert lf["failed_rounds"] == 0 and lf["error_attempts"] == lf["errors"] == 0
    assert lf["by_arm"] == {a: {**ZERO_FAILURES, "refusal_attempts": 2} for a in "ABD"}
    assert lf["d0"] == ZERO_ATTEMPTS
    for rounds in res["arms"].values():
        for e in rounds[1:]:
            assert e["accepted"] and e["record"]["n_attempts"] == 2
            assert [a["outcome"] for a in e["record"]["editor_attempts"]] == ["refusal", "ok"]
    md = (tmp_path / "results.md").read_text()
    assert "| refusal_attempts | 2 | 2 | 2 | 0 | 6 |" in md and "| failed_rounds | 0 | 0 | 0 | 0 | 0 |" in md
    assert "editor attempts: 12 / failed rounds: 0" in md


class _TimesOutThenOk:
    """Attempt 1 spins past the sandbox wall timeout; attempt 2 is the mock constructor."""

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, system, user):
        self.prompts.append(user)
        if len(self.prompts) == 1:
            return ("```python\nimport numpy as np\n\n\ndef build_weights():\n    while True:\n        pass\n```\n")
        return f"```python\n{run_mod.sandbox.mock_constructor_source(4, 3)}```\n"


def test_d0_sandbox_timeout_is_retried_with_the_timeout_in_the_prompt(tmp_path):
    editor = _TimesOutThenOk()
    src = write_llm_d0(editor, [("01", 1), ("1", 0)], 4, tmp_path / "d0.npz", tmp_path, timeout_s=1)
    assert src == run_mod.sandbox.mock_constructor_source(4, 3)
    attempts = json.loads((tmp_path / "d0_attempts.json").read_text())
    assert [a["outcome"] for a in attempts] == ["sandbox", "ok"]
    assert attempts[0]["reason"].endswith("wall timeout after 1s") and "wall timeout after 1s" in editor.prompts[1]
    assert "return code" not in attempts[0]["reason"] and "return code" not in editor.prompts[1]
    assert rnn.weights_equal(load_d0(tmp_path / "d0.npz"), make_d0(4, 3))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d0.npz", "d0_attempts.json", "d0_calls.json"]


def test_mock_d0_matches_fixture_for_h16_seed1(tmp_path):
    src = write_mock_d0(16, 1, tmp_path / "d0.npz")
    assert "build_weights" in src
    assert rnn.weights_equal(load_d0(tmp_path / "d0.npz"), load_d0(FIXTURE))
    assert rnn.weights_equal(load_d0(tmp_path / "d0.npz"), make_d0(16, 1))
    write_mock_d0(4, 2, tmp_path / "d0b.npz")
    assert rnn.weights_equal(load_d0(tmp_path / "d0b.npz"), make_d0(4, 2))
