import ast
import json
import random
from pathlib import Path

import numpy as np

from wsp import rnn
import pytest

from wsp.arms import (ArmA, ArmB, ArmD, _failure, _uniform_fill, counterexamples_per_query_by_length,
                      length_histogram, target_histogram)
from wsp.d0 import load_d0
from wsp.dfa import random_dfa
from wsp.llm import LLMError, LLMRefusal, MockLLM
from wsp.oracle import Oracle
from wsp.prompts import EMPTY_RESPONSE, FAILURE_HEADER
from wsp.sandbox import AuditReport, SandboxResult, mock_constructor_source

HERE = Path(__file__).parent
FIXTURE = HERE / "fixtures" / "d0_h16_seed1.npz"
BOUNDARY_MODULES = ("arms", "prompts", "llm")
FORBIDDEN_MODULES = ("wsp.dfa", "wsp.sat_dfa", "wsp.evaluate")
FORBIDDEN_NAMES = ("distinguishing_string", "equivalent")
N, N_INITIAL = 8, 24


def test_module_boundary_keeps_equivalence_checker_out():
    for mod in BOUNDARY_MODULES:
        text = (HERE / f"{mod}.py").read_text()
        for name in FORBIDDEN_NAMES:
            assert name not in text, (mod, name)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            for n in names:
                assert not any(n == f or n.startswith(f + ".") for f in FORBIDDEN_MODULES), (mod, n)


def _arms(tmp_path, n=N):
    oracle = Oracle(random_dfa(4, seed=1), seed=3)
    init = oracle.initial_dataset(N_INITIAL)

    def make(cls, name):
        return cls(name, oracle, MockLLM(seed=10, h=16, n_candidates=n), MockLLM(seed=20, h=16, n_candidates=n),
                   FIXTURE, init, n, 16, random.Random(5), tmp_path)

    return oracle, init, {"A": make(ArmA, "A"), "B": make(ArmB, "B"), "D": make(ArmD, "D")}


def test_all_arms_start_from_the_d0_file(tmp_path):
    _, _, arms = _arms(tmp_path)
    d0 = load_d0(FIXTURE)
    for arm in arms.values():
        assert rnn.weights_equal(arm.weights, d0)
        assert all(np.array_equal(arm.weights[k], d0[k]) for k in rnn.KEYS)
        assert arm.state.round == 0 and "build_weights" in arm.state.constructor_src


def test_one_round_each_arm(tmp_path):
    oracle, init, arms = _arms(tmp_path)
    d0 = load_d0(FIXTURE)
    target = target_histogram([s for s, _ in init], N)
    records = {name: arm.run_round() for name, arm in arms.items()}
    ledger = oracle.ledger()
    assert ledger["A"]["total"] == N and ledger["B"]["total"] == N
    assert "D" not in ledger
    for name, rec in records.items():
        arm = arms[name]
        assert rec["round"] == 1 and arm.state.round == 1 and arm.state.history == [rec]
        assert "build_weights" in rec["constructor_src"]
        assert rec["weights_path"] == f"{name}/round_1/weights.npz"  # relative to the run directory
        assert (tmp_path / rec["weights_path"]).is_file()
        assert rnn.weights_equal(np.load(tmp_path / rec["weights_path"]), arm.weights)
        assert (tmp_path / name / "round_1" / "constructor.py").is_file()
        assert (tmp_path / name / "calls.json").is_file()
        assert rec["accepted"] and rec["audit"]["ok"] and not rnn.weights_equal(arm.weights, d0)
    for name in "AB":
        rec = records[name]
        assert length_histogram(rec["queried"]) == target
        assert ledger[name]["by_length"] == target
        assert len(rec["labels"]) == N and all(len(c) == 3 for c in rec["counterexamples"])
        assert set(counterexamples_per_query_by_length(rec)) == set(target)
        assert arms[name].seen == rec["queried"]
    assert records["D"]["queried"] == [] and records["D"]["counterexamples"] == []
    assert 0 <= records["A"]["n_filled_random"] <= N and len(records["A"]["probe_predictions"]) <= 64


def test_uniform_fill_never_duplicates_under_saturation():
    rng = random.Random(0)
    avoid = {"000", "001", "010"}
    out = _uniform_fill(rng, 3, 5, avoid)  # exactly the 5 unused strings of length 3
    assert sorted(out) == ["011", "100", "101", "110", "111"] and len(avoid) == 8
    assert sorted(_uniform_fill(rng, 2, 4, set())) == ["00", "01", "10", "11"]
    assert _uniform_fill(rng, 0, 1, set()) == [""]
    big = _uniform_fill(rng, 20, 50, {"0" * 20})  # retry path: bucket too large to enumerate
    assert len(set(big)) == 50 and all(len(s) == 20 for s in big) and "0" * 20 not in big
    with pytest.raises(ValueError):
        _uniform_fill(rng, 1, 3, set())
    with pytest.raises(ValueError):
        _uniform_fill(rng, 2, 1, {"00", "01", "10", "11"})
    with pytest.raises(ValueError):
        _uniform_fill(rng, 3, 6, avoid)  # avoid now holds all 8 strings of length 3


class _EmptyGenerator:
    def complete(self, system, user):
        return "```strings\n```\n"


def test_arm_a_fallback_fills_saturated_buckets_without_duplicates(tmp_path):
    # Every string of length 1 and 2 is in the dataset, so the fallback must exhaust both buckets.
    oracle = Oracle(random_dfa(4, seed=1), seed=3)
    init = [("0", 1), ("1", 0), ("00", 0), ("01", 1), ("10", 1), ("11", 0)]
    n = len(init)
    arm = ArmA("A", oracle, MockLLM(seed=10, h=16, n_candidates=n), _EmptyGenerator(),
               FIXTURE, init, n, 16, random.Random(5), tmp_path)
    rec = arm.run_round()
    assert rec["n_filled_random"] == n and len(set(rec["queried"])) == n
    assert sorted(rec["queried"]) == sorted(s for s, _ in init)
    assert oracle.ledger()["A"]["total"] == n


class _BadEditor:
    def complete(self, system, user):
        return "```python\nimport torch\ndef build_weights():\n    return {}\n```"


def test_failed_constructor_keeps_previous_weights(tmp_path):
    _, _, arms = _arms(tmp_path)
    arm = arms["D"]
    arm.editor_llm = _BadEditor()
    rec = arm.run_round()
    assert not rec["accepted"] and not rec["audit"]["ok"] and rec["failed"] is False
    assert rec["n_attempts"] == 3 and [a["outcome"] for a in rec["editor_attempts"]] == ["audit"] * 3
    assert rec["final_outcome"] == "audit"
    assert rnn.weights_equal(arm.weights, load_d0(FIXTURE))
    assert rnn.weights_equal(np.load(tmp_path / rec["weights_path"]), arm.weights)


def test_failure_feedback_is_return_code_or_timeout_never_both():
    ok = AuditReport(True)
    # A wall timeout: the canonical timeout line only, no return-code line of any kind.
    kind, reason = _failure(SandboxResult(None, ok, stderr="spinning...\nwall timeout after 7s\n",
                                          confined=True, timed_out=True))
    assert kind == "sandbox" and reason == "spinning...\nwall timeout after 7s"
    assert "return code" not in reason
    # A completed process: the stderr tail plus its return code.
    kind, reason = _failure(SandboxResult(None, ok, stderr="Traceback\nZeroDivisionError\nconstructor exited 1\n",
                                          returncode=1, confined=True))
    assert kind == "sandbox" and reason == "Traceback\nZeroDivisionError\nconstructor exited 1\nreturn code: 1"
    # A process that never ran to completion (failed launch): the reason alone.
    kind, reason = _failure(SandboxResult(None, ok, stderr="sandbox launch failed: FileNotFoundError(...)"))
    assert kind == "sandbox" and reason == "sandbox launch failed: FileNotFoundError(...)"
    assert "no return code" not in reason and "return code" not in reason
    # Audit failures are the reasons verbatim, whatever the sandbox fields say.
    kind, reason = _failure(SandboxResult(None, AuditReport(False, ["import not allowed: sys"]), stderr="x"))
    assert (kind, reason) == ("audit", "import not allowed: sys")


class _AuditFailThenOk:
    """Attempt 1 is rejected by the audit (`import sys`); attempt 2 is a valid constructor."""

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, system, user):
        self.prompts.append(user)
        if len(self.prompts) == 1:
            return "```python\nimport sys\nimport numpy as np\n" + mock_constructor_source(16, 7) + "```\n"
        return f"```python\n{mock_constructor_source(16, 7)}```\n"


def test_failed_attempt_is_retried_with_the_reason_in_the_prompt(tmp_path):
    oracle, _, arms = _arms(tmp_path)
    d0 = load_d0(FIXTURE)
    for name, arm in arms.items():
        editor = _AuditFailThenOk()
        arm.editor_llm = editor
        rec = arm.run_round()
        assert rec["n_attempts"] == 2 and rec["final_outcome"] == "ok" and rec["accepted"] and not rec["failed"]
        outcomes = [a["outcome"] for a in rec["editor_attempts"]]
        assert outcomes == ["audit", "ok"] and [a["attempt"] for a in rec["editor_attempts"]] == [1, 2]
        first, second = rec["editor_attempts"]
        assert "import not allowed: sys" in first["reason"] and second["reason"] == ""
        assert first["src_lines"] > second["src_lines"] > 0 and second["literal_matches"] == 0
        assert all(a["elapsed_s"] >= 0 for a in rec["editor_attempts"])
        assert len(editor.prompts) == 2
        assert FAILURE_HEADER not in editor.prompts[0]
        block = editor.prompts[1][editor.prompts[1].index(FAILURE_HEADER):]
        assert "import not allowed: sys" in block
        assert editor.prompts[1].startswith(editor.prompts[0].rstrip("\n"))  # same prompt plus the block
        assert not rnn.weights_equal(arm.weights, d0)
        assert rnn.weights_equal(arm.weights, rnn.random_weights(16, 7))
        assert (tmp_path / name / "round_1" / "constructor.py").read_text() == mock_constructor_source(16, 7)
    assert oracle.ledger()["A"]["total"] == N and oracle.ledger()["B"]["total"] == N  # queried once per round


class _Refusing:
    def __init__(self, exc=LLMRefusal):
        self.exc = exc
        self.prompts: list[str] = []

    def complete(self, system, user):
        self.prompts.append(user)
        raise self.exc("refused by provider", finish_reason="content_filter", native_finish_reason="refusal",
                       refusal_text="blocked", id="gen-r")


def test_refused_editor_call_is_recorded_not_raised(tmp_path):
    oracle, _, arms = _arms(tmp_path)
    for name, arm in arms.items():
        before = {k: v.copy() for k, v in arm.weights.items()}
        src_before = arm.state.constructor_src
        arm.editor_llm = _Refusing()
        rec = arm.run_round()
        assert rec["failed"] is True and rec["error_kind"] == "refusal" and rec["finish_reason"] == "content_filter"
        assert rec["detail"].startswith("refused by provider") and rec["accepted"] is False
        assert rec["n_queried"] == len(rec["queried"])
        # The attempt cap holds: three attempts, every one a refusal, then the round is failed.
        assert rec["n_attempts"] == 3 and rec["final_outcome"] == "refusal"
        assert [a["outcome"] for a in rec["editor_attempts"]] == ["refusal"] * 3
        assert all(a["src_lines"] == 0 and a["literal_matches"] == 0 for a in rec["editor_attempts"])
        assert len(arm.editor_llm.prompts) == 3 and FAILURE_HEADER not in arm.editor_llm.prompts[0]
        assert all(EMPTY_RESPONSE in p and "refused by provider" not in p for p in arm.editor_llm.prompts[1:])
        assert rnn.weights_equal(arm.weights, before) and arm.state.constructor_src == src_before
        assert all(np.array_equal(arm.weights[k], before[k]) for k in rnn.KEYS)
        assert rnn.weights_equal(np.load(tmp_path / rec["weights_path"]), before)
        assert arm.state.round == 1 and arm.state.history == [rec]
        assert not (tmp_path / name / "round_1" / "constructor.py").exists()
        calls = json.loads((tmp_path / name / "calls.json").read_text())
        assert calls[-1]["role"] == "editor" and calls[-1]["response"] is None and "refused" in calls[-1]["error"]
    # B's random generator queried before the editor refused; those charges stand and nothing more:
    # the round's single query is not repeated per attempt.
    assert oracle.ledger()["B"]["total"] == N and oracle.ledger()["A"]["total"] == N
    assert arms["B"].run_round()["failed"]
    assert oracle.ledger()["B"]["total"] == 2 * N


def test_max_attempts_is_configurable(tmp_path):
    oracle = Oracle(random_dfa(4, seed=1), seed=3)
    init = oracle.initial_dataset(N_INITIAL)
    arm = ArmD("D", oracle, _Refusing(), MockLLM(seed=20, h=16, n_candidates=N), FIXTURE, init, N, 16,
               random.Random(5), tmp_path, max_attempts=1)
    rec = arm.run_round()
    assert rec["n_attempts"] == 1 and rec["failed"] and len(arm.editor_llm.prompts) == 1
    assert rnn.weights_equal(arm.weights, load_d0(FIXTURE))


def test_refused_generator_leaves_ledger_unchanged(tmp_path):
    oracle, _, arms = _arms(tmp_path)
    arm = arms["A"]
    arm.generator_llm = _Refusing()
    d0 = load_d0(FIXTURE)
    rec = arm.run_round()
    assert rec["failed"] and rec["error_kind"] == "refusal" and rec["n_queried"] == 0 and rec["queried"] == []
    assert rec["n_attempts"] == 0 and rec["editor_attempts"] == [] and rec["final_outcome"] == "refusal"
    assert "A" not in oracle.ledger() and arm.seen == []
    assert rnn.weights_equal(arm.weights, d0)
    arm.generator_llm = _Refusing(LLMError)
    rec = arm.run_round()
    assert rec["failed"] and rec["error_kind"] == "llm_error" and rec["round"] == 2
    assert "A" not in oracle.ledger()
