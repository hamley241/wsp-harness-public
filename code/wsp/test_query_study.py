import itertools
import json

import pytest

from wsp import query_study as qs
from wsp import sat_dfa
from wsp.dfa import DFA, equivalent, random_dfa
from wsp.llm import LLMError, LLMRefusal, MockLLM


def all_strings(max_len):
    for n in range(max_len + 1):
        for bits in itertools.product("01", repeat=n):
            yield "".join(bits)


# ------------------------------------------------------------------ phases


def test_phases_deterministic():
    samples = [("", 0), ("1", 1), ("01", 0)]
    for phases in ("default", "inverted"):
        a = sat_dfa.identify(samples, 3, phases=phases).dfa
        b = sat_dfa.identify(samples, 3, phases=phases).dfa
        assert a == b
        assert all(a.label(s) == y for s, y in samples)


def test_phases_conformance_decode_different_dfas():
    """Ambiguous instance at k=3: the two phase settings decode non-equivalent DFAs."""
    samples = [("", 0), ("1", 1)]
    assert sat_dfa.ambiguity(samples, 3, sat_dfa.identify(samples, 3).dfa) == "ambiguous"
    default = sat_dfa.identify(samples, 3, phases="default").dfa
    inverted = sat_dfa.identify(samples, 3, phases="inverted").dfa
    assert not equivalent(default, inverted)
    assert sat_dfa.identifiability(samples, 3, phases="inverted")["dfa"] == inverted.to_dict()
    with pytest.raises(ValueError):
        sat_dfa.identify(samples, 3, phases="sideways")


# ---------------------------------------------------------- eligibility


def test_eligible():
    # states 1 and 2 are equivalent (both accepting sinks): minimal size 2, not 3
    assert qs.eligible(DFA(3, ((1, 2), (1, 1), (2, 2)), (False, True, True), 0), 3) == "not_minimal"
    # state 2 is unreachable
    assert qs.eligible(DFA(3, ((0, 1), (1, 0), (2, 0)), (False, True, True), 0), 3) == "unreachable"
    assert qs.eligible(random_dfa(3, seed=1), 3) is None


def test_roster_disjoint_and_records_ineligibles(monkeypatch):
    real = qs.random_dfa
    unreachable = DFA(3, ((0, 1), (1, 0), (2, 0)), (False, True, True), 0)
    monkeypatch.setattr(qs, "random_dfa", lambda k, seed: unreachable if seed == 2 else real(k, seed))
    ros = qs.roster(3, 4, 3, 2, sat_timeout=10)
    assert len(ros["main"]) == 3 and len(ros["extension"]) == 2
    assert not set(ros["main"]) & set(ros["extension"])
    assert {"seed": 2, "reason": "unreachable"} in ros["ineligible"]
    scanned = sorted(ros["main"] + ros["extension"] + [e["seed"] for e in ros["ineligible"]])
    assert scanned == list(range(1, len(scanned) + 1))


def test_sat_samples_drops_requeries_keeps_order():
    initial = [("0", 0), ("1", 1)]
    charged = [("01", 1), ("10", 0), ("01", 1), ("11", 1), ("10", 0)]
    assert qs.sat_samples(initial, charged) == [("0", 0), ("1", 1), ("01", 1), ("10", 0), ("11", 1)]


# ------------------------------------------------------------------ replay


def _write_pilot(tmp_path, initial, probe_predictions, counterexamples=()):
    (tmp_path / "results.json").write_text(json.dumps({"initial_dataset": initial}))
    for r in qs.REPLAY_ROUNDS:
        d = tmp_path / "A" / f"round_{r}"
        d.mkdir(parents=True)
        (d / "record.json").write_text(json.dumps(
            {"round": r, "probe_predictions": probe_predictions, "counterexamples": list(counterexamples)}))
    return tmp_path


@pytest.fixture
def pilot3(tmp_path):
    hidden = random_dfa(3, seed=1)
    initial = [(s, hidden.label(s)) for s in all_strings(4)]
    assert sat_dfa.identifiability(initial, 3)["ambiguity"] == "unique"
    probes = ["".join(b) for b in itertools.product("01", repeat=5)][:8]
    return _write_pilot(tmp_path, initial, [[s, hidden.label(s)] for s in probes])


def test_replay_passes_on_consistent_pilot(pilot3):
    rep = qs.replay(pilot3, 3, timeout_s=10)
    assert rep["pass"] and not rep["vacuous"]
    assert [r["round"] for r in rep["rounds"]] == [1, 2, 3]
    for r in rep["rounds"]:
        assert r["standin_status"] == "sat" and r["n_probes"] == 8
        assert r["n_determined"] == r["agree_determined"] == 8 and r["agree_undetermined"] == 0


def test_replay_fails_on_corrupted_determined_prediction(pilot3):
    path = pilot3 / "A" / "round_2" / "record.json"
    rec = json.loads(path.read_text())
    rec["probe_predictions"][0][1] = 1 - rec["probe_predictions"][0][1]
    path.write_text(json.dumps(rec))
    rep = qs.replay(pilot3, 3, timeout_s=10)
    assert not rep["pass"]
    assert rep["rounds"][1]["agree_determined"] == 7 and rep["rounds"][0]["agree_determined"] == 8


def test_replay_vacuous_when_nothing_determined(tmp_path):
    rep = qs.replay(_write_pilot(tmp_path, [["0", 0]], [["1", 1], ["11", 0]]), 3, timeout_s=10)
    assert rep["vacuous"] and rep["pass"] and rep["n_determined"] == 0
    assert all(r["agree_determined"] == 0 for r in rep["rounds"])


# ---------------------------------------------------------------- run_seed

ROUND_KEYS = {"round", "queried", "labels", "n_filled_random", "fallback_random", "n_refusals", "n_errors",
              "attempts", "ledger_total", "sat"}
A_KEYS = ROUND_KEYS | {"standin_predictions", "n_counterexamples", "standin_status", "standin_timeout"}


def test_run_seed_mock_both_arms():
    res = qs.run_seed(3, 4, 16, 2, 8, MockLLM(seed=311, h=16, n_candidates=8), sat_timeout=10)
    assert res["seed"] == 3 and res["k"] == 4 and res["phases"] == "default"
    assert res["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    for name, keys in (("A", A_KEYS), ("B", ROUND_KEYS)):
        arm = res["arms"][name]
        assert arm["final_status"] in {"unique", "censored", "indeterminate"}
        assert 1 <= len(arm["rounds"]) <= 2
        for i, rec in enumerate(arm["rounds"], 1):
            assert keys <= set(rec)
            assert len(rec["queried"]) == len(rec["labels"]) == 8 and rec["ledger_total"] == 8 * i
            assert rec["attempts"] == 1 and not rec["fallback_random"]
        assert arm["event_time"] is None or arm["event_time"] in {8, 16}
        if arm["final_status"] == "unique":
            assert arm["event_time"] == arm["rounds"][-1]["ledger_total"] and arm["event_set"] == [arm["event_time"]]
        else:
            assert len(arm["rounds"]) == 2 and arm["event_time"] is None
    assert all(len(rec["standin_predictions"]) > 0 for rec in res["arms"]["A"]["rounds"])
    assert any(c["role"] == "generator" for c in res["arms"]["A"]["calls"])


class _Refusing:
    def complete(self, system, user):
        raise LLMRefusal("blocked", finish_reason="content_filter", native_finish_reason="refusal")


class _ErrorsTwice:
    def __init__(self):
        self.n = 0

    def complete(self, system, user):
        self.n += 1
        raise LLMError("empty", finish_reason="length")


def test_refusals_fall_back_to_random():
    res = qs.run_seed(3, 4, 16, 2, 8, _Refusing(), sat_timeout=10)
    a = res["arms"]["A"]
    assert a["final_status"] != "failed"
    for rec in a["rounds"]:
        assert rec["attempts"] == 3 and rec["n_refusals"] == 3 and rec["fallback_random"]
        assert rec["n_filled_random"] == len(rec["queried"]) == 8
    assert res["arms"]["B"]["final_status"] != "failed"


def test_errors_before_charging_mark_arm_failed():
    gen = _ErrorsTwice()
    res = qs.run_seed(3, 4, 16, 2, 8, gen, sat_timeout=10)
    a, b = res["arms"]["A"], res["arms"]["B"]
    assert a["final_status"] == "failed" and a["event_time"] is None and a["event_set"] == []
    assert gen.n == 2 and a["ledger_total"] == 0
    assert a["rounds"][0]["n_errors"] == 2 and a["rounds"][0]["queried"] == []
    assert b["final_status"] != "failed" and b["rounds"][0]["ledger_total"] == 8


class _ErrorsOncePerRound:
    """Raises on the first probe call of every round (the probe prompt carries no predictions yet)."""

    def __init__(self, inner):
        self.inner = inner
        self.armed = True
        self.n_errors = 0

    def complete(self, system, user):
        probe_call = "(none yet)" in user
        if probe_call and self.armed:
            self.armed = False
            self.n_errors += 1
            raise LLMError("empty", finish_reason="length")
        if not probe_call:
            self.armed = True  # the next round's first probe call errors again
        return self.inner.complete(system, user)


def test_error_limit_is_per_round():
    """One recovered error per round: the second round gets its own rerun instead of failing the arm."""
    gen = _ErrorsOncePerRound(MockLLM(seed=311, h=16, n_candidates=8))
    res = qs.run_seed(3, 4, 16, 3, 8, gen, sat_timeout=10)
    a = res["arms"]["A"]
    assert a["final_status"] != "failed" and a["failure"] is None
    assert len(a["rounds"]) >= 2 and gen.n_errors == len(a["rounds"])
    for i, rec in enumerate(a["rounds"], 1):
        assert rec["n_errors"] == 1 and rec["attempts"] == 2
        assert len(rec["queried"]) == 8 and rec["ledger_total"] == 8 * i


# ------------------------------------------------------------------ budget


def test_budget_reserve_and_cap_zero_stops_before_first_call():
    b = qs.Budget(10.0, prompt_rate=4e-6, completion_rate=20e-6, max_output_tokens=1000)
    assert b.reserve("x" * 300) == pytest.approx(100 * 4e-6 + 1000 * 20e-6)
    b.record({"prompt_tokens": 1000, "completion_tokens": 100})
    assert b.spent_usd == pytest.approx(0.004 + 0.002) and b.prompt_tokens == 1000
    assert b.can("x") and not qs.Budget(0.0).can("x")

    class _Never:
        def complete(self, system, user):
            raise AssertionError("must not be called")

    with pytest.raises(qs.BudgetStop):
        qs.run_seed(3, 4, 16, 1, 8, _Never(), budget=qs.Budget(0.0))


def test_budgeted_wrapper_delegates_usage_and_attributes():
    class _Fake:
        model = "fake/model"

        def __init__(self):
            self.last_usage = None

        def complete(self, system, user):
            self.last_usage = {"prompt_tokens": 7, "completion_tokens": 3, "finish_reason": "stop"}
            return "ok"

    budget = qs.Budget(10.0, prompt_rate=1.0, completion_rate=1.0)
    wrapped = qs._BudgetedLLM(_Fake(), budget)
    assert wrapped.last_usage is None and wrapped.model == "fake/model"
    assert wrapped.complete("s", "u") == "ok"
    assert wrapped.last_usage == {"prompt_tokens": 7, "completion_tokens": 3, "finish_reason": "stop"}
    assert budget.spent_usd == 10.0
    with pytest.raises(AttributeError):
        wrapped.no_such_attribute


def _cli(out, cap, roster="main"):
    return qs.main(["run", "--roster", roster, "--n-main", "2", "--n-ext", "1", "--backend", "mock", "--k", "4",
                    "--n-initial", "16", "--rounds", "2", "--n-candidates", "8", "--sat-timeout", "10",
                    "--spend-cap-usd", str(cap), "--out", str(out)])


def test_cli_stops_at_cap(tmp_path):
    assert _cli(tmp_path, 0) == 0
    stop = json.loads((tmp_path / "stopped_at_cap.json").read_text())
    assert stop["roster"] == "main" and stop["cap_usd"] == 0
    assert not list((tmp_path / "main").glob("seed_*.json"))


class _BillingMock:
    """A MockLLM that reports 1000 completion tokens per call, like a provider would."""

    n_calls = 0

    def __init__(self, seed, h, n_candidates):
        self.inner = MockLLM(seed=seed, h=h, n_candidates=n_candidates)
        self.last_usage = None

    def complete(self, system, user):
        type(self).n_calls += 1
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 1000}
        return self.inner.complete(system, user)


def test_cli_mid_seed_cap_persists_partial_usage(tmp_path, monkeypatch):
    """A seed aborted at the cap still cost money; a later invocation must count it, not spend it again."""
    monkeypatch.setattr(qs, "MockLLM", _BillingMock)
    monkeypatch.setitem(qs.RATES, "mock", (0.0, 1e-3))  # $1 per call, reserve $0.001 per call
    monkeypatch.setattr(_BillingMock, "n_calls", 0)
    argv = ["run", "--roster", "main", "--n-main", "2", "--n-ext", "1", "--backend", "mock", "--k", "4",
            "--n-initial", "16", "--rounds", "2", "--n-candidates", "8", "--sat-timeout", "10",
            "--max-output-tokens", "1", "--spend-cap-usd", "2.5", "--out", str(tmp_path)]
    assert qs.main(argv) == 0
    assert _BillingMock.n_calls == 3  # calls 1-3 fit under the cap; the 4th reserve exceeds it
    assert not list((tmp_path / "main").glob("seed_*.json"))
    lines = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["aborted"] and lines[0]["usd"] == pytest.approx(3.0)
    assert lines[0]["completion_tokens"] == 3000
    stop = json.loads((tmp_path / "stopped_at_cap.json").read_text())
    assert stop["spent_usd"] == pytest.approx(3.0) and stop["seed"] == lines[0]["seed"]
    # Second invocation: the accrued $3 already exceeds the cap, so no call is made and nothing more is spent.
    assert qs.main(argv) == 0
    assert _BillingMock.n_calls == 3
    lines = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and lines[1]["aborted"] and lines[1]["usd"] == 0
    assert qs._accrued_usd(tmp_path / "usage.jsonl") == pytest.approx(3.0)


def test_cli_runs_and_skips_existing(tmp_path):
    assert _cli(tmp_path, 10) == 0
    ros = json.loads((tmp_path / "roster.json").read_text())
    files = sorted((tmp_path / "main").glob("seed_*.json"))
    assert [int(p.stem.split("_")[1]) for p in files] == ros["main"] and len(files) == 2
    meta = json.loads((tmp_path / "study_meta.json").read_text())
    assert meta["config"]["k"] == 4 and meta["python_sat_version"]
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 2
    before = {p: p.stat().st_mtime_ns for p in files}
    assert _cli(tmp_path, 10) == 0
    assert {p: p.stat().st_mtime_ns for p in files} == before
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 2
    assert not (tmp_path / "stopped_at_cap.json").exists()
    # the sensitivity rosters take the first main seeds, arm A only
    assert _cli(tmp_path, 10, "sensitivity-inverted") == 0
    seed = json.loads((tmp_path / "sensitivity-inverted" / f"seed_{ros['main'][0]}.json").read_text())
    assert list(seed["arms"]) == ["A"] and seed["phases"] == "inverted"


def test_rates_for_known_models_and_refuses_unknown():
    from wsp.query_study import rates_for
    assert rates_for("mock", None) == (0.0, 0.0)
    assert rates_for("openrouter", "openai/gpt-6-astra") == (10e-6, 50e-6)
    assert rates_for("openrouter", "anthropic/claude-opus-5.5") == (4e-6, 20e-6)
    import pytest
    with pytest.raises(SystemExit):
        rates_for("openrouter", "openai/nonexistent")
