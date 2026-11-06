import json

import numpy as np
import pytest

from wsp import projected_gradient as pg
from wsp import query_study as qs
from wsp import repair_study as rs
from wsp import rnn
from wsp.dfa import random_dfa
from wsp.llm import MockLLM
from wsp.oracle import Oracle

CFG = {"k": 4, "h": 8, "n_initial": 32, "rounds": 2, "n_queries": 8, "n_heldout": 100, "n_long": 40,
       "train_steps": 50, "adam_steps": 50, "per_round": 24, "distinct": 48, "max_norm": 1.0}
ROUND_KEYS = {"round", "queried", "labels", "heldout", "heldout_n", "heldout_noise_floor", "long", "long_n",
              "long_noise_floor", "coordinates_changed", "changed_float32", "touched", "n_touched", "displacement",
              "blocks_touched", "edits"}
E_KEYS = ROUND_KEYS | {"counterexamples", "n_counterexamples", "readout_only", "attempts", "n_attempts", "noop",
                       "usage"}


@pytest.fixture(scope="module")
def seed_record():
    return rs.run_seed(3, CFG, MockLLM(seed=311, h=8, n_candidates=8))


# ---------------------------------------------------------------- run_seed


def test_run_seed_arms_batches_and_budgets(seed_record):
    res = seed_record
    assert res["status"] == "ok" and res["failure"] is None and not res["stopped_at_cap"]
    assert list(res["arms"]) == ["E1", "E2", "G", "N"]
    assert len(res["batches"]) == 2 and all(len(b) == 8 == len(set(b)) for b in res["batches"])
    assert res["D0"]["heldout_n"] == 100 and res["D0"]["coordinates_changed"] == 0
    # P9 / Amendment 2 item 7: E1, E2, G and N each charged the byte-identical frozen batches, round by round,
    # under their own ledger name, so the oracle ledger shows parity across the four arms
    assert set(res["ledger"]) == set(rs.ARMS)
    for name in rs.ARMS:
        queried = [rd["queried"] for rd in res["arms"][name]["rounds"]]
        assert json.dumps(queried) == json.dumps(res["batches"])
        assert res["ledger"][name]["total"] == 16
    assert res["ledger"]["E1"] == res["ledger"]["E2"] == res["ledger"]["G"] == res["ledger"]["N"]
    for name in rs.ARMS:
        for rd in res["arms"][name]["rounds"]:
            assert 0 <= rd["changed_float32"] <= len(rd["edits"])
    for name in rs.EDITED_ARMS:
        arm = res["arms"][name]
        assert arm["budget"]["n_touched"] <= 48 and arm["budget"]["displacement"] <= 1.0
        assert len(arm["rounds"]) == 2
        for rd in arm["rounds"]:
            assert len(rd["edits"]) <= 24 and rd["n_touched"] <= 48 and rd["displacement"] <= 1.0
            assert rd["coordinates_changed"] <= rd["n_touched"]
            assert set(rd["blocks_touched"]) == {p for p, _ in rd["touched"]}
    for rd in res["arms"]["G"]["rounds"]:
        assert rd["n_selected"] == 24 == len(rd["edits"])
    for name in ("E1", "E2"):
        arm = res["arms"][name]
        assert len(arm["calls"]) == 2
        for rd in arm["rounds"]:
            assert E_KEYS <= set(rd)
            assert not rd["noop"] and rd["n_attempts"] == 1 and rd["attempts"][0]["outcome"] == "ok"
            assert len(rd["edits"]) == 8 and isinstance(rd["readout_only"], bool)
    for rd in res["arms"]["N"]["rounds"]:
        assert ROUND_KEYS <= set(rd)
        assert rd["coordinates_changed"] == 0 == rd["changed_float32"] and rd["edits"] == []
        assert rd["heldout"] == res["D0"]["heldout"]
    assert res["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}


def test_run_seed_two_realizations_differ(seed_record):
    e1, e2 = seed_record["arms"]["E1"], seed_record["arms"]["E2"]
    assert e1["rounds"][0]["edits"] != e2["rounds"][0]["edits"]
    assert e1["rounds"][0]["counterexamples"] == e2["rounds"][0]["counterexamples"]  # same inputs in round 1
    new = [rd["n_new_labels"] for rd in e1["rounds"]]  # a batch string already labeled adds nothing
    assert all(0 <= n <= 8 for n in new) and e1["rounds"][-1]["n_labels"] == 32 + sum(new)
    assert [rd["n_new_labels"] for rd in seed_record["arms"]["G"]["rounds"]] == new


def test_run_seed_charges_each_batch_before_its_edit(monkeypatch):
    """Amendment 2 round semantics: for every arm and round, the oracle batch is charged under the arm's name
    before that round's edit runs. Instrumented oracle, editor and gradient step record one shared event order."""
    events, holder = [], {}
    real_editor, real_gradient = rs.editor_round, rs.pg.gradient_round

    def totals():
        return {arm: e["total"] for arm, e in holder["oracle"].ledger().items()}

    class _Spy(Oracle):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder["oracle"] = self

        def query(self, arm, strings):
            ys = super().query(arm, strings)
            events.append(("query", arm, totals()))
            return ys

    def editor(*a, **k):
        events.append(("editor", totals()))
        return real_editor(*a, **k)

    def gradient(*a, **k):
        events.append(("gradient", totals()))
        return real_gradient(*a, **k)

    monkeypatch.setattr(rs, "Oracle", _Spy)
    monkeypatch.setattr(rs, "editor_round", editor)
    monkeypatch.setattr(rs.pg, "gradient_round", gradient)
    res = rs.run_seed(3, CFG, MockLLM(seed=311, h=8, n_candidates=8))
    assert res["status"] == "ok"
    e1, e2, g = {"E1": 16}, {"E1": 16, "E2": 16}, {"E1": 16, "E2": 16, "G": 16}
    assert events == [
        ("query", "E1", {"E1": 8}), ("editor", {"E1": 8}),
        ("query", "E1", {"E1": 16}), ("editor", {"E1": 16}),
        ("query", "E2", {**e1, "E2": 8}), ("editor", {**e1, "E2": 8}),
        ("query", "E2", {**e1, "E2": 16}), ("editor", {**e1, "E2": 16}),
        ("query", "G", {**e2, "G": 8}), ("gradient", {**e2, "G": 8}),
        ("query", "G", {**e2, "G": 16}), ("gradient", {**e2, "G": 16}),
        ("query", "N", {**g, "N": 8}), ("query", "N", {**g, "N": 16})]


def test_run_seed_counterexamples_use_round_start_weights(monkeypatch):
    """Amendment 2 round semantics: round r's counterexamples are the new labels mispredicted by the weights at
    the start of round r, and the editor receives exactly that list. The stubbed editor negates the readout, so
    every prediction flips: the round-start set and the post-edit set are complements of each other."""
    received = []

    def editor(llm, weights, labels, cex, budget, calls, **kw):
        received.append((weights, list(cex)))
        new = {**weights, "w": -weights["w"], "c": -weights["c"]}
        state = budget.state()
        info = {"edits": [], "noop": False, "attempts": [{"outcome": "ok"}], "n_attempts": 1,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
                **{k: state[k] for k in ("touched", "n_touched", "displacement")}}
        return new, info

    monkeypatch.setattr(rs, "editor_round", editor)
    res = rs.run_seed(3, CFG, _Never())
    assert res["status"] == "ok" and len(received) == 4  # E1 r1, E1 r2, E2 r1, E2 r2
    for name, (r1, r2) in (("E1", received[:2]), ("E2", received[2:])):
        rounds = res["arms"][name]["rounds"]
        w1, cex1 = r1
        w2, cex2 = r2
        # round 2 starts from round 1's edited weights, not from D0 and not from round 2's own edit
        assert all(np.array_equal(w2[k], w1[k]) for k in ("Wx", "Wh", "bh"))
        assert np.array_equal(w2["w"], -w1["w"]) and np.array_equal(w2["c"], -w1["c"])
        labels = [tuple(t) for t in res["initial"]]
        for start, cex, rd in ((w1, cex1, rounds[0]), (w2, cex2, rounds[1])):
            new = pg.add_labels(labels, rd["queried"], rd["labels"])  # P9: the same accumulation as the runner
            from_start = [(s, y, rnn.predict(start, s)) for s, y in new if rnn.predict(start, s) != y]
            post = {**start, "w": -start["w"], "c": -start["c"]}
            from_post = [(s, y, rnn.predict(post, s)) for s, y in new if rnn.predict(post, s) != y]
            assert new and cex == from_start and [list(c) for c in cex] == rd["counterexamples"]
            assert cex != from_post and len(from_start) + len(from_post) == len(new)


def _label_lines(user: str) -> list[str]:
    body = user.split("### All labels so far", 1)[1].split("\n", 1)[1]
    return body.split("\n\n", 1)[0].splitlines()


def test_run_seed_requeries_charged_but_add_nothing(monkeypatch):
    """P9: strings already labeled (in the initial set or an earlier batch) are charged again, but neither
    G's training set nor E's prompt label list nor E's feedback grows."""
    initial = [s for s, _ in Oracle(random_dfa(4, 3), 3).initial_dataset(32)]
    fresh = ["0" * 14 + format(i, "04b") for i in range(6)]
    assert not set(fresh) & set(initial)
    batch = initial[:2] + fresh
    monkeypatch.setattr(pg, "query_batches", lambda *a, **k: [batch, list(batch)])
    res = rs.run_seed(3, CFG, MockLLM(seed=311, h=8, n_candidates=8))
    assert res["status"] == "ok"
    for name in rs.ARMS:
        assert res["ledger"][name]["total"] == 16
        assert [rd["n_new_labels"] for rd in res["arms"][name]["rounds"]] == [6, 0]
        assert [rd["n_labels"] for rd in res["arms"][name]["rounds"]] == [38, 38]
        assert res["arms"][name]["rounds"][0]["labels"] == res["arms"][name]["rounds"][1]["labels"]
    for name in ("E1", "E2"):
        arm = res["arms"][name]
        assert arm["rounds"][1]["counterexamples"] == [] and arm["rounds"][1]["n_counterexamples"] == 0
        for call in arm["calls"]:
            lines = _label_lines(call["user"])
            strings = [ln.split("\t")[0] for ln in lines]
            assert len(lines) == 38 and len(set(strings)) == 38 and strings[:32] == initial


class _Never:
    def complete(self, system, user):
        raise AssertionError("must not be called")


def test_run_seed_cap_marks_failed_not_imputed():
    res = rs.run_seed(3, CFG, _Never(), qs.Budget(0.0))
    assert res["status"] == "failed" and res["stopped_at_cap"] and res["failure"].startswith("BudgetStop")
    assert res["arms"]["E1"]["rounds"] == [] and res["arms"]["G"]["rounds"] == []
    assert len(res["batches"]) == 2 and res["ledger"]["E1"]["total"] == 8  # the round was charged before the call


class _Boom:
    def complete(self, system, user):
        raise RuntimeError("OpenRouter HTTP 500: boom")


def test_run_seed_client_errors_consume_attempts_not_the_seed():
    """Amendment 2 item 5: a client failure other than BudgetStop is retried within the three attempts;
    after the third the round is a no-op and the seed stays analyzable."""
    res = rs.run_seed(3, CFG, _Boom())
    assert res["status"] == "ok" and res["failure"] is None
    for name in ("E1", "E2"):
        for rd in res["arms"][name]["rounds"]:
            assert rd["noop"] and not rd["readout_only"] and rd["edits"] == [] and rd["changed_float32"] == 0
            assert [a["outcome"] for a in rd["attempts"]] == ["client_error"] * 3
            assert all("boom" in a["reason"] for a in rd["attempts"])
    assert res["ledger"]["E1"]["total"] == 16 == res["ledger"]["N"]["total"]


def test_run_seed_local_invariant_error_marks_failed(monkeypatch):
    def broken(*a, **k):
        raise AssertionError("invariant violated")

    monkeypatch.setattr(rs.pg, "gradient_round", broken)
    res = rs.run_seed(3, CFG, MockLLM(seed=311, h=8, n_candidates=8))
    assert res["status"] == "failed" and not res["stopped_at_cap"] and "invariant violated" in res["failure"]
    assert len(res["arms"]["E1"]["rounds"]) == 2 and res["arms"]["G"]["rounds"] == []
    assert rs.outcome([res])["failed"][0]["seed"] == 3


class _Scripted:
    def __init__(self, script):
        self.script = list(script)

    def complete(self, system, user):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fence(edits):
    return "```json\n" + json.dumps({"edits": edits}) + "\n```\n"


def test_readout_only_requires_a_successful_nonempty_w_or_c_round():
    wc = _fence([{"param": "w", "index": 1, "value": 0.1}, {"param": "c", "index": 0, "value": -0.2}])
    mixed = _fence([{"param": "w", "index": 1, "value": 0.1}, {"param": "Wh", "index": 3, "value": 0.0}])
    # call order: E1 r1, E1 r2, E2 r1 (three failed attempts), E2 r2
    llm = _Scripted([wc, _fence([]), RuntimeError("x"), RuntimeError("y"), RuntimeError("z"), mixed])
    res = rs.run_seed(3, {**CFG, "max_norm": 100.0}, llm)
    assert res["status"] == "ok"
    e1, e2 = res["arms"]["E1"]["rounds"], res["arms"]["E2"]["rounds"]
    assert e1[0]["readout_only"] and not e1[0]["noop"] and e1[0]["blocks_touched"] == ["w", "c"]
    assert not e1[1]["readout_only"] and not e1[1]["noop"] and e1[1]["edits"] == []       # empty
    assert not e2[0]["readout_only"] and e2[0]["noop"]                                    # failed
    assert not e2[1]["readout_only"] and e2[1]["edits"] and e2[1]["blocks_touched"] == ["Wh", "w"]
    # cumulative touched set and displacement, and float32 changes from round start
    assert e1[0]["changed_float32"] == 2 and e1[1]["changed_float32"] == 0 and e1[1]["touched"] == e1[0]["touched"]
    assert e2[0]["changed_float32"] == 0 and e2[1]["changed_float32"] <= 2
    assert e1[1]["displacement"] == e1[0]["displacement"] > 0


# ----------------------------------------------------------------- outcome


def _rec(seed, N, G, E1, E2, D0=0.6):
    return {"seed": seed, "status": "ok", "D0": {"heldout": D0},
            "arms": {a: {"rounds": [{"heldout": v}]} for a, v in (("E1", E1), ("E2", E2), ("G", G), ("N", N))}}


def _failed(seed):
    return {"seed": seed, "status": "failed", "failure": "BudgetStop: cap", "arms": {}}


def test_outcome_labels_and_precedence():
    f = 0.0224
    # E far above N and G on all five -> E_beats_gradient
    beats = [_rec(s, 0.6, 0.65, 0.8, 0.8) for s in range(1, 6)]
    assert rs.outcome(beats)["outcome"] == "E_beats_gradient"
    assert rs.outcome(beats)["seeds"][0]["d"] == pytest.approx(0.15)
    # gE within f on four seeds takes precedence even though d > f on all five (G below N)
    nr = [_rec(s, 0.6, 0.5, 0.61, 0.61) for s in range(1, 5)] + [_rec(5, 0.6, 0.5, 0.8, 0.8)]
    oc = rs.outcome(nr)
    assert oc["counts"]["d_above_f"] == 5 and oc["counts"]["gE_at_most_f"] == 4 and oc["outcome"] == "no_repair"
    assert rs.outcome([_rec(s, 0.6, 0.8, 0.65, 0.65) for s in range(1, 6)])["outcome"] == "E_below_gradient"
    # |d| <= f on four of five, gE above f on all
    match = [_rec(s, 0.6, 0.7, 0.7, 0.72) for s in range(1, 5)] + [_rec(5, 0.6, 0.7, 0.9, 0.9)]
    assert rs.outcome(match)["outcome"] == "E_matches_gradient"
    mixed = [_rec(1, 0.6, 0.7, 0.9, 0.9), _rec(2, 0.6, 0.7, 0.9, 0.9), _rec(3, 0.6, 0.9, 0.7, 0.7),
             _rec(4, 0.6, 0.9, 0.7, 0.7), _rec(5, 0.6, 0.7, 0.7, 0.7)]
    assert rs.outcome(mixed)["outcome"] == "mixed"
    assert rs.outcome(mixed, f)["f"] == f and all(o in rs.OUTCOMES for o in ("mixed", "no_result"))


def test_outcome_failed_seeds_excluded_and_no_result():
    oc = rs.outcome([_rec(s, 0.6, 0.65, 0.8, 0.8) for s in (1, 4, 6)] + [_failed(15), _failed(20)])
    assert oc["outcome"] == "no_result" and oc["n_analyzable"] == 3
    assert [x["seed"] for x in oc["failed"]] == [15, 20] and len(oc["seeds"]) == 5
    assert oc["seeds"][3] == {"seed": 15, "status": "failed"}
    # Amendment 2 item 6, exactly four analyzable: the all-five beat and below predicates are false ...
    four = [_rec(s, 0.6, 0.65, 0.8, 0.8) for s in (1, 4, 6, 15)] + [_failed(20)]
    assert rs.outcome(four)["outcome"] == "mixed" and rs.outcome(four)["counts"]["d_above_f"] == 4
    below = [_rec(s, 0.6, 0.8, 0.65, 0.65) for s in (1, 4, 6, 15)] + [_failed(20)]
    assert rs.outcome(below)["outcome"] == "mixed" and rs.outcome(below)["counts"]["d_below_minus_f"] == 4
    # ... and no_repair / E_matches_gradient need all four
    assert rs.outcome([_rec(s, 0.6, 0.65, 0.61, 0.61) for s in (1, 4, 6, 15)] + [_failed(20)])["outcome"] == "no_repair"
    match = [_rec(s, 0.6, 0.7, 0.7, 0.72) for s in (1, 4, 6, 15)] + [_failed(20)]
    assert rs.outcome(match)["outcome"] == "E_matches_gradient" and rs.outcome(match)["n_analyzable"] == 4
    three_match = match[:3] + [_rec(15, 0.6, 0.7, 0.9, 0.9), _failed(20)]
    assert rs.outcome(three_match)["outcome"] == "mixed"
    assert rs.outcome(match[:2] + [_rec(6, 0.6, 0.65, 0.61, 0.61), _rec(15, 0.6, 0.65, 0.61, 0.61), _failed(20)])["outcome"] == "mixed"


# ------------------------------------------------------------ descriptives


def _round(touched, editor=False, attempts=({"outcome": "ok"},)):
    rd = {"touched": [list(t) for t in touched]}
    if editor:
        rd.update(noop=False, readout_only=False, attempts=list(attempts))
    return rd


def test_descriptives_list_every_attempt_category_with_reason():
    """Amendment 2 item 7: every non-ok attempt category (refusal, llm_error, client_error, parse, budget) is
    counted and listed with its own reason, in the descriptives and in the rendered report."""
    round1 = [{"attempt": 1, "outcome": "client_error", "reason": "RuntimeError: HTTP 500", "finish_reason": None},
              {"attempt": 2, "outcome": "refusal", "reason": "blocked", "finish_reason": "content_filter"},
              {"attempt": 3, "outcome": "ok", "reason": "", "finish_reason": "stop"}]
    round2 = [{"attempt": 1, "outcome": "llm_error", "reason": "truncated output", "finish_reason": "length"},
              {"attempt": 2, "outcome": "parse", "reason": "no ```json fence", "finish_reason": "stop"},
              {"attempt": 3, "outcome": "budget", "reason": "25 edits exceed per_round 24", "finish_reason": "stop"}]
    rec = {"seed": 9, "status": "ok", "failure": None, "config": {"rounds": 2}, "D0": {"heldout": 0.6},
           "arms": {"E1": {"rounds": [{"round": 1, "heldout": 0.7, **_round([("w", 0)], True, round1)},
                                      {"round": 2, "heldout": 0.7, **_round([("w", 0)], True, round2), "noop": True}]},
                    "E2": {"rounds": [{"round": 1, "heldout": 0.7, **_round([("c", 0)], True)}]},
                    "G": {"rounds": [{"heldout": 0.7, **_round([("w", 0), ("Wh", 2)])}]},
                    "N": {"rounds": [{"heldout": 0.6, **_round([])}]}}}
    rec["arms"]["E2"]["rounds"][0]["readout_only"] = True
    d = rs._descriptives([rec])
    e1 = d["editor"]["E1"]
    assert e1["rounds"] == 2 and e1["noop_rounds"] == 1 and e1["attempts"] == 6
    assert {k: e1[k] for k in rs.ATTEMPT_OUTCOMES} == {k: 1 for k in rs.ATTEMPT_OUTCOMES}
    expected = [{"seed": 9, "round": 1, "attempt": 1, "outcome": "client_error", "finish_reason": None, "reason": "RuntimeError: HTTP 500"},
                {"seed": 9, "round": 1, "attempt": 2, "outcome": "refusal", "finish_reason": "content_filter", "reason": "blocked"},
                {"seed": 9, "round": 2, "attempt": 1, "outcome": "llm_error", "finish_reason": "length", "reason": "truncated output"},
                {"seed": 9, "round": 2, "attempt": 2, "outcome": "parse", "finish_reason": "stop", "reason": "no ```json fence"},
                {"seed": 9, "round": 2, "attempt": 3, "outcome": "budget", "finish_reason": "stop", "reason": "25 edits exceed per_round 24"}]
    assert e1["failures"] == expected
    assert {f["outcome"] for f in expected} == set(rs.ATTEMPT_OUTCOMES) - {"ok"}
    assert len({f["reason"] for f in expected}) == len(expected)  # each category carries its own reason
    assert d["editor"]["E2"]["failures"] == [] and d["editor"]["E2"]["readout_only_rounds"] == 1
    assert d["rounds"][0]["overlap"] == {"E1_G": 1, "E2_G": 0, "E1_E2": 0}  # separate cumulative E1–G and E2–G
    spend = {"entries": 0, "aborted": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    md = rs._markdown({"outcome": rs.outcome([rec]), **d, "spend": spend, "study_meta": None})
    assert "| realization | rounds | noop_rounds | readout_only_rounds | attempts | ok | refusal | llm_error | client_error | parse | budget |" in md
    assert "| E1 | 2 | 1 | 0 | 6 | 1 | 1 | 1 | 1 | 1 | 1 |" in md
    assert "| realization | seed | round | attempt | outcome | finish_reason | reason |" in md
    for f in expected:
        assert f"| E1 | 9 | {f['round']} | {f['attempt']} | {f['outcome']} | {f['finish_reason'] or ''} | {f['reason']} |" in md


def test_descriptives_keep_failed_and_partial_seeds_per_round():
    partial = {"seed": 7, "status": "failed", "failure": "RuntimeError: boom", "config": {"rounds": 2},
               "arms": {"E1": {"rounds": [_round([("Wx", 0), ("w", 1)], True), _round([("Wx", 0), ("w", 1), ("Wh", 3)], True)]},
                        "E2": {"rounds": [_round([("Wx", 0), ("c", 0)], True)]},
                        "G": {"rounds": []}, "N": {"rounds": []}}}
    d = rs._descriptives([partial, _failed(15)])
    assert [s["seed"] for s in d["seeds"]] == [7, 15]
    assert d["seeds"][0]["rounds_completed"] == {"E1": 2, "E2": 1, "G": 0, "N": 0} and d["seeds"][0]["n_rounds"] == 2
    assert d["seeds"][1] == {"seed": 15, "status": "failed", "failure": "BudgetStop: cap", "n_rounds": 0,
                             "rounds_completed": {"E1": 0, "E2": 0, "G": 0, "N": 0}}
    assert [(r["seed"], r["round"]) for r in d["rounds"]] == [(7, 1), (7, 2)]
    r1, r2 = d["rounds"]
    assert r1["available"] == {"E1": True, "E2": True, "G": False}
    assert r1["blocks"]["E1"] == {"Wx": 1, "Wh": 0, "bh": 0, "w": 1, "c": 0} and r1["blocks"]["G"] is None
    assert r1["n_touched"] == {"E1": 2, "E2": 2, "G": None}
    assert r1["overlap"] == {"E1_G": None, "E2_G": None, "E1_E2": 1}
    assert r2["available"] == {"E1": True, "E2": False, "G": False} and r2["n_touched"]["E1"] == 3
    assert r2["overlap"] == {"E1_G": None, "E2_G": None, "E1_E2": None}
    assert d["blocks_touched"]["E1"]["2"] == {"n_available": 1, "Wx": 1, "Wh": 1, "bh": 0, "w": 1, "c": 0}
    assert d["blocks_touched"]["G"]["1"]["n_available"] == 0 and d["blocks_touched"]["E2"]["2"]["n_available"] == 0
    assert d["editor"]["E1"]["rounds"] == 2 and d["editor"]["E2"]["attempts"] == 1
    spend = {"entries": 0, "aborted": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    md = rs._markdown({"outcome": rs.outcome([partial, _failed(15)]), **d, "spend": spend, "study_meta": None})
    assert "| 7 | 1 | failed | Wx,w | Wx,c | n/a | 2 | 2 |  |  |  | 1 |" in md
    assert "| 7 | 2 | failed | Wx,Wh,w | n/a | n/a | 3 |  |  |  |  |  |" in md
    assert "| 15 | failed | BudgetStop: cap | 0 | 0 | 0 | 0 | 0 |" in md


# --------------------------------------------------------------------- CLI


def _run(out, cap, seeds="1,2", extra=()):
    return rs.main(["run", "--seeds", seeds, "--backend", "mock", "--k", "4", "--hidden", "8", "--rounds", "2",
                    "--n-candidates", "8", "--n-heldout", "100", "--n-long", "40", "--train-steps", "50",
                    "--per-round", "24", "--distinct", "48", "--max-norm", "1.0", "--spend-cap-usd", str(cap),
                    "--out", str(out), *extra])


def test_cli_run_report_and_resume(tmp_path):
    assert _run(tmp_path, 50) == 0
    files = sorted(tmp_path.glob("seed_*.json"))
    assert [p.name for p in files] == ["seed_1.json", "seed_2.json"]
    assert all(json.loads(p.read_text())["status"] == "ok" for p in files)
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 2
    meta = json.loads((tmp_path / "study_meta.json").read_text())
    assert meta["config"]["hidden"] == 8 and meta["config"]["spend_cap_usd"] == 50 and meta["resumes"] == []
    assert not (tmp_path / "stopped_at_cap.json").exists()
    assert rs.main(["report", "--out", str(tmp_path)]) == 0
    study = json.loads((tmp_path / "study.json").read_text())
    md = (tmp_path / "study.md").read_text()
    assert study["outcome"]["outcome"] == "no_result" and "- outcome: no_result" in md
    assert study["outcome"]["n_analyzable"] == 2
    assert [(r["seed"], r["round"]) for r in study["rounds"]] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    for r in study["rounds"]:
        assert all(r["available"].values()) and all(isinstance(v, int) for v in r["overlap"].values())
        assert all(isinstance(v, int) for v in r["n_touched"].values())
    assert [s["rounds_completed"] for s in study["seeds"]] == [{a: 2 for a in rs.ARMS}] * 2
    assert study["blocks_touched"]["G"]["1"]["n_available"] == 2 == study["blocks_touched"]["E2"]["2"]["n_available"]
    assert study["editor"]["E1"]["rounds"] == 4 == study["editor"]["E1"]["ok"] == study["editor"]["E1"]["attempts"]
    assert set(study["blocks_touched"]) == {"E1", "E2", "G"} and study["spend"]["entries"] == 2
    assert study["study_meta"]["config"]["k"] == 4 and "| seed | status | D0 |" in md
    assert "| seed | round | status | E1 blocks |" in md and "| arm | round | n_available |" in md
    # a second run with the same experiment (a higher cap) skips both seeds and logs the resume
    before = {p: p.stat().st_mtime_ns for p in files}
    assert _run(tmp_path, 60) == 0
    assert {p: p.stat().st_mtime_ns for p in files} == before
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 2
    meta2 = json.loads((tmp_path / "study_meta.json").read_text())
    assert meta2["config"] == meta["config"] and meta2["written"] == meta["written"]
    assert len(meta2["resumes"]) == 1 and meta2["resumes"][0]["spend_cap_usd"] == 60
    assert meta2["resumes"][0]["seeds"] == "1,2"


def test_cli_resume_with_different_experiment_refuses(tmp_path, capsys):
    assert _run(tmp_path, 50, seeds="1") == 0
    meta = (tmp_path / "study_meta.json").read_text()
    rec = (tmp_path / "seed_1.json").read_text()
    assert _run(tmp_path, 50, seeds="1,2", extra=["--hidden", "4"]) == 1
    err = capsys.readouterr().err
    assert "different experiment" in err and "hidden" in err
    assert (tmp_path / "study_meta.json").read_text() == meta and (tmp_path / "seed_1.json").read_text() == rec
    assert not (tmp_path / "seed_2.json").exists()
    # a changed model is experiment-defining as well
    assert _run(tmp_path, 50, seeds="1", extra=["--model", "other/model"]) == 1
    assert (tmp_path / "study_meta.json").read_text() == meta
    # adding a seed under the same experiment is a resume
    assert _run(tmp_path, 50, seeds="1,2") == 0
    assert (tmp_path / "seed_2.json").exists() and (tmp_path / "seed_1.json").read_text() == rec


PRODUCTION_ARGV = ["run", "--production", "--seeds", "15,20,21,24,30", "--model", "anthropic/claude-opus-5.5",
                   "--reasoning-tokens", "16000", "--max-output-tokens", "24000", "--per-round", "24",
                   "--distinct", "48", "--max-norm", "1.0", "--spend-cap-usd", "50", "--backend", "openrouter"]


def test_production_guard_refuses_before_any_call(tmp_path, capsys, monkeypatch):
    class _NoBackend:
        def __init__(self, *a, **k):
            raise AssertionError("the backend must not be built before the guard passes")

    monkeypatch.setattr(rs, "OpenRouterClient", _NoBackend)
    monkeypatch.setattr(rs, "run_seed", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    for extra in (["--seeds", "1,2", "--backend", "mock"], ["--seeds", "1,2"], ["--model", "other/model"],
                  ["--seeds", "20,15,21,24,30"], ["--spend-cap-usd", "60"], ["--reasoning-tokens", "8000"]):
        out = tmp_path / "-".join(extra).replace("/", "_")
        assert rs.main(PRODUCTION_ARGV + ["--out", str(out)] + extra) == 1
        err = capsys.readouterr().err
        assert "rules/redesign-rule.md" in err and "Amendment 2 item 3" in err and extra[0].lstrip("-").replace("-", "_") in err
        assert not out.exists()
    assert rs.production_violations(rs.parse_args(PRODUCTION_ARGV + ["--out", "x"])) == {}
    assert set(rs.production_violations(rs.parse_args(PRODUCTION_ARGV + ["--out", "x", "--seeds", "1,2"]))) == {"seeds"}


def test_production_guard_passes_exact_configuration_and_records_commit(tmp_path, monkeypatch):
    class _FakeBackend:
        def __init__(self, *a, **k):
            pass

    seen = []

    def fake_run_seed(seed, cfg, client, spend, record_path=None):
        seen.append((seed, cfg["per_round"], cfg["distinct"], cfg["max_norm"]))
        return {"seed": seed, "status": "ok", "failure": None, "stopped_at_cap": False}

    monkeypatch.setattr(rs, "OpenRouterClient", _FakeBackend)
    monkeypatch.setattr(rs, "run_seed", fake_run_seed)
    monkeypatch.setattr(rs, "_git_commit", lambda: "abc123")
    assert rs.main(PRODUCTION_ARGV + ["--out", str(tmp_path)]) == 0
    assert [s for s, *_ in seen] == [15, 20, 21, 24, 30] and {t[1:] for t in seen} == {(24, 48, 1.0)}
    meta = json.loads((tmp_path / "study_meta.json").read_text())
    assert meta["git_commit"] == "abc123" and meta["config"]["production"] and meta["config"]["spend_cap_usd"] == 50
    assert meta["config"]["model"] == "anthropic/claude-opus-5.5" and meta["config"]["seeds"] == "15,20,21,24,30"


def test_cli_refuses_seed_records_without_meta(tmp_path):
    (tmp_path / "seed_1.json").write_text(json.dumps({"seed": 1, "status": "ok"}))
    assert _run(tmp_path, 50) == 1
    assert not (tmp_path / "study_meta.json").exists()


def test_cli_cap_zero_marks_seed_failed_and_stops(tmp_path, monkeypatch):
    monkeypatch.setitem(qs.RATES, "mock", (4e-6, 20e-6))  # priced like a provider; cap 0 permits no call
    assert _run(tmp_path, 0) == 0
    rec = json.loads((tmp_path / "seed_1.json").read_text())
    assert rec["status"] == "failed" and rec["stopped_at_cap"] and rec["failure"].startswith("BudgetStop")
    assert len(rec["batches"]) == 2 and rec["arms"]["E1"]["rounds"] == []
    assert not (tmp_path / "seed_2.json").exists()
    stop = json.loads((tmp_path / "stopped_at_cap.json").read_text())
    assert stop["seed"] == 1 and stop["cap_usd"] == 0
    lines = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["aborted"] and lines[0]["usd"] == 0
    study, md = rs.report(tmp_path)
    assert study["outcome"]["outcome"] == "no_result" and study["outcome"]["failed"][0]["seed"] == 1
    assert "| 1 | failed |" in md
    # the failed seed stays in the descriptives, with explicit availability
    assert study["seeds"][0]["rounds_completed"] == {a: 0 for a in rs.ARMS} and study["seeds"][0]["n_rounds"] == 2
    assert [(r["seed"], r["round"]) for r in study["rounds"]] == [(1, 1), (1, 2)]
    for r in study["rounds"]:
        assert not any(r["available"].values()) and set(r["overlap"].values()) == {None}
        assert set(r["n_touched"].values()) == {None} and set(r["blocks"].values()) == {None}
    assert study["blocks_touched"]["E1"]["1"]["n_available"] == 0
    assert "| 1 | 1 | failed | n/a | n/a | n/a |" in md
