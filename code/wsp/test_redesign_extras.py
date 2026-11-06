import copy
import json

import numpy as np
import pytest

from wsp import redesign_extras as rx
from wsp import repair_study as rs
from wsp import trained_baseline as tb
from wsp.dfa import random_dfa
from wsp.llm import MockLLM
from wsp.numeric_edit import _f32
from wsp.oracle import Oracle

CFG = {"k": 4, "h": 8, "n_initial": 16, "rounds": 2, "n_queries": 8, "n_heldout": 64, "n_long": 40,
       "train_steps": 100, "adam_steps": 50, "per_round": 24, "distinct": 48, "max_norm": 1.0}


class _Scripted:
    def __init__(self, script):
        self.script = list(script)

    def complete(self, system, user):
        return self.script.pop(0)


def _fence(edits, prose=""):
    body = "```json\n" + json.dumps({"edits": [{"param": p, "index": i, "value": v} for p, i, v in edits]}) + "\n```\n"
    return prose + body


def _construction(seed):
    oracle = Oracle(random_dfa(CFG["k"], seed), seed)
    initial = oracle.initial_dataset(CFG["n_initial"])
    heldout = oracle.heldout(CFG["n_heldout"])
    return _f32(tb.train(initial, CFG["h"], seed, CFG["train_steps"])[0]), heldout


def _good_edit(d0, heldout):
    """A single-coordinate edit within the max-norm that raises held-out accuracy, found by search."""
    start, best = tb.accuracy(d0, heldout), None
    for param, n in (("c", 1), ("w", CFG["h"]), ("bh", CFG["h"])):
        for i in range(n):
            for delta in np.linspace(-1.0, 1.0, 21):
                w = _f32(d0)
                w[param].flat[i] = np.float32(d0[param].flat[i] + delta)
                acc = tb.accuracy(w, heldout)
                if acc > start and (best is None or acc > best[0]):
                    best = (acc, (param, i, float(w[param].flat[i])))
    assert best is not None, "no improving single edit for this construction"
    return best[1]


@pytest.fixture(scope="module")
def mock_record():
    return rs.run_seed(15, CFG, MockLLM(seed=1511, h=8, n_candidates=8))


@pytest.fixture(scope="module")
def scripted():
    """Seed 1 with rejected first attempts. E1 r1: prose-wrapped good edit (parse) then empty ok; E1 r2: prose-wrapped
    edit (parse) then an accepted edit; E2 r1: out-of-norm edit (budget) then empty ok; E2 r2: prose (parse) then ok."""
    d0, heldout = _construction(1)
    good = _good_edit(d0, heldout)
    c0 = float(d0["c"])
    script = [_fence([good], "Prose before the fence. Second sentence.\n"), _fence([]),
              _fence([("w", 1, float(d0["w"][1]) + 0.3)], "More prose.\n"), _fence([("w", 2, float(d0["w"][2]) - 0.2)]),
              _fence([("c", 0, c0 + 5.0)]), _fence([]),
              _fence([("bh", 3, float(d0["bh"][3]) + 0.1)], "Prose.\n"), _fence([("bh", 4, float(d0["bh"][4]) + 0.1)])]
    rec = rs.run_seed(1, CFG, _Scripted(script))
    assert rec["status"] == "ok"
    return {"record": rec, "good": good, "d0": d0, "heldout": heldout}


def _dirs(tmp_path, *recs):
    out = {}
    for label, rec in recs:
        d = tmp_path / label
        d.mkdir()
        (d / f"seed_{rec['seed']}.json").write_text(json.dumps(rec))
        out[label] = d
    return out


# --------------------------------------------------------------- both_sets


def test_both_sets_constants_and_missing_seeds(tmp_path, mock_record, scripted):
    dirs = _dirs(tmp_path, ("main", mock_record), ("seed1", scripted["record"]))
    both = rx.both_sets(dirs["main"], dirs["seed1"])
    assert rx.AS_RUN == (15, 20, 21, 24, 30) and rx.PRODUCTION_ORDER == (1, 15, 20, 21, 24)
    assert both["sets"] == {"as_run": list(rx.AS_RUN), "production_order": list(rx.PRODUCTION_ORDER)}
    assert set(both) == {"as_run", "production_order", "sets"}
    for name, present in (("as_run", [15]), ("production_order", [1, 15])):
        oc = both[name]
        assert oc["outcome"] == "no_result" and oc["n_analyzable"] == len(present)
        assert [r["seed"] for r in oc["seeds"]] == list(getattr(rx, name.upper()))
        assert [x["seed"] for x in oc["failed"]] == [s for s in getattr(rx, name.upper()) if s not in present]
        assert all(x["failure"] == "no seed record" for x in oc["failed"])
    assert both["as_run"]["seeds"][0]["D0"] == mock_record["D0"]["heldout"]


# ---------------------------------------------------- replay_first_attempts


def _call(response):
    return {"role": "editor-numeric", "user": "", "response": response, "finish_reason": None, "error": None}


def _attempts(outcomes):
    return [{"attempt": n, "outcome": o, "reason": "", "n_edits": 0} for n, o in enumerate(outcomes, 1)]


def test_replay_maps_calls_to_round_and_attempt(tmp_path):
    wrong = _fence([("w", i, 0.0) for i in range(5)])
    calls = [_call(_fence([("w", 0, 0.1)], "One edit. Then more.\n")),                   # r1 a1
             _call(_fence([("w", 0, 0.1), ("w", 1, 0.1)], "Two edits!\n")), _call(wrong),  # r2 a1, a2
             _call(_fence([("c", 0, 0.1), ("w", 1, 0.1), ("w", 2, 0.1)], "Three\nedits? Yes.\n")),
             _call(wrong), _call(wrong),                                                   # r3 a1..a3
             _call(None), _call(_fence([("w", 3, 0.1)], "Fourth round, second attempt.\n"))]  # r4: no response, then
    rounds = [{"round": 1, "n_attempts": 1, "attempts": _attempts(["parse"]), "edits": [], "heldout": 0.5},
              {"round": 2, "n_attempts": 2, "attempts": _attempts(["parse", "ok"]), "edits": [], "heldout": 0.5},
              {"round": 3, "n_attempts": 3, "attempts": _attempts(["parse", "parse", "ok"]), "edits": [], "heldout": 0.5},
              {"round": 4, "n_attempts": 2, "attempts": _attempts(["llm_error", "parse"]), "edits": [], "heldout": 0.5}]
    rec = {"seed": 9, "status": "ok", "config": {}, "D0": {"heldout": 0.5},
           "arms": {"E1": {"rounds": rounds, "calls": calls}, "E2": {"rounds": [], "calls": []}}}
    d = tmp_path / "x"
    d.mkdir()
    (d / "seed_9.json").write_text(json.dumps(rec))
    rows = rx.replay_first_attempts({"hand": d})
    assert [(r["run"], r["seed"], r["arm"], r["round"], r["attempt"]) for r in rows] == [
        ("hand", 9, "E1", 1, 1), ("hand", 9, "E1", 2, 1), ("hand", 9, "E1", 3, 1), ("hand", 9, "E1", 4, 2)]
    assert [r["n_edits"] for r in rows] == [1, 2, 3, 1]
    assert [r["strict_outcome"] for r in rows] == ["parse", "parse", "parse", "parse"]
    assert all(r["relaxed_parse"] == "ok" for r in rows)
    assert [r["opening_sentence"] for r in rows] == ["One edit.", "Two edits!", "Three edits?",
                                                     "Fourth round, second attempt."]
    assert [r["prose_chars"] for r in rows] == [len("One edit. Then more."), len("Two edits!"),
                                                len("Three\nedits? Yes."), len("Fourth round, second attempt.")]
    # a call count that does not match the attempts is an error, not a silent misalignment
    rec["arms"]["E1"]["calls"] = calls[:-1]
    (d / "seed_9.json").write_text(json.dumps(rec))
    with pytest.raises(ValueError):
        rx.replay_first_attempts({"hand": d})


def test_replay_on_run_records(tmp_path, mock_record, scripted):
    dirs = _dirs(tmp_path, ("main", mock_record), ("seed1", scripted["record"]))
    rows = rx.replay_first_attempts(dirs)
    assert [(r["run"], r["arm"], r["round"]) for r in rows] == [
        (run, arm, rd) for run in ("main", "seed1") for arm in ("E1", "E2") for rd in (1, 2)]
    for r in rows[:4]:  # MockLLM: a bare fence, accepted first time
        assert r["strict_outcome"] == "ok" and r["relaxed_parse"] == "ok" and r["n_edits"] == 8
        assert r["prose_chars"] == 0 and r["opening_sentence"] == ""
    s = {(r["arm"], r["round"]): r for r in rows[4:]}
    assert s[("E1", 1)]["strict_outcome"] == "parse" and s[("E1", 1)]["n_edits"] == 1
    assert s[("E1", 1)]["opening_sentence"] == "Prose before the fence." and s[("E1", 1)]["prose_chars"] == 40
    assert s[("E2", 1)]["strict_outcome"] == "budget" and s[("E2", 1)]["prose_chars"] == 0
    assert all(r["relaxed_parse"] == "ok" and r["attempt"] == 1 for r in rows)


# ------------------------------------------------------------ score_rejected


def test_score_rejected_reproduces_round_start_and_scores_known_good_proposal(tmp_path, scripted):
    rec, d0, heldout = scripted["record"], scripted["d0"], scripted["heldout"]
    dirs = _dirs(tmp_path, ("seed1", rec))
    rows = rx.score_rejected(dirs)
    assert [(r["arm"], r["round"], r["attempt"]) for r in rows] == [("E1", 1, 1), ("E1", 2, 1), ("E2", 1, 1), ("E2", 2, 1)]
    e1 = rec["arms"]["E1"]["rounds"]
    assert rows[0]["start_heldout"] == rec["D0"]["heldout"] == tb.accuracy(d0, heldout)
    assert rows[1]["start_heldout"] == e1[0]["heldout"]  # round 2 starts from round 1's (empty) edit
    # the known good proposal scores above its start
    good = rows[0]
    assert good["budget_ok"] and good["n_edits"] == 1 and good["reason"] == ""
    w = _f32(d0)
    p, i, v = scripted["good"]
    w[p].flat[i] = np.float32(v)
    assert good["after_heldout"] == tb.accuracy(w, heldout) > good["start_heldout"]
    assert good["gain"] == pytest.approx(good["after_heldout"] - good["start_heldout"]) and good["gain"] > 0
    assert good["above_f"] == (good["gain"] > rx.F)
    # the budget-rejected proposal is reported with the run's reason and no score
    bad = rows[2]
    assert not bad["budget_ok"] and bad["n_edits"] == 1 and "max-norm" in bad["reason"]
    assert bad["after_heldout"] is None and bad["gain"] is None and bad["above_f"] is None
    assert rows[1]["budget_ok"] and rows[3]["budget_ok"]
    # a tampered record (round-start held-out that the reconstruction cannot reproduce) fires the assertion
    for tamper in ("D0", "round1"):
        bent = copy.deepcopy(rec)
        if tamper == "D0":
            bent["D0"]["heldout"] += 0.5
        else:
            bent["arms"]["E1"]["rounds"][0]["heldout"] += 0.5
        (dirs["seed1"] / "seed_1.json").write_text(json.dumps(bent))
        with pytest.raises(AssertionError):
            rx.score_rejected(dirs)


def test_score_rejected_without_rejected_attempts(tmp_path, mock_record):
    assert rx.score_rejected(_dirs(tmp_path, ("main", mock_record))) == []


# --------------------------------------------------------------------- CLI


def test_cli_writes_three_files(tmp_path, mock_record, scripted):
    dirs = _dirs(tmp_path, ("main", mock_record), ("seed1", scripted["record"]))
    out = tmp_path / "out"
    assert rx.main(["--main", str(dirs["main"]), "--seed1", str(dirs["seed1"]), "--relaxed", str(dirs["seed1"]),
                    "--out", str(out)]) == 0
    both = json.loads((out / "outcomes-both-sets.json").read_text())
    assert both["as_run"]["outcome"] == "no_result" and both["production_order"]["outcome"] == "no_result"
    replay = (out / "replay-first-attempts.md").read_text()
    assert "| run | seed | arm | round | attempt | strict outcome | relaxed parse |" in replay
    assert replay.count("\n| main |") == 4 and replay.count("\n| seed1 |") == 4 and replay.count("\n| relaxed |") == 4
    assert "Summary: 12 of 12 first attempts parse under the relaxed rule." in replay
    scoring = (out / "offline-scoring.md").read_text()
    assert "| run | seed | arm | round | attempt | n_edits | budget ok | round-start held-out |" in scoring
    assert scoring.count("\n| seed1 |") == 4 and scoring.count("\n| relaxed |") == 4
    assert "Summary: 8 proposals scored, 6 within budget; gains above f = " in scoring
    assert "within ±f = " in scoring and "zero-edit proposals 0; proposals with zero held-out gain" in scoring
