import json

import pytest

from wsp import query_analysis as qa


def arm(status, event_time=None, event_set=None, rounds=()):
    if status == "unique":
        event_set = [event_time]
    return {"name": "?", "final_status": status, "event_time": event_time, "event_set": list(event_set or []),
            "rounds": list(rounds)}


def seed_rec(seed, a, b=None):
    arms = {"A": a} if b is None else {"A": a, "B": b}
    return {"seed": seed, "arms": arms}


def unique(t):
    return arm("unique", t)


def censored():
    return arm("censored")


def indeterminate(*totals):
    return arm("indeterminate", event_set=totals)


def failed():
    return arm("failed")


# ----------------------------------------------------------------- pairing


def test_event_values():
    assert qa.event_values(unique(64)) == {64}
    assert qa.event_values(censored()) == "censored"
    assert qa.event_values(indeterminate(32, 64)) == {32, 64}
    assert qa.event_values(indeterminate()) == set()
    assert qa.event_values(failed()) is None


@pytest.mark.parametrize("a, b, informative, winner, reason", [
    (unique(32), unique(64), True, "A", None),                 # unique vs unique, different times
    (unique(64), unique(32), True, "B", None),
    (unique(64), censored(), True, "A", None),                 # unique vs censored: unique wins
    (censored(), unique(96), True, "B", None),
    (indeterminate(32, 64), unique(96), True, "A", None),      # set vs unique, max below min
    (unique(32), indeterminate(64, 96), True, "A", None),
    (indeterminate(32, 64), indeterminate(96), True, "A", None),   # disjoint sets
    (indeterminate(96), indeterminate(32, 64), True, "B", None),
    (indeterminate(32, 64), censored(), True, "A", None),      # non-empty set vs censored: set wins
    (censored(), indeterminate(64, 96), True, "B", None),
    (unique(64), unique(64), False, None, "equal"),
    (censored(), censored(), False, None, "both_censored"),
    (indeterminate(32, 64), indeterminate(64, 96), False, None, "overlap"),   # overlapping sets
    (indeterminate(32, 96), unique(64), False, None, "overlap"),              # set contains the other time
    (failed(), unique(32), False, None, "failed"),
    (unique(32), failed(), False, None, "failed"),
    (indeterminate(), unique(32), False, None, "indeterminate_unknown"),
    (censored(), indeterminate(), False, None, "indeterminate_unknown"),
])
def test_pair(a, b, informative, winner, reason):
    p = qa.pair(seed_rec(7, a, b))
    assert (p["seed"], p["informative"], p["winner"], p["reason"]) == (7, informative, winner, reason)
    assert p["status_A"] == a["final_status"] and p["status_B"] == b["final_status"]


def test_pair_event_fields_are_json_friendly():
    p = qa.pair(seed_rec(1, unique(32), indeterminate(64, 96)))
    assert p["event_A"] == 32 and p["event_B"] == [64, 96]
    p = qa.pair(seed_rec(1, censored(), failed()))
    assert p["event_A"] == "censored" and p["event_B"] is None


# -------------------------------------------------------------- sign test


def test_sign_test():
    assert qa.sign_test(15, 20) < 0.05
    assert qa.sign_test(14, 20) >= 0.05
    assert qa.sign_test(0, 0) == 1.0
    assert qa.sign_test(0, 5) == 1.0
    assert qa.sign_test(5, 5) == pytest.approx(1 / 32)


# ------------------------------------------------------------------ decide


def _pairs(wins_a, wins_b, ties=0):
    out = []
    for i in range(wins_a):
        out.append(qa.pair(seed_rec(i, unique(32), censored())))
    for i in range(wins_b):
        out.append(qa.pair(seed_rec(100 + i, censored(), unique(32))))
    for i in range(ties):
        out.append(qa.pair(seed_rec(200 + i, censored(), censored())))
    return out


def test_decide_outcomes():
    d = qa.decide(_pairs(15, 5), extension_done=True)
    assert d["outcome"] == "A_beats_random" and d["n_informative"] == 20 and d["wins_A"] == 15
    assert d["p"] < 0.05 and d["margin_ok"] and d["rejected"]
    d = qa.decide(_pairs(14, 6, ties=3), extension_done=True)
    assert d["outcome"] == "no_advantage_shown" and d["n_informative"] == 20 and d["n_pairs"] == 23
    assert qa.decide(_pairs(12, 0), extension_done=True)["outcome"] == "A_beats_random"
    d = qa.decide(_pairs(9, 6), extension_done=True)
    assert d["outcome"] == "underpowered_no_result" and d["n_informative"] == 15 and not d["rejected"]
    assert qa.decide(_pairs(11, 0), extension_done=True)["outcome"] == "no_result"
    assert qa.decide([], extension_done=False)["outcome"] == "no_result"
    # B ahead: never a rejection, whatever the p of the upper tail
    d = qa.decide(_pairs(5, 15), extension_done=True)
    assert d["outcome"] == "no_advantage_shown" and not d["margin_ok"]


def test_decide_needs_extension_and_cap():
    assert qa.decide(_pairs(10, 5), extension_done=False)["needs_extension"]
    assert not qa.decide(_pairs(10, 5), extension_done=True)["needs_extension"]
    assert not qa.decide(_pairs(15, 5), extension_done=False)["needs_extension"]
    d = qa.decide(_pairs(15, 5), extension_done=True, stopped_at_cap=True)
    assert d["outcome"] == "no_result" and d["stopped_at_cap"]
    # an incomplete main roster is never read, and does not trigger the extension yet
    d = qa.decide(_pairs(15, 5), extension_done=False, main_done=False)
    assert d["outcome"] == "no_result" and not d["rejected"] and not d["main_done"] and not d["needs_extension"]


# ------------------------------------------------------------- sensitivity


def _write(d, seed, a):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"seed_{seed}.json").write_text(json.dumps(seed_rec(seed, a, censored())))


def _sens_dirs(tmp_path, main, replicate, inverted):
    for name, arms in (("main", main), ("sensitivity-replicate", replicate), ("sensitivity-inverted", inverted)):
        for seed, a in enumerate(arms, 1):
            _write(tmp_path / name, seed, a)
    return tmp_path / "main", tmp_path / "sensitivity-replicate", tmp_path / "sensitivity-inverted"


def test_discordant():
    assert qa.discordant(unique(32), unique(64))
    assert qa.discordant(unique(32), censored()) and qa.discordant(censored(), unique(96))
    assert not qa.discordant(censored(), censored())
    assert not qa.discordant(unique(64), unique(64))
    assert not qa.discordant(unique(64), failed()) and not qa.discordant(indeterminate(32), unique(64))
    assert not qa.discordant(None, unique(64))


def test_sensitivity_sensitive(tmp_path):
    main = [unique(32), unique(64), censored(), unique(96), censored()]
    s = qa.sensitivity(*_sens_dirs(tmp_path, main, list(main), [unique(64), unique(64), censored(), censored(),
                                                                    censored()]))
    assert (s["d_inv"], s["d_rep"]) == (2, 0)
    assert s["tiebreak_sensitive"] and not s["inconclusive"] and s["bars_adversary_arm"]
    assert s["seeds"] == [1, 2, 3, 4, 5] and len(s["rows"]) == 5
    assert [r["discordant_inverted"] for r in s["rows"]] == [True, False, False, True, False]


def test_sensitivity_not_sensitive_when_replicate_as_discordant(tmp_path):
    main = [unique(32), unique(64), censored(), unique(96), censored()]
    other = [unique(64), unique(64), censored(), censored(), censored()]
    s = qa.sensitivity(*_sens_dirs(tmp_path, main, other, other))
    assert (s["d_inv"], s["d_rep"]) == (2, 2)
    assert not s["tiebreak_sensitive"] and not s["bars_adversary_arm"]


def test_sensitivity_failed_run_is_inconclusive(tmp_path):
    main = [unique(32), unique(64), censored(), unique(96), censored()]
    s = qa.sensitivity(*_sens_dirs(tmp_path, main, list(main), main[:4] + [failed()]))
    assert s["inconclusive"] and s["bars_adversary_arm"] and not s["tiebreak_sensitive"]
    assert s["rows"][4]["status_inverted"] == "failed"


def test_sensitivity_missing_run_is_inconclusive(tmp_path):
    main = [unique(32), unique(64), censored(), unique(96), censored()]
    s = qa.sensitivity(*_sens_dirs(tmp_path, main, list(main), main[:4]))
    assert s["inconclusive"] and s["rows"][4]["status_inverted"] == "missing"


# ------------------------------------------------------- reliability, report


def _round(n, **kw):
    r = {"round": n, "attempts": 1, "n_refusals": 0, "n_errors": 0, "fallback_random": False, "n_filled_random": 0,
         "queried": ["0"] * 8, "labels": [0] * 8, "ledger_total": 8 * n, "sat": {"status": "sat", "ambiguity": "ambiguous"}}
    r.update(kw)
    return r


@pytest.fixture
def study_dir(tmp_path):
    (tmp_path / "roster.json").write_text(json.dumps(
        {"main": [1, 2, 4], "extension": [5], "ineligible": [{"seed": 3, "reason": "not_minimal"}]}))
    (tmp_path / "study_meta.json").write_text(json.dumps({"git_commit": "abc", "python_sat_version": "1.8"}))
    a1 = arm("unique", 16, rounds=[_round(1, attempts=2, n_refusals=1, standin_timeout=True, n_filled_random=3),
                                   _round(2, sat={"status": "sat", "ambiguity": "unique"})])
    b1 = arm("censored", rounds=[_round(1), _round(2)])
    a2 = arm("censored", rounds=[_round(1, attempts=3, n_refusals=3, fallback_random=True, n_filled_random=8),
                                 _round(2)])
    b2 = arm("unique", 8, rounds=[_round(1, sat={"status": "sat", "ambiguity": "unique"})])
    a4 = arm("indeterminate", event_set=[8, 16],
             rounds=[_round(1, attempts=2, n_errors=1), _round(2, sat={"status": "timeout", "ambiguity": "no_signal"})])
    b4 = arm("failed", rounds=[_round(1, attempts=2, n_errors=2, queried=[], labels=[], sat=None)])
    d = tmp_path / "main"
    d.mkdir()
    for seed, a, b in ((1, a1, b1), (2, a2, b2), (4, a4, b4)):
        (d / f"seed_{seed}.json").write_text(json.dumps(seed_rec(seed, a, b)))
    with open(tmp_path / "usage.jsonl", "w") as f:
        for seed, usd in ((1, 0.2), (2, 0.25), (4, 0.1)):
            f.write(json.dumps({"roster": "main", "seed": seed, "prompt_tokens": 1000, "completion_tokens": 500,
                                "usd": usd, "aborted": False}) + "\n")
    return tmp_path


def test_reliability(study_dir):
    rel = qa.reliability(study_dir)
    assert rel["ineligible"] == {"not_minimal": 1}
    assert rel["refusals"] == 4 and rel["errors"] == 3 and rel["attempts"] == 15
    assert rel["fallback_random_rounds"] == 1 and rel["filled_strings"] == 11
    assert rel["standin_timeouts"] == 1 and rel["endpoint_timeouts"] == 1
    assert rel["indeterminate_arms"] == 1 and rel["failed_arms"] == 1 and rel["reruns"] == 1
    assert rel["arms"] == 6 and rel["rounds"] == 10
    assert rel["usage"]["prompt_tokens"] == 3000 and rel["usage"]["completion_tokens"] == 1500
    assert rel["usage"]["usd"] == pytest.approx(0.55) and rel["usage"]["entries"] == 3
    assert rel["stopped_at_cap"] is None


def test_report_writes_study_files(study_dir):
    study, text = qa.report(study_dir)
    assert (study_dir / "study.json").is_file() and (study_dir / "study.md").is_file()
    assert json.loads((study_dir / "study.json").read_text()) == study
    assert [p["seed"] for p in study["main"]] == [1, 2, 4]
    assert [p["winner"] or p["reason"] for p in study["main"]] == ["A", "B", "failed"]
    d = study["decision"]
    assert d["outcome"] == "no_result" and d["n_informative"] == 2 and d["wins_A"] == 1
    assert d["needs_extension"] and d["main_done"] and not d["extension_done"] and not d["stopped_at_cap"]
    assert study["roster_check"] == {"main_missing": [], "main_unrostered": [], "extension_missing": [5],
                                     "extension_unrostered": []}
    assert "sensitivity" not in study and study["study_meta"]["git_commit"] == "abc"
    assert study["spend"]["usd"] == pytest.approx(0.55)
    md = (study_dir / "study.md").read_text()
    assert md == text
    assert "| seed | event_A | status_A | event_B | status_B | informative | winner/reason |" in md
    assert "| 1 | 16 | unique | censored | censored | True | A |" in md
    assert "| 4 | [8, 16] | indeterminate | None | failed | False | failed |" in md
    assert "- outcome: \"no_result\"" in md and "## Extension roster" in md


def test_report_with_extension_cap_and_sensitivity(study_dir):
    d = study_dir / "extension"
    d.mkdir()
    (d / "seed_5.json").write_text(json.dumps(seed_rec(5, unique(8), unique(16))))
    (study_dir / "stopped_at_cap.json").write_text(json.dumps({"roster": "extension", "seed": 5}))
    for name in ("sensitivity-replicate", "sensitivity-inverted"):
        (study_dir / name).mkdir()
        for seed in (1, 2, 4):
            (study_dir / name / f"seed_{seed}.json").write_text(json.dumps(seed_rec(seed, unique(16))))
    study, md = qa.report(study_dir)
    assert [p["seed"] for p in study["extension"]] == [5] and study["decision"]["extension_done"]
    assert study["decision"]["outcome"] == "no_result" and study["decision"]["stopped_at_cap"]
    assert study["reliability"]["stopped_at_cap"]["seed"] == 5
    sens = study["sensitivity"]
    assert sens["seeds"] == [1, 2, 4] and sens["d_inv"] == 1 and sens["d_rep"] == 1
    assert sens["inconclusive"] and sens["bars_adversary_arm"]  # seed 4's main A run is indeterminate
    assert "## Sensitivity" in md and "- outcome: \"no_result\"" in md


def test_report_ignores_unrostered_seed_files(study_dir):
    # stale or unrelated seed files (ineligible seed 3, never-rostered 9 and 11) must not enter the analysis
    for seed in (3, 9):
        (study_dir / "main" / f"seed_{seed}.json").write_text(json.dumps(seed_rec(seed, unique(8), censored())))
    (study_dir / "extension").mkdir()
    (study_dir / "extension" / "seed_11.json").write_text(json.dumps(seed_rec(11, unique(8), censored())))
    (study_dir / "scratch").mkdir()
    (study_dir / "scratch" / "seed_1.json").write_text(json.dumps(seed_rec(1, unique(8), censored(),)))
    study, md = qa.report(study_dir)
    assert [p["seed"] for p in study["main"]] == [1, 2, 4] and study["extension"] == []
    d = study["decision"]
    assert d["n_pairs"] == 3 and d["n_informative"] == 2 and d["wins_A"] == 1 and not d["extension_done"]
    assert study["roster_check"]["main_unrostered"] == [3, 9]
    assert study["roster_check"]["extension_unrostered"] == [11] and study["roster_check"]["extension_missing"] == [5]
    assert "| 9 |" not in md and "| 11 |" not in md
    rel = study["reliability"]
    assert rel["arms"] == 6 and rel["rounds"] == 10  # unchanged by the unrostered files
    assert qa.reliability(study_dir) == rel


def test_report_missing_rostered_seed_is_incomplete(study_dir):
    (study_dir / "roster.json").write_text(json.dumps({"main": [1, 2, 4, 6], "extension": [5], "ineligible": []}))
    study, md = qa.report(study_dir)
    assert [p["seed"] for p in study["main"]] == [1, 2, 4]
    assert study["roster_check"]["main_missing"] == [6]
    d = study["decision"]
    assert not d["main_done"] and d["outcome"] == "no_result" and not d["needs_extension"]
    assert "- main_missing: [6]" in md


def test_report_follows_roster_order(study_dir):
    (study_dir / "roster.json").write_text(json.dumps({"main": [4, 1, 2], "extension": [], "ineligible": []}))
    study, _ = qa.report(study_dir)
    assert [p["seed"] for p in study["main"]] == [4, 1, 2] and study["decision"]["main_done"]
    assert not study["decision"]["extension_done"]


def test_cli(study_dir, capsys):
    assert qa.main(["--out", str(study_dir)]) == 0
    assert "outcome=no_result" in capsys.readouterr().out
    assert (study_dir / "study.md").is_file()
