import numpy as np
import pytest

from wsp import numeric_edit as ne
from wsp import rnn
from wsp.llm import BudgetStop, CallLog, LLMError, LLMRefusal, MockLLM

H = 16
LABELS = [("", 1), ("0", 0), ("01", 1), ("110", 0)]
CEX = [("0", 0, 1), ("110", 0, 1)]


def _d0(seed=0):
    d0 = {k: v.astype(np.float32) for k, v in rnn.random_weights(H, seed).items()}
    d0["w"][0] = 0.25  # exact in float32: 1.25 - 0.25 == 1.0 exactly
    return d0


def _edits(d0, n, start=0, param="Wh", delta=0.01):
    """n entries on consecutive coordinates of `param`, each displaced `delta` from D0."""
    return [(param, i, float(d0[param].flat[i]) + delta) for i in range(start, start + n)]


# ------------------------------------------------------------ coordinates


def test_flat_index_order_and_names():
    assert ne.flat_index("Wx", 0) == 0
    assert ne.flat_index("Wh", 0) == 32
    assert ne.flat_index("bh", 0) == 288
    assert ne.flat_index("w", 15) == 319
    assert ne.flat_index("c", 0) == 320
    names = ne.coordinate_names()
    assert len(names) == 321 == rnn.n_params(H)
    assert names[32 + 17] == "Wh[1,1]" and names[-1] == "c"
    with pytest.raises(ValueError):
        ne.flat_index("Wh", 256)


# ------------------------------------------------------------ budget conformance (P3, P4)


def test_per_round_limit():
    d0 = _d0()
    b = ne.EditBudget(d0)
    assert "per-round" in b.check(_edits(d0, 25), d0)
    assert b.check(_edits(d0, 24), d0) is None


def test_distinct_limit_over_rounds():
    d0 = _d0()
    b = ne.EditBudget(d0)
    cur = b.apply(_edits(d0, 24, 0), d0)
    assert b.state()["n_touched"] == 24
    assert b.check(_edits(d0, 24, 24), cur) is None
    cur = b.apply(_edits(d0, 24, 24), cur)
    assert b.state()["n_touched"] == 48
    # one new coordinate beyond 48 rejected; re-editing touched coords accepted
    assert "distinct" in b.check(_edits(d0, 1, 48), cur)
    assert "distinct" in b.check(_edits(d0, 23, 0) + _edits(d0, 1, 48), cur)
    assert b.check(_edits(d0, 24, 10, delta=0.05), cur) is None
    cur = b.apply(_edits(d0, 24, 10, delta=0.05), cur)
    assert b.state()["n_touched"] == 48


def test_max_norm_from_d0():
    d0 = _d0()
    b = ne.EditBudget(d0)
    assert "max-norm" in b.check([("w", 0, 1.2501)], d0)
    assert b.check([("w", 0, 1.25)], d0) is None
    cur = b.apply([("w", 0, 1.25)], d0)
    assert cur["w"].dtype == np.float32 and cur["w"][0] == np.float32(1.25)
    assert b.state()["displacement"] == pytest.approx(1.0)
    # the limit is measured from D0, not from the current weights: another +1.0 step is rejected
    assert "max-norm" in b.check([("w", 0, 2.25)], cur)
    assert "max-norm" in b.check([("w", 0, -0.7501)], cur)


def test_duplicates_unknowns_and_nonfinite():
    b = ne.EditBudget(_d0())
    assert "duplicate" in b.check([("Wh", 3, 0.1), ("Wh", 3, 0.2)], b.d0)
    assert "unknown param" in b.check([("Wz", 0, 0.1)], b.d0)
    assert "out of range" in b.check([("bh", 16, 0.1)], b.d0)
    assert "out of range" in b.check([("c", 1, 0.1)], b.d0)
    assert "non-finite" in b.check([("w", 1, float("nan"))], b.d0)
    assert "non-finite" in b.check([("w", 1, float("inf"))], b.d0)


def test_unchanged_value_still_counts():
    d0 = _d0()
    b = ne.EditBudget(d0)
    same = [("Wh", i, float(d0["Wh"].flat[i])) for i in range(24)]
    assert b.check(same + [("Wh", 24, float(d0["Wh"].flat[24]))], d0) is not None  # 25 entries
    cur = b.apply(same, d0)
    assert rnn.weights_equal(cur, d0)
    assert b.state()["n_touched"] == 24 and b.state()["displacement"] == 0.0
    b.apply([("Wh", i, float(d0["Wh"].flat[i])) for i in range(24, 48)], cur)
    assert "distinct" in b.check([("Wh", 48, float(d0["Wh"].flat[48]))], cur)


# ------------------------------------------------------------ parse_edits


def _fence(body, tag="json"):
    return f"```{tag}\n{body}\n```\n"


def test_parse_edits_valid():
    out = ne.parse_edits(_fence('{"edits": [{"param": "Wh", "index": 17, "value": -0.25}, '
                                '{"param": "c", "index": 0, "value": 1}]}'))
    assert out == [("Wh", 17, -0.25), ("c", 0, 1.0)]
    assert ne.parse_edits(_fence('{"edits": []}')) == []


def test_parse_edits_failures():
    good = '{"edits": [{"param": "w", "index": 1, "value": 0.5}]}'
    assert "exactly one" in ne.parse_edits(_fence(good) + _fence(good))
    assert "exactly one" in ne.parse_edits(_fence(good, tag="python"))
    assert "exactly one" in ne.parse_edits(good)
    assert "edits" in ne.parse_edits(_fence('{"changes": []}'))
    assert "outside" in ne.parse_edits("here\n" + _fence(good))
    assert "exactly the key" in ne.parse_edits(_fence('{"edits": [], "note": 1}'))
    assert "number" in ne.parse_edits(_fence('{"edits": [{"param": "w", "index": 1, "value": "0.5"}]}'))
    assert "unknown param" in ne.parse_edits(_fence('{"edits": [{"param": "W", "index": 1, "value": 0.5}]}'))
    assert "integer" in ne.parse_edits(_fence('{"edits": [{"param": "w", "index": 1.0, "value": 0.5}]}'))
    assert "invalid JSON" in ne.parse_edits(_fence('{"edits": ['))


def test_parse_edits_rejects_duplicate_member_names():
    top = ne.parse_edits(_fence('{"edits": [], "edits": [{"param": "w", "index": 1, "value": 0.5}]}'))
    assert "duplicate JSON member name" in top and "edits" in top
    inner = ne.parse_edits(_fence('{"edits": [{"param": "w", "index": 1, "value": 0.5, "value": 0.7}]}'))
    assert "duplicate JSON member name" in inner and "value" in inner


# ------------------------------------------------------------ prompts


def test_prompts_carry_weights_labels_budget_and_failure():
    d0 = _d0()
    b = ne.EditBudget(d0)
    b.apply([("w", 0, 0.5)], d0)
    user = ne.editor_user(d0, LABELS, CEX, b.state(), failure="duplicate coordinate Wh[3] in one batch")
    assert "ROLE: editor-numeric" in ne.editor_system()
    assert f"w[0] = {np.float32(0.25)}" in user and "c[0] = " in user
    assert len(ne.weight_lines(d0)) == 321
    assert "\t1\n0\t0\n01\t1\n110\t0" in user  # labels in ledger order
    assert "0\t0\t1" in user and "distinct coordinates not yet touched that you may still add: 47" in user
    assert "current displacement: 0.25" in user and "touched so far: w[0]" in user
    assert user.rstrip().endswith("duplicate coordinate Wh[3] in one batch")
    assert ne.FAILURE_HEADER not in ne.editor_user(d0, LABELS, [], b.state())


# ------------------------------------------------------------ editor_round


class _Scripted:
    """Replays a list of outcomes: an exception instance to raise, or a string to return."""

    def __init__(self, script):
        self.script = list(script)
        self.users: list[str] = []
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 0}

    def complete(self, system, user):
        self.users.append(user)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_editor_round_recovers_on_third_attempt():
    d0 = _d0()
    b = ne.EditBudget(d0)
    llm = _Scripted([LLMRefusal("provider refusal", finish_reason="content_filter"), "```json\n{oops\n```",
                     _fence('{"edits": [{"param": "w", "index": 2, "value": 0.75}]}')])
    log = CallLog()
    new, rec = ne.editor_round(llm, d0, LABELS, CEX, b, log)
    assert rec["n_attempts"] == 3 and not rec["noop"]
    assert [a["outcome"] for a in rec["attempts"]] == ["refusal", "parse", "ok"]
    assert "provider refusal" in rec["attempts"][0]["reason"] and "invalid JSON" in rec["attempts"][1]["reason"]
    assert rec["attempts"][0]["finish_reason"] == "content_filter"
    # the retry carries the previous failure reason verbatim
    assert llm.users[1].rstrip().endswith(rec["attempts"][0]["reason"])
    assert llm.users[2].rstrip().endswith(rec["attempts"][1]["reason"])
    assert ne.FAILURE_HEADER not in llm.users[0]
    assert rec["edits"] == [["w", 2, 0.75]] and rec["touched"] == [["w", 2]] and rec["n_touched"] == 1
    assert rec["displacement"] == pytest.approx(abs(0.75 - float(d0["w"][2])))
    assert rec["usage"] == {"prompt_tokens": 30, "completion_tokens": 15, "reasoning_tokens": 0}
    assert new["w"][2] == np.float32(0.75) and not rnn.weights_equal(new, d0)
    assert len(log.calls) == 3 and log.calls[0]["role"] == "editor-numeric"


def test_editor_round_noop_after_three_failures():
    d0 = _d0()
    b = ne.EditBudget(d0)
    over = _fence('{"edits": [' + ", ".join(f'{{"param": "Wh", "index": {i}, "value": 0.0}}' for i in range(25)) + "]}")
    llm = _Scripted([LLMError("empty", finish_reason="length"), over, "no fence at all"])
    new, rec = ne.editor_round(llm, d0, LABELS, [], b, CallLog())
    assert rec["noop"] and rec["n_attempts"] == 3
    assert [a["outcome"] for a in rec["attempts"]] == ["llm_error", "budget", "parse"]
    assert "per-round" in rec["attempts"][1]["reason"]
    assert rec["edits"] == [] and rec["touched"] == [] and rec["displacement"] == 0.0
    assert rnn.weights_equal(new, d0) and b.state()["n_touched"] == 0


def test_editor_round_client_error_consumes_attempt_and_retries():
    d0 = _d0()
    b = ne.EditBudget(d0)
    llm = _Scripted([RuntimeError("OpenRouter HTTP 500: boom"),
                     _fence('{"edits": [{"param": "w", "index": 2, "value": 0.75}]}')])
    new, rec = ne.editor_round(llm, d0, LABELS, CEX, b, CallLog())
    assert rec["n_attempts"] == 2 and not rec["noop"]
    assert [a["outcome"] for a in rec["attempts"]] == ["client_error", "ok"]
    assert rec["attempts"][0]["reason"] == "RuntimeError: OpenRouter HTTP 500: boom"
    assert llm.users[1].rstrip().endswith("OpenRouter HTTP 500: boom")
    assert rec["attempts"][1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 0}
    assert new["w"][2] == np.float32(0.75) and rec["edits"] == [["w", 2, 0.75]]


def test_editor_round_budget_stop_propagates_and_rejections_touch_no_budget():
    d0 = _d0()
    b = ne.EditBudget(d0)
    with pytest.raises(BudgetStop):
        ne.editor_round(_Scripted([BudgetStop("cap")]), d0, LABELS, CEX, b, CallLog())
    assert b.state()["n_touched"] == 0
    # a budget-rejected attempt (25 entries) and a parse failure advance nothing; only the applied batch counts
    over = _fence('{"edits": [' + ", ".join(f'{{"param": "Wh", "index": {i}, "value": 0.0}}' for i in range(25)) + "]}")
    llm = _Scripted([over, "no fence", _fence('{"edits": [{"param": "c", "index": 0, "value": 0.1}]}')])
    _, rec = ne.editor_round(llm, d0, LABELS, CEX, b, CallLog())
    assert [a["outcome"] for a in rec["attempts"]] == ["budget", "parse", "ok"]
    assert b.state()["touched"] == [["c", 0]] and rec["n_touched"] == 1


def test_mock_editor_round_respects_budget():
    d0 = _d0(seed=3)
    b = ne.EditBudget(d0)
    llm = MockLLM(seed=5, h=H, n_candidates=4)
    cur = d0
    for _ in range(3):
        cur, rec = ne.editor_round(llm, cur, LABELS, CEX, b, CallLog())
        assert not rec["noop"] and rec["n_attempts"] == 1
        assert 0 < len(rec["edits"]) <= 8
        assert sum(int(np.sum(cur[k] != np.asarray(d0[k], dtype=np.float32))) for k in rnn.KEYS) <= rec["n_touched"]
    assert b.state()["n_touched"] <= 24 and b.state()["displacement"] <= 1.0
    # deterministic from the seed
    again, _ = ne.editor_round(MockLLM(seed=5, h=H, n_candidates=4), d0, LABELS, CEX, ne.EditBudget(d0), CallLog())
    first, _ = ne.editor_round(MockLLM(seed=5, h=H, n_candidates=4), d0, LABELS, CEX, ne.EditBudget(d0), CallLog())
    assert rnn.weights_equal(again, first)


def test_parse_edits_relaxed_fence_ignores_prose():
    good = '{"edits": [{"param": "w", "index": 1, "value": 0.5}]}'
    text = "I could not simulate the network.\n" + _fence(good) + "\nUntested."
    assert "outside" in ne.parse_edits(text)
    assert ne.parse_edits(text, fence="relaxed") == [("w", 1, 0.5)]
    assert "exactly one" in ne.parse_edits(_fence(good) + _fence(good), fence="relaxed")
