from wsp.arms import target_histogram
from wsp.prompts import (EMPTY_RESPONSE, FAILURE_HEADER, FEEDBACK_HEADER, editor_system, editor_user,
                         failure_block, generator_system, generator_user)
from wsp.sandbox import mock_constructor_source

DATA = [("0", 1), ("10", 0), ("011", 1)]


def test_failure_block_is_appended_after_the_feedback_block():
    src = mock_constructor_source(16, 1)
    feedback = [("1101", 1, 0)]
    plain = editor_user(src, DATA, 16, feedback)
    assert editor_user(src, DATA, 16, feedback, failure=None) == plain and FAILURE_HEADER not in plain
    assert editor_user(src, DATA, 16, None, failure=None) == editor_user(src, DATA, 16, None)
    block = failure_block("audit", "import not allowed: sys\nname not allowed: os")
    with_failure = editor_user(src, DATA, 16, feedback, failure=block)
    assert with_failure.startswith(plain.rstrip("\n")) and with_failure.endswith(block + "\n")
    assert with_failure.index(FEEDBACK_HEADER) < with_failure.index(FAILURE_HEADER)
    assert "import not allowed: sys\nname not allowed: os" in block and "audit" in block
    sb = failure_block("sandbox", "Traceback ...\nZeroDivisionError\nreturn code: 1")
    assert sb.startswith(FAILURE_HEADER) and "return code: 1" in sb and "sandbox" in sb
    for kind in ("refusal", "llm_error"):
        fb = failure_block(kind, "OpenRouter refusal: secret details")
        assert fb == f"{FAILURE_HEADER}\n{EMPTY_RESPONSE}" and "secret" not in fb


def test_editor_prompts_differ_only_by_feedback_block():
    src = mock_constructor_source(16, 1)
    ab = editor_user(src, DATA, 16, [("1101", 1, 0), ("00", 0, 1)])
    d = editor_user(src, DATA, 16, None)
    assert FEEDBACK_HEADER in ab and FEEDBACK_HEADER not in d
    assert ab[:ab.index(FEEDBACK_HEADER)].rstrip() == d.rstrip()
    block = ab[ab.index(FEEDBACK_HEADER):]
    assert "1101\t1\t0" in block and "00\t0\t1" in block
    assert FEEDBACK_HEADER in editor_user(src, DATA, 16, [])  # empty feedback still gets the block
    assert "0\t1\n10\t0\n011\t1\n" in d and src.rstrip() in d and "h = 16" in d


def test_system_prompts_have_roles_and_contents():
    es = editor_system()
    assert "ROLE: editor" in es and "build_weights" in es and "tanh" in es and "300" in es
    assert "hidden" in es and "numpy" in es
    # Must match rnn.step: Wx is (2, h) and is indexed by the integer symbol, not a one-hot vector.
    assert "h_t = tanh(Wx[i_t] + Wh @ h_{t-1} + bh)" in es and "Wx[x_t]" not in es
    assert "ROLE: generator" in generator_system()
    gu = generator_user(DATA, [("111", 1)], {3: 2, 7: 1})
    assert "- length 3: 2\n- length 7: 1\n" in gu and "111\t1" in gu and "exactly 3" in gu
    assert "```strings" in gu


def test_target_histogram_sums_to_n():
    strings = [s for s, _ in DATA] * 4 + ["1111"]
    for n in (1, 5, 13, 32, 100):
        hist = target_histogram(strings, n)
        assert sum(hist.values()) == n and all(c > 0 for c in hist.values())
    assert target_histogram(["0"] * 8 + ["00"] * 8, 32) == {1: 16, 2: 16}
