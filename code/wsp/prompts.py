"""Prompts for the editor (all arms) and the adversarial generator (arm A)."""
from __future__ import annotations

from wsp import rnn
from wsp.sandbox import ALLOWED, FORBIDDEN

FEEDBACK_HEADER = "### Counterexamples (string, true label, your prediction)"
FAILURE_HEADER = "### Your previous attempt failed"
EMPTY_RESPONSE = "The previous response was empty; reply with the constructor only."
LINE_CAP = 300


def _dataset_lines(dataset) -> list[str]:
    return [f"{s}\t{label}" for s, label in dataset]


def editor_system() -> str:
    return "\n".join([
        "ROLE: editor",
        "",
        "You repair the weights of a small recurrent discriminator over binary strings.",
        "A hidden deterministic finite automaton (DFA) over {0,1} decides which strings are accepted.",
        "The DFA is hidden; reason from labeled strings only.",
        "",
        "Task: write a Python constructor `def build_weights() -> dict[str, np.ndarray]` returning",
        "numpy float arrays of EXACTLY these shapes for the hidden size h given in the user message:",
        "  Wx (2, h), Wh (h, h), bh (h,), w (h,), c ().",
        "",
        "Architecture (h0 = zeros(h); i_t in {0, 1} is the integer value of the t-th character,",
        "so Wx[i_t] is row i_t of Wx, i.e. the one-hot input times Wx):",
        "  h_t = tanh(Wx[i_t] + Wh @ h_{t-1} + bh)",
        "  logit = w @ h_T + c ; accept iff logit > 0",
        "",
        f"Allowed imports: {', '.join(sorted(ALLOWED))}. Forbidden: {', '.join(sorted(FORBIDDEN))}",
        "(no gradient-based optimizers; no file, network or process access).",
        f"Line cap: {LINE_CAP} lines. The constructor runs once in an isolated subprocess.",
        "Return the whole constructor in a single ```python fenced block.",
    ]) + "\n"


def failure_block(kind: str, reason: str) -> str:
    """Why the previous attempt failed: audit reasons or sandbox stderr verbatim, never hidden data."""
    if kind in ("refusal", "llm_error"):
        return "\n".join([FAILURE_HEADER, EMPTY_RESPONSE])
    what = "static audit" if kind == "audit" else "sandbox run"
    return "\n".join([FAILURE_HEADER, f"The constructor failed the {what}:", reason.rstrip("\n")])


def editor_user(previous_src: str, initial_dataset, h: int, feedback=None, failure: str | None = None) -> str:
    shapes = ", ".join(f"{k} {s}" for k, s in rnn.shapes(h).items())
    parts = [
        f"h = {h}. Required shapes: {shapes}.",
        "",
        "### Previous constructor",
        "```python",
        previous_src.rstrip("\n"),
        "```",
        "",
        "### Initial labeled dataset (string<TAB>label; 1 = accept)",
        *_dataset_lines(initial_dataset),
        "",
        "Write an improved build_weights() so the discriminator labels these strings correctly",
        "and generalizes to unseen strings. Return it in a single ```python fence.",
    ]
    if feedback is not None:
        rows = [f"{s}\t{label}\t{pred}" for s, label, pred in feedback] or ["(none this round)"]
        parts += ["", FEEDBACK_HEADER, *rows]
    if failure is not None:
        parts += ["", failure]
    return "\n".join(parts) + "\n"


def generator_system() -> str:
    return "\n".join([
        "ROLE: generator",
        "",
        "You propose binary strings over {0,1} to test a discriminator against a hidden DFA.",
        "The DFA is hidden; reason from labeled strings only. You may see the discriminator's",
        "predictions on strings you proposed earlier (predictions, not true labels).",
        "Propose strings the discriminator is most likely to get wrong.",
        "Output one string per line inside a single ```strings fenced block, nothing else.",
    ]) + "\n"


def generator_user(initial_dataset, probe_predictions, length_histogram: dict[int, int]) -> str:
    total = sum(length_histogram.values())
    parts = [
        "### Initial labeled dataset (string<TAB>label; 1 = accept)",
        *_dataset_lines(initial_dataset),
        "",
        "### Current discriminator predictions on your probes (string<TAB>prediction)",
        *(_dataset_lines(probe_predictions) or ["(none yet)"]),
        "",
        "### Required number of strings per length",
        *[f"- length {L}: {c}" for L, c in sorted(length_histogram.items())],
        "",
        f"Return exactly {total} binary strings, exactly the counts above per length, no duplicates,",
        "one per line inside a ```strings fence.",
    ]
    return "\n".join(parts) + "\n"
