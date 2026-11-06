"""Numeric editor interface and budget accounting (redesign decision record P3, P4, P8).

The editor returns JSON weight edits addressed by (param, flat index within the param); EditBudget enforces
the per-round, distinct-coordinate and max-norm limits cumulatively over a run.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from wsp import rnn
from wsp.llm import _FENCE, BudgetStop, LLMError, LLMRefusal

FEEDBACK_HEADER = "### Counterexamples from the last round (string, true label, current prediction)"
FAILURE_HEADER = "### Your previous attempt failed"
H = 16


def param_sizes(h: int = H) -> dict[str, int]:
    return {k: int(np.prod(s, dtype=int)) for k, s in rnn.shapes(h).items()}


def flat_index(param: str, index: int, h: int = H) -> int:
    """Position of (param, index) in the fixed global order Wx, Wh, bh, w, c (0..320 for h=16)."""
    sizes = param_sizes(h)
    if param not in sizes or not 0 <= index < sizes[param]:
        raise ValueError(f"unknown coordinate {param}[{index}]")
    return sum(sizes[k] for k in rnn.KEYS[:rnn.KEYS.index(param)]) + index


def coordinate_name(param: str, index: int, h: int = H) -> str:
    """Human name with the row-major 2-D index for matrices, e.g. Wh[17] -> 'Wh[1,1]'."""
    shape = rnn.shapes(h)[param]
    if len(shape) == 2:
        return f"{param}[{index // shape[1]},{index % shape[1]}]"
    return f"{param}[{index}]" if shape else param


def coordinate_names(h: int = H) -> list[str]:
    return [coordinate_name(k, i, h) for k in rnn.KEYS for i in range(param_sizes(h)[k])]


def _f32(weights: dict) -> dict[str, np.ndarray]:
    return {k: np.array(v, dtype=np.float32) for k, v in weights.items()}


class EditBudget:
    """P4: per_round entries per round, distinct coordinates over the run, max-norm displacement from d0."""

    def __init__(self, d0: dict, per_round: int = 24, distinct: int = 48, max_norm: float = 1.0):
        self.h = int(np.asarray(d0["Wh"]).shape[0])
        self.d0 = _f32(rnn.check_weights(d0, self.h))
        self.per_round = per_round
        self.distinct = distinct
        self.max_norm = max_norm
        self.touched: set[tuple[str, int]] = set()
        self.displacement = 0.0

    def check(self, edits, current) -> str | None:
        """None when the batch is admissible against `current`, else the reason. Every entry counts."""
        if len(edits) > self.per_round:
            return f"{len(edits)} entries exceed the per-round limit of {self.per_round}"
        sizes = param_sizes(self.h)
        seen: set[tuple[str, int]] = set()
        for param, index, value in edits:
            if param not in sizes:
                return f"unknown param {param!r}"
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < sizes[param]:
                return f"index {index!r} out of range for {param} (0..{sizes[param] - 1})"
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                return f"non-finite or non-numeric value {value!r} for {param}[{index}]"
            if (param, index) in seen:
                return f"duplicate coordinate {param}[{index}] in one batch"
            seen.add((param, index))
            delta = abs(float(np.float32(value)) - float(self.d0[param].flat[index]))
            if delta > self.max_norm:
                return (f"{param}[{index}] = {value} is {delta:.6g} from D0, over the max-norm limit "
                        f"{self.max_norm}")
        new = seen - self.touched
        if len(self.touched) + len(new) > self.distinct:
            return (f"{len(new)} new coordinates would bring the touched set to {len(self.touched) + len(new)}, "
                    f"over the distinct limit of {self.distinct}")
        return None

    def apply(self, edits, current) -> dict[str, np.ndarray]:
        reason = self.check(edits, current)
        if reason is not None:
            raise ValueError(reason)
        out = _f32(current)
        for param, index, value in edits:
            out[param].flat[index] = np.float32(value)
            self.touched.add((param, index))
        self.displacement = max(float(np.max(np.abs(out[k].astype(np.float64) - self.d0[k]), initial=0.0))
                                for k in rnn.KEYS)
        return out

    def state(self) -> dict:
        return {"per_round": self.per_round, "distinct": self.distinct, "max_norm": self.max_norm,
                "touched": sorted([p, i] for p, i in self.touched), "n_touched": len(self.touched),
                "displacement": self.displacement}


class _DuplicateMember(ValueError):
    pass


def _no_duplicate_members(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateMember(f"duplicate JSON member name {key!r}")
        out[key] = value
    return out


def parse_edits(text: str, fence: str = "strict"):
    """list of (param, index, value) from exactly one ```json fence {"edits": [...]}, else a reason string.

    fence="strict" (rule P3): nothing may appear outside the single fence. fence="relaxed" (rule
    Amendment 3, the rerun): prose around the single fence is ignored; still exactly one json fence.
    """
    blocks = _FENCE.findall(text)
    if len(blocks) != 1 or blocks[0][0] != "json":
        return f"expected exactly one ```json fence, found {len(blocks)} fence(s) {[t for t, _ in blocks]}"
    if fence == "strict" and _FENCE.sub("", text).strip():
        return "nothing may appear outside the single ```json fence"
    if fence not in ("strict", "relaxed"):
        raise ValueError(f"fence must be strict or relaxed, got {fence!r}")
    try:
        data = json.loads(blocks[0][1], object_pairs_hook=_no_duplicate_members)
    except _DuplicateMember as e:
        return str(e)
    except ValueError as e:
        return f"invalid JSON: {e}"
    if not isinstance(data, dict) or set(data) != {"edits"}:
        return 'top-level JSON must be an object with exactly the key "edits"'
    if not isinstance(data["edits"], list):
        return '"edits" must be a list'
    out = []
    for n, e in enumerate(data["edits"]):
        if not isinstance(e, dict) or set(e) != {"param", "index", "value"}:
            return f"edit {n} must be an object with exactly the keys param, index, value"
        param, index, value = e["param"], e["index"], e["value"]
        if param not in rnn.KEYS:
            return f"edit {n}: unknown param {param!r} (one of {', '.join(rnn.KEYS)})"
        if not isinstance(index, int) or isinstance(index, bool):
            return f"edit {n}: index must be an integer, got {index!r}"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"edit {n}: value must be a number, got {value!r}"
        out.append((param, index, float(value)))
    return out


def editor_system() -> str:
    return "\n".join([
        "ROLE: editor-numeric",
        "",
        "You repair the weights of a small recurrent discriminator over binary strings by editing",
        "individual weight values. A hidden deterministic finite automaton (DFA) over {0,1} decides which",
        "strings are accepted. The DFA is hidden; reason from labeled strings only.",
        "",
        "Architecture (h0 = zeros(h); i_t in {0, 1} is the integer value of the t-th character):",
        "  h_t = tanh(Wx[i_t] + Wh @ h_{t-1} + bh)",
        "  logit = w @ h_T + c ; accept iff logit > 0",
        "Params and shapes for h = 16: Wx (2, 16), Wh (16, 16), bh (16,), w (16,), c ().",
        "A coordinate is (param, index) with index the row-major flat index within the param:",
        "  Wx[r, k] -> index 16*r + k ; Wh[j, k] -> index 16*j + k ; c -> index 0.",
        "",
        "Budget (cumulative over the run): at most 24 entries per round; at most 48 distinct coordinates",
        "over all rounds; every edited value must stay within 1.0 (max-norm) of its original value in D0.",
        "Every submitted entry consumes the per-round and distinct budgets even if it equals the current value.",
        "",
        'Output exactly one JSON object inside a single ```json fence and nothing else:',
        "```json",
        '{"edits": [{"param": "Wh", "index": 17, "value": -0.25}, ...]}',
        "```",
        "param is one of Wx, Wh, bh, w, c; index is an integer; value is a number (parsed as float32).",
        "Two entries with the same (param, index) fail the attempt.",
    ]) + "\n"


def weight_lines(weights: dict) -> list[str]:
    """One line per coordinate, `param[index] = value` as float32, in the fixed order."""
    w = _f32(weights)
    return [f"{k}[{i}] = {w[k].flat[i]}" for k in rnn.KEYS for i in range(w[k].size)]


def editor_user(weights, labels, counterexamples, budget_state: dict, failure: str | None = None) -> str:
    remaining_distinct = budget_state["distinct"] - budget_state["n_touched"]
    parts = [
        "### Current weights (float32; param[index] = value)",
        *weight_lines(weights),
        "",
        "### All labels so far (ledger order; string<TAB>label; 1 = accept)",
        *[f"{s}\t{label}" for s, label in labels],
        "",
        FEEDBACK_HEADER,
        *([f"{s}\t{label}\t{pred}" for s, label, pred in counterexamples] or ["(none)"]),
        "",
        "### Remaining budget",
        f"- entries this round: {budget_state['per_round']}",
        f"- distinct coordinates not yet touched that you may still add: {remaining_distinct}",
        f"  (touched so far: {', '.join(f'{p}[{i}]' for p, i in budget_state['touched']) or 'none'})",
        f"- max-norm limit from D0: {budget_state['max_norm']}; current displacement: {budget_state['displacement']}",
        "",
        "Return the edits that make the discriminator label the strings above correctly and generalize,",
        'as exactly one ```json fence containing {"edits": [{"param", "index", "value"}, ...]}.',
    ]
    if failure is not None:
        parts += ["", FAILURE_HEADER, failure.rstrip("\n")]
    return "\n".join(parts) + "\n"


def editor_round(llm, weights, labels, counterexamples, budget: EditBudget, calls, max_attempts: int = 3,
                 fence: str = "strict"):
    """Up to max_attempts editor calls; retries carry the previous failure reason verbatim.

    Returns (new_weights float32, record). After the last failure the round is a no-op (weights unchanged).
    Rule Amendment 2 item 5: a provider or client failure other than BudgetStop consumes an attempt
    (category client_error) and is retried; BudgetStop propagates. A rejected attempt touches no budget.
    """
    system = editor_system()
    attempts: list[dict] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    failure = None
    applied: list = []
    new_weights = _f32(weights)
    for n in range(1, max_attempts + 1):
        t0 = time.monotonic()
        rec = {"attempt": n, "outcome": None, "reason": "", "finish_reason": None, "n_edits": 0}
        try:
            text = calls.complete(llm, system, editor_user(weights, labels, counterexamples, budget.state(), failure))
        except LLMError as e:
            kind, reason = ("refusal" if isinstance(e, LLMRefusal) else "llm_error"), str(e)
            rec["finish_reason"] = e.finish_reason
        except (BudgetStop, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001 - transport/client failure: one attempt, retried
            kind, reason = "client_error", f"{type(e).__name__}: {e}"
        else:
            edits = parse_edits(text, fence)
            if isinstance(edits, str):
                kind, reason = "parse", edits
            else:
                rec["n_edits"] = len(edits)
                reason = budget.check(edits, weights)
                if reason is not None:
                    kind = "budget"
                else:
                    kind, reason = "ok", ""
                    new_weights = budget.apply(edits, weights)
                    applied = [[p, i, float(np.float32(v))] for p, i, v in edits]
        last = getattr(llm, "last_usage", None) or {}
        for key in usage:
            usage[key] += int(last.get(key) or 0)
        if rec["finish_reason"] is None:
            rec["finish_reason"] = last.get("finish_reason")
        rec.update(outcome=kind, reason=reason[:300], elapsed_s=time.monotonic() - t0,
                   usage={key: int(last.get(key) or 0) for key in usage})
        attempts.append(rec)
        if kind == "ok":
            break
        failure = reason
    state = budget.state()
    record = {"attempts": attempts, "n_attempts": len(attempts), "noop": kind != "ok", "edits": applied,
              "touched": state["touched"], "n_touched": state["n_touched"],
              "displacement": state["displacement"], "usage": usage}
    return new_weights, record
