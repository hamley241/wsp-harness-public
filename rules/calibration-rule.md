# Calibration rule — frozen 2026-11-04 before the first real call

Frozen 2026-11-04 (OpenRouter, Fable 5.1, 16k reasoning tokens per call).

Rule, applied mechanically:
1. Try k in the order 6, 8, 10. For each k: draw the hidden DFA with
   `random_dfa(k, seed=1)`, draw the 96-string initial dataset, make ONE
   round-0 editor call (no previous constructor, no feedback), run the
   constructor through the sandbox, and score D0 on the 500-string held-out
   set. One editor call per k; a sandbox failure counts as that k's attempt
   (held-out = none) and is not retried.
2. Stop at the first k whose D0 held-out accuracy is in [0.60, 0.90]. Freeze
   that k, and reuse that D0 (same constructor, same tensors) as round 0 of
   all three arms.
3. If no k qualifies, run at k=6 with its D0 and record "calibration failed"
   with all three accuracies in results.md.
4. Everything else at the ruled defaults: h=16, rounds=3, seed=1, n_initial
   96, n_candidates 32, n_heldout 500, n_long 200, sat_timeout 30,
   train_steps 2000, reasoning 16000, max output 20000.

Spend ceiling: calibration ≤ 3 editor calls; the run ≤ 11 editor + ~9
generator calls. Expected ≤ $15 at $10/M in, $50/M out.

## Amendment 1 — 2026-11-04, before the first calibration call
Model: `anthropic/claude-opus-5.5` via OpenRouter instead of Fable 5.1, because
the API-side content filter refuses the spec's editor prompt for Fable (evidence
and bisection recorded at the time). Nothing else changes.
The three Fable calibration attempts at 19:19Z failed inside the client (content
null) before any constructor was produced; they are not counted as attempts.

## Amendment 2 — 2026-11-04 20:05Z, before any further call
Sweep 2 (records/calibration/NOTES-2026-11-04.md) found no k in the window:
k=6 solves (1.000), k=10 is chance (0.500, SAT ambiguous), k=8 produced no
scored D0 in 3 attempts (refusal; reasoning overflow; `import sys`). The
representation ceiling is known: the model's one-hot compile holds ≤ 8 states
in h=16. Recalibration, same window, same scoring:
1. k in the order 7, 8. D0 call gets 5 attempts (`--max-attempts 5`; D0 is
   not an arm, fairness is unaffected) and `--max-output-tokens 32000` so a
   16k reasoning budget cannot starve the answer. Everything else unchanged.
2. First k with D0 held-out in [0.60, 0.90] is frozen; its D0 is reused.
3. If neither qualifies: take whichever of k=7, 8 has a scored D0 closest to
   the window (ties -> 8), record "calibration failed, nearest k", and run.
   If neither has a scored D0 at all, stop and report.

## Amendment 3 — 2026-11-04 20:20Z, before any further call
Sweep 3: k=7 and k=8 both give D0 = 1.000 (records/calibration/NOTES-2026-11-04.md). k alone
cannot reach the window: this editor is an exact learner, so D0 is perfect
whenever the initial labels identify the DFA and near chance when they do not.
The amendment-2 fallback (buy the k=8 run) is withdrawn: its outcome is fixed
by its calibration row (D0_already_solves). New knob: n_initial, chosen
OFFLINE (no model calls) from the grid in records/calibration/NOTES-2026-11-04.md:
1. Admissible (k, n_initial): SAT ambiguity of the initial set is
   "ambiguous" at bound k, AND the gradient-trained baseline (h=16, 2000
   steps, seed 1) scores >= 0.60 on held-out (the interpret-spec rule treats a
   baseline at chance as "recalibrate, do not interpret arms").
   Grid result: only k=6 with n in {24, 32} is admissible.
2. Choose the largest admissible k, then the largest admissible n: k=6,
   n_initial=32. h=16, rounds 3, n_candidates 32, everything else unchanged.
3. D0 call at that config, 5 attempts. If D0 held-out >= 0.95 (the model
   happened to pick the true DFA), repeat once at n_initial=24 and use that.
   Otherwise the D0 is used as scored, inside the window or not; the score is
   recorded. The full run reuses it via --d0-from.
