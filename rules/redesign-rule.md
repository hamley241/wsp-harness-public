# Repair-redesign rule — frozen before the first editor call

Source: the redesign design record plus Amendment 1 below, which replaces
the seed-selection and feasibility clauses of P7 after the offline
preflight. Nothing here changes after the first editor call; later
amendments are appended with a timestamp and apply only to calls after
them. This file fixes what the run does.

## Question (P1)
Can an LLM make useful numeric edits to the weights of a network it did
not build, from labeled counterexamples, under a cumulative edit budget
that excludes the pilot's measured failure mode, and how do its edits
compare with projected dense Adam under the identical budget and labels?
The design does not test private symbolic inference and excludes only the
measured one-hot compile; a readout refit is a legitimate repair for both
arms.

## Fixed objects
- Hidden DFA `random_dfa(6, seed)`; `Oracle(dfa, seed)`; 32 initial labels
  from `initial_dataset(32)`; held-out 500 and long 200 from the oracle;
  f = 0.0224 (held-out noise floor, p = 0.5, n = 500).
- D0 = `trained_baseline.train(initial, h=16, seed)` (2000 Adam steps, lr
  0.01), the same for every arm of a seed.
- Query batches: `projected_gradient.query_batches(seed, initial strings,
  3, 32)`: `random.Random(seed + 1000)`, the seed's initial histogram,
  without replacement within a batch, with replacement across batches,
  generated once per seed and written to the seed record before any call;
  byte-identical for E1, E2, G and N. Re-queries are charged and add
  nothing.
- Budget (P4): 24 entries per round, 48 distinct coordinates over the run,
  max-norm displacement from D0 at most 1.0 at every point; every submitted
  entry counts whether or not its value changes. Measured compile footprint
  on every candidate seed: 321 coordinates, displacement 19.6 to 20.9
  (records/redesign/PREFLIGHT-2026-11-05.md); readout refit 17.
- Editor (P3, P8): anthropic/claude-opus-5.5 via OpenRouter through
  wsp/llm.py, reasoning max_tokens 16000, max_output_tokens 24000, provider
  default sampling, prompts `numeric_edit.editor_system/editor_user`, exactly
  one ```json fence {"edits": [{"param","index","value"}...]} and nothing
  else; three attempts per round with the failure reason fed back verbatim;
  after the third failure the round is a no-op and marked. Two independent
  realizations E1, E2 per seed with identical inputs.
- G (P5): `projected_gradient.gradient_round` each round: Adam lr 0.01, 200
  steps, fresh state, mean BCE on every label so far from the round's
  starting weights; rank |delta| descending, ties by lower flat index;
  select the first 24 that are already touched or, while fewer than 48 are
  touched, new; clip to the 1.0 ball around D0. N: no edit.
- Call order per seed (P8): E1 rounds 1–3, E2 rounds 1–3, then G and N
  offline. Every label charged so far, initial included, is delivered to
  the editor in ledger order each round, with the previous round's
  counterexamples (string, true label, current prediction).

## Seeds (Amendment 1 replaces P7's seed clause)
Seeds are the first five, in roster order (the query study's eligible main
roster 1–28, then its extension roster 29–38), that satisfy both frozen
criteria offline: trained D0 held-out accuracy at least 0.60, and G feasible
at the base budget (gain over N after three rounds greater than f on the
seed's own batches). Measured 2026-11-05 (PREFLIGHT-2026-11-05.md):
**15, 20, 21, 24, 30** (D0 0.738, 0.750, 0.728, 0.722, 0.608; G gain
+0.062, +0.054, +0.072, +0.046, +0.046). Seed 38 also qualifies and is the
reserve if a chosen seed fails before its first editor call. Excluded with
reason: 1, 4, 6, 25, 28 (D0 ≥ 0.60 but G infeasible at the base budget; the
record's one permitted doubling of the per-round budget left 4 and 6
infeasible at 48/48 and 48/96, and dense unprojected Adam from these D0s
also fails to improve held-out while retraining from scratch on the same
128 labels does, so these are a hard-repair stratum for a follow-up, not
a budget problem); 2, 3, 7, 10, 12, 14, 16, 18, 22, 23, 26, 29, 31–37 (D0
below 0.60). The base budget 24/48/1.0 is kept; selection by feasibility
replaces the record's budget-widening schedule, and this is Amendment 1 to
the record, filed before any editor call.

## Outcome vocabulary (P7), descriptive feasibility pilot
Per seed gE = mean(E1, E2 held-out after round 3) − N; gG = G − N; d = gE −
gG. Precedence: `no_repair` when gE ≤ f on at least 4 of 5 seeds; else
`E_beats_gradient` when d > f on all 5; `E_below_gradient` when d < −f on
all 5; `E_matches_gradient` when |d| ≤ f on at least 4 of 5; else `mixed`.
Under independent fair signs with no ties the all-5 rules fire with
probability 1/32 each. A seed whose E or G arm fails or is stopped by the
cap is `failed`, reported, never imputed; fewer than 4 analyzable seeds →
`no_result`. Descriptive reports (P6): blocks touched per arm and round,
E–G coordinate overlap, readout-only E edits, per-attempt refusals,
parse failures and budget rejections.

## Spend (P8)
Per attempt about $0.37 (3,000 prompt + 18,000 output tokens at $4/$20 per
million). Expected about 39 attempts, about $15. Hard worst case 90
attempts at the maxima (4,000 + 24,000 tokens), about $0.50 each, about
$45. Cap $50, enforced per call by the reservation guard; a cap stop makes
the affected seed `failed`. All provider-reported usage is counted,
refusals included.

## Reading
study.json and study.md are read mechanically with this file's outcome
vocabulary; the result is a descriptive feasibility finding about numeric
editing under this budget on these seeds, conditional on the frozen
comparator G.

## Amendment 2 — 2026-11-05, before any editor call

The seven items below are adopted. Where they conflict with the text
above, this amendment governs.

1. Population. The target population is the comparator-responsive stratum:
   the first five query-study eligible seeds, in main-then-extension roster
   order, whose observed D0 accuracy on the study held-out set is at least
   0.60 and whose observed round-3 gain from this exact G implementation,
   on the frozen batches and the same held-out set, exceeds 0.0224.
   Findings are conditional on this selected G-feasible stratum and do not
   estimate E-versus-G performance over the original eligible roster or the
   excluded hard-repair stratum. After G-based selection the 1/32 figure is
   a reference under independent fair signs, not a calibrated error rate.
2. Compile gate, measured 2026-11-05 at tolerance 1e-6 on the seeds
   actually used: one-hot compile footprint 321 coordinates on every seed;
   max-norm displacement 20.52 (15), 20.56 (20), 39.26 (21), 20.57 (24),
   21.69 (30), 29.08 (reserve 38). All exceed 48 and 1.0; the gate passes.
   No real call starts if any seed used has footprint at most 48 or
   displacement at most 1.0.
3. Production configuration. Seeds run in the order 15, 20, 21, 24, 30.
   Prompts and code semantics are pinned to the commit recorded in
   study_meta.json before the first call; the runner refuses a different
   model, token limits, budget, seed order or sampling controls than this
   file states. Any earlier seed example is superseded by this order.
4. Reserve. Seed 38 may replace a seed only for a logged pre-call
   operational failure unrelated to any observed D0, G, N, held-out or
   long-set value. A criterion or compile-gate failure aborts the run; it
   never triggers substitution.
5. Round and failure semantics. Batch r is charged before edit r; its
   counterexamples are computed from the round-start weights. A rejected
   attempt consumes no edit budget. Provider or client failures other than
   BudgetStop consume an attempt and are retried within the three; BudgetStop
   fails the seed; an unexpected local invariant error fails the seed
   immediately and is reported.
6. Analysis and reporting. With exactly four analyzable seeds the all-five
   beat and below predicates are false; no_repair and matches then require
   all four. Reported per round: held-out and long accuracy, actual float32
   changes from round start, cumulative touched set and displacement,
   separate cumulative E1–G and E2–G coordinate intersections, and every
   attempt category and reason. A readout-only round is a successful,
   non-empty round whose edits are all in w or c. Diagnostics never affect
   inclusion or outcome.
7. Implementation gate, before the first call: parse_edits rejects
   duplicate JSON member names; the spend wrapper preserves per-call usage
   and finish reason; N's frozen batches are charged and recorded for
   ledger parity; the reading of the outcome is mechanical under the
   precedence above. The phrase "regardless of budget" in the
   seed clause reads "under the tested projected budgets and the
   dense-continuation procedure".

## Amendment 3 — relaxed-fence rerun. FILED 2026-11-05 AFTER THE RERUN: a freeze violation, recorded as such

A rerun of the editor arms with `--fence relaxed` (parse_edits(fence="relaxed"):
prose around exactly one json fence is ignored; the three-attempt retry is
unchanged) was launched on 2026-11-05 at commit 41f43a1 on seeds 1, 15, 20,
21, 24 with every other setting identical and a $10 cap, without
--production since the guard pins the strict configuration. This amendment
was meant to be appended before the launch; the command that should have
appended it aborted on a failing test before the append, and the rerun
therefore ran with its rule stated only in the launch log and the design
record, not in this file. Status: the rerun is reported beside the
strict-fence result (run-relaxed/) and never replaces it; the primary
outcome of the redesign stays the strict-fence production-order result; the
rerun's own result is descriptive.
