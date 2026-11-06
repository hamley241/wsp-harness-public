# Comparator calibration for a possible execution-enabled-editor program (frozen 2026-11-05, before any run)

Purpose. Before any design for an editor that can execute the network is
written or funded, measure whether a black-box optimiser with
forward-pass-only access repairs the trained network at all under a
realistic evaluation budget, and at which budget. This is a calibration
of the comparator, not an experiment on an editor: it has no model calls
and no API spend. Its reading decides whether an execution-enabled
program is proposed with these budgets, and nothing else. Rule filed
before the first run; amendments are appended with a timestamp; the
reading is applied mechanically.

## Construction (identical to wsp/repair_study.run_seed)
Per seed: hidden DFA `random_dfa(6, seed)`, oracle, 32 initial labels,
500 held-out strings (lengths 1 to 12) and 200 long strings (13 to 24)
never used by any optimiser, D0 = gradient-trained h=16 tanh RNN on the
initial labels (2000 steps, float32), three random query batches of 32
from `query_batches(seed, initial, 3, 32)`, labels accumulated by
`add_labels` (first occurrence). Three rounds; batch r is charged and
added to the labelled set before round r's search. N = no edit = D0.
f = 0.0224 (held-out noise floor at n = 500).

## Development seeds
4, 6, 25, 28: the eligible roster (trained D0 held-out ≥ 0.60, from
records/redesign/PREFLIGHT-2026-11-05.md) minus every seed used
as a test seed in any run of the repair redesign or the second-model
validation (1, 15, 20, 21, 24, 30). They are development seeds: any
program designed after this calibration selects its test seeds afresh and
independently of these results. Seeds 4 and 6 were G-infeasible under the
edit budget by continuation from D0; this is known and not a selection.

## Access and objective
The optimiser sees only: the current labelled set, and the logit of any
candidate weight vector on any string (forward pass). One evaluation =
the objective of one candidate on the full current labelled set (mean
binary cross-entropy of the logits, `trained_baseline.loss`); the
evaluation count is the rollout budget B per round, and the forward-pass
count (evaluations × labels) is reported beside it. No gradients, no
held-out access, no oracle access beyond the charged batches.

## Optimisers (three, fixed; search randomness from `Random(seed*1000 + replicate)`)
- `rs-random24`: the redesign's edit budget (24 changes per round, 48
  distinct coordinates over the run, every coordinate within 1.0 of D0).
  At round start a support of 24 coordinates is drawn uniformly from the
  coordinates still allowed under the 48-distinct cap; (1+1) hill climbing
  on that support with Gaussian steps σ = 0.1, clipped to the max-norm
  box, accepting a candidate iff its objective is strictly lower.
- `rs-probe24`: as `rs-random24`, but the support is the 24 allowed
  coordinates with the largest |Δobjective| under a single finite
  difference (+0.1) per allowed coordinate; the probe evaluations count
  against the round budget B (a budget smaller than the probe cost makes
  the round a no-op, recorded).
- `es-dense`: no per-round or distinct limit, only the max-norm box
  (every coordinate within 1.0 of D0); (1+1)-ES with dense Gaussian
  steps, σ0 = 0.05, one-fifth success rule (σ ×1.5 after a success,
  ×0.85 after a failure, bounded to [1e-4, 1.0]).
Budgets per round: B ∈ {100, 1000, 10000} evaluations. Replicates: 2 per
(seed, optimiser, B). Reference, reported and never a comparator here:
projected dense Adam under the edit budget (`gradient_round`, 200 steps),
which uses gradients the optimisers do not have.

## Endpoints (reported per cell = seed × optimiser × B × replicate, per round)
Held-out accuracy and long accuracy of the round-end weights; labelled-set
objective and accuracy; evaluations used; forward passes; coordinates
changed from round start; cumulative touched set; max displacement from D0.

## Reading (mechanical)
A cell (optimiser, B) repairs a seed iff the mean over its 2 replicates of
(held-out accuracy after round 3 − D0 held-out) > f. For each optimiser and
B, count the development seeds repaired (of 4). Outcome vocabulary, in
precedence order:
- `calibrated_at_100`, `calibrated_at_1000`, `calibrated_at_10000`: the
  smallest B at which some optimiser repairs at least 3 of 4 seeds; name
  the optimiser(s).
- `not_calibrated`: no optimiser repairs 3 of 4 at any B.
Under `not_calibrated` an execution-enabled program is not proposed with
these budgets or optimisers; a different comparator or budget would be a
new calibration with its own rule. Under `calibrated_at_B` the design
document fixes that B (or larger) and that optimiser as the primary
comparator, and a one-seed editor pilot is the next
proposal, costed separately. Descriptive quantities (overfitting of the
labelled objective, the budgeted versus dense contrast, the gradient
reference) never change the outcome.

## Not tested
Any language model; any seed in the test roster; budgets above 10000 per
round; population-based ES or CMA-ES; coordinate supports chosen by
anything but uniform draw or a single finite-difference probe.

## Amendment 1 — 2026-11-05 13:05Z, filed before any run; the thirteen items below replace the corresponding sentences above
1. For a proposed 24/48/1.0 editor, only `rs-random24` and `rs-probe24` may satisfy the calibration gate; `es-dense` is descriptive unless the later editor receives the identical dense constraint.
2. `rs-probe24` counts one incumbent evaluation plus one box-clipped +0.1 evaluation per eligible coordinate, reuses the incumbent value, no-ops iff B < 1 + #eligible, and gives the exact remaining budget to hill climbing.
3. When fewer than 24 new coordinates remain under the 48-distinct cap, ranked or random support selection may include at most that many new coordinates and fills the remainder from previously touched coordinates.
4. `es-dense` applies σ × 1.5 after a success and σ × 1.5^(−1/4) after a failure, bounded to [1e-4, 1.0], which has the one-fifth equilibrium (the earlier × 0.85 targeted about 29% success).
5. Support draws use one persistent `random.Random(S)` and Gaussian draws one persistent `np.random.default_rng(S)` per cell, S = 1000·seed + replicate.
6. Three replicates; a seed is repaired by a cell only if at least two of its three individual held-out gains exceed f and their mean exceeds f.
7. f = 0.0224 is a frozen practical-effect threshold, not a significance or paired-gain noise bound.
8. Any incomplete development seed produces `no_result`; `calibrated_at_B` is development tuning, fixes exactly that B, and does not imply that larger budgets work.
9. If several gate-eligible optimisers qualify at the smallest B, choose by repaired-seed count, then four-seed mean gain, then lexical optimiser name.
10. The development roster was selected by D0 eligibility and prior-test exclusion, not by G success; the known production-semantics G gains are disclosed: seed 4 −0.064, 6 +0.006, 25 +0.006, 28 +0.030 (records/redesign/PREFLIGHT-2026-11-05.md, production recheck).
11. Before any later test run, fresh seed IDs, the sampling frame, and any D0 eligibility rule are frozen without comparator or gradient screening, and every exclusion or failure is reported.
12. Repair means endpoint behavioural improvement; any in-box route, including synthesis or recompilation, counts, because the box excludes the documented one-hot compiler where measured but not every dense compilation route.
13. B is a full-labelled-set BCE-call budget; every uncached call, including clipped duplicates and probes, counts; the report separately totals string logits by search and diagnostic purpose, oracle labels, D0 and reference work, and CPU wall time.
Runtime note from the check: about 0.8 million objective calls for the grid, 1 to 4 serial CPU hours at two replicates, half as much again at three; per-cell checkpointing required, vectorisation optional.
