# Repair redesign — results (run 2026-11-05)

Question (rules/redesign-rule.md): can an LLM make useful numeric edits to
the weights of a network it did not build, from labeled counterexamples,
under a cumulative edit budget (24 entries per round, 48 distinct
coordinates, max-norm 1.0 from D0) that excludes the pilot's measured
compile route, and how do its edits compare with projected dense Adam
under the identical budget and labels? Rule frozen with Amendments 1 and
2 before the first call; the rule was amended before the first call with
amendments, all appended. Population: the comparator-responsive stratum
(seeds 15, 20, 21, 24, 30), chosen offline by trained-D0 accuracy at
least 0.60 and gradient feasibility at the base budget; findings are
conditional on that stratum.

## Design as run
- D0: gradient-trained baseline on the 32 initial labels (h=16, 321
  parameters). Arms on the same D0, seed, three frozen random batches of
  32 labels, and held-out sets: E1, E2 (two independent runs of the Opus
  5.5 editor returning JSON numeric edits only), G (projected dense Adam
  under the same budget), N (no edit).
- Editor: anthropic/claude-opus-5.5 via OpenRouter, reasoning 16,000,
  max output 24,000, three attempts per round with the failure reason fed
  back; a round is a no-op after three failures.
- Endpoint: held-out accuracy (500 strings) after round 3; f = 0.0224.

## Result
| seed | D0 = N | E1 | E2 | G | gE (mean E − N) | gG (G − N) | d = gE − gG |
|---|---|---|---|---|---|---|---|
| 15 | 0.738 | 0.780 | 0.828 | 0.856 | +0.066 | +0.118 | −0.052 |
| 20 | 0.750 | 0.756 | 0.760 | 0.784 | +0.008 | +0.034 | −0.026 |
| 21 | 0.728 | 0.728 | 0.728 | 0.780 | 0.000 | +0.052 | −0.052 |
| 24 | 0.722 | 0.736 | 0.722 | 0.766 | +0.007 | +0.044 | −0.037 |
| 30 | 0.608 | 0.608 | 0.602 | 0.652 | −0.003 | +0.044 | −0.047 |
| 1 (correction) | 0.692 | 0.688 | 0.688 | 0.720 | −0.004 | +0.028 | −0.032 |

Seed selection correction: the feasibility
preflight that chose the seeds ran on pre-production label semantics; under
the production code the frozen "first five" are 1, 15, 20, 21, 24. Seed 1
was then run under the identical configuration (run-seed1/). Primary
outcome on the production-order set 1, 15, 20, 21, 24: **no_repair** (gE at
most f on 4 of 5). The as-run set 15, 20, 21, 24, 30, retained separately:
**no_repair** as well. Secondary, reported not labelled: d below −f on every
seed in both sets, which would read E_below_gradient had no_repair not taken
precedence.

## What the editor did
- Accepted edits per round were tiny: 26 of 29 accepted attempts changed
  one or two coordinates, almost always the output bias c; the largest
  edit was 10 coordinates (seed 15, E2, round 2, which produced the only
  gain above f, +0.090). G used its full 48-coordinate budget on every
  seed.
- Attempts: 64 editor attempts, 29 accepted, 32 rejected for text
  outside the single json fence (the strict P3 format), 3 returned no
  content at the output limit. **No round was accepted on its first
  attempt**: every accepted edit followed a rejection and the fed-back
  reason "nothing may appear outside the single json fence". The editor
  arm therefore ran under the retry condition, not the prompt as
  frozen. The rejected first-attempt texts are stored in the seed records;
  replayed offline with a relaxed fence rule (REPLAY-first-attempts.md), all
  32 parse and propose edits of the same size as the accepted retries (11
  no-edit, 14 one coordinate, one of 24), and the model's prose states it
  could not simulate the network or recover the automaton by hand. The
  retry changed the format, not the behaviour; the 24-coordinate proposal
  was never evaluated.
- Median editor call 13 s; the 16,000-token reasoning budget was rarely
  used.

Reliability: 5 of 5 seeds analyzable, 0 failed arms, 0 budget rejections,
0 refusals. Spend $5.32 (429,739 prompt, 180,039 completion tokens)
for the main run plus $0.67 for seed 1 and $2.60 for the relaxed-fence rerun, $8.59 in all, against the $50 cap.

Raw: run/ (per-seed JSON with every attempt, usage.jsonl, study_meta.json
at commit 0b8d55d, study.json, study.md). Preflight and compile gate:
PREFLIGHT-2026-11-05.md.

## Relaxed-fence rerun (rule Amendment 3, run-relaxed/)
Editor arms rerun on the production-order seeds with prose around the
single json fence accepted; everything else identical. 30 accepted rounds
from 31 attempts: 29 accepted on the first attempt, one output-limit
error followed by a successful retry. The format artifact is gone.
Procedural note: the amendment text was filed in rules/redesign-rule.md
only after the rerun (a freeze violation, recorded there); the rerun is
descriptive beside the strict result and does not replace it.

| seed | D0 = N | E1 | E2 | G | gE | gG | d |
|---|---|---|---|---|---|---|---|
| 1 | 0.692 | 0.688 | 0.688 | 0.720 | −0.004 | +0.028 | −0.032 |
| 15 | 0.738 | 0.800 | 0.814 | 0.856 | +0.069 | +0.118 | −0.049 |
| 20 | 0.750 | 0.756 | 0.756 | 0.784 | +0.006 | +0.034 | −0.028 |
| 21 | 0.728 | 0.728 | 0.728 | 0.780 | 0.000 | +0.052 | −0.052 |
| 24 | 0.722 | 0.734 | 0.734 | 0.766 | +0.012 | +0.044 | −0.032 |

Outcome under the frozen precedence: **no_repair** (gE at most f on 4 of
5; d below −f on all 5). Accepted proposals: 10 of 30 rounds submitted no edit, 16 submitted a
single coordinate; the largest effective change in a round was 6
coordinates and the largest cumulative touched set 12 (seed 15, E2),
against 10 and 11 in the strict run. Spend $2.60.
Reported beside the strict-fence result; the primary outcome is unchanged.

## Offline scoring of the rejected proposals (OFFLINE-SCORING-rejected-proposals.md)
All 39 stored parse-rejected proposals fit the budget when applied to
their reconstructed round-start weights. Five would have gained more than
f, all on seed 15 (max +0.094; smaller positive gains, at most +0.012,
occurred on seeds 1, 20, 21 and 24), the seed where the deployed editor also
succeeded; one, the 24-coordinate proposal on seed 21, would have lost
0.254; 33 were within ±f, 14 of them no-op proposals and 20 with zero gain. Descriptive only: post-hoc,
scored repeatedly on the same 500-string held-out set, no multiplicity
correction; these are not validated discoveries.

## Reading (mechanical, under the frozen rule)
Outcome no_repair on the primary production-order set and on the as-run
set: given the network's weights as numbers, every label so far, and the
counterexamples, the editor deployed only small edits, mostly a single
output-bias nudge, never used more than 11 of its 48 coordinates, and in
its own words declined to attempt a repair it could not simulate. The
gradient comparator under the identical budget improved every seed. The
finding is conditional on the comparator-responsive stratum, the frozen
budget and comparator, and this editor and provider; it is not evidence
that the budget prevented repair, and it says nothing about editors that
can execute the network.
