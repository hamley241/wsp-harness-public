# Redesign preflight measurements — 2026-11-05 (offline, no model calls)

Compile footprint: the pilot's one-hot compiler (records/calibration/opus-d0-k6-attempt1.py, K=10) applied to each seed's true 6-state DFA, compared with that seed's gradient-trained D0 (wsp/trained_baseline.train on the 32 initial labels, h=16).

| seed | D0 held-out | compile held-out | coordinates differing (>1e-6) | max-norm displacement | readout-only coords |
|---|---|---|---|---|---|
| 1 | 0.692 | 1.000 | 321 | 19.98 | 17 |
| 2 | 0.566 | 1.000 | 321 | 20.23 | 17 |
| 3 | 0.432 | 1.000 | 321 | 20.63 | 17 |
| 4 | 0.670 | 1.000 | 321 | 20.87 | 17 |
| 6 | 0.658 | 1.000 | 321 | 19.64 | 17 |

Trained-D0 held-out accuracy over the query study's eligible main roster: 1 0.692, 2 0.566, 3 0.432, 4 0.670, 6 0.658, 7 0.568, 10 0.546, 12 0.472, 14 0.504, 15 0.738, 16 0.480, 18 0.564, 20 0.750, 21 0.728, 22 0.562, 23 0.532, 24 0.722, 25 0.608, 26 0.560, 28 0.674. Seeds with D0 >= 0.60: 1, 4, 6, 15, 20, 21, 24, 25, 28. First five: 1, 4, 6, 15, 20.

## Feasibility preflight for arm G (record P7 / C3), measured 2026-11-05

G = projected dense Adam under the budget on the seed's own three random
batches; gain = G − N after round 3; feasible iff gain > f = 0.0224.

Base budget 24 entries/round, 48 distinct, max-norm 1.0:

| seed | D0 | G r3 | gain | feasible |
|---|---|---|---|---|
| 1 | 0.692 | 0.704 | +0.012 | no |
| 4 | 0.670 | 0.626 | −0.044 | no |
| 6 | 0.658 | 0.642 | −0.016 | no |
| 15 | 0.738 | 0.800 | +0.062 | yes |
| 20 | 0.750 | 0.804 | +0.054 | yes |
| 21 | 0.728 | 0.800 | +0.072 | yes |
| 24 | 0.722 | 0.768 | +0.046 | yes |
| 25 | 0.608 | 0.604 | −0.004 | no |
| 28 | 0.674 | 0.666 | −0.008 | no |
| 29 | 0.524 | 0.578 | +0.054 | yes (D0 < 0.60) |
| 30 | 0.608 | 0.654 | +0.046 | yes |
| 38 | 0.668 | 0.740 | +0.072 | yes |

Seeds 31–37: D0 0.42–0.58, none feasible. Doubled per-round budget on
seeds 1, 4, 6, 15, 20: at 48/48 feasible 15, 20 only; at 48/96 feasible
1, 15, 20. Diagnostic, dense unprojected Adam (200 steps/round, same
labels) held-out after round 3 versus retraining from scratch on all 128
labels: seed 1 0.696 vs 0.752; seed 4 0.648 vs 0.850; seed 6 0.644 vs
0.674; seed 15 0.910 vs 0.944; seed 20 0.824 vs 0.816. Continuation from
D0 is stuck on 1, 4, 6 regardless of budget.

Chosen seeds (Amendment 1 to the record): 15, 20, 21, 24, 30; reserve 38.

## Compile gate on the chosen seeds (rule Amendment 2 item 2), tolerance 1e-6

| seed | D0 held-out | compile held-out | coordinates differing | max-norm displacement |
|---|---|---|---|---|
| 15 | 0.738 | 1.000 | 321 | 20.52 |
| 20 | 0.750 | 1.000 | 321 | 20.56 |
| 21 | 0.728 | 1.000 | 321 | 39.26 |
| 24 | 0.722 | 1.000 | 321 | 20.57 |
| 30 | 0.608 | 1.000 | 321 | 21.69 |
| 38 | 0.668 | 1.000 | 321 | 29.08 |

## Recheck at the production code (run after the study; a post-hoc check, not a selection)

The feasibility preflight above was run at commit a36a7de, whose preflight
accumulated re-queried labels with duplicates. The production code, commit 8d9965c,
changed accumulation to first occurrence (P9), and the production run used
that. Batches and D0 are identical between the two; only G's trajectory
differs. Rerun at commit b6a953f on every candidate seed, base budget,
f = 0.0224:

| seed | D0 | G r1 | G r2 | G r3 | N | gain | feasible |
|---|---|---|---|---|---|---|---|
| 1 | 0.692 | 0.726 | 0.698 | 0.720 | 0.692 | +0.0280 | True |
| 4 | 0.670 | 0.662 | 0.632 | 0.606 | 0.670 | -0.0640 | False |
| 6 | 0.658 | 0.654 | 0.644 | 0.664 | 0.658 | +0.0060 | False |
| 15 | 0.738 | 0.788 | 0.794 | 0.856 | 0.738 | +0.1180 | True |
| 20 | 0.750 | 0.778 | 0.786 | 0.784 | 0.750 | +0.0340 | True |
| 21 | 0.728 | 0.744 | 0.768 | 0.780 | 0.728 | +0.0520 | True |
| 24 | 0.722 | 0.722 | 0.748 | 0.766 | 0.722 | +0.0440 | True |
| 25 | 0.608 | 0.592 | 0.606 | 0.614 | 0.608 | +0.0060 | False |
| 28 | 0.674 | 0.674 | 0.672 | 0.704 | 0.674 | +0.0300 | True |
| 29 | 0.524 | 0.550 | 0.546 | 0.550 | 0.524 | +0.0260 | True |
| 30 | 0.608 | 0.616 | 0.646 | 0.652 | 0.608 | +0.0440 | True |
| 38 | 0.668 | 0.676 | 0.714 | 0.722 | 0.668 | +0.0540 | True |

Under the production G the first five seeds in roster order with D0 >= 0.60
and G feasible are 1, 15, 20, 21, 24; the study ran 15, 20, 21, 24, 30.
All five run seeds are feasible under the production G. The deviation and
its correction are recorded in RESULTS.md.
