# Offline scoring of the stored rejected editor proposals

Each parse- or budget-rejected proposal is applied to its recorded round-start weights (D0 reproduced by the frozen trainer, plus the edits the record shows applied in earlier rounds of the same realization) and scored as held-out gain over that round-start accuracy, after a budget check identical to the run's. The reconstruction reproduces every recorded held-out accuracy exactly. Recomputed by `wsp.redesign_extras`; no model calls.

| run | seed | arm | round | attempt | n_edits | budget ok | round-start held-out | after proposal | gain | > f | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main | 15 | E1 | 1 | 2 | 1 | yes | 0.738 | 0.730 | -0.008 | no |  |
| main | 15 | E1 | 2 | 1 | 3 | yes | 0.768 | 0.840 | +0.072 | yes |  |
| main | 15 | E1 | 3 | 1 | 5 | yes | 0.770 | 0.864 | +0.094 | yes |  |
| main | 15 | E2 | 1 | 1 | 2 | yes | 0.738 | 0.774 | +0.036 | yes |  |
| main | 15 | E2 | 2 | 1 | 7 | yes | 0.732 | 0.796 | +0.064 | yes |  |
| main | 15 | E2 | 3 | 3 | 2 | yes | 0.828 | 0.856 | +0.028 | yes |  |
| main | 20 | E1 | 1 | 1 | 1 | yes | 0.750 | 0.756 | +0.006 | no |  |
| main | 20 | E1 | 2 | 1 | 0 | yes | 0.756 | 0.756 | +0.000 | no |  |
| main | 20 | E1 | 3 | 1 | 1 | yes | 0.756 | 0.760 | +0.004 | no |  |
| main | 20 | E2 | 1 | 1 | 1 | yes | 0.750 | 0.756 | +0.006 | no |  |
| main | 20 | E2 | 2 | 1 | 0 | yes | 0.756 | 0.756 | +0.000 | no |  |
| main | 20 | E2 | 3 | 1 | 0 | yes | 0.760 | 0.760 | +0.000 | no |  |
| main | 21 | E1 | 1 | 1 | 1 | yes | 0.728 | 0.730 | +0.002 | no |  |
| main | 21 | E1 | 2 | 1 | 1 | yes | 0.730 | 0.728 | -0.002 | no |  |
| main | 21 | E1 | 3 | 1 | 24 | yes | 0.728 | 0.474 | -0.254 | no |  |
| main | 21 | E2 | 1 | 1 | 1 | yes | 0.728 | 0.730 | +0.002 | no |  |
| main | 21 | E2 | 2 | 1 | 1 | yes | 0.730 | 0.728 | -0.002 | no |  |
| main | 21 | E2 | 3 | 1 | 2 | yes | 0.728 | 0.730 | +0.002 | no |  |
| main | 24 | E1 | 1 | 1 | 1 | yes | 0.722 | 0.734 | +0.012 | no |  |
| main | 24 | E1 | 2 | 1 | 0 | yes | 0.734 | 0.734 | +0.000 | no |  |
| main | 24 | E1 | 3 | 1 | 0 | yes | 0.734 | 0.734 | +0.000 | no |  |
| main | 24 | E2 | 1 | 1 | 1 | yes | 0.722 | 0.726 | +0.004 | no |  |
| main | 24 | E2 | 2 | 1 | 1 | yes | 0.722 | 0.722 | +0.000 | no |  |
| main | 24 | E2 | 3 | 1 | 1 | yes | 0.722 | 0.722 | +0.000 | no |  |
| main | 30 | E1 | 1 | 1 | 1 | yes | 0.608 | 0.608 | +0.000 | no |  |
| main | 30 | E1 | 1 | 2 | 0 | yes | 0.608 | 0.608 | +0.000 | no |  |
| main | 30 | E1 | 2 | 1 | 0 | yes | 0.608 | 0.608 | +0.000 | no |  |
| main | 30 | E1 | 3 | 1 | 0 | yes | 0.608 | 0.608 | +0.000 | no |  |
| main | 30 | E2 | 1 | 1 | 0 | yes | 0.608 | 0.608 | +0.000 | no |  |
| main | 30 | E2 | 2 | 1 | 0 | yes | 0.606 | 0.606 | +0.000 | no |  |
| main | 30 | E2 | 2 | 2 | 1 | yes | 0.606 | 0.606 | +0.000 | no |  |
| main | 30 | E2 | 3 | 1 | 0 | yes | 0.606 | 0.606 | +0.000 | no |  |
| seed1 | 1 | E1 | 1 | 1 | 0 | yes | 0.692 | 0.692 | +0.000 | no |  |
| seed1 | 1 | E1 | 2 | 1 | 0 | yes | 0.686 | 0.686 | +0.000 | no |  |
| seed1 | 1 | E1 | 2 | 2 | 1 | yes | 0.686 | 0.686 | +0.000 | no |  |
| seed1 | 1 | E1 | 3 | 1 | 1 | yes | 0.686 | 0.688 | +0.002 | no |  |
| seed1 | 1 | E2 | 1 | 1 | 0 | yes | 0.692 | 0.692 | +0.000 | no |  |
| seed1 | 1 | E2 | 2 | 1 | 1 | yes | 0.692 | 0.692 | +0.000 | no |  |
| seed1 | 1 | E2 | 3 | 1 | 1 | yes | 0.690 | 0.686 | -0.004 | no |  |

Summary: 39 proposals scored, 39 within budget; gains above f = 5; gains at or below −f = 1; within ±f = 33; max gain +0.094, min -0.254; zero-edit proposals 14; proposals with zero held-out gain 20.
