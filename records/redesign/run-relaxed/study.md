# Repair study (redesign decision record P5–P9)

## Held-out accuracy after the last round, per seed

| seed | status | D0 | N | G | E1 | E2 | gE | gG | d |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 0.6920 | 0.6920 | 0.7200 | 0.6880 | 0.6880 | -0.0040 | 0.0280 | -0.0320 |
| 15 | ok | 0.7380 | 0.7380 | 0.8560 | 0.8000 | 0.8140 | 0.0690 | 0.1180 | -0.0490 |
| 20 | ok | 0.7500 | 0.7500 | 0.7840 | 0.7560 | 0.7560 | 0.0060 | 0.0340 | -0.0280 |
| 21 | ok | 0.7280 | 0.7280 | 0.7800 | 0.7280 | 0.7280 | 0.0000 | 0.0520 | -0.0520 |
| 24 | ok | 0.7220 | 0.7220 | 0.7660 | 0.7340 | 0.7340 | 0.0120 | 0.0440 | -0.0320 |

## Outcome

- f: 0.0224
- analyzable seeds: 5 of 5 (minimum 4)
- failed seeds: []
- gE_at_most_f: 4
- d_above_f: 0
- d_below_minus_f: 5
- abs_d_at_most_f: 0
- outcome: no_repair

## Seeds (status and rounds completed per arm; no seed is excluded)

| seed | status | failure | n_rounds | E1 done | E2 done | G done | N done |
|---|---|---|---|---|---|---|---|
| 1 | ok |  | 3 | 3 | 3 | 3 | 3 |
| 15 | ok |  | 3 | 3 | 3 | 3 | 3 |
| 20 | ok |  | 3 | 3 | 3 | 3 | 3 |
| 21 | ok |  | 3 | 3 | 3 | 3 | 3 |
| 24 | ok |  | 3 | 3 | 3 | 3 | 3 |

## Per seed and round (P6): blocks touched, touched-set size, coordinate overlap

n/a: the arm did not complete the round. Overlaps are coordinate counts for that round.

| seed | round | status | E1 blocks | E2 blocks | G blocks | n_E1 | n_E2 | n_G | E1∩G | E2∩G | E1∩E2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 24 | 1 | 1 | 1 |
| 1 | 2 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 40 | 1 | 1 | 1 |
| 1 | 3 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 48 | 1 | 1 | 1 |
| 15 | 1 | ok | Wx | Wx,w | Wx,Wh,bh,w,c | 3 | 6 | 24 | 1 | 2 | 1 |
| 15 | 2 | ok | Wx,Wh | Wx,w | Wx,Wh,bh,w,c | 5 | 11 | 37 | 1 | 3 | 4 |
| 15 | 3 | ok | Wx,Wh,bh | Wx,w,c | Wx,Wh,bh,w,c | 6 | 12 | 48 | 2 | 6 | 4 |
| 20 | 1 | ok | c | c | Wh,w,c | 1 | 1 | 24 | 1 | 1 | 1 |
| 20 | 2 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 39 | 1 | 1 | 1 |
| 20 | 3 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 48 | 1 | 1 | 1 |
| 21 | 1 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 24 | 1 | 1 | 1 |
| 21 | 2 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 34 | 1 | 1 | 1 |
| 21 | 3 | ok | c | c | Wx,Wh,bh,w,c | 1 | 1 | 48 | 1 | 1 | 1 |
| 24 | 1 | ok | Wx | Wx | Wx,Wh,w,c | 1 | 1 | 24 | 0 | 0 | 1 |
| 24 | 2 | ok | Wx | Wx | Wx,Wh,w,c | 1 | 1 | 36 | 0 | 0 | 1 |
| 24 | 3 | ok | Wx | Wx | Wx,Wh,w,c | 1 | 1 | 48 | 0 | 0 | 1 |

## Blocks touched per arm and round (seeds with the round available; per block, seeds touching it)

| arm | round | n_available | Wx | Wh | bh | w | c |
|---|---|---|---|---|---|---|---|
| E1 | 1 | 5 | 2 | 0 | 0 | 0 | 3 |
| E1 | 2 | 5 | 2 | 1 | 0 | 0 | 3 |
| E1 | 3 | 5 | 2 | 1 | 1 | 0 | 3 |
| E2 | 1 | 5 | 2 | 0 | 0 | 1 | 3 |
| E2 | 2 | 5 | 2 | 0 | 0 | 1 | 3 |
| E2 | 3 | 5 | 2 | 0 | 0 | 1 | 4 |
| G | 1 | 5 | 4 | 5 | 1 | 5 | 5 |
| G | 2 | 5 | 5 | 5 | 1 | 5 | 5 |
| G | 3 | 5 | 5 | 5 | 2 | 5 | 5 |

## Editor rounds and attempts

| realization | rounds | noop_rounds | readout_only_rounds | attempts | ok | refusal | llm_error | client_error | parse | budget |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 15 | 0 | 6 | 15 | 15 | 0 | 0 | 0 | 0 | 0 |
| E2 | 15 | 0 | 7 | 16 | 15 | 0 | 1 | 0 | 0 | 0 |

## Editor attempt failures (every non-ok attempt, with its reason)

| realization | seed | round | attempt | outcome | finish_reason | reason |
|---|---|---|---|---|---|---|
| E2 | 20 | 1 | 1 | llm_error | length | OpenRouter returned no content (finish_reason=length) |

## Spend (usage.jsonl)

| entries | aborted | prompt_tokens | completion_tokens | usd |
|---|---|---|---|---|
| 5 | 0 | 207433 | 88461 | 2.5990 |

## Study meta

- git_commit: 41f43a1d3fb00ec7d3bfd5e93891e36a04b2a423
- config: {"seeds": "1,15,20,21,24", "production": false, "backend": "openrouter", "model": "anthropic/claude-opus-5.5", "reasoning_tokens": 16000, "max_output_tokens": 24000, "k": 6, "hidden": 16, "n_initial": 32, "rounds": 3, "n_candidates": 32, "n_heldout": 500, "n_long": 200, "train_steps": 2000, "fence": "relaxed", "per_round": 24, "distinct": 48, "max_norm": 1.0, "spend_cap_usd": 10.0}
- written: 2026-11-05T08:28:14.367022+00:00
- resumes: []
