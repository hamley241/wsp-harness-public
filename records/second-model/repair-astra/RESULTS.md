# Repair study (redesign decision record P5–P9)

## Held-out accuracy after the last round, per seed

| seed | status | D0 | N | G | E1 | E2 | gE | gG | d |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 0.6920 | 0.6920 | 0.7200 | 0.6940 | 0.6760 | -0.0070 | 0.0280 | -0.0350 |
| 15 | ok | 0.7380 | 0.7380 | 0.8560 | 0.7720 | 0.7400 | 0.0180 | 0.1180 | -0.1000 |
| 20 | ok | 0.7500 | 0.7500 | 0.7840 | 0.7660 | 0.7860 | 0.0260 | 0.0340 | -0.0080 |
| 21 | ok | 0.7280 | 0.7280 | 0.7800 | 0.6140 | 0.7680 | -0.0370 | 0.0520 | -0.0890 |
| 24 | ok | 0.7220 | 0.7220 | 0.7660 | 0.5920 | 0.6360 | -0.1080 | 0.0440 | -0.1520 |

## Outcome

- f: 0.0224
- analyzable seeds: 5 of 5 (minimum 4)
- failed seeds: []
- gE_at_most_f: 4
- d_above_f: 0
- d_below_minus_f: 4
- abs_d_at_most_f: 1
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
| 1 | 1 | ok | Wx,Wh,c | w,c | Wx,Wh,w,c | 13 | 17 | 24 | 1 | 12 | 1 |
| 1 | 2 | ok | Wx,Wh,w,c | Wh,w,c | Wx,Wh,w,c | 23 | 25 | 40 | 4 | 14 | 4 |
| 1 | 3 | ok | Wx,Wh,w,c | Wx,Wh,w,c | Wx,Wh,w,c | 31 | 27 | 48 | 13 | 16 | 13 |
| 15 | 1 | ok | Wx,Wh,w,c | w,c | Wx,Wh,bh,w,c | 17 | 14 | 24 | 7 | 12 | 7 |
| 15 | 2 | ok | Wx,Wh,bh,w,c | Wx,Wh,w,c | Wx,Wh,bh,w,c | 29 | 21 | 37 | 11 | 13 | 12 |
| 15 | 3 | ok | Wx,Wh,bh,w,c | Wx,Wh,w,c | Wx,Wh,bh,w,c | 34 | 24 | 48 | 13 | 17 | 14 |
| 20 | 1 | ok | w | w | Wh,w,c | 1 | 1 | 24 | 0 | 0 | 1 |
| 20 | 2 | ok | Wx,Wh,w | w,c | Wx,Wh,w,c | 25 | 10 | 39 | 5 | 7 | 1 |
| 20 | 3 | ok | Wx,Wh,w,c | w,c | Wx,Wh,w,c | 36 | 17 | 48 | 17 | 16 | 12 |
| 21 | 1 | ok | Wh,w,c | Wx,w,c | Wx,Wh,w,c | 24 | 21 | 24 | 13 | 13 | 17 |
| 21 | 2 | ok | Wx,Wh,w,c | Wx,Wh,bh,w,c | Wx,Wh,w,c | 27 | 24 | 34 | 16 | 16 | 18 |
| 21 | 3 | ok | Wx,Wh,w,c | Wx,Wh,bh,w,c | Wx,Wh,bh,w,c | 32 | 29 | 48 | 18 | 17 | 19 |
| 24 | 1 | ok | w,c | Wh,w,c | Wx,Wh,w,c | 13 | 19 | 24 | 11 | 14 | 13 |
| 24 | 2 | ok | Wh,w,c | Wh,w,c | Wx,Wh,w,c | 31 | 31 | 36 | 13 | 17 | 13 |
| 24 | 3 | ok | Wx,Wh,w,c | Wh,w,c | Wx,Wh,w,c | 35 | 32 | 48 | 15 | 17 | 13 |

## Blocks touched per arm and round (seeds with the round available; per block, seeds touching it)

| arm | round | n_available | Wx | Wh | bh | w | c |
|---|---|---|---|---|---|---|---|
| E1 | 1 | 5 | 2 | 3 | 0 | 4 | 4 |
| E1 | 2 | 5 | 4 | 5 | 1 | 5 | 4 |
| E1 | 3 | 5 | 5 | 5 | 1 | 5 | 5 |
| E2 | 1 | 5 | 1 | 1 | 0 | 5 | 4 |
| E2 | 2 | 5 | 2 | 4 | 1 | 5 | 5 |
| E2 | 3 | 5 | 3 | 4 | 1 | 5 | 5 |
| G | 1 | 5 | 4 | 5 | 1 | 5 | 5 |
| G | 2 | 5 | 5 | 5 | 1 | 5 | 5 |
| G | 3 | 5 | 5 | 5 | 2 | 5 | 5 |

## Editor rounds and attempts

| realization | rounds | noop_rounds | readout_only_rounds | attempts | ok | refusal | llm_error | client_error | parse | budget |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 15 | 0 | 4 | 18 | 15 | 0 | 1 | 0 | 0 | 2 |
| E2 | 15 | 0 | 6 | 18 | 15 | 0 | 3 | 0 | 0 | 0 |

## Editor attempt failures (every non-ok attempt, with its reason)

| realization | seed | round | attempt | outcome | finish_reason | reason |
|---|---|---|---|---|---|---|
| E1 | 1 | 3 | 1 | llm_error | length | OpenRouter returned no content (finish_reason=length) |
| E1 | 21 | 3 | 1 | budget | stop | w[3] = -0.463140896 is 1.6 from D0, over the max-norm limit 1.0 |
| E1 | 24 | 3 | 1 | budget | stop | w[4] = 0.0279351472854614 is 2 from D0, over the max-norm limit 1.0 |
| E2 | 20 | 2 | 1 | llm_error | length | OpenRouter returned no content (finish_reason=length) |
| E2 | 20 | 2 | 2 | llm_error | length | OpenRouter returned no content (finish_reason=length) |
| E2 | 24 | 1 | 1 | llm_error | length | OpenRouter returned no content (finish_reason=length) |

## Spend (usage.jsonl)

| entries | aborted | prompt_tokens | completion_tokens | usd |
|---|---|---|---|---|
| 6 | 1 | 213944 | 393964 | 21.8376 |

## Study meta

- git_commit: 144406f95715752cdbef91bc6f7fd9ffb13e4909
- config: {"seeds": "1,15,20,21,24", "production": false, "backend": "openrouter", "model": "openai/gpt-6-astra", "reasoning_tokens": 16000, "max_output_tokens": 24000, "k": 6, "hidden": 16, "n_initial": 32, "rounds": 3, "n_candidates": 32, "n_heldout": 500, "n_long": 200, "train_steps": 2000, "fence": "relaxed", "per_round": 24, "distinct": 48, "max_norm": 1.0, "spend_cap_usd": 18.0}
- written: 2026-11-05T09:09:01.093396+00:00
- resumes: [{'time': '2026-11-05T11:15:10.879400+00:00', 'git_commit': 'd76c38ea39d0eedc59a68614d0f0bd2b1fca1c6f', 'seeds': '1,15,20,21,24', 'spend_cap_usd': 25.0}]
