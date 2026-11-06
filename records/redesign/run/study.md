# Repair study (redesign decision record P5–P9)

## Held-out accuracy after the last round, per seed

| seed | status | D0 | N | G | E1 | E2 | gE | gG | d |
|---|---|---|---|---|---|---|---|---|---|
| 15 | ok | 0.7380 | 0.7380 | 0.8560 | 0.7800 | 0.8280 | 0.0660 | 0.1180 | -0.0520 |
| 20 | ok | 0.7500 | 0.7500 | 0.7840 | 0.7560 | 0.7600 | 0.0080 | 0.0340 | -0.0260 |
| 21 | ok | 0.7280 | 0.7280 | 0.7800 | 0.7280 | 0.7280 | 0.0000 | 0.0520 | -0.0520 |
| 24 | ok | 0.7220 | 0.7220 | 0.7660 | 0.7360 | 0.7220 | 0.0070 | 0.0440 | -0.0370 |
| 30 | ok | 0.6080 | 0.6080 | 0.6520 | 0.6080 | 0.6020 | -0.0030 | 0.0440 | -0.0470 |

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
| 15 | ok |  | 3 | 3 | 3 | 3 | 3 |
| 20 | ok |  | 3 | 3 | 3 | 3 | 3 |
| 21 | ok |  | 3 | 3 | 3 | 3 | 3 |
| 24 | ok |  | 3 | 3 | 3 | 3 | 3 |
| 30 | ok |  | 3 | 3 | 3 | 3 | 3 |

## Per seed and round (P6): blocks touched, touched-set size, coordinate overlap

n/a: the arm did not complete the round. Overlaps are coordinate counts for that round.

| seed | round | status | E1 blocks | E2 blocks | G blocks | n_E1 | n_E2 | n_G | E1∩G | E2∩G | E1∩E2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 1 | ok | Wx | c | Wx,Wh,bh,w,c | 2 | 1 | 24 | 0 | 1 | 0 |
| 15 | 2 | ok | Wx,c | Wx,w,c | Wx,Wh,bh,w,c | 3 | 11 | 37 | 1 | 5 | 2 |
| 15 | 3 | ok | Wx,c | Wx,w,c | Wx,Wh,bh,w,c | 3 | 11 | 48 | 2 | 7 | 2 |
| 20 | 1 | ok | c | c | Wh,w,c | 1 | 1 | 24 | 1 | 1 | 1 |
| 20 | 2 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 39 | 1 | 1 | 1 |
| 20 | 3 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 48 | 1 | 1 | 1 |
| 21 | 1 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 24 | 1 | 1 | 1 |
| 21 | 2 | ok | c | c | Wx,Wh,w,c | 1 | 1 | 34 | 1 | 1 | 1 |
| 21 | 3 | ok | c | c | Wx,Wh,bh,w,c | 1 | 1 | 48 | 1 | 1 | 1 |
| 24 | 1 | ok | Wx | c | Wx,Wh,w,c | 1 | 1 | 24 | 0 | 1 | 0 |
| 24 | 2 | ok | Wx,c | c | Wx,Wh,w,c | 2 | 1 | 36 | 1 | 1 | 1 |
| 24 | 3 | ok | Wx,w,c | c | Wx,Wh,w,c | 6 | 1 | 48 | 5 | 1 | 1 |
| 30 | 1 | ok | - | bh,w,c | Wx,Wh,w,c | 0 | 4 | 24 | 0 | 1 | 0 |
| 30 | 2 | ok | c | bh,w,c | Wx,Wh,w,c | 1 | 4 | 38 | 1 | 3 | 1 |
| 30 | 3 | ok | c | bh,w,c | Wx,Wh,w,c | 1 | 6 | 48 | 1 | 5 | 1 |

## Blocks touched per arm and round (seeds with the round available; per block, seeds touching it)

| arm | round | n_available | Wx | Wh | bh | w | c |
|---|---|---|---|---|---|---|---|
| E1 | 1 | 5 | 2 | 0 | 0 | 0 | 2 |
| E1 | 2 | 5 | 2 | 0 | 0 | 0 | 5 |
| E1 | 3 | 5 | 2 | 0 | 0 | 1 | 5 |
| E2 | 1 | 5 | 0 | 0 | 1 | 1 | 5 |
| E2 | 2 | 5 | 1 | 0 | 1 | 2 | 5 |
| E2 | 3 | 5 | 1 | 0 | 1 | 2 | 5 |
| G | 1 | 5 | 4 | 5 | 1 | 5 | 5 |
| G | 2 | 5 | 5 | 5 | 1 | 5 | 5 |
| G | 3 | 5 | 5 | 5 | 2 | 5 | 5 |

## Editor rounds and attempts

| realization | rounds | noop_rounds | readout_only_rounds | attempts | ok | refusal | llm_error | client_error | parse | budget |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 15 | 0 | 12 | 32 | 15 | 0 | 1 | 0 | 16 | 0 |
| E2 | 15 | 1 | 12 | 32 | 14 | 0 | 2 | 0 | 16 | 0 |

## Editor attempt failures (every non-ok attempt, with its reason)

| realization | seed | round | attempt | outcome | finish_reason | reason |
|---|---|---|---|---|---|---|
| E1 | 15 | 1 | 1 | llm_error | length | OpenRouter returned no content (finish_reason=length) |
| E1 | 15 | 1 | 2 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 15 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 15 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 20 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 20 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 20 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 21 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 21 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 21 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 24 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 24 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 24 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 30 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 30 | 1 | 2 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 30 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E1 | 30 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 15 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 15 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 15 | 3 | 1 | llm_error | length | OpenRouter returned no content (finish_reason=length) |
| E2 | 15 | 3 | 2 | llm_error | length | OpenRouter returned no content (finish_reason=length) |
| E2 | 15 | 3 | 3 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 20 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 20 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 20 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 21 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 21 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 21 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 24 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 24 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 24 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 30 | 1 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 30 | 2 | 1 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 30 | 2 | 2 | parse | stop | nothing may appear outside the single ```json fence |
| E2 | 30 | 3 | 1 | parse | stop | nothing may appear outside the single ```json fence |

## Spend (usage.jsonl)

| entries | aborted | prompt_tokens | completion_tokens | usd |
|---|---|---|---|---|
| 5 | 0 | 429739 | 180039 | 5.3197 |

## Study meta

- git_commit: 0b8d55dce60f25253d30a33850fd5063e7c8f1ff
- config: {"seeds": "15,20,21,24,30", "production": true, "backend": "openrouter", "model": "anthropic/claude-opus-5.5", "reasoning_tokens": 16000, "max_output_tokens": 24000, "k": 6, "hidden": 16, "n_initial": 32, "rounds": 3, "n_candidates": 32, "n_heldout": 500, "n_long": 200, "train_steps": 2000, "per_round": 24, "distinct": 48, "max_norm": 1.0, "spend_cap_usd": 50.0}
- written: 2026-11-05T06:41:17.414905+00:00
- resumes: []
