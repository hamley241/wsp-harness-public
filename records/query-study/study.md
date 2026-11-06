# Query-policy study

## Main roster

| seed | event_A | status_A | event_B | status_B | informative | winner/reason |
|---|---|---|---|---|---|---|
| 1 | 32 | unique | 32 | unique | False | equal |
| 2 | 32 | unique | 32 | unique | False | equal |
| 3 | 64 | unique | 32 | unique | True | B |
| 4 | 32 | unique | 64 | unique | True | A |
| 6 | 64 | unique | 32 | unique | True | B |
| 7 | 32 | unique | 96 | unique | True | A |
| 10 | 32 | unique | 32 | unique | False | equal |
| 12 | censored | censored | censored | censored | False | both_censored |
| 14 | censored | censored | censored | censored | False | both_censored |
| 15 | censored | censored | censored | censored | False | both_censored |
| 16 | 32 | unique | 32 | unique | False | equal |
| 18 | 64 | unique | 32 | unique | True | B |
| 20 | 32 | unique | 32 | unique | False | equal |
| 21 | 96 | unique | censored | censored | True | A |
| 22 | 64 | unique | 64 | unique | False | equal |
| 23 | 64 | unique | 64 | unique | False | equal |
| 24 | 32 | unique | 32 | unique | False | equal |
| 25 | 32 | unique | 64 | unique | True | A |
| 26 | 32 | unique | 32 | unique | False | equal |
| 28 | censored | censored | 32 | unique | True | B |

## Extension roster

| seed | event_A | status_A | event_B | status_B | informative | winner/reason |
|---|---|---|---|---|---|---|
| 29 | censored | censored | 64 | unique | True | B |
| 30 | censored | censored | censored | censored | False | both_censored |
| 31 | 32 | unique | 32 | unique | False | equal |
| 32 | 32 | unique | 32 | unique | False | equal |
| 33 | 32 | unique | 32 | unique | False | equal |
| 34 | censored | censored | censored | censored | False | both_censored |
| 35 | censored | censored | 64 | unique | True | B |
| 36 | 32 | unique | censored | censored | True | A |
| 37 | 32 | unique | 32 | unique | False | equal |
| 38 | censored | censored | 32 | unique | True | B |

## Roster check

- main_missing: []
- main_unrostered: []
- extension_missing: []
- extension_unrostered: []

## Decision

- n_pairs: 30
- n_informative: 12
- wins_A: 5
- wins_B: 7
- p: 0.80615234375
- alpha: 0.05
- margin_ok: false
- rejected: false
- outcome: "underpowered_no_result"
- stopped_at_cap: false
- main_done: true
- extension_done: true
- needs_extension: false

## Sensitivity (arm A, seeds 1, 2, 3, 4, 6)

| seed | event_main | status_main | event_replicate | status_replicate | event_inverted | status_inverted | discordant_inverted | discordant_replicate |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 | unique | 32 | unique | 32 | unique | False | False |
| 2 | 32 | unique | 32 | unique | 32 | unique | False | False |
| 3 | 64 | unique | 32 | unique | 64 | unique | False | True |
| 4 | 32 | unique | 32 | unique | 96 | unique | True | False |
| 6 | 64 | unique | 64 | unique | 64 | unique | False | False |

- d_inv: 1
- d_rep: 1
- inconclusive: false
- tiebreak_sensitive: false
- bars_adversary_arm: false

## Reliability

- ineligible: {"initial_unique": 8}
- refusals: 0
- errors: 0
- attempts: 122
- fallback_random_rounds: 0
- filled_strings: 0
- standin_timeouts: 0
- endpoint_timeouts: 0
- indeterminate_arms: 0
- failed_arms: 0
- reruns: 0
- arms: 70
- rounds: 122
- stopped_at_cap: null

## Spend

- prompt_tokens: 98691
- completion_tokens: 160427
- usd: 3.6033039999999996
- entries: 40
- aborted: 0

## Study meta

```json
{
 "git_commit": "b8f63d823f0badf0199a44b8195063bbe34fc88e",
 "python_sat_version": "1.9.dev20",
 "config": {
  "roster": "sensitivity-inverted",
  "n_main": 20,
  "n_ext": 10,
  "k": 6,
  "n_initial": 32,
  "rounds": 3,
  "n_candidates": 32,
  "backend": "openrouter",
  "model": "anthropic/claude-opus-5.5",
  "reasoning_tokens": 16000,
  "max_output_tokens": 32000,
  "sat_timeout": 30.0,
  "spend_cap_usd": 10.0
 },
 "written": "2026-11-05T04:58:20.592849+00:00"
}
```
