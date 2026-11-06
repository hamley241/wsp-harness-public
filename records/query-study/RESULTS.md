# Query-policy study — results (2026-11-05)

Question: does an LLM adversary (arm A, the pilot's generator, Opus 5.5 via
OpenRouter, prompts unchanged) reach SAT uniqueness of a hidden 6-state DFA
in fewer charged labels than random queries (arm B), against the same
stand-in discriminator (the SAT-minimal DFA over every label charged so far)?
Rule frozen before the first call: rules/query-study-rule.md + Amendment 1
+ note of 2026-11-05. Precondition replay against the pilot passed
(replay-precondition.json: 184 determined probes, 184 agree).

## Design as run
- 30 eligible seeds: main 1–28 (20 eligible; 8 skipped because 32 initial
  labels already identified the DFA), extension 29–38 (10), triggered by
  the rule because the main roster yielded under 20 informative pairs.
- Both arms per seed: same DFA, same 32 initial labels, same length
  histogram, budget 3 rounds × 32 charged labels, stop at first uniqueness.
- Endpoint: charged labels (32, 64, 96) at the first round after which the
  arm's labels give SAT uniqueness at k=6; censored = not unique after 96.
- Test: exact one-sided paired sign test on informative pairs, alpha 0.05.
- Sensitivity: seeds 1, 2, 3, 4, 6, arm A rerun with a default-phase
  replicate and with inverted solver phases.

## Result
| quantity | value |
|---|---|
| pairs | 30 |
| informative pairs | 12 |
| A wins | 5 |
| B wins | 7 |
| exact one-sided p | 0.806 |
| outcome (frozen vocabulary) | underpowered_no_result |
| redesign query arm (frozen rule) | random queries only |

Pair structure: unique–unique 19 (11 equal at 32 labels, 2 equal at 64, 6 ordered),
both censored 5, A censored while B unique 4 (seeds 28, 29, 35, 38), B
censored while A unique 2 (seeds 21, 36).

Sensitivity: inverted-phase discordance 1 of 5, replicate discordance 1 of
5; not tie-break sensitive; does not bar an adversary arm on its own.

Reliability: 0 refusals, 0 errors, 0 fallback rounds, 0 stand-in or
endpoint timeouts, 0 failed or indeterminate arms. Rounds charged: 69 main,
38 extension, 6 sensitivity-replicate, 9 sensitivity-inverted.

Spend: $3.60 (98,691 prompt, 160,427 completion tokens) against a $10 cap.

Raw: main/, extension/, sensitivity-*/ (per-seed JSON), roster.json,
usage.jsonl, study_meta.json (commit b8f63d8, python-sat 1.9.dev20),
study.json, study.md. Launch script: query-study.sh.

## Reading (mechanical, under the frozen rule)
Outcome underpowered_no_result: the data do not support an advantage for
the LLM adversary over random queries, and do not show the reverse. Under
the frozen rule the numeric-delta redesign uses random queries only. The
result is conditional on the frozen stand-in. Two observations recorded as
open questions, not findings: in 8 of the 9 censored A arms, rounds after
the first found zero counterexamples against the exact-fit stand-in, and A
re-queried 192 prior-round strings against B's 99; and 11 of 19
unique–unique pairs tied at the first 32-label endpoint, so a 32-label
round is a coarse clock and smaller rounds would raise the informative
yield at the same cost.
