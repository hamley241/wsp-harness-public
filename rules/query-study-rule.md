# Query-policy study rule — frozen 2026-11-04 before the first call

Frozen 2026-11-04. This file states the mechanical rule for the run. Where this
file is more specific than the record, the record governs intent and this
file governs execution. Nothing here is changed after the first model call;
amendments are appended with a timestamp and apply only to calls after them.

## Question
Does arm A's LLM query policy (the pilot's generator, unchanged) reach SAT
uniqueness of the hidden DFA in fewer charged labels than random queries
(arm B), against the same stand-in discriminator?

## Population (P9)
- Hidden DFA: `random_dfa(k=6, seed)`, seeds 1..20 in order for the main
  study, 21..30 for the one declared extension. A seed is ineligible, skipped
  and counted when the DFA is not minimal at 6 states or has an unreachable
  state, or when the 32 initial labels already give SAT uniqueness at k=6.
- Initial labels: `Oracle(dfa, seed).initial_dataset(32)`, the pilot's
  sampler. Histogram: `target_histogram(initial strings, 32)`, the pilot's.
- Both arms share the DFA, the initial labels, the histogram, the budget of
  3 rounds of 32 charged labels, and the seed.
- Generator: `anthropic/claude-opus-5.5` via OpenRouter, reasoning 16000,
  max output 32000, the pilot's prompts unchanged (wsp/prompts.py).
- Re-queries of an already labeled string are charged and add nothing.
- SAT: pysat Glucose4 through wsp/sat_dfa.py, 30 s per solve, the pilot's
  ambiguity re-solve. Uniqueness is computed once per round over every label
  charged so far by that arm, never within a round.

## Stand-in discriminator (P2)
- The SAT DFA at k=6 consistent with every label the arm has charged so far,
  decoded from the first model Glucose4 returns with default phases
  (`identify(..., phases="default")`). A sees its predictions on its probes,
  exactly where the pilot's generator saw the network's predictions. B never
  sees predictions.
- If the stand-in solve times out in a round, A's second generator call
  receives no predictions, and the round is marked `standin_timeout`.
- Precondition (P1), run before any model call: `replay` the pilot's three
  A rounds with the stand-in over the pilot's own information set (initial
  labels plus counterexamples before that round). Pass: every round solves,
  and on every probe whose label is determined by that information set the
  stand-in agrees with the pilot's recorded prediction. Agreement on
  undetermined probes is reported, not required: on those strings the pilot's
  compiler and the stand-in differ only by tie-break, which is the choice this
  rule freezes. On failure the stand-in is revised, this file is amended, and
  the replay is rerun; no model call until it passes.
- Sensitivity (P2): seeds 1..5 are rerun for arm A only with
  `phases="inverted"`. Discordance: if A's event time differs between the two
  phases on 2 or more of the 5 seeds, the result is marked
  `tiebreak_sensitive` and cannot support an LLM-adversary arm in the
  redesign.

## Per-seed procedure (P3)
- Round r = 1..3 for each arm: A as in the pilot's ArmA (probe call, stand-in
  predictions, candidate call, one re-prompt for missing buckets, uniform
  fill for the rest, counted); B uniform per bucket.
- Refusals and errors (intention-to-treat): a generator refusal or error is
  retried up to 2 extra times within the round. If the round still fails, A's
  32 candidates are all uniform-filled and the round is marked
  `fallback_random`. The seed stays paired. This fill is neutral in
  distribution, not a penalty.
- Event time: the ledger total at the first round after which the arm's
  charged labels give SAT uniqueness. Final status: `unique`; `censored`
  (three rounds, last solve sat and ambiguous: event time exceeds 96);
  `indeterminate` (last solve timed out); `failed` (a non-refusal failure
  that a single rerun of the round did not clear).
- An intermediate timeout leaves the event time interval-unknown between the
  last solved round and the next solved round.

## Pairing and test (P3, P8)
- A pair is informative when the two arms' event times are ordered: both
  unique with different event times, or one unique and the other censored.
  Equal event times, both censored, either indeterminate or failed, or an
  interval-unknown arm whose interval contains the other arm's event time: tie,
  uninformative.
- Estimand: the per-seed sign of B's event time minus A's. Meaningful effect:
  A earlier by at least one full round (32 labels), which is every ordered
  pair under this design.
- Test: one-sided paired sign test on informative pairs, alpha 0.05, H1 "A
  wins more pairs than B".
- Thresholds: 20 or more informative pairs, the test is read. 12 to 19: a
  rejection counts; a non-rejection reports `underpowered_no_result`. Below
  12: `no_result`. If the first 20 seeds yield under 20 informative pairs the
  study extends once to seeds 21..30. Nominal power: 20 informative pairs
  give about 0.8 at alpha 0.05 when A's per-pair win probability is 0.8.
- Outcomes: `A_beats_random` only when the test rejects and the median
  informative difference is at least one round. `no_advantage_shown` only
  with 20 or more informative pairs and a non-rejection. Otherwise
  `underpowered_no_result` or `no_result`. Under `A_beats_random` the
  redesign spec may carry an LLM-adversary arm and must argue transfer; under
  any other outcome it uses random queries only.
- Reliability tally reported beside the result: ineligible seeds by reason,
  refusals, errors, retries, fallback rounds, fills, stand-in and endpoint
  timeouts, indeterminate and failed arms, reruns.

## Spend (P7, C4)
- Estimate about $7: 20 seeds at about $0.20 each (pilot rate, 2 generator
  calls per round, $0.197 for 6 calls), about $1 for the 5-seed sensitivity,
  about $2 for the extension if needed. Frozen cap $10 including retries,
  enforced by `--spend-cap-usd 10` across the study's output directory; the
  run stops before the seed that would exceed it.
- A refusal returns no billed tokens; retries that do bill are counted.

## Reading
- The outcome JSON and the reliability tally are read mechanically with the
  outcome vocabulary above. The result is conditional on the frozen stand-in;
  the sensitivity check is reported with it.

## Amendment 1 — 2026-11-04, before any model call

The seven items below are adopted as written except item 5, which asked
for deterministic provider seeds; Anthropic models through OpenRouter accept
no seed parameter and wsp/llm.py sends none, so item 5 is replaced by a
replicate baseline. Where this amendment conflicts with the text above, the
amendment governs.

1. SAT identity and inputs. The python-sat version is the one pinned in
   uv.lock at the commit recorded in the study's `study.json` before the
   first call, Glucose4, and the `_Encoding` in wsp/sat_dfa.py at that
   commit. Every SAT sample list is the 32 initial pairs in Oracle-return
   order followed by each charged query's first occurrence in ledger order;
   re-queries stay charged and are omitted after their first occurrence.
   `default` means no phase override. `inverted` sets every variable's
   preferred phase to positive (Glucose's default preference is negative),
   and a conformance test must show the two settings decode different DFAs
   on at least one ambiguous instance.
2. Seed roster and randomness. Scan integer seeds upward from 1. The first
   20 eligible seeds are the main roster; the next 10 eligible are the
   extension roster; every ineligible seed is recorded with its reason and
   replaced by the next. Sensitivity uses the first five eligible main seeds.
   An eligibility SAT timeout aborts preflight rather than skipping the seed.
   A and B use separate `random.Random(seed)` instances; sensitivity reruns
   of A reuse A's stream. Uniform fill is without replacement within a round
   only, so a re-query of an earlier string remains possible and is charged.
3. Failures. A refusal at any generator call restarts the whole uncharged
   round, up to three attempts; after the third refused attempt the whole
   candidate batch is random-filled and the round is `fallback_random`. A
   non-refusal failure before any label is charged discards the partial round
   and permits one whole-round rerun; a second such failure marks the arm
   `failed`. A no-label failed attempt consumes no budget. An unexpected
   failure after labels are charged marks the arm `failed` and is never
   rerun.
4. Endpoint and analysis. SAT always uses the initial labels plus charged
   labels. An arm stops at its first unique endpoint. `unsat` is a failed
   arm. When a round's solve timed out, the arm's event-time set is the
   discrete round totals consistent with the solved rounds. A pair is
   informative whenever every value consistent with both arms gives the same
   strict ordering; overlapping sets are uninformative. The test is the exact
   one-sided binomial sign test, run once after the required roster (and the
   complete ten-seed extension when triggered) finishes. The median-margin
   condition is replaced by: A wins strictly more than half of the
   informative pairs. No censored event time is imputed.
5. Sensitivity (replaces the text above). On the five sensitivity seeds, A
   runs three times: the main default run, a default replicate, and the
   inverted run. d_inv counts seeds where main and inverted differ (unique
   totals differ, or one unique and the other censored; both censored is
   concordant); d_rep counts the same between main and replicate. The result
   is `tiebreak_sensitive` iff d_inv >= 2 and d_inv > d_rep. Any failed or
   indeterminate sensitivity run makes sensitivity inconclusive, which bars
   an LLM-adversary arm in the redesign. Otherwise the main result stands as
   conditional on the default stand-in.
6. Replay. A probe is determined iff adding the opposite of the stand-in's
   prediction makes the k=6 instance UNSAT. A timeout in the stand-in or in
   a determination solve fails the replay. The number determined is
   reported; zero determined is reported as a vacuous machinery check. The
   replay is a forced-label machinery check, not evidence of transfer or of
   full reproduction of the pilot's predictions.
7. Spend and implementation gate. All provider-reported usage is counted,
   refusals and errors included. Before every call the runner reserves that
   call's maximum charge (prompt characters / 3 at the prompt rate plus
   max_output_tokens at the completion rate, rates $4 and $20 per million
   for anthropic/claude-opus-5.5) and starts it only if accrued spend plus
   the reserve is at most $10, reconciling afterwards from reported usage. A
   cap stop during an unfinished required roster yields `no_result`. Order:
   main, required extension, then sensitivity. No real call until the phase
   API, the runner, the exact analysis, the accounting guard and the
   conformance tests are implemented and reviewed.

Revised estimate: about $8 (20 main seeds at about $0.20, 10 extension
seeds at about $0.20 if triggered, two extra A runs on five seeds at about
$0.07 each). Cap unchanged at $10.

Note 2026-11-05 (before any model call): event times are charged-label
ledger totals, as in the pilot (32, 64, 96), not initial plus charged; the
initial 32 are identical for both arms, so the pairing is unaffected. The
replay precondition passed on the pilot: 184 determined probes, 184 agree
(records/pilot/replay-precondition.json).
