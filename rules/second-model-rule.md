# Second-model rule — gpt-6-astra as editor. Frozen 2026-11-05 before the first call

Frozen 2026-11-05. Two runs,
both with `openai/gpt-6-astra` via OpenRouter through wsp/llm.py, the
pilot's unified reasoning budget (`reasoning.max_tokens` 16000), provider
default sampling, no seed parameter. Prices verified against
openrouter.ai/api/v1/models on 2026-11-05: $10 per million prompt tokens,
$50 per million completion tokens including reasoning (wsp/query_study.py
MODEL_RATES, commit 68a437b). Nothing here changes after the first call;
amendments are appended with a timestamp.

## Run A — repair redesign on gpt-6-astra (runs/repair-study-astra)
Everything as rules/redesign-rule.md with Amendments 1 to 3 except the
model: seeds 1, 15, 20, 21, 24 in that order, relaxed single-json-fence
format (Amendment 3) as the protocol, three attempts with fed-back reasons,
two editor realizations per seed, G and N as before, same batches, same
D0, same held-out sets, same budget 24/48/1.0, same outcome precedence and
f = 0.0224. Command:
`python -m wsp.repair_study run --seeds 1,15,20,21,24 --fence relaxed
--backend openrouter --model openai/gpt-6-astra --reasoning-tokens 16000
--max-output-tokens 24000 --per-round 24 --distinct 48 --max-norm 1.0
--spend-cap-usd 18 --out runs/repair-study-astra` (the production guard
pins the Opus configuration, so --production is off; the configuration
diff is recorded in study_meta.json). Expected about $13 (Opus token
counts at astra rates); cap $18 enforced per call by the reservation
guard; a cap stop fails the affected seed.

## Run B — constructor pilot replication on gpt-6-astra (runs/pilot-astra)
The pilot's protocol (records/pilot/pilot-frozen.txt and
RESULTS.md) with the model changed and D0 written by gpt-6-astra itself
from the 32 initial labels (no --d0-from; the Opus-written D0 is not
reused): `python -m wsp.run --backend openrouter --model
openai/gpt-6-astra --rounds 3 --k 6 --hidden 16 --seed 1 --n-initial 32
--n-candidates 32 --n-heldout 500 --n-long 200 --sat-timeout 30
--train-steps 2000 --constructor-timeout 60 --max-attempts 3
--reasoning-tokens 16000 --max-output-tokens 32000 --out runs/pilot-astra`.
Question: does gpt-6-astra also embed the dataset and compile an exact-fit
DFA (literal-match column, SAT equality after identification), and does
its D0 start in the pilot's window? wsp.run has no per-call spend guard
(it predates it); the call count is bounded by the protocol (at most 5 D0
attempts, 3 rounds × 3 arms × 3 attempts, the parity check), so the worst
case is about $16 at the output maximum and the expected cost about $4.
Usage is checked after the run; this gap is recorded, not fixed,
for this run.

## Reading
Outcomes are read mechanically under the existing precedences; the
comparison with the Opus runs is descriptive.

## Not changed
No new seeds, budgets, prompts or comparators. The Opus results remain the
primary record of the redesign; this rule adds a second-model record.

## Amendment 1 — 2026-11-05 09:45Z, run A's cap raised to $25, filed before any cap stop

Run A's spend cap is $25, replacing $18. The running process carries the
old cap; if it stops a seed at $18, that seed's partial spend counts and
the seed is rerun from scratch under the new cap with the identical
configuration, by relaunching the same command with --spend-cap-usd 25
(the runner resumes per seed; the stopped seed's record and the cap marker
are removed first so it is rerun, and both are kept under
runs/repair-study-astra/cap-stop-18/ for the record). The $25 is a hard
cap on run A's total, enforced per call. Run B is complete ($3.93).
