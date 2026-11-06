# Pilot results — counterexample-guided weight repair without gradients (2026-11-04)

**Question.** Can a pretrained LLM use a few labeled failures to modify RNN
weights so that the fix generalizes, and does LLM-chosen (adversarial)
feedback beat equally costly random feedback?

**Setup.** Hidden minimal 6-state DFA over {0,1}; tanh RNN discriminator with
h=16 (321 parameters; this line originally said 337, an arithmetic slip corrected 2026-11-05: 2·16 + 16·16 + 16 + 16 + 1 = 321, `rnn.n_params`); weights come only from an LLM-written Python
constructor run in an OS-confined sandbox. Editor: Opus 5.5 (16k reasoning).
Initial labels: 32 (chosen so that they do not identify the DFA; the
gradient-trained baseline scores 0.692 on them). Three arms from one shared
D0: A (LLM adversary proposes 32 strings per round, oracle labels them),
B (32 random strings per round), D (self-refinement, no new labels). Three
rounds. Hidden evaluation: 500 held-out strings (lengths 1..12), 200 long
strings (13..24). Baselines: gradient-trained RNN; minimum consistent DFA by
SAT with an ambiguity re-solve (identifiability bit). One seed.

**Results** (run/results.md has every number).

| labels charged | A held-out | B held-out | D held-out | SAT identifiable (A / B) |
|---|---|---|---|---|
| 0 (D0) | 0.888 | 0.888 | 0.888 | no / no |
| 32 | 1.000 | 0.888 | 0.888 | yes / no |
| 64 | 1.000 | 1.000 | 0.888 | yes / yes |
| 96 | 1.000 | 1.000 | 0.500 | yes / yes |

Noise floor (SE at p=0.5): 0.022 held-out. Long strings 13..24 track
held-out exactly in every cell. Counterexamples found: A 4/32 in round 1,
then 0; B 0/32, 5/32, 0/32. Probability that random feedback finds zero
counterexamples in round 1 under D0's error profile: 0.049.

**What the editor did.** In every accepted constructor the model embedded the
labeled strings as a literal (32 to 37 of them per round), ran an exact DFA
identification (backtracking over the prefix-tree acceptor) and compiled the
found automaton into one-hot RNN weights. Accuracy jumps to 1.000 in exactly
the round where the arm's labels first identify the DFA uniquely, and the
compiled network then equals the SAT baseline's DFA on every hidden set.
Self-refinement never improved and, in round 3, returned a reject-all
network after its search hit its own step limit.

**Reading** (mechanical, under the frozen rule).
- Counterexample-guided repair works here: +0.112 over self-refinement,
  five noise floors.
- Adversarial vs random feedback: inconclusive. A led by one round (32
  labels) at equal cost; the later tie is a ceiling effect. One seed cannot
  separate A's structured probing (three of its four counterexamples share
  the prefix 010010110) from B's unlucky first draw.
- Attribution: information. The remaining error at every stage is exactly
  what the labels leave unidentified; no representation or inference gap was
  measured once labels identify the DFA.
- Gate: stop this design. Because the editor re-identifies the DFA exactly,
  further seeds measure labels-to-identification, which the SAT baseline
  computes offline without a model. The follow-up that would test weight
  repair rather than program synthesis must prevent the constructor from
  embedding the data and re-solving.

**Calibration facts** (records/calibration/): with 96 initial labels the same
editor produced a perfect D0 at k = 6, 7 and 8 and a chance-level one at
k = 10; h=16 holds at most 8 one-hot states. Fable 5.1 could not be used:
Anthropic's API content filter refused the editor prompt
(recorded at the time).

**Cost.** Pilot run: 19 model calls, 30k prompt and 56k completion tokens,
about $1.25. Calibration sweeps and retries: about $4. Total about $5.
