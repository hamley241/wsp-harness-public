# wsp run: seed 1, k=6, h=16, rounds=3

## Config

| flag | value |
|---|---|
| rounds | 3 |
| k | 6 |
| hidden | 16 |
| seed | 1 |
| backend | openrouter |
| mock | false |
| model | anthropic/claude-opus-5.5 |
| reasoning_tokens | 16000 |
| max_output_tokens | 32000 |
| d0_from | runs/calib-k6-n32 |
| n_initial | 32 |
| n_candidates | 32 |
| n_heldout | 500 |
| n_long | 200 |
| sat_timeout | 30 |
| constructor_timeout | 60 |
| max_attempts | 3 |
| train_steps | 2000 |

## Hidden sets and noise floor (SE at p = 0.5)

| set | n | noise floor |
|---|---|---|
| initial | 32 | 0.088 |
| heldout | 500 | 0.022 |
| long | 200 | 0.035 |

d0_shared: true
d0_source: kind=reused from=runs/calib-k6-n32 sha256=ecee5e022abc77a98f019d6d02dd5a27d940ebb53ee57288b874b0a7f609ea2a
editor attempts: 12 / failed rounds: 0
hidden DFA states: 6
held-out / long / SAT held-out / SAT long noise floor: 0.022 / 0.035 / 0.022 / 0.035

## Arms, per round

| arm | round | held-out | long | reported failures | n failures | failures noise floor | regressions | n D0-correct | regressions noise floor | ledger total | n counterexamples | accepted | attempts | src lines | literal matches | SAT status | SAT states | ambiguity | SAT held-out | SAT long | frac weights changed | max |delta| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 0 | 0 | - | - | - | - | sat | 6 | ambiguous | 0.888 | 0.825 | - | - |
| A | 1 | 1.000 | 1.000 | 1.000 | 4 | 0.250 | 0.000 | 32 | 0.088 | 32 | 4 | true | 1 | 140 | 36 | sat | 6 | unique | 1.000 | 1.000 | 0.642 | 40.000 |
| A | 2 | 1.000 | 1.000 | 1.000 | 4 | 0.250 | 0.000 | 32 | 0.088 | 64 | 0 | true | 1 | 140 | 36 | sat | 6 | unique | 1.000 | 1.000 | 0.000 | 0.000 |
| A | 3 | 1.000 | 1.000 | 1.000 | 4 | 0.250 | 0.000 | 32 | 0.088 | 96 | 0 | true | 1 | 139 | 36 | sat | 6 | unique | 1.000 | 1.000 | 0.000 | 0.000 |
| B | 0 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 0 | 0 | - | - | - | - | sat | 6 | ambiguous | 0.888 | 0.825 | - | - |
| B | 1 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 32 | 0 | true | 1 | 142 | 32 | sat | 6 | ambiguous | 0.874 | 0.810 | 0.000 | 0.000 |
| B | 2 | 1.000 | 1.000 | 1.000 | 5 | 0.224 | 0.000 | 32 | 0.088 | 64 | 5 | true | 2 | 147 | 37 | sat | 6 | unique | 1.000 | 1.000 | 0.037 | 12.000 |
| B | 3 | 1.000 | 1.000 | 1.000 | 5 | 0.224 | 0.000 | 32 | 0.088 | 96 | 0 | true | 1 | 161 | 37 | sat | 6 | unique | 1.000 | 1.000 | 0.000 | 0.000 |
| D | 0 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 0 | 0 | - | - | - | - | sat | 6 | ambiguous | 0.888 | 0.825 | - | - |
| D | 1 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 0 | 0 | true | 2 | 146 | 32 | sat | 6 | ambiguous | 0.888 | 0.825 | 0.592 | 10.000 |
| D | 2 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 0 | 0 | true | 2 | 147 | 32 | sat | 6 | ambiguous | 0.888 | 0.825 | 0.000 | 0.000 |
| D | 3 | 0.500 | 0.500 | - | 0 | - | 0.500 | 32 | 0.088 | 0 | 0 | true | 1 | 212 | 32 | sat | 6 | ambiguous | 0.888 | 0.825 | 0.629 | 40.000 |

## Per-length held-out accuracy (final round; n and noise floor per length)

| length | n | noise floor | A held-out | B held-out | D held-out | trained RNN |
|---|---|---|---|---|---|---|
| 1 | 0 | - | - | - | - | - |
| 2 | 2 | 0.354 | 1.000 | 1.000 | 0.000 | 0.500 |
| 3 | 6 | 0.204 | 1.000 | 1.000 | 0.333 | 0.667 |
| 4 | 14 | 0.134 | 1.000 | 1.000 | 0.429 | 0.714 |
| 5 | 29 | 0.093 | 1.000 | 1.000 | 0.276 | 0.655 |
| 6 | 61 | 0.064 | 1.000 | 1.000 | 0.410 | 0.738 |
| 7 | 114 | 0.047 | 1.000 | 1.000 | 0.351 | 0.667 |
| 8 | 106 | 0.049 | 1.000 | 1.000 | 0.802 | 0.717 |
| 9 | 42 | 0.077 | 1.000 | 1.000 | 0.500 | 0.595 |
| 10 | 42 | 0.077 | 1.000 | 1.000 | 0.500 | 0.595 |
| 11 | 42 | 0.077 | 1.000 | 1.000 | 0.500 | 0.786 |
| 12 | 42 | 0.077 | 1.000 | 1.000 | 0.500 | 0.762 |

## Baselines

| baseline | held-out | held-out noise floor | long | long noise floor | train acc | n train | train noise floor | steps |
|---|---|---|---|---|---|---|---|---|
| trained RNN h=16 | 0.692 | 0.022 | 0.610 | 0.035 | 1.000 | 32 | 0.088 | 2000 |

| parity one-shot | ran | accepted | accuracy | n test | noise floor |
|---|---|---|---|---|---|
|  | true | true | 1.000 | 200 | 0.035 |

## Cross-play (row: weights arm/round; column: counterexample set arm/round)

| weights \ set | A/0 | A/1 | A/2 | A/3 | B/0 | B/1 | B/2 | B/3 | D/0 | D/1 | D/2 | D/3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 0 | 4 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| noise floor | - | 0.250 | - | - | - | - | 0.224 | - | - | - | - | - |
| A/0 | - | 0.000 | - | - | - | - | 0.000 | - | - | - | - | - |
| A/1 | - | 1.000 | - | - | - | - | 1.000 | - | - | - | - | - |
| A/2 | - | 1.000 | - | - | - | - | 1.000 | - | - | - | - | - |
| A/3 | - | 1.000 | - | - | - | - | 1.000 | - | - | - | - | - |
| B/0 | - | 0.000 | - | - | - | - | 0.000 | - | - | - | - | - |
| B/1 | - | 0.000 | - | - | - | - | 0.000 | - | - | - | - | - |
| B/2 | - | 1.000 | - | - | - | - | 1.000 | - | - | - | - | - |
| B/3 | - | 1.000 | - | - | - | - | 1.000 | - | - | - | - | - |
| D/0 | - | 0.000 | - | - | - | - | 0.000 | - | - | - | - | - |
| D/1 | - | 0.000 | - | - | - | - | 0.000 | - | - | - | - | - |
| D/2 | - | 0.000 | - | - | - | - | 0.000 | - | - | - | - | - |
| D/3 | - | 0.500 | - | - | - | - | 0.800 | - | - | - | - | - |

## Oracle ledger

| arm | total | by length |
|---|---|---|
| A | 96 | {"1": 6, "2": 6, "3": 6, "4": 6, "5": 9, "6": 9, "7": 9, "8": 9, "9": 9, "10": 9, "11": 9, "12": 9} |
| B | 96 | {"1": 6, "2": 6, "3": 6, "4": 6, "5": 9, "6": 9, "7": 9, "8": 9, "9": 9, "10": 9, "11": 9, "12": 9} |

## LLM usage

| calls | prompt_tokens | completion_tokens | reasoning_tokens |
|---|---|---|---|
| 19 | 30179 | 56429 | 37105 |
| LLM refusals / errors | 0 / 0 |  |  |

## Timing

| field | seconds |
|---|---|
| total_s | 567.274 |
| round_1_s | 243.285 |
| round_2_s | 127.740 |
| round_3_s | 161.197 |
