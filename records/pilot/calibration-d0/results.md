# wsp run: seed 1, k=6, h=16, rounds=0

## Config

| flag | value |
|---|---|
| rounds | 0 |
| k | 6 |
| hidden | 16 |
| seed | 1 |
| backend | openrouter |
| mock | false |
| model | anthropic/claude-opus-5.5 |
| reasoning_tokens | 16000 |
| max_output_tokens | 32000 |
| d0_from | - |
| n_initial | 32 |
| n_candidates | 32 |
| n_heldout | 500 |
| n_long | 200 |
| sat_timeout | 30 |
| constructor_timeout | 60 |
| max_attempts | 5 |
| train_steps | 2000 |

## Hidden sets and noise floor (SE at p = 0.5)

| set | n | noise floor |
|---|---|---|
| initial | 32 | 0.088 |
| heldout | 500 | 0.022 |
| long | 200 | 0.035 |

d0_shared: true
d0_source: kind=editor from=- sha256=ecee5e022abc77a98f019d6d02dd5a27d940ebb53ee57288b874b0a7f609ea2a
editor attempts: 0 / failed rounds: 0
hidden DFA states: 6
held-out / long / SAT held-out / SAT long noise floor: 0.022 / 0.035 / 0.022 / 0.035

## Arms, per round

| arm | round | held-out | long | reported failures | n failures | failures noise floor | regressions | n D0-correct | regressions noise floor | ledger total | n counterexamples | accepted | attempts | src lines | literal matches | SAT status | SAT states | ambiguity | SAT held-out | SAT long | frac weights changed | max |delta| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 0 | 0 | - | - | - | - | sat | 6 | ambiguous | 0.888 | 0.825 | - | - |
| B | 0 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 0 | 0 | - | - | - | - | sat | 6 | ambiguous | 0.888 | 0.825 | - | - |
| D | 0 | 0.888 | 0.825 | - | 0 | - | 0.000 | 32 | 0.088 | 0 | 0 | - | - | - | - | sat | 6 | ambiguous | 0.888 | 0.825 | - | - |

## Per-length held-out accuracy (final round; n and noise floor per length)

| length | n | noise floor | A held-out | B held-out | D held-out | trained RNN |
|---|---|---|---|---|---|---|
| 1 | 0 | - | - | - | - | - |
| 2 | 2 | 0.354 | 1.000 | 1.000 | 1.000 | 0.500 |
| 3 | 6 | 0.204 | 1.000 | 1.000 | 1.000 | 0.667 |
| 4 | 14 | 0.134 | 0.929 | 0.929 | 0.929 | 0.714 |
| 5 | 29 | 0.093 | 0.966 | 0.966 | 0.966 | 0.655 |
| 6 | 61 | 0.064 | 0.885 | 0.885 | 0.885 | 0.738 |
| 7 | 114 | 0.047 | 0.912 | 0.912 | 0.912 | 0.667 |
| 8 | 106 | 0.049 | 0.849 | 0.849 | 0.849 | 0.717 |
| 9 | 42 | 0.077 | 0.881 | 0.881 | 0.881 | 0.595 |
| 10 | 42 | 0.077 | 0.952 | 0.952 | 0.952 | 0.595 |
| 11 | 42 | 0.077 | 0.833 | 0.833 | 0.833 | 0.786 |
| 12 | 42 | 0.077 | 0.833 | 0.833 | 0.833 | 0.762 |

## Baselines

| baseline | held-out | held-out noise floor | long | long noise floor | train acc | n train | train noise floor | steps |
|---|---|---|---|---|---|---|---|---|
| trained RNN h=16 | 0.692 | 0.022 | 0.610 | 0.035 | 1.000 | 32 | 0.088 | 2000 |

| parity one-shot | ran | accepted | accuracy | n test | noise floor |
|---|---|---|---|---|---|
|  | true | true | 0.595 | 200 | 0.035 |

## Cross-play (row: weights arm/round; column: counterexample set arm/round)

| weights \ set | A/0 | B/0 | D/0 |
|---|---|---|---|
| n | 0 | 0 | 0 |
| noise floor | - | - | - |
| A/0 | - | - | - |
| B/0 | - | - | - |
| D/0 | - | - | - |

## Oracle ledger

| arm | total | by length |
|---|---|---|

## LLM usage

| calls | prompt_tokens | completion_tokens | reasoning_tokens |
|---|---|---|---|
| 3 | 2392 | 28279 | 25675 |
| LLM refusals / errors | 0 / 0 |  |  |

## Timing

| field | seconds |
|---|---|
| total_s | 280.459 |
