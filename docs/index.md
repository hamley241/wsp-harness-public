---
layout: default
title: Can a language model repair a network's weights by reading them?
---

**Short answer: not in our setting, and when it looked like it could, it was doing something else.**

A trained neural network is a list of numbers. Language models read and write text, and numbers are text. So it is tempting to hand a model a small network's weights and its mistakes and ask it to fix the numbers directly, the way one patches code. We tested this under controlled conditions: the decision rules for each experiment were fixed before its model calls, and the two exceptions (a seed-set correction after the first run, and a secondary rerun specified afterwards and reported as descriptive) are documented in the released rules. Two findings matter.

1. **Unconstrained construction can masquerade as repair.** On the one seed tested, two different models reached perfect accuracy in the query arms by writing a program that infers the hidden automaton from the labelled examples and compiles it into new weights. That is reconstruction, not evidence of repair.
2. **Under the constrained-edit rule, neither model met the prespecified criterion.** Each model's mean gain over no edit was at or below the threshold of 0.0224 on four of five seeds. Projected Adam, which has gradient access, exceeded it on all five; the seeds had been selected for Adam's feasibility, and the budget excludes the measured rewrite route rather than every conceivable one.

Paper: [extended abstract (PDF)](https://github.com/hamley241/wsp-harness-public/blob/main/paper/main.pdf). Code, decision rules (with their documented post-run amendments) and complete records: [the repository](https://github.com/hamley241/wsp-harness-public). Every number on this page can be recomputed offline from those records; the figures are produced by `docs/make_figures.py`.

## What the models actually did

![Repair as asked for, versus the reconstruction route both models took](assets/fig0-reconstruction.png)

*Top: the route we asked for. Bottom: the route both models took on the tested seed, as read from the programs they wrote.*

This is the part worth remembering. We asked for a repair: read the trained weights, find what is wrong, change a few numbers. What both models did instead was write a program that searches the labelled examples for the smallest consistent automaton and generates replacement weights. In plain terms, the program solves the problem and fills in weights; it does not read the weights it was given. Those programs are in the records: they embed the labelled strings literally, search for the smallest automaton consistent with them, and write a fresh network, with weights chosen to drive the units into saturation, that encodes it. An offline solver confirms the timing: in each query arm, accuracy reaches 1.000 in exactly the round in which the labels queried so far first identify a single automaton with at most six states, for both models and both query strategies. The model itself saw only the mistakes from each batch, not the full label set the solver checks. Before that round, accuracy stays at or near that of the solver's own consistent automaton.

That is a real and impressive result in this small setting: the emitted programs implement induction from labelled examples and programmatic network construction. But it means a perfect score established nothing about the network the model was given. The programs returned complete replacement weight arrays and took no supplied weights as input; in a separate measurement, compiled weights differed from trained ones on all 321 coordinates.

## The setting

- **Task.** A small six-state rule machine, an automaton, reads a binary string one bit at a time and decides whether to accept it. Because the rule is known to the experimenter, every prediction can be scored exactly.
- **Network.** A 16-unit recurrent network with 321 weights.
- **Measurement.** Accuracy on 500 held-out strings never shown to any arm, plus 200 longer ones. A gain counts only above a threshold of 0.0224, one standard error of an accuracy at chance with 500 strings, fixed in advance.
- **Experiment 1.** Each model saw its previous constructor program, 32 initial labels and the counterexamples from each query batch, and wrote a new program returning the weights; it could not run the network. Three arms per model, one seed.
- **Experiment 2.** Each model saw the numeric weights, all labels so far and the current counterexamples, and could only list numeric edits within a fixed budget; it could not run the network. Compared with doing nothing and with projected Adam under the same budget, on five seeds. A seed is one random choice of hidden automaton, training labels and starting network.

## Experiment 1: a free hand, and what the accuracy hid

![Per-round held-out accuracy for three construction arms, two models](assets/fig1-construction.png)

*Held-out accuracy by round for one seed. Arms: queries chosen by a second model call, random queries, and self-refinement with no new labels. The annotation marks the first round in which the queried labels identify the automaton uniquely among automata with at most six states, as checked offline; the model did not see that full label set.*

With queries that bring in new labelled strings, accuracy jumps to 1.000 in exactly that round. Without new labels nothing durable happens: the Opus arm stays flat and then collapses to 0.500, and the astra arm rises briefly from 0.890 to 0.924 before returning to 0.890.

## Experiment 2: small edits only

![Per-seed held-out gain for two models and projected Adam under the same budget](assets/fig2-editing.png)

*Gain after round 3 over no edit, averaged over two editor runs per seed. Opus used the primary strict output protocol and astra the relaxed one; projected Adam had gradient access; the five seeds were selected for projected-Adam feasibility.*

Now the model may change at most 24 numbers per round, 48 distinct coordinates over the three rounds, none by more than 1.0 from its trained value. This budget excludes the rewrite route measured in Experiment 1. Under the study's rule an editor reads *no repair* when its mean gain over no edit is at most 0.0224 on at least four of five seeds; it does not mean no seed improved. Both models read *no repair*. Projected Adam under the identical budget cleared the threshold on every seed. Among the Opus attempts that were rejected for format, several notes say the model could not simulate the network and was estimating.

## Follow-up: is the budget simply too tight for anything without gradients?

![Seeds repaired versus evaluations per round for three black-box searches](assets/fig3-search.png)

*Four development seeds disjoint from Experiment 2. The two sparse searches obey the 24/48/1.0 budget; the dense evolution strategy ignores the sparsity constraint and is shown for comparison only.*

Two sparse black-box searches with forward-pass access only failed a 3-of-4 criterion at 100, 1,000 and 10,000 objective evaluations per round; the best reached 2 of 4 at 10,000, and the dense strategy tied it there. Projected Adam itself cleared the threshold on only one of these four seeds. These searches show that the tested forward-pass-only methods struggled under these constraints; they do not bound what an editor with execution access could do.

## What this does and does not show

Experiment 2 found that neither tested editor met the prespecified five-seed criterion. That result is conditional on seeds chosen for comparator feasibility, on different output protocols for the two models, on one task, one architecture, one budget, and on no execution access. Experiment 1 is a separate one-seed construction study. Neither shows anything about language models and weights in general.

The practical lesson is for anyone evaluating a claim that a language model edited a network's weights. When the labelled examples and the stated task assumptions may suffice to reconstruct a solution, report the size and route of the change and compare against a labels-only reconstruction baseline. Held-out accuracy alone does not distinguish a local repair from a replacement.

## What is next

A controlled re-evaluation of a published one-shot weight-repair demonstration on a different task is in preparation. Its decision rule was fixed on 2026-11-07 before any model call; the SHA-256 of that rule text is `04a95f2cb96ebf2d3b7c7681f8c68aa9fca7ea1775596d921ddc19fd1b5d90bf`, and the text will be published with the records. It is a design, not a result, until its model arms have run; nothing from it is claimed on this page.

## How to check any number here

Every record, every model attempt, every charged query and every spend ledger is in the repository. The README lists the offline commands that regenerate each analysis from the raw records, and `docs/make_figures.py` regenerates the figures.

## Citing

Please cite the extended abstract in the repository's `paper/` directory and link this page. Experiments ran on 2026-11-04 and 2026-11-05; the repository was released on 2026-11-06 Pacific time (2026-11-07 UTC).
