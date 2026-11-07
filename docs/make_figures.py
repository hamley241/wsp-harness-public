"""Regenerate every figure on the results page from the released records.
Run from the repository root: python docs/make_figures.py  (needs numpy and matplotlib)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R = "records/"; OUT = "docs/assets/"; F = 0.0224
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def last_round(arm):
    rounds = arm["rounds"] if isinstance(arm, dict) else arm
    rounds = [r for r in rounds if isinstance(r, dict) and r.get("heldout") is not None]
    return max(rounds, key=lambda r: r.get("round", 0))["heldout"]

def seed_row_from_record(path):
    d = json.load(open(path)); a = d["arms"]
    n = last_round(a["N"]); e = (last_round(a["E1"]) + last_round(a["E2"])) / 2; g = last_round(a["G"])
    d0 = d["D0"]["heldout"] if isinstance(d["D0"], dict) else d["D0"]
    return {"seed": d["seed"], "D0": d0, "gE": e - n, "gG": g - n}

def per_seed(path):
    d = json.load(open(path)); return {s["seed"]: s for s in d["outcome"]["seeds"] if s.get("status") == "ok"}

# Figure 0: the two routes
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
def box(x, y, w, h, text, fc, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15", fc=fc, ec="#444", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#444"))
ax.text(0.2, 4.25, "What we asked for: repair", fontsize=12, fontweight="bold", color="#4C72B0")
box(0.2, 2.9, 2.6, 0.9, "trained network\n(321 weights) + its mistakes", "#dbe5f3"); box(3.8, 2.9, 2.4, 0.9, "read the weights,\nfind the fault", "#dbe5f3"); box(7.2, 2.9, 2.6, 0.9, "patch a few numbers:\nthe same network, fixed", "#dbe5f3")
arrow(2.85, 3.35, 3.75, 3.35); arrow(6.25, 3.35, 7.15, 3.35)
ax.text(0.2, 2.25, "What both models did on the tested seed: reconstruction", fontsize=12, fontweight="bold", color="#C44E52")
box(0.2, 0.8, 2.6, 0.9, "the labelled examples, embedded literally\n(32 to 37 strings for Opus, 32 to 40 for astra)", "#f6dcdc", fs=9); box(3.8, 0.8, 2.4, 0.9, "the written program searches for\nthe smallest consistent automaton", "#f6dcdc", fs=9.5); box(7.2, 0.8, 2.6, 0.9, "and writes a brand-new network\nthat encodes it", "#f6dcdc", fs=9.5)
arrow(2.85, 1.25, 3.75, 1.25); arrow(6.25, 1.25, 7.15, 1.25)
ax.text(5, 0.25, "Accuracy reaches 1.000 in exactly the round in which the queried labels first identify the automaton (offline SAT check).", ha="center", fontsize=9.5, style="italic")
ax.text(5, -0.05, "The programs returned complete replacement weight arrays and took no supplied weights as input. Same accuracy, different procedure evaluated.", ha="center", fontsize=9.5, style="italic")
fig.tight_layout(); fig.savefig(OUT + "fig0-reconstruction.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# Figure 1: construction pilot
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (f, title) in zip(axes, [(R + "pilot/run/results.json", "Claude Opus 5.5"), (R + "second-model/pilot-astra/results.json", "gpt-6-astra")]):
    arms = json.load(open(f))["arms"]
    for arm, lab, mk in [("A", "adversarial queries", "o"), ("B", "random queries", "s"), ("D", "self-refinement (no new labels)", "^")]:
        rs = arms[arm]; x = [r["round"] for r in rs]; y = [r["heldout"] for r in rs]
        ax.plot(x, y, marker=mk, label=lab)
        for r in rs:
            if (r.get("sat") or {}).get("ambiguity") == "unique":
                ax.annotate("queried labels first\nidentify the automaton", (r["round"], r["heldout"]), textcoords="offset points", xytext=(0, -34), ha="center", fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6)); break
    ax.set_title(title); ax.set_xlabel("round"); ax.set_xticks(x); ax.set_ylim(0.45, 1.03); ax.axhline(0.5, color="grey", lw=0.6, ls=":")
axes[0].set_ylabel("held-out accuracy (500 strings)"); axes[0].legend(loc="lower left", fontsize=9)
fig.suptitle("Experiment 1, unconstrained construction, one seed: accuracy reaches 1.000 in the round the queried labels first identify the automaton", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT + "fig1-construction.png", dpi=160); plt.close(fig)

# Figure 2: constrained editing
opus = per_seed(R + "redesign/run/study.json"); opus[1] = seed_row_from_record(R + "redesign/run-seed1/seed_1.json")
astra = per_seed(R + "second-model/repair-astra/study.json")
seeds = [1, 15, 20, 21, 24]; x = np.arange(len(seeds)); w = 0.26
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(x - w, [opus[s]["gE"] for s in seeds], w, label="Claude Opus 5.5, numeric edits (strict output protocol)", color="#4C72B0")
ax.bar(x, [astra[s]["gE"] for s in seeds], w, label="gpt-6-astra, numeric edits (relaxed output protocol)", color="#DD8452")
ax.bar(x + w, [opus[s]["gG"] for s in seeds], w, label="projected Adam, gradient access, same budget", color="#55A868")
ax.axhline(F, color="k", lw=0.8, ls="--"); ax.text(len(seeds) - 0.55, F + 0.004, "threshold f = 0.0224", fontsize=8, ha="right"); ax.axhline(0, color="grey", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"seed {s}\nstart {opus[s]['D0']:.3f}" for s in seeds]); ax.set_ylabel("held-out gain after round 3 (over no edit)")
ax.set_title("Experiment 2, constrained numeric editing\n(at most 24 edits per round, 48 distinct coordinates over three rounds, no change larger than 1.0)", fontsize=11); ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig2-editing.png", dpi=160); plt.close(fig)
print("figure 2 values:", {s: (round(opus[s]["gE"], 3), round(astra[s]["gE"], 3), round(opus[s]["gG"], 3)) for s in seeds})

# Figure 3: search calibration
cells = json.load(open(R + "search-calibration/study.json"))["outcome"]["cells"]
fig, ax = plt.subplots(figsize=(7, 3.8))
for opt, lab, mk in [("rs-random24", "sparse random search (24/48/1.0 budget)", "o"), ("rs-probe24", "sparse probe search (24/48/1.0 budget)", "s"), ("es-dense", "dense evolution strategy (ignores sparsity)", "^")]:
    bs = sorted({c["budget"] for c in cells.values() if c["optimiser"] == opt})
    ys = [sum(1 for v in cells[f"{opt}/B{b}"]["per_seed"].values() if v.get("repaired")) for b in bs]
    ax.plot(bs, ys, marker=mk, label=lab)
ax.set_xscale("log"); ax.set_xlabel("objective evaluations per round"); ax.set_ylabel("seeds repaired (of 4)"); ax.set_yticks(range(5)); ax.set_ylim(-0.2, 4.3)
ax.axhline(3, color="k", ls="--", lw=0.8); ax.text(9000, 3.1, "criterion: 3 of 4", fontsize=8, ha="right")
ax.set_title("Follow-up: forward-pass-only search on four development seeds"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig3-search.png", dpi=160); plt.close(fig)
print("done")
