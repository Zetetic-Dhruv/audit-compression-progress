#!/usr/bin/env python3
"""Regenerate fig_boundary_witnesses.pdf and fig_e5.pdf from the banked per-seed shards
(results_data/). This reconstructs fig_boundary, whose original generator did not survive,
and fixes two review-flagged sub-panel defects:
  - E4a: the legend overlapped the n=144 bars -> ylim headroom lifts it clear of the bars.
  - E6:  NLL/ECE/Accuracy shared one 0-0.9 axis (calibration metrics squashed) -> a twin
         axis puts NLL/ECE on a readable left scale and Accuracy (full 0-1) on the right,
         with the (small, by-construction) deltas annotated numerically.
fig_e5's legend is emitted at 3-decimal precision (-0.527) to match the text/caption/table.

Usage: python3 regen_paper_figs.py [OUTDIR]   (default: alongside this script)
"""
import json, glob, os, sys
import numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results_data")
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})
GRAY, GREEN, RED, BLUE = "0.6", "#2e8b57", "#c0392b", "#2e86c1"

def load(exp):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, f"{exp}_seed*.json"))):
        r = json.load(open(f)); out.append(r.get(exp, r))
    return out

# =================== fig_boundary_witnesses (2x2) ===================
fig, ((e1ax, e3ax), (e4ax, e6ax)) = plt.subplots(2, 2, figsize=(8.2, 5.6))
w = 0.38

# E1: stream scoring overstates progress
e1 = load("e1"); byk = defaultdict(lambda: defaultdict(list))
for r in e1:
    for row in r["by_k"]:
        byk[row["k"]]["s"].append(row["cum_stream_cp"]); byk[row["k"]]["a"].append(row["cum_fresh_audit_cp"])
ks = sorted(byk); x = np.arange(len(ks))
sm = [np.mean(byk[k]["s"]) for k in ks]; am = [np.mean(byk[k]["a"]) for k in ks]
e1ax.bar(x - w/2, sm, w, color=RED, label="stream-scored CP")
e1ax.bar(x + w/2, am, w, color=BLUE, label="sealed-audit CP")
e1ax.set_xticks(x); e1ax.set_xticklabels([f"{k:g}" for k in ks]); e1ax.set_xlabel("stream bias $k$")
e1ax.set_ylabel("cumulative CP"); e1ax.set_title("E1: stream scoring overstates progress", fontsize=9)
e1ax.set_ylim(top=max(sm) * 1.30); e1ax.legend(fontsize=7.2, loc="upper right", framealpha=0.9)

# E3: clipping farms reward
e3 = load("e3")
cs = np.mean([r["cum_signed"] for r in e3]); cc = np.mean([r["cum_clipped"] for r in e3])
gap = np.mean([r["exploit_gap"] for r in e3])
e3ax.bar([0, 1], [cs, cc], color=[BLUE, RED], width=0.6)
e3ax.axhline(0, color="k", lw=0.6)
e3ax.set_xticks([0, 1]); e3ax.set_xticklabels(["signed\nCP", "clipped\nCP"])
e3ax.set_ylabel("cumulative reward"); e3ax.set_title("E3: clipping farms reward", fontsize=9)
e3ax.set_ylim(min(cs * 1.6, -0.1), cc * 1.25)
e3ax.annotate(f"$+{gap:.2f}$ gap", xy=(0.5, cc), ha="center", va="bottom", fontsize=8)

# E4a: memorization reverses the sign (legend lifted clear of the bars)
e4 = load("e4a"); byn = defaultdict(lambda: defaultdict(list))
for r in e4:
    for row in r["by_n"]:
        byn[row["n_total"]]["e"].append(row["empirical_progress"]); byn[row["n_total"]]["t"].append(row["true_progress"])
nns = sorted(byn); x4 = np.arange(len(nns))
em = [np.mean(byn[n]["e"]) for n in nns]; tm = [np.mean(byn[n]["t"]) for n in nns]
e4ax.bar(x4 - w/2, em, w, color=RED, label="reusable panel CP")
e4ax.bar(x4 + w/2, tm, w, color=BLUE, label="fresh-audit CP")
e4ax.axhline(0, color="k", lw=0.6)
e4ax.set_xticks(x4); e4ax.set_xticklabels([str(n) for n in nns]); e4ax.set_xlabel("panel size $n$")
e4ax.set_ylabel("cumulative CP"); e4ax.set_title("E4a: memorization reverses the sign", fontsize=9)
e4ax.set_ylim(min(tm) * 1.30, max(em) * 1.75)   # headroom so the legend clears the bars
e4ax.legend(fontsize=7.2, loc="upper center", ncol=2, framealpha=0.9)

# E6: calibration without discrimination change (twin axis: NLL/ECE left, Accuracy right)
e6 = load("e6")
nll0, nll1 = np.mean([r["nll"][0] for r in e6]), np.mean([r["nll"][1] for r in e6])
ece0, ece1 = np.mean([r["ece"][0] for r in e6]), np.mean([r["ece"][1] for r in e6])
acc0, acc1 = np.mean([r["acc"][0] for r in e6]), np.mean([r["acc"][1] for r in e6])
wj = 0.34
e6ax.bar(0 - wj/2, nll0, wj, color=GRAY, label="before")
e6ax.bar(0 + wj/2, nll1, wj, color=GREEN, label="after temp. scaling")
e6ax.bar(1 - wj/2, ece0, wj, color=GRAY); e6ax.bar(1 + wj/2, ece1, wj, color=GREEN)
e6ax.set_ylim(0, max(nll0, nll1) * 1.45); e6ax.set_ylabel("NLL / ECE (nats)")
e6r = e6ax.twinx(); e6r.spines["top"].set_visible(False)
e6r.bar(2 - wj/2, acc0, wj, color=GRAY); e6r.bar(2 + wj/2, acc1, wj, color=GREEN)
e6r.set_ylim(0, 1.0); e6r.set_ylabel("accuracy")
e6ax.set_xticks([0, 1, 2]); e6ax.set_xticklabels(["NLL", "ECE", "Accuracy"])
e6ax.set_title("E6: calibration without discrimination change", fontsize=9)
e6ax.legend(fontsize=7.0, loc="upper left", framealpha=0.9)
e6ax.annotate(f"$\\Delta$NLL$={nll1-nll0:+.4f}$\n$\\Delta$ECE$={ece1-ece0:+.4f}$\n$\\Delta$acc$={acc1-acc0:+.4f}$",
              xy=(0.37, 0.56), xycoords="axes fraction", fontsize=6.4, va="center")

fig.suptitle("Boundary witnesses: each broken proxy violates a different assumption", fontsize=10, y=1.005)
fig.tight_layout()
bpath = os.path.join(OUTDIR, "fig_boundary_witnesses.pdf"); fig.savefig(bpath); plt.close(fig)
print("wrote", bpath)

# =================== fig_e5 (legend at 3 decimals -> -0.527) ===================
e5 = load("e5"); byn5 = defaultdict(list)
for r in e5:
    for row in r["by_n"]: byn5[row["n"]].append(row["delta_n_median"])
ns5 = np.array(sorted(byn5)); dd = np.array([np.mean(byn5[n]) for n in ns5]); ds = np.array([np.std(byn5[n]) for n in ns5])
slope, intc = np.polyfit(np.log(ns5), np.log(dd), 1)
f5, ax = plt.subplots(figsize=(3.2, 2.5))
ax.errorbar(ns5, dd, yerr=ds, fmt="o", color="C0", capsize=2, label=f"observed (slope {slope:.3f})")
xx = np.array([ns5.min(), ns5.max()]); ax.plot(xx, np.exp(intc) * xx**slope, "C0-", lw=1)
ax.plot(xx, dd[0] * (xx / ns5[0])**-0.5, "k--", lw=1, label=r"theory $n^{-1/2}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("audit panel size $n$")
ax.set_ylabel(r"uniform deviation $\Delta_n$")
ax.legend(frameon=False, fontsize=8); f5.tight_layout()
epath = os.path.join(OUTDIR, "fig_e5.pdf"); f5.savefig(epath); plt.close(f5)
print("wrote", epath, "| slope=%.4f" % slope)
