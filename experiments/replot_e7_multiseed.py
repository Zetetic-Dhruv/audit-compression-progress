#!/usr/bin/env python3
"""Regenerate fig_e7.pdf from the 20-seed E7 summary, WITH 95% CI error bars (the prior
figure was single-seed and carried none). Two panels:
  left  = power + defenses at Q=8000 (per-release gap mean +/- 95% CI vs 2*Delta_n)
  right = panel-size scaling curve (gap mean +/- 95% CI vs 2*Delta_n, log n)
Writes straight into the arXiv source package figures directory.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "camera_ready", "camera_ready_results", "e7_multiseed_summary.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fig_e7.pdf")
assert os.path.isdir(os.path.dirname(OUT)), f"figures dir not found: {os.path.dirname(OUT)}"

d = json.load(open(SUMMARY))
nseeds = d["nseeds"]
defs = d["calibration_defenses_maxQ"]
thr = defs[0]["threshold"]

fig, (b1, b2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

labels = {"naive": "naive", "fresh_subsample": "fresh\nsubsample", "ladder": "ladder",
          "rounded": "rounded", "one_shot": "one-shot"}
names = [labels.get(x["defense"], x["defense"]) for x in defs]
gaps = [x["gap_mean"] for x in defs]
errs = [x["gap_sd"] for x in defs]
colz = ["#c0392b" if x["wins"] > 0 else "#2e86c1" for x in defs]
b1.bar(range(len(defs)), gaps, yerr=errs, capsize=3, color=colz)
for i, x in enumerate(defs):
    if x["gap_mean"] < 0.05:
        b1.text(i, 0.06, "0", ha="center", va="bottom", fontsize=7)
b1.axhline(thr, ls="--", color="k", lw=1)
b1.text(len(defs) - 1, thr + 0.10, r"$2\Delta_n=%.2f$" % thr, ha="right", va="bottom", fontsize=8)
b1.set_xticks(range(len(defs)))
b1.set_xticklabels(names, fontsize=7)
b1.set_ylabel("overfit gap (Q=8000)")
b1.set_title(f"power + defenses ({nseeds} seeds)", fontsize=8.5)

sc = d["scaling_curve"]
nn = [x["n"] for x in sc]
gg = [x["gap_mean"] for x in sc]
ge = [x["gap_sd"] for x in sc]
tt = [x["threshold"] for x in sc]
b2.errorbar(nn, gg, yerr=ge, fmt="o-", color="#c0392b", capsize=3, label="attacker gap")
b2.plot(nn, tt, "s--", color="k", lw=1, label=r"$2\Delta_n$")
b2.set_xscale("log")
b2.set_xlabel("panel size $n$")
b2.set_ylabel("gap")
b2.set_title(f"budget vs capacity ({nseeds} seeds)", fontsize=8.5)
b2.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUT)
plt.close(fig)
print("wrote", OUT)
