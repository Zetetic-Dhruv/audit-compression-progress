#!/usr/bin/env python3
"""Generate camera-ready figures + table numbers for the Audit-CP results from the banked
per-seed JSONs (results_data/). Figures -> P3/figures/*.pdf. Table numbers printed to stdout.
Run: python3 make_results_figures.py
"""
import json, glob, os, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "results_data")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.5})

def load(exp):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, f"{exp}_seed*.json"))):
        r = json.load(open(f)); out.append(r.get(exp, r))
    return out

def line(s): print(s)

# ---------- E5: uniform-deviation concentration ----------
e5 = load("e5")
byn = defaultdict(list)
for r in e5:
    for row in r["by_n"]: byn[row["n"]].append(row["delta_n_median"])
ns = np.array(sorted(byn)); dd = np.array([np.mean(byn[n]) for n in ns]); ds = np.array([np.std(byn[n]) for n in ns])
slope, intc = np.polyfit(np.log(ns), np.log(dd), 1)
fig, ax = plt.subplots(figsize=(3.2, 2.5))
ax.errorbar(ns, dd, yerr=ds, fmt="o", color="C0", capsize=2, label=f"observed (slope {slope:.2f})")
xx = np.array([ns.min(), ns.max()]); ax.plot(xx, np.exp(intc)*xx**slope, "C0-", lw=1)
ref = dd[0]*(xx/ns[0])**-0.5; ax.plot(xx, ref, "k--", lw=1, label=r"theory $n^{-1/2}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("audit panel size $n$"); ax.set_ylabel(r"uniform deviation $\Delta_n$")
ax.legend(frameon=False, fontsize=8); fig.savefig(os.path.join(FIG, "fig_e5.pdf")); plt.close(fig)
line(f"[E5] n={len(e5)} slope={slope:.3f} (theory -0.5); Delta_n by n: " + ", ".join(f"{n}:{np.mean(byn[n]):.4f}" for n in ns))

# ---------- C1: noisy-TV reward ablation ----------
c1 = load("c1")
pols = ["audit_cp","prediction_error","rnd","icm","uniform","oracle_learnable"]
labels = ["audit-CP","pred-err","RND","ICM","uniform","oracle"]
acc = {p: [] for p in pols}; dfr = {p: [] for p in pols}
for r in c1:
    bp = r["by_policy"]
    for p in pols:
        if p in bp: acc[p].append(bp[p]["final_active_cell_acc"]); dfr[p].append(bp[p]["distractor_fraction"])
am = [np.mean(acc[p]) for p in pols]; asd = [np.std(acc[p]) for p in pols]; dm = [np.mean(dfr[p]) for p in pols]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.5))
cols = ["C2","C3","C0","C1","0.6","0.3"]
a1.bar(range(len(pols)), am, yerr=asd, color=cols, capsize=2)
a1.set_xticks(range(len(pols))); a1.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
a1.set_ylabel("active-cell accuracy")
_ylo = min(m - s for m, s in zip(am, asd)); _yhi = max(m + s for m, s in zip(am, asd))
a1.set_ylim(_ylo - 0.01, _yhi + 0.01)  # widen so error-bar whiskers are not clipped (fixes fig_c1 artifact)
a2.bar(range(len(pols)), dm, color=cols)
a2.axhline(0.25, color="k", ls=":", lw=0.8); a2.text(len(pols)-1, 0.265, "uniform", fontsize=6.5, ha="right")
a2.set_xticks(range(len(pols))); a2.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
a2.set_ylabel("distractor fraction")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_c1.pdf")); plt.close(fig)
contr = defaultdict(list)
for r in c1:
    for k,v in r.get("audit_cp_minus_curiosity_active_acc",{}).items(): contr[k].append(v)
line(f"[C1] n={len(c1)} active_acc: " + ", ".join(f"{l}:{m:.3f}" for l,m in zip(labels,am)))
line("[C1] distractor_frac: " + ", ".join(f"{l}:{d:.3f}" for l,d in zip(labels,dm)))
line("[C1] contrast audit_cp-curiosity: " + ", ".join(f"{k}:{np.mean(v):+.4f}+/-{np.std(v):.4f}(pos {int((np.array(v)>0).sum())}/{len(v)})" for k,v in contr.items()))

# ---------- E7: power + defenses (cal) and scaling (scal) ----------
cal = load("e7cal")[0]; scal = load("e7scal")[0]
dmax = cal["defenses_at_maxQ"]; defs = [d["defense"] for d in dmax]; gaps = [d["gap"] for d in dmax]; thr = dmax[0]["threshold_2delta"]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(5.4, 2.5))
colz = ["C3" if d["attacker_wins"] else "C2" for d in dmax]
b1.bar(range(len(defs)), gaps, color=colz)
for _i, _g in enumerate(gaps):  # annotate zero-gap defenses so they read as 0, not missing (fixes fig_e7 artifact)
    if _g < 0.05: b1.text(_i, 0.06, "0", ha="center", va="bottom", fontsize=7, color="0.35")
b1.axhline(thr, color="k", ls="--", lw=1); b1.text(len(defs)-1, thr*1.05, r"$2\Delta_n$", fontsize=7.5, ha="right")
b1.set_xticks(range(len(defs))); b1.set_xticklabels([d.replace("_","\n") for d in defs], fontsize=6.8)
b1.set_ylabel(f"overfit gap (Q={dmax[0]['Q']})"); b1.set_title("power + defenses", fontsize=8.5)
byn7 = scal["by_n"]; nn = [r["n"] for r in byn7]; gg = [r["gap"] for r in byn7]; tt = [r["threshold_2delta"] for r in byn7]
b2.plot(nn, gg, "o-", color="C3", label="attacker gap"); b2.plot(nn, tt, "s--", color="k", label=r"$2\Delta_n$")
b2.set_xscale("log"); b2.set_xlabel("panel size $n$"); b2.set_ylabel("gap"); b2.set_title("budget vs capacity", fontsize=8.5)
b2.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_e7.pdf")); plt.close(fig)
line(f"[E7cal] HAS_POWER={cal['HAS_POWER']} defenses_below_threshold={cal['defenses_below_threshold']}")
line("[E7cal] gap@maxQ by defense (thr=%.3f): "%thr + ", ".join(f"{d['defense']}:{d['gap']:.3f}{'(WIN)' if d['attacker_wins'] else ''}" for d in dmax))
line("[E7scal] " + ", ".join(f"n{r['n']}:gap{r['gap']:.2f}/thr{r['threshold_2delta']:.2f}{'X' if r['crosses'] else ''}" for r in byn7))

# ---------- Tables (numbers) ----------
e3 = load("e3")
cs=[r["cum_signed"] for r in e3]; cc=[r["cum_clipped"] for r in e3]; eg=[r["exploit_gap"] for r in e3]; ei=[r["endpoint_improvement"] for r in e3]
line(f"[E3] n={len(e3)} cum_signed={np.mean(cs):.3f} endpoint={np.mean(ei):.3f} cum_clipped={np.mean(cc):.3f} exploit_gap={np.mean(eg):.3f}+/-{np.std(eg):.3f} (pos {int((np.array(eg)>0).sum())}/{len(eg)})")
e4a = load("e4a"); byn4=defaultdict(lambda: defaultdict(list))
for r in e4a:
    for row in r["by_n"]:
        for k in ("empirical_progress","true_progress","false_positive_excess"): byn4[row["n_total"]][k].append(row[k])
for n in sorted(byn4):
    d=byn4[n]; line(f"[E4a] n_total={n}: emp={np.mean(d['empirical_progress']):.3f} true={np.mean(d['true_progress']):.3f} false_pos_excess={np.mean(d['false_positive_excess']):.3f}+/-{np.std(d['false_positive_excess']):.3f}")
e6 = load("e6")
T=[r["temperature"] for r in e6]; n0=[r["nll"][0] for r in e6]; n1=[r["nll"][1] for r in e6]; e0=[r["ece"][0] for r in e6]; e1_=[r["ece"][1] for r in e6]; da=[r["delta_acc"] for r in e6]; a0=[r["acc"][0] for r in e6]
line(f"[E6] n={len(e6)} T={np.mean(T):.3f} acc={np.mean(a0):.4f} d_acc={np.mean(da):.5f} maxabs_dacc={np.max(np.abs(da)):.5f} NLL {np.mean(n0):.4f}->{np.mean(n1):.4f} ECE {np.mean(e0):.4f}->{np.mean(e1_):.4f}")
e1 = load("e1"); byk=defaultdict(lambda: defaultdict(list))
for r in e1:
    for row in r["by_k"]:
        for k in ("cum_stream_cp","cum_fresh_audit_cp","gamma_SA"): byk[row["k"]][k].append(row[k])
for k in sorted(byk):
    d=byk[k]; line(f"[E1] k={k}: stream_CP={np.mean(d['cum_stream_cp']):.2f} audit_CP={np.mean(d['cum_fresh_audit_cp']):.3f} gamma_SA={np.mean(d['gamma_SA']):.2f}")
line(f"\nFigures written to {FIG}: " + ", ".join(sorted(os.listdir(FIG))))
