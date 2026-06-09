#!/usr/bin/env python3
"""Multi-seed re-run of the E7 power calibration (cells=1) and the panel-size scaling curve.

Why: the red-team review noted the headline E7 numbers (naive gap 3.28 > 2*Delta_n = 1.64 at
Q=8000, n=200) were reported from a SINGLE seed (seed 0). The calibration is pure NumPy
(zeros base logits + RANDOM integer panel labels drawn from cfg.seed + a NumPy boosting
attacker also seeded from cfg.seed); it is fully seed-DEPENDENT, it was simply only run once.
This script runs the SAME drivers (cp_experiments.run_e7_power_calibration / run_e7_scaling)
across NSEEDS seeds and reports mean +/- 95% CI of the gap and the cross-threshold win-rate,
so the headline becomes a distribution rather than an anecdote.

Faithfulness check: seed 0 must reproduce the banked gap 3.2799 exactly.

Usage:  python3 e7_multiseed.py [NSEEDS]      (default 20, matching the rest of the suite)
"""
import json, math, sys
from pathlib import Path
import numpy as np
import cp_experiments as E
from cp_experiments import Cfg

NSEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = list(range(NSEEDS))


def stats(xs):
    xs = np.asarray(xs, float)
    m = float(xs.mean())
    sd = float(xs.std(ddof=1)) if len(xs) > 1 else 0.0
    half = float(1.96 * sd / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return m, sd, half, float(xs.min()), float(xs.max())


print(f"=== E7 multi-seed re-run, {NSEEDS} seeds ===\n")

cal = [E.run_e7_power_calibration(Cfg.from_scale("full", seed=s)) for s in SEEDS]
Qs = [d["Q"] for d in cal[0]["naive_sweep"]]

print("E7 power calibration (cells=1, naive release):")
naive_out = []
for qi, Q in enumerate(Qs):
    gaps = [r["naive_sweep"][qi]["gap"] for r in cal]
    thr = [r["naive_sweep"][qi]["threshold_2delta"] for r in cal]
    wins = sum(r["naive_sweep"][qi]["attacker_wins"] for r in cal)
    m, sd, half, lo, hi = stats(gaps)
    tm = float(np.mean(thr))
    print(f"  Q={Q:5d}: gap {m:6.3f} +/- {half:.3f} (95%CI) [sd {sd:.3f}, min {lo:.3f}, max {hi:.3f}] "
          f"| thr {tm:.3f} | wins {wins}/{NSEEDS}")
    naive_out.append({"Q": Q, "gap_mean": m, "gap_sd": sd, "gap_ci95": half,
                      "gap_min": lo, "gap_max": hi, "threshold": tm, "wins": wins})

s0 = cal[0]["naive_sweep"][-1]
ok = abs(s0["gap"] - 3.2799) < 1e-3
print(f"\nFAITHFULNESS seed-0 @ Q={s0['Q']}: gap {s0['gap']:.4f} (banked 3.2799) "
      f"reusable {s0['reusable_cp']:.4f} fresh {s0['fresh_cp']:.4f} -> {'MATCH' if ok else 'MISMATCH'}")

print(f"\nDefenses at Q={max(Qs)} (cells=1):")
defnames = [d["defense"] for d in cal[0]["defenses_at_maxQ"]]
def_out = []
for di, dn in enumerate(defnames):
    gaps = [r["defenses_at_maxQ"][di]["gap"] for r in cal]
    thr = [r["defenses_at_maxQ"][di]["threshold_2delta"] for r in cal]
    wins = sum(r["defenses_at_maxQ"][di]["attacker_wins"] for r in cal)
    m, sd, half, lo, hi = stats(gaps)
    print(f"  {dn:16s}: gap {m:6.3f} +/- {half:.3f} | thr {float(np.mean(thr)):.3f} | wins {wins}/{NSEEDS}")
    def_out.append({"defense": dn, "gap_mean": m, "gap_sd": sd, "gap_ci95": half,
                    "threshold": float(np.mean(thr)), "wins": wins})

scal = [E.run_e7_scaling(Cfg.from_scale("full", seed=s)) for s in SEEDS]
ns = [o["n"] for o in scal[0]["by_n"]]
print(f"\nE7 scaling curve (cells=1, Q=8000, panel size n):")
scal_out = []
for ni, n in enumerate(ns):
    gaps = [r["by_n"][ni]["gap"] for r in scal]
    thr = [r["by_n"][ni]["threshold_2delta"] for r in scal]
    crosses = sum(r["by_n"][ni]["crosses"] for r in scal)
    m, sd, half, lo, hi = stats(gaps)
    print(f"  n={n:6d}: gap {m:7.3f} +/- {half:.3f} | thr {float(np.mean(thr)):.3f} | crosses {crosses}/{NSEEDS}")
    scal_out.append({"n": n, "gap_mean": m, "gap_sd": sd, "gap_ci95": half,
                     "threshold": float(np.mean(thr)), "crosses": crosses})

out = {"nseeds": NSEEDS, "seeds": SEEDS,
       "seed0_faithful": bool(ok), "seed0_gap_Q8000": float(s0["gap"]),
       "calibration_naive": naive_out, "calibration_defenses_maxQ": def_out,
       "scaling_curve": scal_out}
dst = Path(__file__).resolve().parent / "camera_ready_results" / "e7_multiseed_summary.json"
dst.write_text(json.dumps(out, indent=2))
print(f"\nwrote {dst}")
