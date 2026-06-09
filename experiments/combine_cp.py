"""Aggregate per-seed cp_experiments JSON shards into a camera-ready summary.

The orchestrator writes one shard per (experiment, seed) at
    <dir>/<exp>_seed<S>.json   ==  {"<exp>": <driver-result-dict>, ...}
This combiner groups by experiment and reports mean / std / 95% CI across seeds for each
experiment's headline metrics (paired-by-seed where relevant), keeping raw per-seed values.

Usage: python3 combine_cp.py --dir camera_ready_results/json --out camera_ready_results/summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def _stats(xs: List) -> Optional[Dict]:
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    a = np.asarray(xs, dtype=float)
    s = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "std": s,
            "ci95": (1.96 * s / math.sqrt(len(a)) if len(a) > 1 else 0.0), "n": len(a)}


def _agg_c1(recs: List[Dict]) -> Dict:
    pols = recs[0].get("by_policy", {}).keys()
    out = {}
    for p in pols:
        out[p] = {
            "final_active_cell_acc": _stats([r["by_policy"][p].get("final_active_cell_acc") for r in recs]),
            "final_fresh_floored_ce": _stats([r["by_policy"][p].get("final_fresh_floored_ce") for r in recs]),
            "distractor_fraction": _stats([r["by_policy"][p].get("distractor_fraction") for r in recs]),
        }
    # primary contrast: MWU(audit_cp) vs curiosity rewards (paired by seed)
    contrast = {}
    for p in ("prediction_error", "rnd", "icm"):
        diffs = [r["by_policy"]["audit_cp"]["final_active_cell_acc"]
                 - r["by_policy"][p]["final_active_cell_acc"] for r in recs
                 if "audit_cp" in r["by_policy"] and p in r["by_policy"]]
        contrast[f"audit_cp_minus_{p}"] = _stats(diffs)
    return {"by_policy": out, "primary_contrast_active_acc": contrast,
            "_headline": f"MWU(audit_cp) active-acc vs curiosity: {contrast}"}


def _agg_e7(recs: List[Dict]) -> Dict:
    keyed = defaultdict(list)
    for r in recs:
        for row in r.get("results", []):
            keyed[(row["attacker"], row["defense"], row["Q"])].append(row)
    cells = {}
    for (atk, dfn, Q), rows in sorted(keyed.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        cells[f"{atk}|{dfn}|Q{Q}"] = {
            "gap": _stats([x["gap"] for x in rows]),
            "threshold_2delta": _stats([x["threshold_2delta"] for x in rows]),
            "win_fraction": float(np.mean([1.0 if x["attacker_wins"] else 0.0 for x in rows])),
        }
    return {"by_attacker_defense_Q": cells,
            "_headline": "gap vs 2*Delta_n per (attacker,defense,Q); win_fraction across seeds"}


def _agg_e3(recs: List[Dict]) -> Dict:
    return {"cum_signed": _stats([r.get("cum_signed") for r in recs]),
            "cum_clipped": _stats([r.get("cum_clipped") for r in recs]),
            "exploit_gap": _stats([r.get("exploit_gap") for r in recs]),
            "endpoint_improvement": _stats([r.get("endpoint_improvement") for r in recs]),
            "_headline": "clipped >= signed (exploit_gap >= 0); signed ~ endpoint"}


def _agg_e1(recs: List[Dict]) -> Dict:
    ks = [d["k"] for d in recs[0].get("by_k", [])]
    by_k = {f"k={k}": _stats([next(d["gamma_SA"] for d in r["by_k"] if d["k"] == k) for r in recs])
            for k in ks}
    return {"gamma_SA_by_k": by_k,
            "slope_in_k": _stats([r.get("gamma_SA_slope_in_k") for r in recs]),
            "_headline": "stream-audit gap gamma_SA increasing in k"}


def _agg_e5(recs: List[Dict]) -> Dict:
    return {"loglog_slope": _stats([r.get("loglog_slope") for r in recs]),
            "expected_slope": -0.5, "_headline": "Delta_n ~ n^slope, expect ~ -0.5"}


def _agg_e6(recs: List[Dict]) -> Dict:
    return {"delta_nll": _stats([r.get("delta_nll") for r in recs]),
            "delta_ece": _stats([r.get("delta_ece") for r in recs]),
            "delta_acc": _stats([r.get("delta_acc") for r in recs]),
            "_headline": "NLL/ECE down, accuracy ~ flat"}


def _agg_e4a(recs: List[Dict]) -> Dict:
    ns = [d["n_total"] for d in recs[0].get("by_n", [])]
    gap_by_n = {f"n={n}": _stats([next(d["memorization_gap"] for d in r["by_n"] if d["n_total"] == n)
                                  for r in recs]) for n in ns}
    return {"memorization_gap_by_n": gap_by_n,
            "_headline": "white-box memorization gap shrinks with n"}


_AGG = {"c1": _agg_c1, "e7": _agg_e7, "e3": _agg_e3, "e1": _agg_e1,
        "e5": _agg_e5, "e6": _agg_e6, "e4a": _agg_e4a}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "*_seed*.json")))
    by_exp: Dict[str, List[Dict]] = defaultdict(list)
    for f in files:
        exp = os.path.basename(f).split("_seed")[0]
        try:
            blob = json.load(open(f))
        except Exception as e:
            print(f"  skip {f}: {e}"); continue
        rec = blob.get(exp, blob)            # CLI writes {exp: result}; unwrap
        by_exp[exp].append(rec)
    summary = {"n_shard_files": len(files),
               "seeds_per_experiment": {e: len(v) for e, v in by_exp.items()},
               "experiments": {}}
    for exp, recs in by_exp.items():
        try:
            summary["experiments"][exp] = _AGG[exp](recs) if exp in _AGG else {"raw_seeds": recs}
        except Exception as e:
            summary["experiments"][exp] = {"aggregation_error": str(e), "n_seeds": len(recs)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2, default=float)
    print(f"combine_cp: wrote {args.out} ({len(by_exp)} experiments, {len(files)} shards)")
    for e, v in summary["experiments"].items():
        print(f"  {e}: {v.get('_headline', list(v)[:3])}")


if __name__ == "__main__":
    main()
