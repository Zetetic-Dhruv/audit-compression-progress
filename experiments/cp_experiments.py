"""Audit-CP camera-ready experiment drivers (C1, E7, E3, E1, E5, E6, E4a).

All drivers run on the SOLE task source ARC-TGI (simon-arc-lab) via arc_tgi_source (imported
as crt): per-module generators, fresh draws/step, disjoint panels by seed, padded to 30x30
with a pad token (n_colors=11). The local procedural generators are retired. Models are the
real architectures in cp_models. Each run_* returns a results dict; the CLI dispatches by
subcommand. Configs default to TINY (smoke) values; pass --scale full for the
camera-ready sizes (PROTOCOL §8.4 / ROADMAP).

Panel hygiene (PROTOCOL §8.3): three disjoint panels by seed offset
  train stream  : fresh draws each step
  reusable A_n  : seed + 1000
  sealed B_m    : seed + 2000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import cp_models as M
import arc_tgi_source as crt        # ARC-TGI (simon-arc-lab) is the SOLE task source (see P3/URS.md)

DEV = M.device_auto()
M.AUDIT_IGNORE_INDEX = crt.PAD     # audit CP scores ARC-TGI CONTENT cells only (exclude pad token)


# -----------------------------------------------------------------------------
# Shared data / training helpers (over the locked generators)
# -----------------------------------------------------------------------------

def panel(seed: int, n_per_family: int, grid: int, task_indices: Sequence[int]
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint-by-seed audit panel over the given families (uses crt generators)."""
    return crt.build_audit_panel(seed, n_per_family, grid, list(task_indices))


def reward_probe_panel(seed: int, total: int, grid: int, task_indices: Sequence[int]
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small held-out probe for the PER-STEP audit-CP reward signal. ~`total` examples spread
    across the families. The per-step reward only needs an UNBIASED compression-progress
    estimate; the FULL `audit_per_family` panel is reserved for FINAL reported metrics.
    Evaluating the full panel every step is O(panel*steps) and explodes at full scale."""
    fams = [int(x) for x in task_indices]
    per = max(1, total // max(1, len(fams)))
    return crt.build_audit_panel(seed, per, grid, fams)


def make_model(kind: str, n_tasks: int, grid: int):
    return M.BASE_LEARNERS[kind](n_tasks=n_tasks, n_colors=crt.N_COLORS, grid_size=grid).to(DEV)


def train_batch(task_idx: int, n: int, rng: np.random.Generator, grid: int):
    x, y, t = crt.sample_task_batch(task_idx, n, rng, size=grid)
    return (torch.as_tensor(x, device=DEV), torch.as_tensor(y, device=DEV),
            torch.as_tensor(t, device=DEV))


def _train_step(model, opt, x, y, t, eps: float = 0.01) -> float:
    model.train()
    opt.zero_grad(set_to_none=True)
    loss = M.floored_cross_entropy(model(x, t), y, eps=eps)
    loss.backward(); opt.step()
    return float(loss.detach().cpu())


@dataclass
class Cfg:
    scale: str = "smoke"
    seed: int = 0
    grid: int = 12
    model: str = "cnn"
    # sizes (smoke); full overrides applied in from_scale
    steps: int = 30
    audit_per_family: int = 8       # FINAL reported metric panel (per family) -- PROTOCOL >=512
    reward_probe: int = 48          # PER-STEP audit-CP reward probe (TOTAL across families).
    # Per-step reward needs only an UNBIASED CP estimate; evaluating the full audit_per_family
    # panel every step is O(panel*steps) and explodes at full scale (the doomed first launch).
    batch: int = 32
    lr: float = 1e-3

    @staticmethod
    def from_scale(scale: str, **over) -> "Cfg":
        if scale == "full":
            c = Cfg(scale="full", grid=30, steps=5000, audit_per_family=512,
                    reward_probe=512, batch=128)
        else:
            c = Cfg(scale="smoke")
        for k, v in over.items():
            setattr(c, k, v)
        return c


# =============================================================================
# C1 — curriculum reward-ablation (EXP3 fixed; vary the reward signal)
# =============================================================================

def run_c1(cfg: Cfg) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    tasks, learn, distract = crt.TASKS, crt.LEARNABLE_TASKS, crt.DISTRACTOR_TASKS
    k = len(tasks)
    # A = small PER-STEP audit-CP reward probe (was the full panel -> O(panel*steps) blowup).
    # B = full sealed panel for the FINAL reported metric (unchanged, PROTOCOL >=512/family).
    A = reward_probe_panel(cfg.seed + 1000, cfg.reward_probe, cfg.grid, learn)
    B = panel(cfg.seed + 2000, cfg.audit_per_family, cfg.grid, learn)   # sealed/fresh (final)

    # policy = (scheduler factory, reward signal name). EXP3 fixed for reward-based ones.
    reward_policies = ["audit_cp", "prediction_error", "rnd", "icm"]
    baseline_policies = ["uniform", "round_robin", "oracle_learnable"]
    out = {}

    for pol in reward_policies + baseline_policies:
        torch.manual_seed(cfg.seed)
        model = make_model(cfg.model, k, cfg.grid)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        prng = np.random.default_rng(cfg.seed * 131 + hash(pol) % 9973)

        if pol in reward_policies:
            sched = M.EXP3Scheduler(k, rng=prng)
        elif pol == "uniform":
            sched = M.UniformScheduler(k, rng=prng)
        elif pol == "round_robin":
            sched = M.RoundRobinScheduler(k)
        else:
            sched = M.OracleScheduler(learn)

        unit = M.Unitizer()
        rnd_mod = icm_mod = aux_opt = reward_obj = None
        if pol == "rnd":
            rnd_mod = M.ConvRND(n_colors=crt.N_COLORS).to(DEV)
            aux_opt = torch.optim.Adam(rnd_mod.predictor.parameters(), lr=cfg.lr)
            reward_obj = M.RNDReward(rnd_mod)
        elif pol == "icm":
            icm_mod = M.ConvICM(n_tasks=k, n_colors=crt.N_COLORS).to(DEV)
            aux_opt = torch.optim.Adam(icm_mod.parameters(), lr=cfg.lr)
            reward_obj = M.ICMReward(icm_mod)
        elif pol == "prediction_error":
            reward_obj = M.PredictionErrorReward()
        elif pol == "audit_cp":
            reward_obj = M.AuditCPReward()

        counts = np.zeros(k); cum_cp = 0.0
        for t in range(cfg.steps):
            arm = sched.select(t)
            counts[arm] += 1
            x, y, tid = train_batch(arm, cfg.batch, prng, cfg.grid)
            ctx = M.StepContext(model=model, x=x, y=y, task_id=tid, device=DEV,
                                aux_opt=aux_opt)
            if pol == "audit_cp":
                before = M.audit_panel_loss(model, A, DEV)["floored_ce"]
                _train_step(model, opt, x, y, tid)
                after = M.audit_panel_loss(model, A, DEV)["floored_ce"]
                ctx.audit_loss_before, ctx.audit_loss_after = before, after
                raw = reward_obj.compute(ctx); cum_cp += raw
            elif pol in ("prediction_error", "rnd", "icm"):
                raw = reward_obj.compute(ctx)
                _train_step(model, opt, x, y, tid)
            else:
                raw = 0.0
                _train_step(model, opt, x, y, tid)
            if isinstance(sched, M.EXP3Scheduler):
                sched.update(arm, unit(raw))

        fin = M.audit_panel_loss(model, B, DEV)   # report on SEALED fresh panel
        dfrac = float(counts[distract].sum() / max(counts.sum(), 1))
        out[pol] = {"final_fresh_floored_ce": fin["floored_ce"],
                    "final_active_cell_acc": fin["active_cell_acc"],
                    "final_exact_grid_acc": fin["exact_grid_acc"],
                    "distractor_fraction": dfrac,
                    "cumulative_audit_cp": cum_cp if pol == "audit_cp" else None}

    # primary contrast: MWU(audit_cp) vs MWU(other rewards)
    contrast = {p: out["audit_cp"]["final_active_cell_acc"] - out[p]["final_active_cell_acc"]
                for p in ["prediction_error", "rnd", "icm"]}
    return {"experiment": "C1_curriculum_reward_ablation", "by_policy": out,
            "audit_cp_minus_curiosity_active_acc": contrast}


# =============================================================================
# E7 — black-box scalar-feedback audit overfitting
# =============================================================================

def _warm_base(cfg: Cfg, warm_steps: int):
    """Warm ONE base model on the train stream and freeze it (shared across panels)."""
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = make_model(cfg.model, len(crt.TASKS), cfg.grid)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for t in range(warm_steps):
        arm = int(crt.LEARNABLE_TASKS[t % len(crt.LEARNABLE_TASKS)])
        x, y, tid = train_batch(arm, cfg.batch, rng, cfg.grid)
        _train_step(model, opt, x, y, tid)
    model.eval()
    return model


@torch.no_grad()
def _logits_on(model, P) -> Tuple[np.ndarray, np.ndarray]:
    X, Y, T = P
    xb = torch.as_tensor(X, device=DEV); tb = torch.as_tensor(T, device=DEV)
    # Batched: running the full panel through the CNN at once spiked ~4GB and (with e6's 16GB
    # un-batched hog) cascaded OOM across all concurrent shards. Chunk it like e6.
    return _batched_logits(model, xb, tb).detach().float().cpu().numpy(), Y


def _floored_ce_np(logits: np.ndarray, target: np.ndarray, eps: float = 0.01) -> float:
    """Mean probability-floored CE for numpy logits (N,C,g,g) / target (N,g,g)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logits = np.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        idx = np.take_along_axis(p, target[:, None, :, :], axis=1).squeeze(1)
        return float(np.mean(-np.log(np.clip(idx, eps, 1.0))))


def _make_attacker(name: str, d: int, seed: int):
    if name == "sa":
        return M.SAAttacker(dim=d, rng=np.random.default_rng(seed + 1))
    if name == "es":
        return M.GaussianESAttacker(dim=d, rng=np.random.default_rng(seed + 2))
    if name == "greedy_select":
        return M.GreedySelectionAttacker(dim=d, rng=np.random.default_rng(seed + 3))
    if name == "boosting":
        return M.BoostingSelectionAttacker(dim=d, rng=np.random.default_rng(seed + 4))
    raise ValueError(name)


def _e7_instrument(base_re, lab_re, base_fr, lab_fr, *, g_a, Q, defense, seed,
                   attacker="boosting", eps=0.01, delta=0.05, step=1.0) -> Dict:
    """UNIFIED E7 instrument over a panel object (n, cells, C). The synthetic calibration
    is just cells=1; the grid E7 is cells=g^2 -- SAME code path, so faithfulness is
    structural. Attacker = per-example, per-class, per-cell-BLOCK additive logit field;
    granularity g_a in {1..cells}, dim = n*C*g_a. Loss = per-example mean floored-CE
    aggregated over examples (n = n_examples = the Hoeffding unit). card_A = 2*dim (honest
    distinct-query count of the sign-selection attacker). Power on low-entropy panels
    (dim fillable within Q), robustness on high-entropy grids (dim >> Q) -- two ends of one
    instrument, governed by output entropy, not a projection artifact."""
    n, cells, C = base_re.shape
    g_a = max(1, min(int(g_a), cells))
    block_of = (np.arange(cells) * g_a // cells).astype(np.int64)   # cell -> block in [0,g_a)
    dim = n * C * g_a

    def apply_field(theta, base):
        f = theta.reshape(n, C, g_a)[:, :, block_of]            # (n,C,cells)
        return base + np.clip(np.transpose(f, (0, 2, 1)), -20.0, 20.0)   # (n,cells,C)

    def per_example_ce(logits, lab, idx):
        z = logits[idx]
        z = z - z.max(2, keepdims=True)
        p = np.exp(z); p /= p.sum(2, keepdims=True)
        li = lab[idx]
        pt = np.take_along_axis(p, li[:, :, None], axis=2).squeeze(2)   # (n_idx,cells)
        nll = -np.log(np.clip(pt, eps, 1.0))
        valid = (li != crt.PAD)                                        # content-only (exclude pad)
        denom = valid.sum(axis=1); denom[denom == 0] = 1
        return float(np.mean((nll * valid).sum(axis=1) / denom))

    def reusable_cp(theta, idx=slice(None)):
        return per_example_ce(base_re, lab_re, idx) - per_example_ce(apply_field(theta, base_re), lab_re, idx)
    base_ce_fr = per_example_ce(base_fr, lab_fr, slice(None))
    def fresh_cp(theta):
        return base_ce_fr - per_example_ce(apply_field(theta, base_fr), lab_fr, slice(None))

    rng2 = np.random.default_rng(seed + 7); best = {"v": -math.inf}; sub_n = max(1, n // 2)
    ladder_thresh = M.floored_ce_range(eps) / math.sqrt(n)   # Blum-Hardt Ladder: only release
    def score_fn(theta):                                     # gains exceeding the noise scale R/sqrt(n)
        if defense == "fresh_subsample":
            r = reusable_cp(theta, idx=rng2.choice(n, size=sub_n, replace=False))
        else:
            r = reusable_cp(theta)
        if defense == "rounded":
            r = M.Defense.rounded(r, bits=4)
        elif defense == "ladder":
            r = M.Defense.ladder(r, best["v"], thresh=ladder_thresh)
        best["v"] = max(best["v"], r)
        return 0.0 if defense == "one_shot" else r
    atk = _make_attacker(attacker, dim, seed)
    if isinstance(atk, M.BoostingSelectionAttacker):
        atk.step = step
    tr = atk.run(score_fn, budget=Q)
    r_cp, f_cp = reusable_cp(tr.best_theta), fresh_cp(tr.best_theta)
    # phase-2 reuses coordinates with replacement, so the honest distinct-holdout-query
    # count is ~Q (which exceeds 2*dim in the winning regime); count the larger of the two.
    card_A = max(2 * dim, tr.queries_used)
    thr = M.threshold_2delta(card_A=card_A, n=n, eps=eps, delta=delta)
    dn = M.delta_n(card_A=card_A, n=n, eps=eps, delta=delta)        # one-sided (reusable alone)
    gap = r_cp - f_cp
    return {"attacker": attacker, "defense": defense, "Q": Q, "g_a": g_a, "dim": dim,
            "card_A": card_A, "reusable_cp": r_cp, "fresh_cp": f_cp, "gap": gap,
            "threshold_2delta": thr, "delta_n_reusable": dn,
            "attacker_wins": bool(gap > thr), "reusable_crosses": bool(r_cp > dn)}


def run_e7(cfg: Cfg, attackers=("boosting", "sa", "es"),
           defenses=("naive", "rounded", "ladder", "fresh_subsample", "one_shot"),
           budgets=(50, 100), label_noise: float = 0.0, g_a: Optional[int] = None) -> Dict:
    """Grid E7: warm a frozen base model, reshape its panel logits to (n, cells=g^2, C),
    and run the unified _e7_instrument. Capacity g_a scales with Q (g_a = Q/(2 n C),
    capped by cells) unless overridden. label_noise corrupts reusable labels only."""
    learn = crt.LEARNABLE_TASKS
    A = panel(cfg.seed + 1000, cfg.audit_per_family, cfg.grid, learn)   # reusable
    B = panel(cfg.seed + 2000, cfg.audit_per_family, cfg.grid, learn)   # sealed
    base = _warm_base(cfg, max(20, cfg.steps))
    Alog, A_tgt = _logits_on(base, A)                  # Alog (n,C,g,g), A_tgt (n,g,g)
    Blog, B_tgt = _logits_on(base, B)
    n_eff, Ccol, g = A[0].shape[0], crt.N_COLORS, cfg.grid
    cells = g * g
    base_re = np.transpose(Alog.reshape(n_eff, Ccol, cells), (0, 2, 1))   # (n,cells,C)
    base_fr = np.transpose(Blog.reshape(n_eff, Ccol, cells), (0, 2, 1))
    lab_re = A_tgt.reshape(n_eff, cells).copy()
    lab_fr = B_tgt.reshape(n_eff, cells)
    arng = np.random.default_rng(cfg.seed + 99)
    if label_noise > 0:
        mask = arng.random(lab_re.shape) < label_noise
        lab_re[mask] = arng.integers(0, Ccol, size=int(mask.sum()))
    results = []
    for Q in budgets:
        ga = g_a if g_a else max(1, min(cells, Q // (2 * n_eff * Ccol)))
        for atk in attackers:
            for d in defenses:
                results.append(_e7_instrument(base_re, lab_re, base_fr, lab_fr,
                                              g_a=ga, Q=Q, defense=d, seed=cfg.seed, attacker=atk))
    return {"experiment": "E7_scalar_feedback_overfit", "n_eff": n_eff, "cells": cells,
            "label_noise": label_noise, "results": results}


def run_e7_power_calibration(cfg: Cfg, n: int = 200, C: int = 10, Qs=(500, 2000, 8000)) -> Dict:
    """Instrument-power certification = run_e7 at the DEGENERATE cells=1 panel (a structural
    special case of the deployed instrument, not a separate/easier one). A capable boosting
    attacker memorizes random per-example labels via scalar feedback and must cross 2*Delta_n
    under naive release with the gap growing in Q; defenses must push it back below."""
    rng = np.random.default_rng(cfg.seed)
    base_re = np.zeros((n, 1, C)); base_fr = np.zeros((n, 1, C))
    lab_re = rng.integers(0, C, (n, 1)); lab_fr = rng.integers(0, C, (n, 1))
    def one(Q, d):
        r = _e7_instrument(base_re, lab_re, base_fr, lab_fr, g_a=1, Q=Q, defense=d, seed=cfg.seed)
        return r
    naive = [one(Q, "naive") for Q in Qs]
    Qm = max(Qs)
    defenses = [one(Qm, d) for d in ("naive", "fresh_subsample", "ladder", "rounded", "one_shot")]
    gaps = [d["gap"] for d in naive]
    return {"experiment": "E7_power_calibration (cells=1 special case of run_e7)", "n": n, "C": C,
            "naive_sweep": naive, "defenses_at_maxQ": defenses,
            "HAS_POWER": any(d["attacker_wins"] for d in naive),
            "gap_grows_with_Q": all(gaps[i] <= gaps[i + 1] + 1e-6 for i in range(len(gaps) - 1)),
            "defenses_below_threshold": [d["defense"] for d in defenses
                                         if d["defense"] != "naive" and not d["attacker_wins"]]}


def run_e7_scaling(cfg: Cfg, C: int = 10, Q: int = 8000, cells: int = 1,
                   n_grid=(50, 200, 800, 3200, 12800)) -> Dict:
    """The headline E7 artifact, with the HONEST mechanism: ONE instrument swept over PANEL
    SIZE n at fixed budget Q (canonical cells=1 holdout, random labels, naive release). The
    gap crosses 2*Delta_n at small n (Q can fill the attacker's n*C*g_a DoF) and falls below
    at large n (n*C >> Q -> the attacker cannot even address all degrees of freedom within Q).
    This budget-vs-capacity ratio (not output entropy) is what makes the full-scale ARC grid
    robust: n_examples*cells*C >> Q. Power at small n, robustness at large n -- two ends of
    one instrument."""
    rng = np.random.default_rng(cfg.seed)
    out = []
    for n in n_grid:
        base_re = np.zeros((n, cells, C)); base_fr = np.zeros((n, cells, C))
        lab_re = rng.integers(0, C, (n, cells)); lab_fr = rng.integers(0, C, (n, cells))
        ga = max(1, min(cells, Q // (2 * n * C)))
        r = _e7_instrument(base_re, lab_re, base_fr, lab_fr, g_a=ga, Q=Q, defense="naive", seed=cfg.seed)
        out.append({"n": n, "g_a": r["g_a"], "dim": r["dim"], "Q_over_2dim": round(Q / (2 * r["dim"]), 3),
                    "gap": r["gap"], "threshold_2delta": r["threshold_2delta"], "crosses": r["attacker_wins"]})
    crossed = [o["n"] for o in out if o["crosses"]]
    gaps = [o["gap"] for o in out]
    return {"experiment": "E7_scaling_curve (panel size n vs budget Q)", "C": C, "Q": Q, "cells": cells,
            "by_n": out, "crosses_at_small_n": bool(crossed),
            "max_n_crossed": (max(crossed) if crossed else None),
            "gap_decreases_with_n": all(gaps[i] >= gaps[i + 1] - 1e-6 for i in range(len(gaps) - 1))}


# =============================================================================
# E3 — signed vs clipped compression-progress cycling
# =============================================================================

def run_e3(cfg: Cfg, cycles: int = 4, steps_per_block: int = 8) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    learn = crt.LEARNABLE_TASKS
    A_fam, B_fam = int(learn[0]), int(learn[len(learn) // 2])
    QA = panel(cfg.seed + 1000, cfg.audit_per_family, cfg.grid, [A_fam])
    QB = panel(cfg.seed + 1001, cfg.audit_per_family, cfg.grid, [B_fam])
    def union_loss(model):
        return 0.5 * (M.audit_panel_loss(model, QA, DEV)["floored_ce"]
                      + M.audit_panel_loss(model, QB, DEV)["floored_ce"])

    torch.manual_seed(cfg.seed)
    model = make_model(cfg.model, len(crt.TASKS), cfg.grid)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    cum_signed = cum_clipped = 0.0
    E0 = union_loss(model)
    traj = []
    for c in range(cycles):
        for fam in (A_fam, B_fam):
            for _ in range(steps_per_block):
                before = union_loss(model)
                x, y, tid = train_batch(fam, cfg.batch, rng, cfg.grid)
                _train_step(model, opt, x, y, tid)
                after = union_loss(model)
                r = before - after
                cum_signed += r
                cum_clipped += max(0.0, r)
        traj.append({"cycle": c, "cum_signed": cum_signed, "cum_clipped": cum_clipped,
                     "union_loss": union_loss(model)})
    E_T = union_loss(model)
    return {"experiment": "E3_signed_vs_clipped", "E0": E0, "E_T": E_T,
            "endpoint_improvement": E0 - E_T, "cum_signed": cum_signed,
            "cum_clipped": cum_clipped, "exploit_gap": cum_clipped - cum_signed,
            "trajectory": traj}


# =============================================================================
# E1 — stream-audit gap under biased streams
# =============================================================================

def run_e1(cfg: Cfg, k_grid=(0.0, 0.5, 1.0)) -> Dict:
    learn = list(crt.LEARNABLE_TASKS)
    shortcut = learn[:max(2, len(learn) // 10)]            # easy/shortcut subset
    # Q = sealed audit PROBE (small, per-step). Was the full panel evaluated 2x/step*5000*3 ->
    # the worst full-scale blowup. The reported quantity is the stream-vs-audit CP GAP, an
    # unbiased estimate on this probe.
    Q = reward_probe_panel(cfg.seed + 2000, cfg.reward_probe, cfg.grid, learn)
    # Per-family stream probes, built ONCE (rebuilding S every step was O(steps) generation).
    s_n = max(8, cfg.reward_probe // 4)
    S_by_fam = {f: reward_probe_panel(cfg.seed + 3000 + f, s_n, cfg.grid, [f])
                for f in {int(x) for x in learn}}
    out = []
    for kbias in k_grid:
        rng = np.random.default_rng(cfg.seed + int(1000 * kbias))
        torch.manual_seed(cfg.seed)
        model = make_model(cfg.model, len(crt.TASKS), cfg.grid)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        sum_stream = sum_audit = 0.0
        for t in range(cfg.steps):
            fam = int(rng.choice(shortcut) if rng.random() < kbias else rng.choice(learn))
            S = S_by_fam[fam]                              # held-out stream probe for THIS family
            s_before = M.audit_panel_loss(model, S, DEV)["floored_ce"]
            a_before = M.audit_panel_loss(model, Q, DEV)["floored_ce"]
            x, y, tid = train_batch(fam, cfg.batch, rng, cfg.grid)
            _train_step(model, opt, x, y, tid)
            s_after = M.audit_panel_loss(model, S, DEV)["floored_ce"]
            a_after = M.audit_panel_loss(model, Q, DEV)["floored_ce"]
            sum_stream += (s_before - s_after)
            sum_audit += (a_before - a_after)
        out.append({"k": kbias, "cum_stream_cp": sum_stream, "cum_fresh_audit_cp": sum_audit,
                    "gamma_SA": sum_stream - sum_audit})
    slope = float(np.polyfit([r["k"] for r in out], [r["gamma_SA"] for r in out], 1)[0]) \
        if len(out) > 1 else 0.0
    return {"experiment": "E1_stream_audit_gap", "by_k": out, "gamma_SA_slope_in_k": slope}


# =============================================================================
# E5 — uniform-deviation scaling Delta_n vs n (finite model family)
# =============================================================================

def run_e5(cfg: Cfg, n_grid=(16, 64, 256), n_models: int = 16, n_panels: int = 12,
           pop_n: int = 1024) -> Dict:
    learn = crt.LEARNABLE_TASKS
    k = len(crt.TASKS)
    models = []
    for j in range(n_models):                      # finite family A of frozen models
        torch.manual_seed(cfg.seed * 1000 + j)
        m = make_model(cfg.model, k, cfg.grid)
        if j > 0:                                   # diversify: a few quick steps
            opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
            rng = np.random.default_rng(j)
            for _ in range(j % 5):
                fam = int(rng.choice(learn))
                x, y, tid = train_batch(fam, cfg.batch, rng, cfg.grid)
                _train_step(m, opt, x, y, tid)
        m.eval(); models.append(m)
    pop = panel(cfg.seed + 5000, max(1, pop_n // len(learn)), cfg.grid, learn)
    pop_loss = np.array([M.audit_panel_loss(m, pop, DEV)["floored_ce"] for m in models])
    rows = []
    rng = np.random.default_rng(cfg.seed)
    Np = pop[0].shape[0]
    for n in n_grid:
        n = min(n, Np)
        devs = []
        for _ in range(n_panels):
            idx = rng.choice(Np, size=n, replace=False)
            sub = (pop[0][idx], pop[1][idx], pop[2][idx])
            emp = np.array([M.audit_panel_loss(m, sub, DEV)["floored_ce"] for m in models])
            devs.append(float(np.max(np.abs(emp - pop_loss))))
        rows.append({"n": n, "delta_n_median": float(np.median(devs)),
                     "delta_n_p90": float(np.quantile(devs, 0.9))})
    ns = np.array([r["n"] for r in rows], float)
    dd = np.array([r["delta_n_median"] for r in rows], float)
    slope = float(np.polyfit(np.log(ns), np.log(dd + 1e-9), 1)[0]) if len(rows) > 1 else 0.0
    return {"experiment": "E5_uniform_deviation", "by_n": rows,
            "loglog_slope": slope, "expected_slope": -0.5}


# =============================================================================
# E6 — calibration vs discrimination (temperature scaling)
# =============================================================================

def _ece(logits: torch.Tensor, target: torch.Tensor, bins: int = 15) -> float:
    p = torch.softmax(logits, 1)
    conf, pred = p.max(1)
    correct = (pred == target).float()
    conf, correct = conf.flatten(), correct.flatten()
    e = 0.0; N = conf.numel()
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += float(m.float().mean()) * abs(float(conf[m].mean()) - float(correct[m].mean()))
    return e

def _batched_logits(model, X, T, bs: int = 512):
    """Forward in chunks + concat. Running a full ~12k-example panel through the CNN at once
    spikes multi-GB of activations -> CUDA OOM under concurrent workers (e6's failure at full
    scale). Chunking caps peak memory; the concatenated logits tensor itself is modest."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            outs.append(model(X[i:i+bs], T[i:i+bs]))
    return torch.cat(outs, 0)


def run_e6(cfg: Cfg) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    learn = crt.LEARNABLE_TASKS
    torch.manual_seed(cfg.seed)
    model = make_model(cfg.model, len(crt.TASKS), cfg.grid)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for t in range(cfg.steps):
        fam = int(rng.choice(learn))
        x, y, tid = train_batch(fam, cfg.batch, rng, cfg.grid)
        _train_step(model, opt, x, y, tid)
    Bp = panel(cfg.seed + 2000, cfg.audit_per_family, cfg.grid, learn)   # sealed
    Xb = torch.as_tensor(Bp[0], device=DEV); Yb = torch.as_tensor(Bp[1], device=DEV)
    Tb = torch.as_tensor(Bp[2], device=DEV)
    logits = _batched_logits(model, Xb, Tb)          # batched: avoids the full-panel OOM spike
    def metrics(scaled):
        return (float(M.floored_cross_entropy(scaled, Yb).cpu()),
                float(_ece(scaled, Yb)),
                float((scaled.argmax(1) == Yb).float().mean().cpu()))
    nll0, ece0, acc0 = metrics(logits)
    # fit temperature T on a calibration split (disjoint seed)
    Cp = panel(cfg.seed + 4000, cfg.audit_per_family, cfg.grid, learn)
    Xc = torch.as_tensor(Cp[0], device=DEV); Yc = torch.as_tensor(Cp[1], device=DEV)
    Tc = torch.as_tensor(Cp[2], device=DEV)
    logits_c = _batched_logits(model, Xc, Tc)        # batched: avoids the full-panel OOM spike
    logT = torch.zeros(1, device=DEV, requires_grad=True)
    topt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
    def closure():
        topt.zero_grad()
        loss = M.floored_cross_entropy(logits_c / logT.exp(), Yc)
        loss.backward(); return loss
    topt.step(closure)
    T = float(logT.exp().detach().cpu())
    nll1, ece1, acc1 = metrics(logits / T)
    return {"experiment": "E6_calibration", "temperature": T,
            "nll": [nll0, nll1], "ece": [ece0, ece1], "acc": [acc0, acc1],
            "delta_nll": nll1 - nll0, "delta_ece": ece1 - ece0, "delta_acc": acc1 - acc0}


# =============================================================================
# E4a — white-box finite audit-panel memorization (upper bound)
# =============================================================================

def run_e4a(cfg: Cfg, n_grid=(16, 128)) -> Dict:
    learn = crt.LEARNABLE_TASKS
    out = []
    for n_pf in n_grid:
        nfit = max(1, n_pf // len(learn) + 1)
        A = panel(cfg.seed + 1000, nfit, cfg.grid, learn)   # reusable (trained ON)
        B = panel(cfg.seed + 2000, 8, cfg.grid, learn)      # fresh
        torch.manual_seed(cfg.seed)
        model = make_model(cfg.model, len(crt.TASKS), cfg.grid)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        eA0 = M.audit_panel_loss(model, A, DEV)["floored_ce"]
        eB0 = M.audit_panel_loss(model, B, DEV)["floored_ce"]
        Xa = torch.as_tensor(A[0], device=DEV); Ya = torch.as_tensor(A[1], device=DEV)
        Ta = torch.as_tensor(A[2], device=DEV)
        for _ in range(cfg.steps * 3):                       # train directly on A_n
            opt.zero_grad(set_to_none=True)
            loss = M.floored_cross_entropy(model(Xa, Ta), Ya)
            loss.backward(); opt.step()
        eA1 = M.audit_panel_loss(model, A, DEV)["floored_ce"]
        eB1 = M.audit_panel_loss(model, B, DEV)["floored_ce"]
        out.append({"n_total": A[0].shape[0],
                    "empirical_progress": eA0 - eA1, "true_progress": eB0 - eB1,
                    "memorization_gap": eB1 - eA1,
                    "false_positive_excess": (eA0 - eA1) - (eB0 - eB1)})
    return {"experiment": "E4a_white_box_memorization", "by_n": out}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

RUNNERS = {"c1": run_c1, "e7": run_e7, "e3": run_e3, "e1": run_e1,
           "e5": run_e5, "e6": run_e6, "e4a": run_e4a,
           "e7cal": run_e7_power_calibration, "e7scal": run_e7_scaling}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=list(RUNNERS) + ["all"])
    ap.add_argument("--scale", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="cnn", choices=["cnn", "convgru"])
    ap.add_argument("--out", default=None)
    # E7 / e7cal conditions (the red-team flagged these were unreachable from the CLI):
    ap.add_argument("--label-noise", type=float, default=0.0)
    ap.add_argument("--budgets", type=str, default=None, help="comma list, e.g. 50,500,5000")
    ap.add_argument("--capacity", type=int, default=None, help="attacker capacity F (default max(Q,64))")
    ap.add_argument("--attackers", type=str, default=None, help="comma list: boosting,sa,es,greedy_select")
    ap.add_argument("--defenses", type=str, default=None,
                    help="comma list: naive,fresh_subsample,ladder,rounded,one_shot")
    args = ap.parse_args()
    cfg = Cfg.from_scale(args.scale, seed=args.seed, model=args.model)
    e7_kwargs = {}
    if args.label_noise: e7_kwargs["label_noise"] = args.label_noise
    if args.budgets: e7_kwargs["budgets"] = tuple(int(x) for x in args.budgets.split(","))
    if args.capacity: e7_kwargs["g_a"] = args.capacity
    if args.attackers: e7_kwargs["attackers"] = tuple(args.attackers.split(","))
    if args.defenses: e7_kwargs["defenses"] = tuple(args.defenses.split(","))
    aux = {"e7cal", "e7scal"}
    targets = [k for k in RUNNERS if k not in aux] if args.experiment == "all" else [args.experiment]
    results = {}
    for name in targets:
        t0 = time.time()
        if name == "e7":
            results[name] = run_e7(cfg, **e7_kwargs)
        elif name in aux:
            results[name] = RUNNERS[name](cfg)
        else:
            results[name] = RUNNERS[name](cfg)
        results[name]["_wallclock_sec"] = round(time.time() - t0, 2)
        print(f"[{name}] done in {results[name]['_wallclock_sec']}s")
    blob = json.dumps(results, indent=2, default=float)
    if args.out:
        Path(args.out).write_text(blob)
    print(blob)


if __name__ == "__main__":
    main()
