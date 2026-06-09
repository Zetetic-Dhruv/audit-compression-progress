"""Non-trivial model + infrastructure library for the Audit-CP camera-ready suite.

This replaces the pilot's flat-MLP `GridTransformNet` with real architectures and
provides the shared infrastructure every experiment driver (C1/E7/E3/E1/E5/E6/E4a)
imports. Dependency surface: torch + numpy only.

Components
----------
Base learners (task-conditioned, grid-size agnostic, output logits (B,C,H,W)):
  * TaskConditionedCNN   - FiLM-conditioned residual conv net (compact).
  * ConvGRURefiner       - iterative ConvGRU refinement model (larger/recurrent).

Intrinsic-reward modules (conv-based, not flat MLP):
  * ConvRND              - frozen random conv target + trained conv predictor.
  * ConvICM              - conv encoder + inverse + forward dynamics (Pathak-style).

Schedulers (reward-agnostic; reward in [0,1]):
  * EXP3Scheduler, UCBScheduler, EpsilonGreedyScheduler,
    UniformScheduler, RoundRobinScheduler, OracleScheduler.

Reward signals (pluggable; the C1 factorial varies THIS with EXP3 fixed):
  * AuditCPReward, PredictionErrorReward, RNDReward, ICMReward, Unitizer.

Audit / threshold:
  * floored_cross_entropy (bounded proper score, range R = -ln eps)
  * audit_panel_loss, delta_n, threshold_2delta.

E7 attackers (scalar-feedback, theory-anchored):
  * SAAttacker  - Robbins-Monro / SPSA zeroth-order, step a/(t+1)^nu, nu=0.67.
  * GaussianESAttacker - dependency-free NES-style search.
  * Defenses    - naive / rounded / ladder release transforms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Grid helpers
# =============================================================================

AUDIT_IGNORE_INDEX = None   # pad token excluded from the audit metric; set by the task source
                            # (ARC-TGI sets it to its PAD) so audit CP scores CONTENT, not padding.


def one_hot_grid(x: torch.Tensor, n_colors: int = 11) -> torch.Tensor:
    """(B,H,W) int -> (B,C,H,W) float one-hot."""
    return F.one_hot(x.long(), num_classes=n_colors).permute(0, 3, 1, 2).float()


# =============================================================================
# Bounded proper audit loss + finite-sample threshold
# =============================================================================

def floored_cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                          eps: float = 0.01, reduction: str = "mean") -> torch.Tensor:
    """Probability-floored cross-entropy.

    Per-cell loss = -log( clamp(p_target, eps, 1) ), bounded in [0, -ln eps] = [0, R].
    Keeps log-loss semantics (proper up to the eps-floor); the floor caps per-example
    surprisal so Hoeffding (`finite_validation_family_bound`) applies with range R.

    logits: (B,C,H,W); target: (B,H,W) int. Returns scalar (mean) or per-cell tensor.
    """
    p = F.softmax(logits, dim=1)                                   # (B,C,H,W)
    p_t = p.gather(1, target.long().unsqueeze(1)).squeeze(1)       # (B,H,W)
    p_t = p_t.clamp(min=eps, max=1.0)
    nll = -torch.log(p_t)                                          # in [0, -ln eps]
    if reduction == "mean":
        return nll.mean()
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    raise ValueError(reduction)


def floored_ce_range(eps: float = 0.01) -> float:
    """R = -ln eps, the per-example loss range for the finite-sample bound."""
    return -math.log(eps)


def delta_n(card_A: float, n: int, eps: float = 0.01, delta: float = 0.05) -> float:
    """Uniform-deviation slack from finite_validation_family_bound (Extended.lean:149):

        P[ sup_{a in A} |Ehat - E_Q| >= eta ] <= |A| * 2 * exp(-2 n eta^2 / R^2)

    Inverting at confidence 1 - delta with loss range R = -ln eps:

        Delta_n = R * sqrt( ln(2|A|/delta) / (2 n) ).
    """
    R = floored_ce_range(eps)
    return R * math.sqrt(math.log(2.0 * card_A / delta) / (2.0 * n))


def threshold_2delta(card_A: float, n: int, eps: float = 0.01, delta: float = 0.05) -> float:
    """E7 falsification threshold X = 2 Delta_n (max manufacturable fake CP)."""
    return 2.0 * delta_n(card_A, n, eps=eps, delta=delta)


# =============================================================================
# FiLM-conditioned residual CNN base learner
# =============================================================================

class FiLM(nn.Module):
    """Feature-wise linear modulation: per-channel (gamma, beta) from a conditioning vec."""
    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, n_channels)
        self.to_beta = nn.Linear(cond_dim, n_channels)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond).unsqueeze(-1).unsqueeze(-1)    # (B,C,1,1)
        beta = self.to_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * h + beta


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, groups: int = 8):
        super().__init__()
        g = min(groups, channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = FiLM(cond_dim, channels)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r = self.conv1(F.relu(self.norm1(h)))
        r = self.film(r, cond)
        r = self.conv2(F.relu(self.norm2(r)))
        return h + r


class TaskConditionedCNN(nn.Module):
    """Compact task-conditioned residual CNN. Preserves spatial dims; output (B,C,H,W).

    Task identity conditions every residual block via FiLM (true conditioning, not a
    flatten-and-concat). Grid-size agnostic (all conv, padding='same').
    """
    def __init__(self, n_tasks: int, n_colors: int = 11, channels: int = 96,
                 n_blocks: int = 6, task_dim: int = 64, grid_size: int = 8):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_colors = n_colors
        self.grid_size = grid_size
        self.task_emb = nn.Embedding(n_tasks, task_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(task_dim, task_dim), nn.ReLU(),
                                      nn.Linear(task_dim, task_dim))
        self.stem = nn.Conv2d(n_colors, channels, 3, padding=1)
        self.blocks = nn.ModuleList([FiLMResBlock(channels, task_dim) for _ in range(n_blocks)])
        self.head_norm = nn.GroupNorm(min(8, channels), channels)
        self.head = nn.Conv2d(channels, n_colors, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(self.task_emb(task_id.long()))
        h = self.stem(one_hot_grid(x, self.n_colors))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.head(F.relu(self.head_norm(h)))


# =============================================================================
# ConvGRU iterative refinement model (larger / recurrent)
# =============================================================================

class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int):
        super().__init__()
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, 3, padding=1)
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        zr = self.conv_zr(torch.cat([x, h], dim=1))
        z, r = torch.chunk(zr, 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_tilde


class ConvGRURefiner(nn.Module):
    """Iterative refinement: K ConvGRU steps, task-conditioned each step via FiLM.

    Larger / recurrent counterpart to TaskConditionedCNN. Output (B,C,H,W).
    """
    def __init__(self, n_tasks: int, n_colors: int = 11, hid_ch: int = 96,
                 n_steps: int = 5, task_dim: int = 64, grid_size: int = 8):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_colors = n_colors
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.task_emb = nn.Embedding(n_tasks, task_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(task_dim, task_dim), nn.ReLU(),
                                      nn.Linear(task_dim, task_dim))
        self.encode = nn.Conv2d(n_colors, hid_ch, 3, padding=1)
        self.film = FiLM(task_dim, hid_ch)
        self.cell = ConvGRUCell(hid_ch, hid_ch)
        self.head_norm = nn.GroupNorm(min(8, hid_ch), hid_ch)
        self.head = nn.Conv2d(hid_ch, n_colors, 1)
        self.hid_ch = hid_ch
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        cond = self.cond_mlp(self.task_emb(task_id.long()))
        feat = self.film(self.encode(one_hot_grid(x, self.n_colors)), cond)
        h = torch.zeros_like(feat)
        for _ in range(self.n_steps):
            h = self.cell(feat, h)
        return self.head(F.relu(self.head_norm(h)))


# =============================================================================
# Conv-based intrinsic-reward modules
# =============================================================================

class ConvPairEncoder(nn.Module):
    """Conv encoder over an (input, output) grid pair -> feature vector (B, feat)."""
    def __init__(self, n_colors: int = 11, channels: int = 64, feat_dim: int = 128):
        super().__init__()
        self.n_colors = n_colors
        self.net = nn.Sequential(
            nn.Conv2d(2 * n_colors, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, feat_dim, 3, padding=1), nn.ReLU(),
        )
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.cat([one_hot_grid(x, self.n_colors), one_hot_grid(y, self.n_colors)], dim=1)
        h = self.net(z)
        return h.mean(dim=(2, 3))            # global average pool -> (B, feat_dim)


class ConvRND(nn.Module):
    """Random Network Distillation: frozen random conv target + trained conv predictor.
    Intrinsic reward = predictor MSE to the target on the (x,y) pair."""
    def __init__(self, n_colors: int = 11, channels: int = 64, feat_dim: int = 128):
        super().__init__()
        self.target = ConvPairEncoder(n_colors, channels, feat_dim)
        self.predictor = ConvPairEncoder(n_colors, channels, feat_dim)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def error(self, x, y) -> float:
        return float(F.mse_loss(self.predictor(x, y), self.target(x, y)).detach().cpu())

    def error_and_update(self, x, y, opt: torch.optim.Optimizer) -> float:
        with torch.no_grad():
            tgt = self.target(x, y)
        pred = self.predictor(x, y)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return float(loss.detach().cpu())


class ConvICM(nn.Module):
    """Pathak-style ICM with a conv encoder.
    phi conv-encodes a grid; inverse model predicts the task from (phi(x), phi(y));
    forward model predicts phi(y) from (phi(x), action-embed). Reward = forward MSE."""
    def __init__(self, n_tasks: int, n_colors: int = 11, channels: int = 64, feat_dim: int = 128):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_colors = n_colors
        self.phi_net = nn.Sequential(
            nn.Conv2d(n_colors, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, feat_dim, 3, padding=1), nn.ReLU(),
        )
        self.feat_dim = feat_dim
        self.inverse = nn.Sequential(nn.Linear(2 * feat_dim, 256), nn.ReLU(), nn.Linear(256, n_tasks))
        self.forward_model = nn.Sequential(nn.Linear(feat_dim + n_tasks, 256), nn.ReLU(),
                                           nn.Linear(256, feat_dim))

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi_net(one_hot_grid(x, self.n_colors)).mean(dim=(2, 3))

    def error_and_update(self, x, y, task_id, opt: torch.optim.Optimizer, beta: float = 0.2) -> float:
        phi_x, phi_y = self.phi(x), self.phi(y)
        inv_logits = self.inverse(torch.cat([phi_x, phi_y], dim=1))
        inv_loss = F.cross_entropy(inv_logits, task_id.long())
        act = F.one_hot(task_id.long(), num_classes=self.n_tasks).float()
        pred_phi_y = self.forward_model(torch.cat([phi_x, act], dim=1))
        fwd_loss = F.mse_loss(pred_phi_y, phi_y.detach())
        loss = (1.0 - beta) * inv_loss + beta * fwd_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return float(fwd_loss.detach().cpu())


# =============================================================================
# Audit-panel evaluation
# =============================================================================

@torch.no_grad()
def audit_panel_loss(model: nn.Module, panel: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     device: torch.device, eps: float = 0.01, batch_size: int = 256
                     ) -> Dict[str, float]:
    """Mean floored-CE (bounded, the accounting metric) + reconstruction accuracies
    over a held-out audit panel. Returns dict with floored_ce, cell_acc, active_cell_acc,
    exact_grid_acc."""
    X, Y, T = panel
    model.eval()
    n = X.shape[0]
    ig = AUDIT_IGNORE_INDEX                      # pad token to exclude (content-only scoring)
    tot_loss = tot_cells = tot_correct = 0.0
    act_correct = act_total = 0.0
    exact = 0
    for i in range(0, n, batch_size):
        xb = torch.as_tensor(X[i:i+batch_size], device=device)
        yb = torch.as_tensor(Y[i:i+batch_size], device=device)
        tb = torch.as_tensor(T[i:i+batch_size], device=device)
        logits = model(xb, tb)
        nll = floored_cross_entropy(logits, yb, eps=eps, reduction="none")   # (B,H,W)
        valid = (yb != ig) if ig is not None else torch.ones_like(yb, dtype=torch.bool)
        vf = valid.float()
        pred = logits.argmax(1)
        corr = (pred == yb) & valid
        tot_loss += float((nll * vf).sum().cpu()); tot_cells += float(vf.sum().cpu())
        tot_correct += float(corr.sum().cpu())
        active = valid & (yb != 0)
        act_total += float(active.sum().cpu()); act_correct += float((corr & active).sum().cpu())
        exact += int(((corr | ~valid).flatten(1).all(1)).sum().cpu())   # exact over content cells
    return {
        "floored_ce": tot_loss / max(tot_cells, 1.0),
        "cell_acc": tot_correct / max(tot_cells, 1.0),
        "active_cell_acc": act_correct / max(act_total, 1.0),
        "exact_grid_acc": exact / max(n, 1),
    }


# =============================================================================
# Reward signals (pluggable). The C1 factorial varies THIS with EXP3 fixed.
# =============================================================================

@dataclass
class StepContext:
    model: nn.Module
    x: torch.Tensor
    y: torch.Tensor
    task_id: torch.Tensor
    device: torch.device
    audit_loss_before: Optional[float] = None
    audit_loss_after: Optional[float] = None
    aux_opt: Optional[torch.optim.Optimizer] = None
    eps: float = 0.01


class RewardSignal:
    """Returns a raw signed reward in natural units. Unitization for the bandit is
    handled separately by `Unitizer` so the scheduler stays reward-agnostic."""
    name = "base"
    def compute(self, ctx: StepContext) -> float:
        raise NotImplementedError


class AuditCPReward(RewardSignal):
    """Signed sealed-audit compression progress: floored-CE drop on the audit panel."""
    name = "audit_cp"
    def compute(self, ctx: StepContext) -> float:
        assert ctx.audit_loss_before is not None and ctx.audit_loss_after is not None
        return ctx.audit_loss_before - ctx.audit_loss_after


class PredictionErrorReward(RewardSignal):
    """Curiosity = model's own loss on the selected batch (high loss => opportunity)."""
    name = "prediction_error"
    @torch.no_grad()
    def compute(self, ctx: StepContext) -> float:
        ctx.model.eval()
        logits = ctx.model(ctx.x, ctx.task_id)
        v = float(floored_cross_entropy(logits, ctx.y, eps=ctx.eps).cpu())
        ctx.model.train()
        return v


class RNDReward(RewardSignal):
    name = "rnd"
    def __init__(self, rnd: ConvRND):
        self.rnd = rnd
    def compute(self, ctx: StepContext) -> float:
        return self.rnd.error_and_update(ctx.x, ctx.y, ctx.aux_opt)


class ICMReward(RewardSignal):
    name = "icm"
    def __init__(self, icm: ConvICM):
        self.icm = icm
    def compute(self, ctx: StepContext) -> float:
        return self.icm.error_and_update(ctx.x, ctx.y, ctx.task_id, ctx.aux_opt)


class Unitizer:
    """Maps raw rewards to [0,1] for the bandit via running [q_lo, q_hi] window
    (PROTOCOL u_t). Never reported as CP. Uses a fixed-capacity buffer of recents."""
    def __init__(self, lo_q: float = 0.05, hi_q: float = 0.95, warmup: int = 32, cap: int = 4096):
        self.lo_q, self.hi_q, self.warmup, self.cap = lo_q, hi_q, warmup, cap
        self.buf: List[float] = []
    def __call__(self, r: float) -> float:
        self.buf.append(float(r))
        if len(self.buf) > self.cap:
            self.buf = self.buf[-self.cap:]
        if len(self.buf) < self.warmup:
            return 0.5
        lo = float(np.quantile(self.buf, self.lo_q))
        hi = float(np.quantile(self.buf, self.hi_q))
        if hi - lo < 1e-12:
            return 0.5
        return float(np.clip((r - lo) / (hi - lo), 0.0, 1.0))


# =============================================================================
# Schedulers (reward-agnostic; .select(t)->arm, .update(arm, reward in [0,1]))
# =============================================================================

class Scheduler:
    name = "base"
    def select(self, t: int) -> int: raise NotImplementedError
    def update(self, arm: int, reward: float) -> None: pass


class EXP3Scheduler(Scheduler):
    """Adversarial-bandit EXP3 with explicit gamma. Importance-weighted exponential
    update; probabilities mix normalized weights with uniform exploration."""
    name = "exp3"
    def __init__(self, k: int, gamma: float = 0.07, rng: Optional[np.random.Generator] = None):
        self.k = k
        self.gamma = gamma
        self.w = np.ones(k, dtype=np.float64)
        self.rng = rng or np.random.default_rng(0)
        self._p = np.full(k, 1.0 / k)
    def probs(self) -> np.ndarray:
        self._p = (1 - self.gamma) * (self.w / self.w.sum()) + self.gamma / self.k
        return self._p
    def select(self, t: int) -> int:
        return int(self.rng.choice(self.k, p=self.probs()))
    def update(self, arm: int, reward: float) -> None:
        reward = float(np.clip(reward, 0.0, 1.0))
        est = reward / self._p[arm]
        self.w[arm] *= math.exp(self.gamma * est / self.k)
        # renormalize to avoid overflow
        self.w /= self.w.max()


class UCBScheduler(Scheduler):
    """UCB1 on the empirical mean reward per arm."""
    name = "ucb"
    def __init__(self, k: int, c: float = 1.0):
        self.k, self.c = k, c
        self.counts = np.zeros(k); self.means = np.zeros(k); self.t = 0
    def select(self, t: int) -> int:
        self.t += 1
        if (self.counts == 0).any():
            return int(np.argmin(self.counts))
        ucb = self.means + self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(ucb))
    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]


class EpsilonGreedyScheduler(Scheduler):
    name = "greedy"
    def __init__(self, k: int, eps: float = 0.1, ema: float = 0.1,
                 rng: Optional[np.random.Generator] = None):
        self.k, self.eps, self.ema = k, eps, ema
        self.q = np.zeros(k); self.rng = rng or np.random.default_rng(0)
    def select(self, t: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.q))
    def update(self, arm: int, reward: float) -> None:
        self.q[arm] = (1 - self.ema) * self.q[arm] + self.ema * reward


class UniformScheduler(Scheduler):
    name = "uniform"
    def __init__(self, k: int, rng: Optional[np.random.Generator] = None):
        self.k = k; self.rng = rng or np.random.default_rng(0)
    def select(self, t: int) -> int:
        return int(self.rng.integers(self.k))


class RoundRobinScheduler(Scheduler):
    name = "round_robin"
    def __init__(self, k: int):
        self.k = k
    def select(self, t: int) -> int:
        return t % self.k


class OracleScheduler(Scheduler):
    """Round-robins over a privileged set of learnable arms only (no-reward baseline)."""
    name = "oracle_learnable"
    def __init__(self, learnable_arms: Sequence[int]):
        self.arms = list(learnable_arms); self.i = 0
    def select(self, t: int) -> int:
        a = self.arms[self.i % len(self.arms)]; self.i += 1
        return int(a)


# =============================================================================
# E7 scalar-feedback attackers (theory-anchored)
# =============================================================================

@dataclass
class AttackTrace:
    best_theta: np.ndarray
    best_score: float
    score_history: List[float] = field(default_factory=list)
    best_history: List[float] = field(default_factory=list)
    queries_used: int = 0


class SAAttacker:
    """Robbins-Monro / SPSA zeroth-order stochastic approximation on scalar feedback.

    Maximizes a scalar objective `score_fn(theta) -> float` observed only through the
    (defense-transformed) reusable-audit feedback. Per iteration:
        c_t = c0 / (t+1)^gamma_pert ,  Delta ~ Rademacher
        ghat = (score(theta + c_t Delta) - score(theta - c_t Delta)) / (2 c_t) * Delta
        theta <- theta + a_t * ghat ,  a_t = a0 / (t+1)^nu   (nu in (2/3,1))
    Step schedule matches the RL library's RobbinsMonro / inv_poly anchor
    (nu=0.67 default). Two score queries per iteration.
    """
    name = "sa"
    def __init__(self, dim: int, nu: float = 0.67, a0: float = 0.1, c0: float = 0.1,
                 gamma_pert: float = 0.101, rng: Optional[np.random.Generator] = None):
        self.dim, self.nu, self.a0, self.c0, self.gamma_pert = dim, nu, a0, c0, gamma_pert
        self.rng = rng or np.random.default_rng(0)

    def run(self, score_fn: Callable[[np.ndarray], float], budget: int,
            theta0: Optional[np.ndarray] = None) -> AttackTrace:
        theta = np.zeros(self.dim) if theta0 is None else theta0.astype(float).copy()
        best_theta, best_score, used = theta.copy(), -math.inf, 0
        tr = AttackTrace(best_theta=best_theta, best_score=best_score)
        t = 0
        while used + 2 <= budget:
            a_t = self.a0 / (t + 1) ** self.nu
            c_t = self.c0 / (t + 1) ** self.gamma_pert
            delta = self.rng.choice([-1.0, 1.0], size=self.dim)
            s_plus = score_fn(theta + c_t * delta); used += 1
            s_minus = score_fn(theta - c_t * delta); used += 1
            ghat = ((s_plus - s_minus) / (2.0 * c_t)) * delta
            theta = theta + a_t * ghat
            for s, th in ((s_plus, theta + c_t * delta), (s_minus, theta - c_t * delta)):
                tr.score_history.append(s)
                if s > best_score:
                    best_score, best_theta = s, th.copy()
                tr.best_history.append(best_score)
            t += 1
        tr.best_theta, tr.best_score, tr.queries_used = best_theta, best_score, used
        return tr


class GaussianESAttacker:
    """Dependency-free NES-style evolution strategy over theta (breadth adversary).
    Maintains mean mu, fixed sigma; each generation samples a population, evaluates the
    scalar score, and updates mu by a fitness-weighted (rank-normalized) average."""
    name = "es"
    def __init__(self, dim: int, pop: int = 8, sigma: float = 0.1, lr: float = 0.2,
                 rng: Optional[np.random.Generator] = None):
        self.dim, self.pop, self.sigma, self.lr = dim, pop, sigma, lr
        self.rng = rng or np.random.default_rng(0)

    def run(self, score_fn: Callable[[np.ndarray], float], budget: int,
            theta0: Optional[np.ndarray] = None) -> AttackTrace:
        mu = np.zeros(self.dim) if theta0 is None else theta0.astype(float).copy()
        best_theta, best_score, used = mu.copy(), -math.inf, 0
        tr = AttackTrace(best_theta=best_theta, best_score=best_score)
        while used + self.pop <= budget:
            eps = self.rng.standard_normal((self.pop, self.dim))
            cand = mu[None, :] + self.sigma * eps
            scores = np.empty(self.pop)
            for i in range(self.pop):
                scores[i] = score_fn(cand[i]); used += 1
                tr.score_history.append(scores[i])
                if scores[i] > best_score:
                    best_score, best_theta = scores[i], cand[i].copy()
                tr.best_history.append(best_score)
            # rank-normalize fitness to [-0.5, 0.5] for a robust NES gradient
            ranks = np.argsort(np.argsort(scores)) / (self.pop - 1) - 0.5
            grad = (ranks[:, None] * eps).mean(0) / self.sigma
            mu = mu + self.lr * grad
        tr.best_theta, tr.best_score, tr.queries_used = best_theta, best_score, used
        return tr


class GreedySelectionAttacker:
    """Adaptive-selection / hill-climbing attack (Blum-Hardt reusable-holdout style).

    The canonical scalar-feedback holdout-breaker: repeatedly propose a perturbed
    candidate, query its (defense-transformed) reusable score, and ACCEPT it iff it
    improves — so every accepted move is selected using holdout feedback. This is the
    mechanism that provably overfits a finite reusable panel; it is the steel-man
    adversary (the SA attacker is the theory-anchored honest optimizer for comparison).
    Perturbs a random block of coordinates per step (efficient zeroth-order ascent).
    """
    name = "greedy_select"
    def __init__(self, dim: int, sigma: float = 0.5, block: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None):
        self.dim, self.sigma = dim, sigma
        self.block = block or max(1, dim // 8)
        self.rng = rng or np.random.default_rng(0)

    def run(self, score_fn: Callable[[np.ndarray], float], budget: int,
            theta0: Optional[np.ndarray] = None) -> AttackTrace:
        theta = np.zeros(self.dim) if theta0 is None else theta0.astype(float).copy()
        cur = score_fn(theta); used = 1
        best_theta, best_score = theta.copy(), cur
        tr = AttackTrace(best_theta=best_theta, best_score=best_score)
        tr.score_history.append(cur); tr.best_history.append(best_score)
        while used < budget:
            cand = theta.copy()
            idx = self.rng.choice(self.dim, size=self.block, replace=False)
            cand[idx] += self.sigma * self.rng.standard_normal(self.block)
            s = score_fn(cand); used += 1
            tr.score_history.append(s)
            if s > cur:                       # accept improvements (select on holdout)
                theta, cur = cand, s
            if s > best_score:
                best_score, best_theta = s, cand.copy()
            tr.best_history.append(best_score)
        tr.best_theta, tr.best_score, tr.queries_used = best_theta, best_score, used
        return tr


class BoostingSelectionAttacker:
    """Blum-Hardt-style boosting/selection attack that attains the adaptive-data-
    analysis worst-case overfitting rate Theta(sqrt(Q/n)).

    Coordinate-wise sign selection: cycle through the theta coordinates (each is one
    input-dependent random-feature DoF); for each, probe +/- and KEEP the sign whose
    (defense-transformed) reusable score improves. After Q queries it has selected ~Q/2
    coordinates on holdout feedback, so the aggregate accumulates ~Q/2 selected-on-noise
    terms -> reusable overfit grows like sqrt(Q/n). REQUIRES dim to scale with Q (else it
    saturates at sqrt(dim/n)); run_e7 sets F = max(Q, .). This is the steel-man adversary
    whose existence gives the E7 test statistical power; SA is the honest-optimizer foil.
    """
    name = "boosting"
    def __init__(self, dim: int, step: float = 1.0, rng: Optional[np.random.Generator] = None):
        self.dim, self.step = dim, step
        self.rng = rng or np.random.default_rng(0)

    def run(self, score_fn: Callable[[np.ndarray], float], budget: int,
            theta0: Optional[np.ndarray] = None) -> AttackTrace:
        # Blum-Hardt averaging: measure each coordinate's marginal direction in ISOLATION
        # from the baseline (so the selected signs are independent), then SUM all
        # sign-selected coordinates. The aggregate of ~Q/2 selected-on-noise directions
        # is what attains the Theta(sqrt(Q/n)) adaptive overfitting rate.
        dim = self.dim
        base = np.zeros(dim) if theta0 is None else theta0.astype(float).copy()
        s0 = score_fn(base); used = 1
        tr = AttackTrace(best_theta=base.copy(), best_score=s0)
        tr.score_history.append(s0); tr.best_history.append(s0)
        theta = base.copy()
        order = self.rng.permutation(dim); i = 0
        probe = base.copy()
        # Phase 1: marginal-sign aggregation -- each coord measured in ISOLATION from the
        # baseline so the selected signs are independent (the sqrt(Q/n) construction).
        while used + 2 <= budget and i < dim:
            j = int(order[i]); i += 1
            probe[j] = base[j] + self.step; s_p = score_fn(probe); used += 1
            probe[j] = base[j] - self.step; s_m = score_fn(probe); used += 1
            probe[j] = base[j]                                   # restore for isolation
            theta[j] = base[j] + (self.step if s_p >= s_m else -self.step)
            tr.score_history.extend([s_p, s_m]); tr.best_history.extend([s0, s0])
        # Phase 2: coordinate-ascent refinement WITH REPLACEMENT -- spends any budget beyond
        # 2*dim by re-probing random coords from the CURRENT aggregate (magnitude grows),
        # so capacity is not capped at one pass (fixes the 2*dim saturation).
        cur = score_fn(theta); used += 1
        while used + 2 <= budget:
            j = int(self.rng.integers(dim)); old = theta[j]
            theta[j] = old + self.step; s_p = score_fn(theta); used += 1
            theta[j] = old - self.step; s_m = score_fn(theta); used += 1
            if max(s_p, s_m) > cur:
                theta[j] = old + self.step if s_p >= s_m else old - self.step
                cur = max(s_p, s_m)
            else:
                theta[j] = old
            tr.score_history.extend([s_p, s_m]); tr.best_history.extend([cur, cur])
        final = score_fn(theta); used += 1
        tr.best_theta, tr.best_score, tr.queries_used = theta.copy(), final, used
        return tr


# =============================================================================
# E7 defenses: transforms on the released scalar feedback
# =============================================================================

class Defense:
    """Release transforms for E7. fresh-subsample / one-shot are handled at the driver
    level (they change WHICH panel is evaluated, not just the scalar)."""
    @staticmethod
    def naive(r: float, **kw) -> float:
        return r
    @staticmethod
    def rounded(r: float, bits: int = 4, scale: float = 1.0, **kw) -> float:
        q = 2 ** bits
        return float(round(r / scale * q) / q * scale)
    @staticmethod
    def ladder(r: float, best_so_far: float, thresh: float = 0.0, **kw) -> float:
        """Blum-Hardt Ladder: reveal the improved value only if it beats best+thresh."""
        return r if r > best_so_far + thresh else best_so_far


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


BASE_LEARNERS = {"cnn": TaskConditionedCNN, "convgru": ConvGRURefiner}
