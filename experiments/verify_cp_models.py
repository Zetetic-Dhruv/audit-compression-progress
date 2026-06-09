"""Smoke verification for cp_models.py. Exits nonzero on any failure.

Checks: model shapes, finite grads, that the real models can LEARN (overfit a batch),
floored-CE boundedness, audit panel eval, every scheduler, the unitizer, the Delta_n
threshold math, and both E7 attackers on a known objective.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

import cp_models as M

dev = M.device_auto()
print(f"device = {dev}")
G, C, NT, B = 12, 11, 11, 8          # grid, colors (10 ARC + 1 pad), tasks, batch
rng = np.random.default_rng(0)
torch.manual_seed(0)

def rand_batch(b=B):
    x = torch.randint(0, C, (b, G, G), device=dev)
    y = torch.randint(0, C, (b, G, G), device=dev)
    t = torch.randint(0, NT, (b,), device=dev)
    return x, y, t

# ---- 1. base learners: shape, finite grad, AND can learn (overfit a batch) ----
for key, Cls in M.BASE_LEARNERS.items():
    model = Cls(n_tasks=NT, n_colors=C, grid_size=G).to(dev)
    x, y, t = rand_batch()
    logits = model(x, t)
    assert logits.shape == (B, C, G, G), f"{key} bad shape {logits.shape}"
    loss0 = M.floored_cross_entropy(logits, y)
    loss0.backward()
    gnorm = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    assert math.isfinite(gnorm) and gnorm > 0, f"{key} grad not finite/zero ({gnorm})"
    # overfit the fixed batch for a few steps -> loss must drop substantially
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    init = float(loss0.detach().cpu())
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        l = M.floored_cross_entropy(model(x, t), y)
        l.backward(); opt.step()
    fin = float(M.floored_cross_entropy(model(x, t), y).detach().cpu())
    assert fin < 0.5 * init, f"{key} did not learn: {init:.3f} -> {fin:.3f}"
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  [{key}] shape OK, learns {init:.3f}->{fin:.3f}, params={nparams:,}")

# ---- 2. floored-CE boundedness ----
R = M.floored_ce_range(0.01)
x, y, t = rand_batch()
logits = torch.randn(B, C, G, G, device=dev) * 5
percell = M.floored_cross_entropy(logits, y, eps=0.01, reduction="none")
assert float(percell.min()) >= -1e-6 and float(percell.max()) <= R + 1e-4, \
    f"floored-CE out of [0,{R:.3f}]: [{float(percell.min()):.3f},{float(percell.max()):.3f}]"
print(f"  floored-CE in [0, R={R:.3f}] OK (max={float(percell.max()):.3f})")

# ---- 3. intrinsic-reward modules ----
rndm = M.ConvRND(n_colors=C).to(dev)
ropt = torch.optim.Adam(rndm.predictor.parameters(), lr=1e-3)
r_errs = [rndm.error_and_update(*rand_batch()[:2], ropt) for _ in range(20)]
assert all(math.isfinite(e) and e >= 0 for e in r_errs), "RND error bad"
icm = M.ConvICM(n_tasks=NT, n_colors=C).to(dev)
iopt = torch.optim.Adam(icm.parameters(), lr=1e-3)
i_errs = [icm.error_and_update(*rand_batch(), iopt) for _ in range(20)]
assert all(math.isfinite(e) and e >= 0 for e in i_errs), "ICM error bad"
print(f"  ConvRND mean err {np.mean(r_errs):.4f} | ConvICM mean fwd err {np.mean(i_errs):.4f} OK")

# ---- 4. audit panel eval ----
panel = (rng.integers(0, C, (40, G, G)), rng.integers(0, C, (40, G, G)),
         rng.integers(0, NT, (40,)))
model = M.TaskConditionedCNN(n_tasks=NT, n_colors=C, grid_size=G).to(dev)
ap = M.audit_panel_loss(model, panel, dev)
assert set(ap) >= {"floored_ce", "cell_acc", "active_cell_acc", "exact_grid_acc"}
assert 0 <= ap["floored_ce"] <= R + 1e-4 and 0 <= ap["cell_acc"] <= 1
print(f"  audit_panel_loss { {k: round(v,3) for k,v in ap.items()} } OK")

# ---- 5. schedulers ----
k = NT
scheds = [M.EXP3Scheduler(k, rng=np.random.default_rng(1)), M.UCBScheduler(k),
          M.EpsilonGreedyScheduler(k, rng=np.random.default_rng(2)),
          M.UniformScheduler(k, rng=np.random.default_rng(3)),
          M.RoundRobinScheduler(k), M.OracleScheduler(list(range(3)))]
for s in scheds:
    for tt in range(50):
        a = s.select(tt)
        assert 0 <= a < k, f"{s.name} bad arm {a}"
        s.update(a, float(rng.random()))
exp3 = M.EXP3Scheduler(k); p = exp3.probs()
assert abs(p.sum() - 1.0) < 1e-9 and (p > 0).all(), "EXP3 probs invalid"
# EXP3 must concentrate on a consistently-rewarded arm
for _ in range(300):
    a = exp3.select(0); exp3.update(a, 1.0 if a == 5 else 0.0)
assert exp3.probs().argmax() == 5, "EXP3 failed to learn best arm"
print("  schedulers (exp3/ucb/greedy/uniform/round_robin/oracle) OK; EXP3 concentrates")

# ---- 6. unitizer ----
u = M.Unitizer()
outs = [u(float(v)) for v in rng.normal(0, 1, 500)]
assert all(0 <= o <= 1 for o in outs), "unitizer out of [0,1]"
print("  unitizer in [0,1] OK")

# ---- 7. threshold math ----
d_agg = M.threshold_2delta(card_A=1000, n=512 * 40)
d_fam = M.threshold_2delta(card_A=1000, n=512)
assert d_agg > 0 and d_fam > d_agg, "threshold monotonicity wrong"
print(f"  2*Delta_n: aggregate(n=20480)={d_agg:.3f} nats | per-family(n=512)={d_fam:.3f} nats OK")

# ---- 8. E7 attackers on a known concave objective ----
target = rng.normal(0, 1, 5)
def score_fn(theta):                # maximized at theta=target
    return -float(np.sum((theta - target) ** 2))
sa = M.SAAttacker(dim=5, rng=np.random.default_rng(7)).run(score_fn, budget=400)
es = M.GaussianESAttacker(dim=5, rng=np.random.default_rng(8)).run(score_fn, budget=400)
for nm, tr in (("SA", sa), ("ES", es)):
    assert tr.queries_used <= 400, f"{nm} over budget"
    assert tr.best_history[-1] > tr.best_history[0], f"{nm} did not improve"
    d0 = np.linalg.norm(np.zeros(5) - target)
    dF = np.linalg.norm(tr.best_theta - target)
    assert dF < d0, f"{nm} did not approach target ({d0:.2f}->{dF:.2f})"
    print(f"  {nm} attacker: best {tr.best_history[0]:.2f}->{tr.best_history[-1]:.2f}, "
          f"dist {d0:.2f}->{dF:.2f}, q={tr.queries_used} OK")

# ---- 9. defenses ----
assert M.Defense.naive(0.3) == 0.3
assert abs(M.Defense.rounded(0.123456, bits=4) - 0.125) < 0.02
assert M.Defense.ladder(0.4, best_so_far=0.5) == 0.5 and M.Defense.ladder(0.6, 0.5) == 0.6
print("  defenses (naive/rounded/ladder) OK")

print("\nALL cp_models CHECKS PASSED")
