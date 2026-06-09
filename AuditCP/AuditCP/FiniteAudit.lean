import AuditCP.Telescope

/-!
# T2 — finite-audit Goodhart-resistance (THE ANCHOR)

The load-bearing theorem. Its depth is not the endpoint algebra but the uniformity over adaptive
policies with **no union bound over the horizon `T`**: the telescoped empirical reward depends only
on the endpoints `g 0, g T`, both in `F` by `Admissible`, so the single uniform-deviation event
controls every horizon at once. The probabilistic instantiation of that event is in `FiniteExperts`.
-/

namespace AuditCP

/-- **T2 (finite-audit Goodhart-resistance).** On the uniform-deviation event, empirical cumulative
compression progress is bounded by true cumulative progress plus `2Δ`, for every admissible
trajectory and every horizon `T`. Unbounded measured progress during true-loss stagnation is
therefore impossible whenever `Δ` is finite and `T`-independent. -/
theorem finite_audit_goodhart {ι : Type*} {F : Set ι} {Ehat E : ι → ℝ} {Δ : ℝ}
    (hUD : UniformDev F Ehat E Δ) {g : ℕ → ι} (hg : Admissible g F) (T : ℕ) :
    cumCP Ehat g T ≤ cumCP E g T + 2 * Δ := by
  rw [cumCP_telescope Ehat g T, cumCP_telescope E g T]
  have h0 := abs_le.1 (hUD (g 0) (hg 0))
  have hT := abs_le.1 (hUD (g T) (hg T))
  linarith [h0.1, h0.2, hT.1, hT.2]

end AuditCP
