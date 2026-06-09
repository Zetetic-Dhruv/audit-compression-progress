import AuditCP.Basic

/-!
# T1 — exact telescoping (the definitional kernel)

A4 honesty: this is a telescoping identity, not a deep theorem. It is the kernel that gives
`cumCP` its meaning, and the substrate for T2 and T3. Mathlib anchor: `Finset.sum_range_sub'`.
-/

namespace AuditCP

/-- **T1 (telescoping).** Cumulative signed CP equals the net change in the potential. -/
theorem cumCP_telescope {ι : Type*} (E : ι → ℝ) (g : ℕ → ι) (T : ℕ) :
    cumCP E g T = E (g 0) - E (g T) := by
  induction T with
  | zero => simp [cumCP]
  | succ n ih =>
      simp only [cumCP, Finset.sum_range_succ] at ih ⊢
      rw [ih]; ring

/-- **T1 (budget).** With a lower bound `Emin ≤ E (g T)`, signed CP has a finite budget: it cannot
diverge while the potential stays above `Emin`. -/
theorem cumCP_le_of_lb {ι : Type*} (E : ι → ℝ) (g : ℕ → ι) (T : ℕ) {Emin : ℝ}
    (hlb : Emin ≤ E (g T)) : cumCP E g T ≤ E (g 0) - Emin := by
  rw [cumCP_telescope]; linarith

end AuditCP
