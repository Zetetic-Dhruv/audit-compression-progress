import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import AuditCP.Vendored.FinitePMF

/-!
# Audit-CP — representation grammar (types)

Types derived via the typing-derivation URS (see ../URS_TYPING.json). The audit loss is kept
ABSTRACT (`E : ι → ℝ`) for the telescoping/uniform-convergence theorems — maximally general,
matching the manuscript; the concrete cross-entropy instantiation is `auditCE`. The admissibility
concept bifurcates (see ../BREAKS.md, BP1): the deterministic half (`Admissible`, iterates-in-`F`)
lives here; the probabilistic half (sealed-audit independence) lives in `FiniteExperts`.
-/

namespace AuditCP

/-- A predictor maps each input to a predictive pmf over labels. Profile: **T-Abbrev**. -/
abbrev Predictor (X Y : Type*) [Fintype Y] := X → FinitePMF Y

/-- Pointwise `ε`-floored cross-entropy audit loss of a predictive pmf at a label; bounded in
`[0, -Real.log ε]` (cap `R ≈ -log ε`). Profile: **T-Abbrev**. -/
noncomputable def auditCE {Y : Type*} [Fintype Y] (ε : ℝ) (p : FinitePMF Y) (y : Y) : ℝ :=
  -Real.log (max ε (p.prob y))

/-- Cumulative **signed** compression progress of a real potential `E` along an index-trajectory
`g`, over `T` steps. Profile: **T-Parametric** (abstract over the index type `ι`).

The signed convention is load-bearing: the clipped variant
`fun t => max 0 (E (g t) - E (g (t+1)))` is a different function and is NOT Goodhart-resistant
(boundary B1; witnessed empirically by E3). It is deliberately not defined here. -/
noncomputable def cumCP {ι : Type*} (E : ι → ℝ) (g : ℕ → ι) (T : ℕ) : ℝ :=
  ∑ t ∈ Finset.range T, (E (g t) - E (g (t + 1)))

/-- Uniform audit deviation: empirical and population audit losses agree to within `Δ` everywhere
on the class `F`. The event (prob ≥ 1−δ) that controls the audit; its probability is discharged in
`AuditCP.FiniteExperts`. Profile: **T-Prop**. -/
def UniformDev {ι : Type*} (F : Set ι) (Ehat E : ι → ℝ) (Δ : ℝ) : Prop :=
  ∀ θ ∈ F, |Ehat θ - E θ| ≤ Δ

/-- Audit-blind trajectory (deterministic half of admissibility): every iterate stays in `F`.
This is all the finite-audit theorem needs — adaptivity is "free" because the uniform event covers
all of `F` at once (no union bound over the horizon `T`). Profile: **T-Prop**. -/
def Admissible {ι : Type*} (g : ℕ → ι) (F : Set ι) : Prop :=
  ∀ t, g t ∈ F

end AuditCP
