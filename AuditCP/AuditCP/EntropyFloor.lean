import AuditCP.Telescope
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# T3 — noisy-TV entropy floor, finite Gibbs, and the cross-entropy composition

`klDivFinitePMF_nonneg` (Gibbs) is assembled from Mathlib `Real.log_le_sub_one_of_pos`.
`entropy_floor_budget` is the abstract budget that takes the floor as a hypothesis (reducing to
`cumCP_le_of_lb`). The cross-entropy case is then mechanized end to end:
`crossEntropyFinitePMF_eq_entropy_add_kl` (H(Q,P)=H(Q)+KL) composed with `klDivFinitePMF_nonneg`
gives the derived floor `entropy_le_crossEntropyFinitePMF` (H(Q) <= H(Q,P)), which
`entropy_floor_budget_crossEntropy` discharges into the trajectory budget: the floor is derived,
not assumed.
-/

namespace AuditCP

/-- **Finite Gibbs.** KL divergence between finite pmfs is nonnegative (positive prior). -/
theorem klDivFinitePMF_nonneg {H : Type*} [Fintype H] (Q P : FinitePMF H)
    [HasPositivePrior P] : 0 ≤ klDivFinitePMF Q P := by
  have key : ∀ h : H,
      Q.prob h - P.prob h ≤
        (if Q.prob h = 0 then 0 else Q.prob h * Real.log (Q.prob h / P.prob h)) := by
    intro h
    have hppos : 0 < P.prob h := ‹HasPositivePrior P›.pos h
    by_cases hq : Q.prob h = 0
    · rw [if_pos hq, hq]; linarith
    · rw [if_neg hq]
      have hqpos : 0 < Q.prob h := lt_of_le_of_ne (Q.prob_nonneg h) (Ne.symm hq)
      have hqne : Q.prob h ≠ 0 := ne_of_gt hqpos
      have hlogqp : Real.log (Q.prob h / P.prob h)
          = Real.log (Q.prob h) - Real.log (P.prob h) :=
        Real.log_div hqne (ne_of_gt hppos)
      have hlogpq : Real.log (P.prob h / Q.prob h)
          = Real.log (P.prob h) - Real.log (Q.prob h) :=
        Real.log_div (ne_of_gt hppos) hqne
      have hb : Real.log (P.prob h / Q.prob h) ≤ P.prob h / Q.prob h - 1 :=
        Real.log_le_sub_one_of_pos (by positivity)
      have hlog : 1 - P.prob h / Q.prob h ≤ Real.log (Q.prob h / P.prob h) := by
        rw [hlogqp]; rw [hlogpq] at hb; linarith
      have hmul := mul_le_mul_of_nonneg_left hlog (le_of_lt hqpos)
      have heq : Q.prob h * (1 - P.prob h / Q.prob h) = Q.prob h - P.prob h := by
        field_simp
      rw [heq] at hmul
      exact hmul
  calc (0 : ℝ)
      = ∑ h : H, (Q.prob h - P.prob h) := by
        rw [Finset.sum_sub_distrib, Q.prob_sum_one, P.prob_sum_one]; ring
    _ ≤ ∑ h : H, (if Q.prob h = 0 then 0 else Q.prob h * Real.log (Q.prob h / P.prob h)) :=
        Finset.sum_le_sum (fun h _ => key h)
    _ = klDivFinitePMF Q P := rfl

/-- **T3 (entropy floor / noisy-TV budget).** On a component whose restricted audit loss `E_S` is
bounded below by its conditional entropy `HS` (which holds by `klDivFinitePMF_nonneg`), cumulative
compression progress restricted to that component is bounded by the initial excess `E_S (g 0) − HS`.
A pure-noise component therefore pays only a finite, vanishing budget.

This is exactly `cumCP_le_of_lb` with `Emin := HS`: T3 is glue over T1. -/
theorem entropy_floor_budget {ι : Type*} (E_S : ι → ℝ) (g : ℕ → ι) (T : ℕ) {HS : ℝ}
    (hfloor : HS ≤ E_S (g T)) : cumCP E_S g T ≤ E_S (g 0) - HS :=
  cumCP_le_of_lb E_S g T hfloor

/-- Shannon entropy of a finite pmf (same `0 * log(1/0) = 0` convention as the cross-entropy). -/
noncomputable def entropyFinitePMF {H : Type*} [Fintype H] (Q : FinitePMF H) : ℝ :=
  ∑ h : H, if Q.prob h = 0 then 0 else Q.prob h * Real.log (1 / Q.prob h)

/-- **Cross-entropy decomposition.** `H(Q,P) = H(Q) + KL(Q || P)` for a positive prior `P`. -/
theorem crossEntropyFinitePMF_eq_entropy_add_kl {H : Type*} [Fintype H]
    (Q P : FinitePMF H) [HasPositivePrior P] :
    crossEntropyFinitePMF Q P = entropyFinitePMF Q + klDivFinitePMF Q P := by
  unfold crossEntropyFinitePMF entropyFinitePMF klDivFinitePMF
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun h _ => ?_)
  by_cases hq : Q.prob h = 0
  · simp [hq]
  · have hppos : 0 < P.prob h := ‹HasPositivePrior P›.pos h
    rw [if_neg hq, if_neg hq, if_neg hq]
    simp only [one_div, Real.log_inv]
    rw [Real.log_div hq hppos.ne']
    ring

/-- **Gibbs floor (cross-entropy).** `H(Q) <= H(Q,P)`: cross-entropy is at least the entropy of the
data distribution. This composes `klDivFinitePMF_nonneg` through the decomposition, so the floor is
*derived*, not assumed. -/
theorem entropy_le_crossEntropyFinitePMF {H : Type*} [Fintype H]
    (Q P : FinitePMF H) [HasPositivePrior P] :
    entropyFinitePMF Q ≤ crossEntropyFinitePMF Q P := by
  have hkl := klDivFinitePMF_nonneg Q P
  rw [crossEntropyFinitePMF_eq_entropy_add_kl Q P]; linarith

/-- **T3 (entropy floor), composition mechanized.** Along any model trajectory `P ∘ g`, cumulative
signed compression progress on the cross-entropy audit loss of a fixed data distribution `Q` is
bounded by the initial excess over the entropy floor `H(Q)`. Unlike `entropy_floor_budget`, the
floor `H(Q) <= E_S(theta)` is **discharged by Gibbs** (`entropy_le_crossEntropyFinitePMF`), not taken
as a hypothesis: the floor-because-Gibbs link is machine-checked here, end to end. -/
theorem entropy_floor_budget_crossEntropy {ι H : Type*} [Fintype H]
    (Q : FinitePMF H) (P : ι → FinitePMF H) [∀ θ, HasPositivePrior (P θ)]
    (g : ℕ → ι) (T : ℕ) :
    cumCP (fun θ => crossEntropyFinitePMF Q (P θ)) g T
      ≤ crossEntropyFinitePMF Q (P (g 0)) - entropyFinitePMF Q :=
  cumCP_le_of_lb (fun θ => crossEntropyFinitePMF Q (P θ)) g T
    (entropy_le_crossEntropyFinitePMF Q (P (g T)))

/-! ## Conditional (input-averaged) entropy floor — the form stated in the paper

Theorem 3 in the paper is stated for a conditional data distribution `Q(Y|X)` with conditional
entropy `H_{Q}(Y|X)` and an input-averaged log-loss risk `E_S(θ)`. The declarations above prove
the single-distribution (unconditional) case. The layer below lifts that case to the conditional
statement by averaging over a finite input space `X` against an input marginal `μ`: the per-input
Gibbs floor is combined under a nonnegative-weighted sum, which is exactly the paper's
`E_S(θ) = H(Y|X) + E_x KL`, `E_x KL ≥ 0`. No new analytic content is introduced; the lift is
monotonicity of a finite expectation. -/
section Conditional
variable {ι X H : Type*} [Fintype X] [Fintype H]

/-- Conditional (input-averaged) Shannon entropy `H(Y|X) = E_{x~μ} H(Q(·|x))`. -/
noncomputable def condEntropyFinitePMF (μ : FinitePMF X) (Q : X → FinitePMF H) : ℝ :=
  ∑ x : X, μ.prob x * entropyFinitePMF (Q x)

/-- Conditional cross-entropy `E_{x~μ} H(Q(·|x), P(·|x))`: the input-averaged log-loss risk. -/
noncomputable def condCrossEntropyFinitePMF (μ : FinitePMF X) (Q P : X → FinitePMF H) : ℝ :=
  ∑ x : X, μ.prob x * crossEntropyFinitePMF (Q x) (P x)

/-- Conditional KL `E_{x~μ} KL(Q(·|x) ‖ P(·|x))`. -/
noncomputable def condKLDivFinitePMF (μ : FinitePMF X) (Q P : X → FinitePMF H) : ℝ :=
  ∑ x : X, μ.prob x * klDivFinitePMF (Q x) (P x)

/-- **Conditional cross-entropy decomposition.** `E_x H(Q,P) = H(Y|X) + E_x KL`: the
input-averaged form of `crossEntropyFinitePMF_eq_entropy_add_kl`. -/
theorem condCrossEntropyFinitePMF_eq_entropy_add_kl (μ : FinitePMF X) (Q P : X → FinitePMF H)
    [hP : ∀ x, HasPositivePrior (P x)] :
    condCrossEntropyFinitePMF μ Q P = condEntropyFinitePMF μ Q + condKLDivFinitePMF μ Q P := by
  unfold condCrossEntropyFinitePMF condEntropyFinitePMF condKLDivFinitePMF
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun x _ => ?_)
  haveI := hP x
  rw [crossEntropyFinitePMF_eq_entropy_add_kl (Q x) (P x)]; ring

/-- **Conditional finite Gibbs.** The input-averaged KL is nonnegative (positive priors). -/
theorem condKLDivFinitePMF_nonneg (μ : FinitePMF X) (Q P : X → FinitePMF H)
    [hP : ∀ x, HasPositivePrior (P x)] : 0 ≤ condKLDivFinitePMF μ Q P := by
  refine Finset.sum_nonneg (fun x _ => ?_)
  haveI := hP x
  exact mul_nonneg (μ.prob_nonneg x) (klDivFinitePMF_nonneg (Q x) (P x))

/-- **Conditional Gibbs floor.** `H(Y|X) ≤ E_{x~μ} H(Q(·|x), P(·|x))`: the input-averaged
cross-entropy is at least the conditional entropy of the data. Derived from conditional Gibbs. -/
theorem condEntropy_le_condCrossEntropy (μ : FinitePMF X) (Q P : X → FinitePMF H)
    [∀ x, HasPositivePrior (P x)] :
    condEntropyFinitePMF μ Q ≤ condCrossEntropyFinitePMF μ Q P := by
  have h := condKLDivFinitePMF_nonneg μ Q P
  rw [condCrossEntropyFinitePMF_eq_entropy_add_kl μ Q P]; linarith

/-- **T3 (conditional entropy floor), composition mechanized.** For a fixed conditional data
distribution `Q(·|x)` with input marginal `μ`, cumulative signed compression progress on the
input-averaged cross-entropy audit loss along a model trajectory `P ∘ g` is bounded by the
initial excess over the conditional entropy floor `H_{Q}(Y|X)`. This is the form stated in the
paper: the floor `H(Y|X) ≤ E_S(θ)` is **discharged by conditional Gibbs**
(`condEntropy_le_condCrossEntropy`, i.e. `E_S = H(Y|X) + E_x KL` with `E_x KL ≥ 0`), not assumed.
The unconditional `entropy_floor_budget_crossEntropy` is the single-input special case. -/
theorem cond_entropy_floor_budget_crossEntropy (μ : FinitePMF X) (Q : X → FinitePMF H)
    (P : ι → X → FinitePMF H) [hP : ∀ θ, ∀ x, HasPositivePrior (P θ x)] (g : ℕ → ι) (T : ℕ) :
    cumCP (fun θ => condCrossEntropyFinitePMF μ Q (P θ)) g T
      ≤ condCrossEntropyFinitePMF μ Q (P (g 0)) - condEntropyFinitePMF μ Q := by
  haveI := hP (g T)
  exact cumCP_le_of_lb (fun θ => condCrossEntropyFinitePMF μ Q (P θ)) g T
    (condEntropy_le_condCrossEntropy μ Q (P (g T)))

end Conditional

end AuditCP
