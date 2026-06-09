# Break points (typing-derivation Γ-output)

Genuine paradigm joints where no single clean Lean type serves all obligations. Documented so the
proof-planning phase (and any reviewer) sees WHERE the type system forced a choice.

## BP1 — Admissibility bifurcates (deterministic ⊥ probabilistic)

The manuscript's single notion **"audit-blind admissibility"** does not have one clean type. It
splits by what each theorem actually needs:

- **Deterministic half — `Admissible g F := ∀ t, g t ∈ F`** (Basic.lean, T-Prop).
  This is ALL that T2 (`finite_audit_goodhart`) needs: the telescoped empirical reward depends only
  on endpoints `g 0, g T ∈ F`, so the uniform event controls every horizon with no union bound over
  `T`. Adaptivity is "free" here.
- **Probabilistic half — sealed-audit independence** (FiniteExperts.lean, via `IndepFun`).
  Needed only to make the uniform-deviation EVENT hold with prob ≥ 1−δ against an *adaptive* policy
  (panel ⟂ training-history σ-algebra).

**Why a break, not a bundle:** bundling both into one `structure`/`class` would attach a
measure-theoretic independence field to a predicate that T2's proof never touches — a decorative
hypothesis (A4 failure). Keeping them separate makes T2 maximally clean and isolates the
measure-theory plumbing in one file. HC>0 at the deterministic/probabilistic joint; the split is
the resolution.

## BP2 — Abstract potential `E : ι → ℝ`  vs  concrete cross-entropy loss (a deliberate abstraction, not a break)

T1/T2 are stated over an ABSTRACT lower-bounded potential `E : ι → ℝ` (substrate-independent;
`cumCP`, `cumCP_telescope`, `finite_audit_goodhart`). The concrete floored cross-entropy
(`auditCE`, and the population/empirical losses built from it) refines INTO that slot. This composes
cleanly (HC ≈ 0) — it is recorded here only to flag that the generality is a deliberate design
choice (A5: do not collapse the abstract theorem into the cross-entropy special case).
