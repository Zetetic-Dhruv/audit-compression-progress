import AuditCP

/-!
# Axiom audit

Each core theorem must depend ONLY on the three standard Lean/Mathlib axioms
`[propext, Classical.choice, Quot.sound]`. If any line below additionally reported `sorryAx`,
that theorem would contain an unfinished proof. None do.

Run with:  `lake env lean Check.lean`
-/

#print axioms AuditCP.cumCP_telescope
#print axioms AuditCP.cumCP_le_of_lb
#print axioms AuditCP.finite_audit_goodhart
#print axioms AuditCP.klDivFinitePMF_nonneg
#print axioms AuditCP.entropy_floor_budget
#print axioms AuditCP.crossEntropyFinitePMF_eq_entropy_add_kl
#print axioms AuditCP.entropy_le_crossEntropyFinitePMF
#print axioms AuditCP.entropy_floor_budget_crossEntropy
#print axioms AuditCP.condCrossEntropyFinitePMF_eq_entropy_add_kl
#print axioms AuditCP.condKLDivFinitePMF_nonneg
#print axioms AuditCP.condEntropy_le_condCrossEntropy
#print axioms AuditCP.cond_entropy_floor_budget_crossEntropy
