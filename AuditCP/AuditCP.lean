/-
Audit-CP: a self-contained (Mathlib-only) formal core for
"signed compression progress on a sealed audit panel is Goodhart-resistant."

Dependency policy: Mathlib is the SOLE `require`. The finite-PMF / cross-entropy layer is
VENDORED (copied, cited) from the author's prior Lean development in AuditCP/Vendored/FinitePMF.lean, not `require`d.
-/
import AuditCP.Vendored.FinitePMF
import AuditCP.Basic
import AuditCP.Telescope
import AuditCP.FiniteAudit
import AuditCP.EntropyFloor
import AuditCP.FiniteExperts
