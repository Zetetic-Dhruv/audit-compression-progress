import Lake
open Lake DSL

/-
AuditCP — self-contained formal core for the compression-progress paper.
Dependency policy: Mathlib is the SOLE `require`. The finite-PMF / cross-entropy layer is
VENDORED (copied, cited) from the author's prior Lean development in AuditCP/Vendored/FinitePMF.lean, NOT `require`d.
Pinned to lean-toolchain v4.29.0-rc6 and mathlib fde0cc50, so the vendored file and the
SubGaussian/IndepFun/telescoping anchors are all guaranteed present.
-/

package «AuditCP» where
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "fde0cc508f5375f278f515cb2f50a34a545a4c5c"

@[default_target]
lean_lib «AuditCP» where
  roots := #[`AuditCP]
