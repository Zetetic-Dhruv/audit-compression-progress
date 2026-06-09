# AuditCP — machine-checked core (Lean 4 / Mathlib)

Formal artifact for the preprint **"Signed Compression Progress on a Sealed Audit is
Goodhart-Resistant."** Self-contained Lean 4 development whose **only build
dependency is Mathlib**, pinned to an exact commit for reproducibility.

The load-bearing theorems are **fully proved** (no `sorry`). This is verified two ways:
1. `lake build` completes with **no errors and no `sorry` warnings**; and
2. an **axiom audit** (`Check.lean`) shows every core theorem depends only on the three standard
   Lean/Mathlib axioms `[propext, Classical.choice, Quot.sound]` — **no `sorryAx`**.

## Claim → mechanization map

| Paper result | Lean declaration | File | Status |
|---|---|---|---|
| **T1** exact telescoping (signed CP) | `cumCP_telescope` | `AuditCP/Telescope.lean` | ✅ proved |
| **T1** finite budget | `cumCP_le_of_lb` | `AuditCP/Telescope.lean` | ✅ proved |
| **T2** finite-audit Goodhart-resistance (the **anchor**, the `2Δₙ` bound) | `finite_audit_goodhart` | `AuditCP/FiniteAudit.lean` | ✅ proved |
| finite Gibbs (`KL ≥ 0`) | `klDivFinitePMF_nonneg` | `AuditCP/EntropyFloor.lean` | ✅ proved |
| **T3** entropy floor (cross-entropy + conditional) | `cond_entropy_floor_budget_crossEntropy` | `AuditCP/EntropyFloor.lean` | ✅ proved (Gibbs-discharged) |
| finite-experts `Δₙ` radius | `deltaFiniteExperts` (def) | `AuditCP/FiniteExperts.lean` | ◑ defined; probabilistic theorem is future work (see below) |

## Scope — what is and is NOT mechanized (no undocumented gaps)

- **Mechanized & axiom-clean:** T1, T2 (anchor), T3, finite Gibbs — the deterministic core.
- **Future work (documented, not a `sorry`):** `finite_experts_uniformDev` — the probabilistic
  instantiation that discharges the `UniformDev` hypothesis via Hoeffding + a union bound. It is a
  written TODO in `FiniteExperts.lean` (with its Mathlib proof route), **not** an unproven stub.
- **Empirical witnesses, not Lean (by design):** the boundary results B1 (clipped-reward farming),
  B2 (stream leakage), B3 (finite-panel memorization) are concrete constructions established by the
  experiments (E3, E1, E4a in the experiments artifact), not mechanized here.
- **Cited as standard mathematics:** the covering-number / RKHS instantiations of `Δₙ` and the
  VC-capacity boundary — Mathlib carries the abstract definitions but not the quantitative
  learning-theoretic bounds, so these are referenced, not re-proved.

## Dependencies / environment

- **Lean toolchain:** `leanprover/lean4:v4.29.0-rc6` (managed by [`elan`](https://github.com/leanprover/elan); the `lean-toolchain` file selects it automatically).
- **Mathlib:** pinned in `lake-manifest.json` to commit `fde0cc508f5375f278f515cb2f50a34a545a4c5c`
  (+ its transitive deps, also pinned). **Do not run `lake update`** — it would move the pin; the
  committed manifest is authoritative.
- **Disk:** ~5–8 GB for the downloaded Mathlib build cache. **OS:** macOS or Linux.
- **License:** Apache-2.0 (see `LICENSE`).

## Build & verify (one command)

```sh
cd AuditCP
bash verify.sh        # = lake exe cache get → lake build → axiom audit (Check.lean)
```
`lake exe cache get` downloads the prebuilt Mathlib oleans for the pinned revision (minutes, not
hours). `lake build` succeeding with no errors means the theorems type-check. The final step prints
the axioms of each core theorem; the expected, correct output is:
```
'AuditCP.finite_audit_goodhart' depends on axioms: [propext, Classical.choice, Quot.sound]
... (same for the other four; NO `sorryAx`)
```

Manual equivalent:
```sh
lake exe cache get && lake build && lake env lean Check.lean
```

## Layout

```
AuditCP/
├── README.md            this file
├── LICENSE              Apache-2.0
├── lakefile.lean        package config (Mathlib is the sole `require`)
├── lean-toolchain       pinned Lean version
├── lake-manifest.json   pinned Mathlib + transitive deps (reproducibility)
├── verify.sh            build + axiom-audit in one command
├── Check.lean           axiom audit (#print axioms for each core theorem)
├── AuditCP.lean         library root (imports the modules below)
├── AuditCP/
│   ├── Basic.lean        types: Predictor, auditCE, cumCP, UniformDev, Admissible
│   ├── Telescope.lean    T1: cumCP_telescope, cumCP_le_of_lb
│   ├── FiniteAudit.lean  T2 anchor: finite_audit_goodhart
│   ├── EntropyFloor.lean finite Gibbs + T3
│   ├── FiniteExperts.lean Δₙ def + the deferred probabilistic theorem (TODO)
│   └── Vendored/
│       └── FinitePMF.lean  finite-PMF / KL / cross-entropy layer (see below)
└── BREAKS.md            documented design break points (e.g. the admissibility split)
```

## Vendoring & attribution

To keep **Mathlib as the sole `require`**, the finite-PMF / KL / cross-entropy layer is **vendored
(copied verbatim, with attribution)** from the author's prior Lean development rather than depended
upon: `AuditCP/Vendored/FinitePMF.lean` (author: Dhruv Gupta, Apache-2.0). No other external code is
used; the entire dependency graph of this artifact is `Mathlib + this one vendored file`.
