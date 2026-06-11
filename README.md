# Audit Compression Progress

Reproducibility artifacts for the preprint **"Signed Compression Progress on a Sealed Audit is Goodhart-Resistant"** ([arXiv:2606.11417](https://arxiv.org/abs/2606.11417)).

Signed compression progress measured on a fixed, sealed audit loss telescopes to endpoint audit improvement, giving a zero false-positive budget; for a finite audit panel the cumulative empirical reward is bounded by the true audit improvement plus a `2*Delta_n` deviation budget. This repository hosts the two artifacts behind the paper.

## Contents

- **`AuditCP/`**: the machine-checked core (Lean 4 / Mathlib). Self-contained, with Mathlib as the only build dependency, pinned to an exact commit. Telescoping, the finite-audit `2*Delta_n` bound, finite Gibbs, and the conditional entropy floor are fully proved and axiom-audited. See [`AuditCP/README.md`](AuditCP/README.md).
- **`experiments/`**: the ARC-TGI experiment code (curriculum reward ablation, finite-audit concentration, the adaptive scalar-feedback holdout attack with its release defenses, and the boundary witnesses).
- **`dist/`**: downloadable snapshots, `AuditCP_lean.zip` (the Lean project) and `auditcp_experiments.zip` (code plus per-seed result shards).

## Lean: build and verify

```sh
cd AuditCP
bash verify.sh        # lake exe cache get, then lake build, then the axiom audit (Check.lean)
```
Expected: `lake build` completes with no errors and no `sorry`, and the axiom audit reports that every core theorem depends only on `[propext, Classical.choice, Quot.sound]`.

## Experiments: requirements and use

Python 3.9+ with `numpy`, `torch`, and `matplotlib`. The task source is the ARC-TGI generators (the `simon-arc-lab` project, pinned commit `d8a2621`); clone it and point the code at it through the `ARC_TGI_ROOT` environment variable.

```sh
export ARC_TGI_ROOT=/path/to/simon-arc-lab
cd experiments
python3 cp_experiments.py c1 --scale full --seed 0      # one experiment, one seed
python3 e7_multiseed.py 20                               # 20-seed E7 power calibration + scaling curve
```
`run_parallel.sh` fans the full battery out over (experiment x seed) shards. `make_results_figures.py` and `replot_e7_multiseed.py` regenerate the figures from the result shards.

## Citation

```bibtex
@article{mittal2026signed,
  title   = {Signed Compression Progress on a Sealed Audit is Goodhart-Resistant},
  author  = {Mittal, Ayush and Gupta, Dhruv},
  journal = {arXiv preprint arXiv:2606.11417},
  year    = {2026}
}
```

## License

Apache-2.0 (see `AuditCP/LICENSE`).
