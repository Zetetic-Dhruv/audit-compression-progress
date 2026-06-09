#!/usr/bin/env bash
# Reproduce + verify the AuditCP machine-checked core.
# Requires `elan` (https://github.com/leanprover/elan); it auto-selects the pinned Lean toolchain.
# Do NOT run `lake update` — the committed lake-manifest.json is the authoritative dependency pin.
set -euo pipefail
cd "$(dirname "$0")"

echo "== [1/3] fetch prebuilt Mathlib oleans for the pinned revision =="
lake exe cache get

echo "== [2/3] build the core (no errors = theorems type-check) =="
lake build

echo "== [3/3] axiom audit: expect only [propext, Classical.choice, Quot.sound], NO sorryAx =="
lake env lean Check.lean

echo "== OK: AuditCP core verified =="
