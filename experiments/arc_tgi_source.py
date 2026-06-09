"""ARC-TGI (simon-arc-lab) bridge — the SOLE task source for the compression-progress suite.

Mirrors the module-level API cp_experiments.py imports from `camera_ready_train` (crt):
  TASKS, LEARNABLE_TASKS, DISTRACTOR_TASKS,
  sample_task_batch(task_idx, n, rng, size),
  build_audit_panel(seed, n_per_family, size, task_indices)
so the repoint is a one-line `import arc_tgi_source as crt`.

Design (locked + literature-grounded; see P3/URS.md):
  - per-module families = simon-arc-lab `dataset_solve_*.py` grid->grid generators, each
    registered ONLY if it passes an empirical conformance gate (produces valid grid pairs);
  - distractor / noisy-TV families = ARC-TGI `image_create_random_advanced` input + an
    INDEPENDENT random output (unlearnable by construction);
  - variable grids padded to a 30x30 canvas with a DEDICATED pad token => N_COLORS = 11;
  - FRESH per-step on-demand generation (RE-ARC standard);
  - repo cloned + commit-pinned; located via ARC_TGI_ROOT / P3/data/arc-tgi / /opt/arc-tgi.
"""
from __future__ import annotations

import ast
import glob
import importlib.util
import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

N_COLORS = 11      # 10 ARC content colors (0..9) + 1 dedicated pad token
PAD = 10           # pad token value (distinct from all content colors)
GRID = 30          # ARC canvas (grids are <= 30x30, so padding is lossless)


# -------------------------------------------------------------------------
# Locate + path-inject the pinned simon-arc-lab repo
# -------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_repo() -> str:
    cands = [os.environ.get("ARC_TGI_ROOT", ""),
             os.path.join(_HERE, "..", "data", "arc-tgi"),
             "/opt/arc-tgi"]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "simon_arc_lab")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "simon-arc-lab repo not found. Set ARC_TGI_ROOT or clone to P3/data/arc-tgi or /opt/arc-tgi.")

_REPO = _find_repo()
for _p in (_REPO, os.path.join(_REPO, "simon_arc_dataset"),
           os.path.join(_REPO, "simon_arc_dataset_run")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simon_arc_lab.image_create_random_advanced import image_create_random_advanced  # noqa: E402


# -------------------------------------------------------------------------
# Canvas padding with the dedicated pad token
# -------------------------------------------------------------------------
def _to_canvas(g: np.ndarray, size: int = GRID) -> np.ndarray:
    g = np.asarray(g, dtype=np.int64)
    out = np.full((size, size), PAD, dtype=np.int64)     # pad token everywhere
    hh, ww = min(g.shape[0], size), min(g.shape[1], size)  # crop only if > size (never at 30)
    out[:hh, :ww] = g[:hh, :ww]
    return out


# -------------------------------------------------------------------------
# Import a generator module by file path (safe: each has an if __name__ guard)
# -------------------------------------------------------------------------
def _import_path(path: str):
    name = "arctgi_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_tids(path: str) -> List[str]:
    """Best-effort extraction of valid transformation_id strings from source: both
    `transformation_ids = [...]` list literals AND string literals passed at generate_task*
    call sites (modules often pass the tid inline rather than via a named list)."""
    try:
        src = open(path).read()
    except Exception:
        return []
    out: List[str] = []
    # (a) named list literals: transformation_ids = ['a', 'b', ...]
    for m in re.finditer(r"transformation_ids\s*=\s*(\[[^\]]*\])", src, re.S):
        try:
            out.extend([x for x in ast.literal_eval(m.group(1)) if isinstance(x, str)])
        except Exception:
            continue
    # (b) string literals passed at generate_task*( ... ) call sites
    for m in re.finditer(r"generate_task\w*\(([^)]*)\)", src, re.S):
        out.extend(re.findall(r"['\"]([A-Za-z][A-Za-z0-9_]+)['\"]", m.group(1)))
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq


def _build_kwargs(sig: inspect.Signature, tids: List[str], rng: np.random.Generator) -> dict:
    """Fill a generate_task signature's params (besides seed) by name heuristic."""
    kw = {}
    for pname, p in list(sig.parameters.items())[1:]:    # skip the first param (the seed slot)
        low = pname.lower()
        if p.default is not inspect.Parameter.empty:
            if pname == "transformation_id" and tids:
                kw[pname] = str(rng.choice(tids))
            elif "noise" in low:
                kw[pname] = 0.0
            continue  # else keep the default
        # required param -> heuristic
        if "transformation_id" in low:
            if not tids:
                raise ValueError("transformation_id required but no tids extracted")
            kw[pname] = str(rng.choice(tids))
        elif "noise" in low:
            kw[pname] = 0.0
        elif low in ("dx", "dy"):
            kw[pname] = int(rng.integers(-3, 4))
        elif "scale" in low:
            kw[pname] = int(rng.integers(1, 4))
        elif "up_down" in low or "flip" in low:
            kw[pname] = int(rng.integers(0, 2))
        else:
            kw[pname] = int(rng.integers(0, 3))
    return kw


def _harvest_pairs(task) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Valid (input, output) example pairs: 2D int grids, <=30, content colors in 0..9."""
    pairs = []
    for i in range(getattr(task, "count_examples", 0)):
        x = task.example_input(i); y = task.example_output(i)
        if x is None or y is None:
            continue
        x = np.asarray(x); y = np.asarray(y)
        if (x.ndim == 2 and y.ndim == 2 and x.size and y.size
                and max(x.shape) <= GRID and max(y.shape) <= GRID
                and int(x.min()) >= 0 and int(x.max()) < 10
                and int(y.min()) >= 0 and int(y.max()) < 10):
            pairs.append((x.astype(np.int64), y.astype(np.int64)))
    return pairs


@dataclass
class TaskSpec:
    name: str
    learnable: bool
    gens: List[Tuple[Callable, inspect.Signature]] = field(default_factory=list)
    gen_names: List[str] = field(default_factory=list)
    tids: List[str] = field(default_factory=list)


def _collect_generators(mod):
    """Module-level functions named generate_task* DEFINED in this module (not imported)."""
    out = []
    for nm, fn in vars(mod).items():
        if (nm.startswith("generate_task") and inspect.isfunction(fn)
                and getattr(fn, "__module__", None) == mod.__name__):
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            if len(sig.parameters) >= 1:
                out.append((nm, fn, sig))
    return out


def _try_family(path: str, rng: np.random.Generator) -> Tuple[Optional[TaskSpec], str]:
    """Register a module as a family iff >=1 of its generate_task* fns yields valid pairs.
    The module's multiple generate_task* variants are the within-family transformations."""
    try:
        mod = _import_path(path)
    except Exception as e:
        return None, f"import failed: {type(e).__name__}: {e}"
    gens = _collect_generators(mod)
    if not gens:
        return None, "no module-level generate_task* function"
    tids = _extract_tids(path)
    conforming, names = [], []
    for (nm, fn, sig) in gens:
        for attempt in range(4):
            try:
                task = fn(7000 + attempt, **_build_kwargs(sig, tids, rng))
                if _harvest_pairs(task):
                    conforming.append((fn, sig)); names.append(nm); break
            except Exception:
                continue
    if not conforming:
        return None, "no generate_task* produced valid pairs"
    return TaskSpec(os.path.basename(path)[:-3], True, conforming, names, tids), "ok"


# -------------------------------------------------------------------------
# Build the registry at import (one-time conformance scan)
# -------------------------------------------------------------------------
N_DISTRACTORS = int(os.environ.get("ARC_TGI_N_DISTRACTORS", "8"))
_reg_rng = np.random.default_rng(20260529)
_PATHS = sorted(glob.glob(os.path.join(_REPO, "simon_arc_dataset_run", "dataset_solve_*.py")))

TASKS: List[TaskSpec] = []
SKIPPED: List[Tuple[str, str]] = []
for _path in _PATHS:
    _spec, _why = _try_family(_path, _reg_rng)
    (TASKS.append(_spec) if _spec else SKIPPED.append((os.path.basename(_path), _why)))

N_LEARNABLE = len(TASKS)
for _j in range(N_DISTRACTORS):
    TASKS.append(TaskSpec(f"distractor_random_{_j}", False))

LEARNABLE_TASKS = [i for i, t in enumerate(TASKS) if t.learnable]
DISTRACTOR_TASKS = [i for i, t in enumerate(TASKS) if not t.learnable]


# -------------------------------------------------------------------------
# crt-compatible sampling API (FRESH per call; padded to the 30x30 token canvas)
# -------------------------------------------------------------------------
def sample_task_batch(task_idx: int, n: int, rng: np.random.Generator,
                      size: int = GRID) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = TASKS[task_idx]
    X = np.empty((n, size, size), np.int64)
    Y = np.empty((n, size, size), np.int64)
    T = np.full(n, task_idx, np.int64)
    if spec.learnable:
        filled, guard = 0, 0
        while filled < n and guard < n * 50 + 50:
            guard += 1
            fn, sig = spec.gens[int(rng.integers(len(spec.gens)))]   # random within-family variant
            seed = int(rng.integers(0, 2_000_000_000))
            try:
                task = fn(seed, **_build_kwargs(sig, spec.tids, rng))
            except Exception:
                continue
            for (x, y) in _harvest_pairs(task):
                if filled >= n:
                    break
                X[filled] = _to_canvas(x, size); Y[filled] = _to_canvas(y, size); filled += 1
        if filled < n:                                   # extremely unlikely; pad-token fill
            X[filled:] = PAD; Y[filled:] = PAD
    else:
        # noisy-TV distractor: ARC-like random INPUT; output = independent UNIFORM iid noise
        # over content colors (0..9), SAME shape as the input. Size is predictable but the
        # CONTENT carries no learnable signal, so the pad-masked content CE stays at ln(10)
        # (genuinely unlearnable). NOTE: a structured random output (image_create_random_*)
        # is NOT a valid distractor -- its color marginal is learnable (do-5 Pl-kill).
        for i in range(n):
            s1 = int(rng.integers(0, 2_000_000_000))
            xi = np.asarray(image_create_random_advanced(s1, 1, size, 1, size))
            yi = rng.integers(0, 10, size=xi.shape)      # iid uniform content noise
            X[i] = _to_canvas(xi, size); Y[i] = _to_canvas(yi, size)
    return X, Y, T


def build_audit_panel(seed: int, n_per_family: int, size: int,
                      task_indices) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xs, Ys, Ts = [], [], []
    for ti in task_indices:
        x, y, t = sample_task_batch(int(ti), n_per_family, rng, size=size)
        Xs.append(x); Ys.append(y); Ts.append(t)
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ts)


def manifest() -> dict:
    """Provenance record: conforming families + transformation_ids + skipped modules."""
    return {
        "repo": _REPO,
        "n_colors": N_COLORS, "pad_token": PAD, "grid": GRID,
        "n_learnable_families": N_LEARNABLE, "n_distractor_families": N_DISTRACTORS,
        "learnable_families": [{"name": TASKS[i].name, "generators": TASKS[i].gen_names,
                                "transformation_ids": TASKS[i].tids} for i in LEARNABLE_TASKS],
        "distractor_spec": "input=image_create_random_advanced; output=independent random (unlearnable)",
        "skipped_modules": SKIPPED,
    }


if __name__ == "__main__":
    import json
    m = manifest()
    print(f"ARC-TGI bridge @ {_REPO}")
    print(f"  learnable families: {m['n_learnable_families']}  distractors: {m['n_distractor_families']}")
    print(f"  conforming: {[f['name'] for f in m['learnable_families']]}")
    print(f"  skipped ({len(SKIPPED)}):")
    for nm, why in SKIPPED:
        print(f"    {nm}: {why}")
