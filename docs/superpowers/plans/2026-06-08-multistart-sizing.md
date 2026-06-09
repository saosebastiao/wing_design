# Multi-start Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A serial, parallel-ready multi-start wrapper around `size_beam_shell_laminate` that runs the sizer from N initial guesses and returns the best feasible result. Verified WITHOUT running real multi-start sizings (stubbed sizer in tests; example compile-checked only).

**Architecture:** Add an `x0` param to the sizer (default None = today's guess), extract `laminate_design_bounds`, add `laminate_result_is_feasible`, then build `size_beam_shell_laminate_multistart` + `MultiStartResult`. The start loop is one serial map (swappable for multiprocessing later).

**Tech Stack:** numpy, scipy SLSQP, pytest (monkeypatch).

**CONSTRAINT (user): do NOT execute any multi-start sizing run.** No example run, no headline. Orchestration is tested with a monkeypatched stub sizer; the only real sizing in tests is one small `x0`-roundtrip backward-compat check (same cost class as the existing suite).

**Domain notes:**
- Sizer DV layout: `x = [radii(n), t_band(B), f0_grp(L), f45_grp(L)]`, `L = B if config.per_band_layup else 1`, `nx = n+B+2L`, `f0_lo = n+B`, `f45_lo = n+B+L`. These locals already exist in `size_beam_shell_laminate`.
- `LaminateSizingResult` fields: `mass_kg, max_beam_vm_Pa, max_skin_vm_Pa, tip_defl_m, tip_twist_deg, max_beam_buckling_util, max_panel_buckling_util, min_skin_strength_ratio` (+ radii/t/f arrays).
- `LaminateSizingConfig` fields: `sigma_allow_Pa, tip_defl_max_m, tip_twist_max_deg, r_min, r_max, t_min, t_max, buckling_safety_factor, ..., n_skin_bands, skin_failure, tsai_wu_safety_factor, per_band_layup`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1: `x0` param, bounds helper, feasibility check

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Test: `tests/beams/test_multistart.py` (new)

- [ ] **Step 1: Write failing tests** — create `tests/beams/test_multistart.py`:

```python
import numpy as np
import pytest

from wing_design.scenario import small_scenario
from wing_design.beams import (
    LaminateSizingConfig, LaminateSizingResult, build_beam_frame, build_beam_shell_model,
    project_panels_to_beam_nodes, size_beam_shell_laminate,
    laminate_design_bounds, laminate_result_is_feasible,
)
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.materials.unidir import T700_EPOXY


def _cfg(**kw):
    base = dict(sigma_allow_Pa=1.0e9, tip_defl_max_m=0.1, tip_twist_max_deg=5.0)
    base.update(kw)
    return LaminateSizingConfig(**base)


def _result(**kw):
    base = dict(
        radii=np.array([0.01]), t_skin=0.002, t_bands=np.array([0.002]),
        f0=0.33, f45=0.34, f90=0.33,
        f0_bands=np.array([0.33]), f45_bands=np.array([0.34]), f90_bands=np.array([0.33]),
        mass_kg=10.0, beam_mass_kg=4.0, skin_mass_kg=6.0, converged=True, n_iter=10,
        max_beam_vm_Pa=1.0e8, max_skin_vm_Pa=1.0e8, tip_defl_m=0.05, tip_twist_deg=2.0,
        max_beam_buckling_util=0.9, max_panel_buckling_util=0.9, min_skin_strength_ratio=None,
    )
    base.update(kw)
    return LaminateSizingResult(**base)


def _small():
    P = small_scenario()
    spec = P.geometry
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    return P, spec, model, loads


def test_design_bounds_shape():
    P, spec, model, loads = _small()
    cfg = _cfg(n_skin_bands=3, per_band_layup=True)
    lo, hi = laminate_design_bounds(model, cfg)
    n = model.beam_elements.shape[0]
    assert lo.shape == hi.shape == (n + 3 + 2 * 3,)
    assert np.all(lo < hi)
    # fraction entries bounded [0,1]
    assert np.all(lo[n + 3:] == 0.0) and np.all(hi[n + 3:] == 1.0)


def test_feasible_check_true_and_false():
    cfg = _cfg(buckling_safety_factor=1.5)
    assert laminate_result_is_feasible(_result(), cfg) is True
    assert laminate_result_is_feasible(_result(max_beam_buckling_util=1.7), cfg) is False
    assert laminate_result_is_feasible(_result(tip_twist_deg=9.9), cfg) is False
    # tsai_wu skin: needs min_skin_strength_ratio >= SF
    cfg_tw = _cfg(skin_failure="tsai_wu", tsai_wu_safety_factor=2.0)
    assert laminate_result_is_feasible(_result(min_skin_strength_ratio=2.5), cfg_tw) is True
    assert laminate_result_is_feasible(_result(min_skin_strength_ratio=1.2), cfg_tw) is False


def test_x0_roundtrip_backward_compat():
    P, spec, model, loads = _small()
    cfg = _cfg(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
               tip_twist_max_deg=5.0)
    a = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15)
    lo, hi = laminate_design_bounds(model, cfg)
    n = model.beam_elements.shape[0]
    default = np.concatenate([np.full(n, cfg.r_max), np.full(1, cfg.t_max), [1/3], [1/3]])
    b = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15, x0=default)
    assert np.allclose(a.radii, b.radii) and np.isclose(a.mass_kg, b.mass_kg)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_multistart.py -k "bounds or feasible or roundtrip" -v`
Expected: FAIL — `ImportError` for `laminate_design_bounds`/`laminate_result_is_feasible`,
and `size_beam_shell_laminate` has no `x0` param.

- [ ] **Step 3: Implement in `src/wing_design/beams/laminate_sizing.py`**

(a) Add the bounds helper and feasibility check at module scope (after the imports,
before `size_beam_shell_laminate`):

```python
def laminate_design_bounds(model: BeamShellModel, config: LaminateSizingConfig):
    """(lo, hi) bound arrays for the sizer design vector [radii(n), t_band(B), f0_grp(L), f45_grp(L)]."""
    n = model.beam_elements.shape[0]
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    lo = np.concatenate([np.full(n, config.r_min), np.full(B, config.t_min), np.zeros(2 * L)])
    hi = np.concatenate([np.full(n, config.r_max), np.full(B, config.t_max), np.ones(2 * L)])
    return lo, hi


def laminate_result_is_feasible(
    result: "LaminateSizingResult", config: LaminateSizingConfig, *, tol: float = 1.0e-3,
) -> bool:
    """True iff every active sizing constraint holds within relative ``tol``."""
    if result.max_beam_vm_Pa > config.sigma_allow_Pa * (1.0 + tol):
        return False
    if config.skin_failure == "tsai_wu":
        r = result.min_skin_strength_ratio
        if r is None or r < config.tsai_wu_safety_factor * (1.0 - tol):
            return False
    else:
        if result.max_skin_vm_Pa > config.sigma_allow_Pa * (1.0 + tol):
            return False
    if result.tip_defl_m > config.tip_defl_max_m * (1.0 + tol):
        return False
    if result.tip_twist_deg > config.tip_twist_max_deg * (1.0 + tol):
        return False
    if config.buckling_safety_factor is not None:
        if result.max_beam_buckling_util > 1.0 + tol:
            return False
        if result.max_panel_buckling_util > 1.0 + tol:
            return False
    return True
```

(b) Add `x0: np.ndarray | None = None` to the `size_beam_shell_laminate` signature
(after `ftol: float = 1.0e-4`).

(c) Replace the internal default-x0 construction. Find (the exact line):
```python
    x0 = np.concatenate([
        np.full(n, config.r_max), np.full(B, config.t_max),
        np.full(L, 1.0 / 3.0), np.full(L, 1.0 / 3.0),
    ])
```
Replace with:
```python
    if x0 is None:
        x0 = np.concatenate([
            np.full(n, config.r_max), np.full(B, config.t_max),
            np.full(L, 1.0 / 3.0), np.full(L, 1.0 / 3.0),
        ])
    else:
        lo_b, hi_b = laminate_design_bounds(model, config)
        x0 = np.clip(np.asarray(x0, dtype=float), lo_b, hi_b).copy()
        if x0.shape != (nx,):
            raise ValueError(f"x0 must have length nx={nx}, got {x0.shape}")
        # project layup groups onto the simplex (f0_g + f45_g <= 1)
        fg = x0[f0_lo:f0_lo + L]
        hg = x0[f45_lo:f45_lo + L]
        scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
        x0[f0_lo:f0_lo + L] = fg / scale
        x0[f45_lo:f45_lo + L] = hg / scale
```
(Note: `nx`, `f0_lo`, `f45_lo`, `L` are already defined above this point in the function.
The `.shape` check after `clip` is fine since clip preserves shape; do the length check
on the raw input instead if you prefer — either way raise on wrong length.)

(d) Replace the inline `bounds = (...)` construction with the helper, to keep one source
of truth. Find:
```python
    bounds = (
        [(config.r_min, config.r_max)] * n
        + [(config.t_min, config.t_max)] * B
        + [(0.0, 1.0)] * (2 * L)
    )
```
Replace with:
```python
    lo_b, hi_b = laminate_design_bounds(model, config)
    bounds = [(float(lo_b[i]), float(hi_b[i])) for i in range(nx)]
```

- [ ] **Step 4: Run the Task-1 tests**

Run: `uv run pytest tests/beams/test_multistart.py -k "bounds or feasible or roundtrip" -v`
Expected: PASS (3 tests). Then `uv run pytest tests/beams/test_laminate_sizing.py -q` to
confirm the sizer's default path is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py tests/beams/test_multistart.py
git commit -m "$(cat <<'EOF'
Multi-start: x0 param, design-bounds helper, feasibility check

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Multi-start wrapper + exports (stubbed tests)

**Files:**
- Modify: `src/wing_design/beams/laminate_sizing.py`
- Modify: `src/wing_design/beams/__init__.py`
- Test: `tests/beams/test_multistart.py` (append)

**Context:** Add `MultiStartResult` + `size_beam_shell_laminate_multistart`. Tests
monkeypatch the sizer so NO real sizing runs.

- [ ] **Step 1: Append failing tests to `tests/beams/test_multistart.py`:**

```python
import wing_design.beams.laminate_sizing as ls_mod
from wing_design.beams import size_beam_shell_laminate_multistart, MultiStartResult


def test_multistart_selects_best_feasible(monkeypatch):
    # stub returns canned results by call order; start 2 is lightest-but-infeasible.
    masses = [10.0, 8.0, 6.0, 9.0]
    feas =   [True, True, False, True]
    calls = {"i": 0}

    def fake(model, loads, config, *, ply, rho, maxiter, ftol, x0=None):
        i = calls["i"]; calls["i"] += 1
        buck = 1.7 if not feas[i] else 0.9
        return _result(mass_kg=masses[i], max_beam_buckling_util=buck)

    monkeypatch.setattr(ls_mod, "size_beam_shell_laminate", fake)
    cfg = _cfg(buckling_safety_factor=1.5)
    res = size_beam_shell_laminate_multistart(None, None, cfg, ply=None, rho=1600.0,
                                              n_starts=4, seed=0, maxiter=5)
    assert isinstance(res, MultiStartResult)
    assert res.n_starts == 4 and res.n_feasible == 3
    assert res.best.mass_kg == 8.0          # lightest FEASIBLE (6.0 was infeasible)
    assert res.best_start_index == 1
    assert len(res.start_masses) == 4 and len(res.start_feasible) == 4


def test_multistart_deterministic(monkeypatch):
    seen = []

    def fake(model, loads, config, *, ply, rho, maxiter, ftol, x0=None):
        # mass depends on x0 so different starts differ; record x0 to check determinism
        seen.append(None if x0 is None else float(np.sum(x0)))
        m = 10.0 if x0 is None else float(5.0 + np.sum(x0))
        return _result(mass_kg=m)

    monkeypatch.setattr(ls_mod, "size_beam_shell_laminate", fake)
    cfg = _cfg(n_skin_bands=2)
    P = small_scenario()
    model = build_beam_shell_model(P.geometry, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    r1 = size_beam_shell_laminate_multistart(model, [], cfg, ply=None, rho=1600.0, n_starts=3, seed=42)
    r2 = size_beam_shell_laminate_multistart(model, [], cfg, ply=None, rho=1600.0, n_starts=3, seed=42)
    assert r1.best_start_index == r2.best_start_index
    assert r1.start_masses == r2.start_masses


def test_multistart_start0_is_default(monkeypatch):
    x0s = []

    def fake(model, loads, config, *, ply, rho, maxiter, ftol, x0=None):
        x0s.append(x0)
        return _result(mass_kg=10.0)

    monkeypatch.setattr(ls_mod, "size_beam_shell_laminate", fake)
    cfg = _cfg(n_skin_bands=2)
    P = small_scenario()
    model = build_beam_shell_model(P.geometry, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    size_beam_shell_laminate_multistart(model, [], cfg, ply=None, rho=1600.0, n_starts=3, seed=1)
    assert x0s[0] is None                    # start 0 uses the sizer's default guess
    assert all(x is not None for x in x0s[1:])
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/beams/test_multistart.py -k multistart -v`
Expected: FAIL with `ImportError: cannot import name 'size_beam_shell_laminate_multistart'`.

- [ ] **Step 3: Implement in `src/wing_design/beams/laminate_sizing.py`**

Add the dataclass and wrapper at the end of the module:

```python
@dataclass(frozen=True)
class MultiStartResult:
    best: LaminateSizingResult
    best_start_index: int
    n_starts: int
    n_feasible: int
    start_masses: tuple[float, ...]
    start_feasible: tuple[bool, ...]


def size_beam_shell_laminate_multistart(
    model: BeamShellModel,
    load_arrays,
    config: LaminateSizingConfig,
    *,
    ply: UDPly,
    rho: float,
    n_starts: int = 8,
    seed: int = 0,
    maxiter: int = 80,
    ftol: float = 1.0e-4,
) -> MultiStartResult:
    """Run the sizer from ``n_starts`` initial guesses; return the best feasible result.

    Start 0 is the sizer's default guess (so the result is never worse than a single
    start); starts 1.. are uniform-random within the design bounds (per-group
    simplex-projected), from a seeded RNG (deterministic for a given ``seed``). The
    start loop is a serial map (swappable for multiprocessing). Selection: min-mass among
    feasible results (``laminate_result_is_feasible``); if none feasible, min-mass overall
    with ``n_feasible == 0``.
    """
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    lo, hi = laminate_design_bounds(model, config)
    n = model.beam_elements.shape[0]
    B = config.n_skin_bands
    L = B if config.per_band_layup else 1
    f0_lo = n + B
    f45_lo = n + B + L
    rng = np.random.default_rng(seed)

    def make_start(k):
        if k == 0:
            return None
        x = rng.uniform(lo, hi)
        fg = x[f0_lo:f0_lo + L]
        hg = x[f45_lo:f45_lo + L]
        scale = np.where(fg + hg > 1.0, fg + hg, 1.0)
        x[f0_lo:f0_lo + L] = fg / scale
        x[f45_lo:f45_lo + L] = hg / scale
        return x

    starts = [make_start(k) for k in range(n_starts)]
    # Serial map (swap for multiprocessing.Pool.map later without interface change).
    results = [
        size_beam_shell_laminate(model, load_arrays, config, ply=ply, rho=rho,
                                 maxiter=maxiter, ftol=ftol, x0=s)
        for s in starts
    ]
    feasible = [laminate_result_is_feasible(r, config) for r in results]
    masses = [float(r.mass_kg) for r in results]

    feasible_idx = [i for i, ok in enumerate(feasible) if ok]
    pool = feasible_idx if feasible_idx else list(range(n_starts))
    best_idx = min(pool, key=lambda i: masses[i])
    return MultiStartResult(
        best=results[best_idx],
        best_start_index=int(best_idx),
        n_starts=int(n_starts),
        n_feasible=int(len(feasible_idx)),
        start_masses=tuple(masses),
        start_feasible=tuple(bool(b) for b in feasible),
    )
```
IMPORTANT: the wrapper calls `size_beam_shell_laminate` by its module-global name (so
the monkeypatch in the tests replaces it). Ensure `dataclass` is imported in this module
(it is — `from dataclasses import dataclass`).

- [ ] **Step 4: Export from `src/wing_design/beams/__init__.py`**

Extend the `from .laminate_sizing import (...)` to add `MultiStartResult`,
`laminate_design_bounds`, `laminate_result_is_feasible`,
`size_beam_shell_laminate_multistart`, and add all four to `__all__`. (Also ensure
`LaminateSizingResult` is exported — Task 1's test imports it; if it isn't already,
add it.)

- [ ] **Step 5: Run the multistart tests + full suite**

Run: `uv run pytest tests/beams/test_multistart.py -v`  → PASS (all, incl. stubbed
orchestration — no real sizing in those).
Run: `uv run pytest -q`  → all green (~8 min). `grep -rn "LaminateSizingResult(" src/`
should still show only the sizer constructs it (the wrapper returns existing results).

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/laminate_sizing.py src/wing_design/beams/__init__.py tests/beams/test_multistart.py
git commit -m "$(cat <<'EOF'
Multi-start: serial parallel-ready wrapper + MultiStartResult + exports

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Validation example (compile-checked only, NOT run)

**Files:**
- Create: `examples/36_multistart.py`

**Context:** A cheap validation harness for the USER to run. The implementer creates it
and byte-compiles it but does NOT execute it (user constraint).

- [ ] **Step 1: Create `examples/36_multistart.py`:**

```python
"""Validate the multi-start sizer machinery on the medium wingsail (small/cheap settings).

Runs size_beam_shell_laminate_multistart at a deliberately SMALL mesh (n_levels=5,
n_beams=8) with n_starts=4 and low maxiter -- this is a machinery check, NOT a converged
headline design. Prints each start's mass + feasibility and the chosen best (multi-start
is never worse than the default single start). Intended to run in minutes.
"""
from __future__ import annotations

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate_multistart,
)
from wing_design.materials.unidir import T700_EPOXY


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 8, 5  # small / cheap -- machinery validation only

    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    load_arrays = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
                   for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5, n_skin_bands=4)

    res = size_beam_shell_laminate_multistart(
        model, load_arrays, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
        n_starts=4, seed=0, maxiter=120)

    print(f"n_starts={res.n_starts}  feasible={res.n_feasible}  best_start={res.best_start_index}")
    for i, (m, ok) in enumerate(zip(res.start_masses, res.start_feasible)):
        tag = "default" if i == 0 else f"random#{i}"
        print(f"  start {i} ({tag}): mass {m:.2f} kg  feasible={ok}")
    b = res.best
    print(f"\n  BEST: {b.mass_kg:.2f} kg (start {res.best_start_index}); "
          f"converged={b.converged} twist {b.tip_twist_deg:.2f} "
          f"buck {b.max_beam_buckling_util:.2f}/{b.max_panel_buckling_util:.2f}")
    print("  (multi-start is never worse than the default single start, by construction.)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile-check ONLY (do NOT run)**

Run: `uv run python -m py_compile examples/36_multistart.py && echo ok`
Expected: `ok`. DO NOT execute the example.

- [ ] **Step 3: Commit**

```bash
git add examples/36_multistart.py
git commit -m "$(cat <<'EOF'
Multi-start: cheap validation example (compile-checked, not run)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Record the finding

**Files:**
- Modify: `docs/plan.md`

- [ ] **Step 1: Add status note** (after the per-band-layup note, before `### Phase F`):

```markdown
**Multi-start sizing done (2026-06-08) — machinery only (not run).** A serial,
parallel-ready `size_beam_shell_laminate_multistart` runs the sizer from N initial
guesses and returns the best feasible result (`laminate_result_is_feasible`); start 0 is
the default guess so it never regresses, starts 1.. are seeded uniform-random within
`laminate_design_bounds` (simplex-projected). The sizer gained an `x0` param (default
None = unchanged). Per the cost realities (a single sizing is hours; multi-start is N×),
**no full multi-start was executed** — orchestration is verified with a stubbed sizer
(start-0-default, best-feasible selection, determinism, never-worse), and
`examples/36_multistart.py` is a minutes-scale validation harness left for on-demand use.
The lever for a real headline is parallel execution (structured for it) and/or an
analytic Jacobian to make each sizing cheaper.
```

- [ ] **Step 2: Add Decisions-log row** (after the per-band-layup row):

```markdown
| Multi-start sizing | Serial parallel-ready `size_beam_shell_laminate_multistart`: N seeded starts (start 0 = default → never worse), best-feasible-by-mass selection (`laminate_result_is_feasible`), `x0` param on the sizer + `laminate_design_bounds` helper. Machinery built + unit-tested via a stubbed sizer; NOT run at scale (cost). Parallel exec + full headline deferred. |
```

- [ ] **Step 3: Commit**

```bash
git add docs/plan.md
git commit -m "$(cat <<'EOF'
Multi-start: record finding (machinery built + tested, not run at scale)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] `uv run pytest -q` — all green (multi-start tests are stubbed/fast + one small x0 roundtrip).
- [ ] `uv run python -m py_compile examples/36_multistart.py` — compiles.
- [ ] Confirm NO multi-start sizing was executed (no example run, no headline).
- [ ] Dispatch a final code review over the diff, then use superpowers:finishing-a-development-branch.

## Notes / risks

- **Do not run the multi-start** (user constraint) — stubbed tests + compile-check only.
- **`x0=None` keeps the sizer byte-identical** to today; full suite green is the gate.
- The wrapper must call `size_beam_shell_laminate` as a module global (so monkeypatch
  works) — do not `from .laminate_sizing import size_beam_shell_laminate as ...` inside
  the wrapper.
- Selection rule: feasible-min-mass; if none feasible, min-mass overall with
  `n_feasible=0` (documented, tested).
