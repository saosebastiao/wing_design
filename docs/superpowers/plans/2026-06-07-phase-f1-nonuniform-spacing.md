# Phase F.1 — Non-Uniform (Stress-Weighted) Beam Spacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Place the N form beams at **stress-weighted arc positions** around each cross-section (clustered where the skin's cumulative principal stress is high) instead of even arc-length spacing, then re-size and compare to even spacing under the same load cases — testing whether stress-driven layout is lighter/stiffer.

**Scope note:** Phase F increment 1 — **non-uniform spacing only** (the chosen scope). The second diagonal beam family is deferred (F.2). The placement is a one-shot: a baseline even-spacing solve provides the skin-membrane-stress field that drives the re-placement; not iterated. Equal-cumulative-stress placement reduces *exactly* to even spacing when the stress weights are uniform (a built-in correctness anchor). Honest reporting: if one-shot stress-weighting does not beat even spacing, that is itself the finding.

**Architecture:** A pure helper `stress_weighted_targets(weights, n)` maps per-perimeter-segment stress weights to N arc-fraction targets at equal cumulative weight (inverse-CDF). `resample_closed_polyline`, `beam_section_points`, `form_beam_grid`, and `build_beam_shell_model` gain an optional `arc_fractions` argument (default `None` ⇒ even spacing, fully backward-compatible) threading the non-uniform targets through. `cross_section_stress_weights(model, load_arrays)` aggregates the baseline skin-membrane von Mises per perimeter segment. The example builds even → weights → non-uniform → sizes both → compares.

**Tech Stack:** numpy, the existing beam-shell model + sizing + skin-stress recovery.

---

## Background facts (verified)

- `beams/cross_section.py::resample_closed_polyline(poly, n)`: arc-length resample, even targets via `np.linspace(0.0, total, n, endpoint=False)` (this is where "even spacing" lives). `beam_section_points(spec, z, n_beams, n_pts=None)` → calls it on `oml_section_polyline(spec, z)`.
- `beams/splines.py::form_beam_grid(spec, z_levels, n_beams)` → `(n_beams, n_levels, 3)`, fills each level via `beam_section_points(spec, z_levels[k], n_beams)`.
- `beams/shell_model.py::build_beam_shell_model(spec, *, n_beams, n_levels, beam_radius, material, knockdown, nu, skin_thickness)` builds nodes from `form_beam_grid`, longitudinal beams + skin triangles. `skin_triangles(n_beams, n_levels)`: for `k in range(n_levels-1): for b in range(n_beams):` appends 2 triangles for the panel between beams b and (b+1)%n_beams — so triangle index `e` belongs to perimeter segment `(e // 2) % n_beams`.
- `beams/shell_model.py::solve_beam_shell_model(model, loads) -> FrameResult`.
- `structural/shell.py::recover_membrane_stress(nodes, tris, displacements, *, E, nu)`, `membrane_von_mises(stress)`.
- `beams/shell_sizing.py::size_beam_shell(model, load_arrays, config, *, rho, ...)` (re-sizes a model's per-element beam radii); `BeamShellSizingConfig`.
- `beams/fea_model.py::build_beam_frame`, `project_panels_to_beam_nodes` (loads); aero wiring as in examples/25.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/beams/layout.py` (create) | `stress_weighted_targets(weights, n)`; `cross_section_stress_weights(model, load_arrays)`. |
| `src/wing_design/beams/cross_section.py` (modify) | `resample_closed_polyline(poly, n, arc_fractions=None)`; `beam_section_points(..., arc_fractions=None)`. |
| `src/wing_design/beams/splines.py` (modify) | `form_beam_grid(spec, z_levels, n_beams, arc_fractions=None)`. |
| `src/wing_design/beams/shell_model.py` (modify) | `build_beam_shell_model(..., arc_fractions=None)`. |
| `src/wing_design/beams/__init__.py` (modify) | Export `stress_weighted_targets`, `cross_section_stress_weights`. |
| `examples/30_nonuniform_spacing.py` (create) | Even vs stress-weighted layout; re-size both; compare. |
| Tests: `tests/beams/test_layout.py` (create); additions to `test_cross_section.py`, `test_shell_model.py`. |
| `docs/plan.md` (modify) | Record F.1. |

---

## Task 1: Arc-fraction placement + stress-weighted targets

**Files:**
- Modify: `src/wing_design/beams/cross_section.py`, `src/wing_design/beams/splines.py`
- Create: `src/wing_design/beams/layout.py`
- Test: `tests/beams/test_layout.py`, `tests/beams/test_cross_section.py`

- [ ] **Step 1: Write failing tests.**

`tests/beams/test_layout.py`:
```python
import numpy as np

from wing_design.beams.layout import stress_weighted_targets


def test_uniform_weights_recover_even_spacing():
    t = stress_weighted_targets(np.ones(8), 8)
    assert np.allclose(t, np.arange(8) / 8.0)


def test_targets_in_unit_interval_and_sorted():
    t = stress_weighted_targets(np.array([1.0, 2.0, 5.0, 1.0, 1.0, 3.0]), 10)
    assert t.shape == (10,)
    assert np.all((t >= 0.0) & (t < 1.0))
    assert np.all(np.diff(t) >= 0.0)


def test_peaked_weights_cluster_targets():
    w = np.ones(8)
    w[2] = 100.0   # one perimeter segment carries almost all the stress
    t = stress_weighted_targets(w, 8)
    near_peak = np.sum((t >= 2.0 / 8.0) & (t < 3.0 / 8.0))
    assert near_peak >= 4   # most beams cluster into the high-stress segment
```

Append to `tests/beams/test_cross_section.py`:
```python
def test_resample_with_arc_fractions_matches_even_default():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    from wing_design.beams.cross_section import resample_closed_polyline
    a = resample_closed_polyline(sq, 8)
    b = resample_closed_polyline(sq, 8, arc_fractions=np.arange(8) / 8.0)
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run** — `uv run pytest tests/beams/test_layout.py tests/beams/test_cross_section.py -v`, expect failures (ImportError / unexpected kwarg).

- [ ] **Step 3: Implement.**

Create `src/wing_design/beams/layout.py`:
```python
"""Stress-driven beam layout helpers (Phase F): place beams where the skin is loaded.

`stress_weighted_targets` turns per-perimeter-segment stress weights into N arc-fraction
targets at equal cumulative weight (clustering beams where stress is high); uniform
weights give exactly even spacing. `cross_section_stress_weights` derives those weights
from a baseline beam-shell solve's skin membrane stress.
"""
from __future__ import annotations

import numpy as np

from ..structural.shell import membrane_von_mises, recover_membrane_stress
from .shell_model import BeamShellModel, solve_beam_shell_model


def stress_weighted_targets(weights: np.ndarray, n: int) -> np.ndarray:
    """(n,) arc fractions in [0,1) placing N beams at equal cumulative ``weights``.

    `weights[i]` is the stress intensity (density) of perimeter segment i of m equal
    arc-length segments. Beams are placed where the cumulative weight reaches
    (b/n)*total, so they cluster where weights are high. Uniform weights -> even
    spacing (b/n). Returns ascending fractions.
    """
    w = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    m = w.shape[0]
    seg = 1.0 / m
    cum = np.concatenate([[0.0], np.cumsum(w * seg)])   # (m+1,) cumulative weight at segment boundaries
    boundaries = np.concatenate([np.arange(m) / m, [1.0]])
    total = cum[-1]
    target_w = (np.arange(n) / n) * total
    return np.interp(target_w, cum, boundaries)


def cross_section_stress_weights(model: BeamShellModel, load_arrays: list[np.ndarray]) -> np.ndarray:
    """(n_beams,) per-perimeter-segment skin stress intensity from a baseline solve.

    For each perimeter segment (the skin panel between beams b and b+1), the peak
    membrane von Mises over its triangles, all z-levels, and all load cases.
    """
    nb = model.n_beams
    w = np.zeros(nb)
    for loads in load_arrays:
        res = solve_beam_shell_model(model, loads)
        vm = membrane_von_mises(
            recover_membrane_stress(model.nodes, model.shell_tris, res.displacements,
                                    E=model.E_skin, nu=model.nu_skin)
        )
        for e in range(model.shell_tris.shape[0]):
            b = (e // 2) % nb
            w[b] = max(w[b], float(vm[e]))
    return w
```

In `beams/cross_section.py`, generalize:
```python
def resample_closed_polyline(poly: np.ndarray, n: int, arc_fractions: np.ndarray | None = None) -> np.ndarray:
```
Keep the existing arc-length setup (`pts`, `loop`, `seg`, `s`, `total`). Replace the targets line:
```python
    if arc_fractions is None:
        targets = np.linspace(0.0, total, n, endpoint=False)
    else:
        af = np.asarray(arc_fractions, dtype=float)
        if af.shape[0] != n:
            raise ValueError(f"arc_fractions must have length n={n}, got {af.shape[0]}")
        targets = af * total
```
And `beam_section_points(spec, z, n_beams, n_pts=None, arc_fractions=None)` passes `arc_fractions` through to `resample_closed_polyline`.

In `beams/splines.py`, `form_beam_grid(spec, z_levels, n_beams, arc_fractions=None)` passes `arc_fractions` to each `beam_section_points(...)` call.

Export `stress_weighted_targets`, `cross_section_stress_weights` from `beams/__init__.py`.

- [ ] **Step 4: Run** the new tests + full suite (existing tests use the even default → unchanged).

- [ ] **Step 5: Commit** — `feat(beams): stress-weighted arc placement + arc_fractions plumbing` + trailer.

---

## Task 2: Non-uniform beam-shell model

**Files:**
- Modify: `src/wing_design/beams/shell_model.py`
- Test: `tests/beams/test_shell_model.py`

- [ ] **Step 1: Write failing tests** (append to `tests/beams/test_shell_model.py`)

```python
def test_build_model_accepts_arc_fractions():
    spec = WingSpec()
    even = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    af = np.linspace(0, 1, 8, endpoint=False)
    same = build_beam_shell_model(spec, n_beams=8, n_levels=5, arc_fractions=af)
    assert np.allclose(even.nodes, same.nodes)   # even fractions == default


def test_nonuniform_model_solves():
    spec = WingSpec()
    af = stress_weighted_targets(np.array([1, 1, 5, 1, 1, 1, 3, 1.0]), 8)
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, arc_fractions=af)
    assert model.nodes.shape == (8 * 5, 3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 20.0
    res = solve_beam_shell_model(model, loads)
    assert np.isfinite(res.displacements).all()
    assert np.allclose(res.displacements[model.fixed_nodes], 0.0)
```
(Add `from wing_design.beams.layout import stress_weighted_targets` to the test imports.)

- [ ] **Step 2: Run**, expect failure (unexpected kwarg).

- [ ] **Step 3: Implement** — `build_beam_shell_model` gains `arc_fractions: np.ndarray | None = None`; pass it to `form_beam_grid(spec, z, n_beams, arc_fractions=arc_fractions)`. Everything else (elements, skin_triangles, fixed/tip nodes, materials) is unchanged — the topology is identical; only node positions move. Default `None` ⇒ even (backward-compatible).

- [ ] **Step 4: Run** the new tests + full suite.

- [ ] **Step 5: Commit** — `feat(beams): build_beam_shell_model accepts non-uniform arc_fractions` + trailer.

---

## Task 3: Demonstrator + plan update

**Files:**
- Create: `examples/30_nonuniform_spacing.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Create `examples/30_nonuniform_spacing.py`**:
  1. Build the even-spacing beam-shell model + frame + LL envelope `load_arrays` (as in examples/25).
  2. `weights = cross_section_stress_weights(even_model, load_arrays)`; print the per-segment weights (MPa) to show the stress concentration.
  3. `af = stress_weighted_targets(weights, n_beams)`; build `nonu_model = build_beam_shell_model(spec, ..., arc_fractions=af)`.
  4. Size BOTH models with the same `BeamShellSizingConfig` (sigma_allow, tip_defl_max=0.02*span, tip_twist_max=5.0, buckling_safety_factor=1.5) via `size_beam_shell(..., rho=P.rho_kgm3, maxiter=150)`.
  5. Print a comparison: even sized mass vs non-uniform sized mass (beam + skin), governing metrics (max stress, defl, twist, buckling util), and the % mass change. State whether stress-weighted layout is lighter.

  (FOREGROUND, ~4–7 min: baseline solve + 2 sizings.)

- [ ] **Step 2: Run** `uv run python examples/30_nonuniform_spacing.py`. Paste full output. Report: the stress-weight distribution (is it concentrated?), the even vs non-uniform sized masses, and whether stress-weighted spacing wins. **Honest reporting**: if it is NOT lighter (e.g. clustering starves low-stress panels into buckling, or even spacing was already near-optimal), say so — a null/negative result is a legitimate Phase-F.1 finding that motivates iteration or the second beam family (F.2).

- [ ] **Step 3: Update `docs/plan.md`** — Phase-F status with the ACTUAL result (stress-weight concentration; even vs non-uniform sized mass; verdict). Decisions-log row:

```markdown
| Phase-F.1 non-uniform spacing | Beams placed at equal-cumulative-skin-stress arc positions (`stress_weighted_targets` from a baseline `cross_section_stress_weights`); `arc_fractions` threaded through cross_section/splines/shell_model (default even, backward-compatible). One-shot (not iterated); second diagonal beam family deferred (F.2). Result: <fill from run>. |
```

- [ ] **Step 4:** `uv run pytest` → green.

- [ ] **Step 5: Commit** — `feat(beams): Phase-F.1 non-uniform-spacing example; record finding` + trailer.

---

## Self-Review

- **Spec coverage:** non-uniform spacing by cumulative principal stress → `stress_weighted_targets` + `cross_section_stress_weights` (Tasks 1–2); demonstrably lighter/stiffer vs even → re-size comparison (Task 3). Second diagonal family explicitly deferred (F.2).
- **Placeholder scan:** none.
- **Backward compatibility:** `arc_fractions=None` everywhere ⇒ even `np.linspace` ⇒ existing geometry/model/sizing behavior byte-identical (anchored by `test_resample_with_arc_fractions_matches_even_default` and `test_build_model_accepts_arc_fractions`). Uniform weights ⇒ even targets (anchored by `test_uniform_weights_recover_even_spacing`).
- **Type consistency:** `stress_weighted_targets(weights (m,), n) -> (n,)` arc fractions feed `build_beam_shell_model(arc_fractions=...)` → `form_beam_grid` → `beam_section_points` → `resample_closed_polyline`; `cross_section_stress_weights` returns `(n_beams,)` matching the perimeter-segment count; triangle→segment map `(e//2)%n_beams` matches `skin_triangles` construction.
- **Known limitations (intentional):** one-shot placement from a baseline (even-spacing) stress field (not iterated to self-consistency); peak-vm-per-segment weight (a simple intensity; cumulative-integral or principal-value variants possible); same arc_fractions at every z (consistent longitudinal beams, no spanwise variation of the layout); skin/beam topology unchanged (only node positions move).
