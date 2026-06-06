# Phase D Finish — Deferred Items Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the two Phase-D deferred items: (1) make the wing transitions manufacturable by *construction* — an eased (smoothstep) airfoil→circle morph with zero slope at both junctions, so there are no creases (which OCC could not fillet) — and retire the non-functional `apply_wing_fillets`; (2) fix the coarse on-surface fidelity by decoupling geometry resolution from sizing resolution: size at a few levels (SLSQP speed) but build the geometry at many levels by interpolating the sized radii.

**Architecture:** A module-level smoothstep `_transition_blend(u)` replaces the linear morph fraction in both `oml_section_polyline` (so beams ride the smooth OML) and `build_wing_solid`'s fairing loop (so the solid surface is C1 at the junctions) — z-placement of sections is unchanged, only the morph easing. `apply_wing_fillets` and its test/export/call are removed (the crease it tried to round no longer exists). A `resample_segment_radii` helper interpolates Phase-C per-segment radii from the sizing level count to a finer geometry level count; the deliverable example builds beams at high resolution and reports the recovered fidelity.

**Tech Stack:** numpy, build123d (existing loft path), the Phase-C/D beam builders.

---

## Background facts (verified)

- Probe results (build123d 0.10.0): `fillet` works on a box but `max_fillet` on the wing transition edges fails outright ("Failed to find the max value") — post-hoc fillets are impossible on the morphing loft at any radius. Root cause: slope crease where the linearly-morphing fairing meets the constant airfoil (z=0) and the constant spar cylinder (z=-transition_length).
- Fidelity vs uniform geometry levels (worst on-surface error): 8→286 mm, 20→223, 40→125, **60→52 mm**. Decoupled from sizing (sizing can stay at 8 levels).
- `geometry/wing.py`:
  - `oml_section_polyline` transition branch (≈ lines 176–178): `blend = -z / spec.transition_length` (linear).
  - `build_wing_solid` fairing loop (≈ lines 216–223): `blend = j/(n_transition_sections+1)`, `z = -transition_length * blend`, section at `blend`.
  - `_section_face(chord, thickness, pivot_frac, diameter, blend, n_pts)` morphs airfoil(blend=0)→circle(blend=1). `np` already imported.
- `beams/build.py`: `apply_wing_fillets(solid, spec)` (best-effort no-op, 0/2), imports `Axis, fillet` (used only by it); `build_sized_lens_beams(spec, long_radii, *, n_beams, n_levels, n_arc)` / `build_sized_circular_beams(...)` take `long_radii` of length `n_beams*(n_levels-1)`; `_check_long_radii`, `_segment_station_radii` helpers.
- `beams/__init__.py` exports `apply_wing_fillets` (to be removed) among the Phase-D names.
- `examples/23_sized_geometry.py` calls `apply_wing_fillets(build_wing_solid(spec), spec)` and builds beams at `n_levels=8`.
- `tests/beams/test_sized_build.py` contains `test_apply_wing_fillets_returns_valid_solid` (to be removed) plus the sized-beam smoke tests (keep).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/geometry/wing.py` (modify) | Add `_transition_blend` (smoothstep); use it in `oml_section_polyline` + `build_wing_solid` fairing. |
| `src/wing_design/beams/build.py` (modify) | Remove `apply_wing_fillets` + its `Axis`/`fillet` imports; add `resample_segment_radii`. |
| `src/wing_design/beams/__init__.py` (modify) | Drop `apply_wing_fillets` export; add `resample_segment_radii`. |
| `examples/23_sized_geometry.py` (modify) | Drop fillets; build beams at high `n_geo` via `resample_segment_radii`; print sizing-res vs geometry-res fidelity. |
| `tests/geometry/test_transition.py` (create) | Junctions are eased (slope→0), morph endpoints correct. |
| `tests/beams/test_sized_build.py` (modify) | Remove the fillet test; add a `resample_segment_radii` test. |
| `docs/plan.md` (modify) | Mark the two deferred items resolved with numbers. |

---

## Task 1: Smoothstep (eased) transition morph

**Files:**
- Modify: `src/wing_design/geometry/wing.py`
- Test: `tests/geometry/test_transition.py`

- [ ] **Step 1: Write the failing test** (`tests/geometry/test_transition.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec, oml_section_polyline


def _xspan(spec, z):
    p = oml_section_polyline(spec, z)
    return float(p[:, 0].max() - p[:, 0].min())


def test_transition_morph_endpoints():
    spec = WingSpec()
    # z=0 is the airfoil (chordwise extent == root_chord); deep in the spar it's the circle.
    assert abs(_xspan(spec, 0.0) - spec.root_chord) < 1e-6
    assert abs(_xspan(spec, -spec.transition_length) - spec.spar_diameter) < 1e-6


def test_transition_is_eased_at_junctions():
    spec = WingSpec()
    T = spec.transition_length
    eps = T * 0.02

    def slope(z0):
        return abs(_xspan(spec, z0 + eps) - _xspan(spec, z0 - eps)) / (2 * eps)

    s_top = slope(-1.5 * eps)        # just below the z=0 junction
    s_mid = slope(-T / 2)            # middle of the transition
    s_bot = slope(-T + 1.5 * eps)    # just above the z=-T junction
    # Smoothstep eases the morph at both ends, so the junction slopes are far
    # smaller than the mid-transition slope (a linear morph would make them equal).
    assert s_top < 0.5 * s_mid
    assert s_bot < 0.5 * s_mid
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/geometry/test_transition.py -v`
Expected: `test_transition_is_eased_at_junctions` FAILS with the current linear morph (junction slopes ≈ mid slope). `test_transition_morph_endpoints` may already pass.

- [ ] **Step 3: Implement** — in `src/wing_design/geometry/wing.py`:

Add a module-level helper (place it above `_airfoil_to_circle_polyline`):
```python
def _transition_blend(u: float) -> float:
    """Smoothstep airfoil→circle morph fraction with zero slope at u=0 and u=1.

    Eases the transition so the lofted surface meets the constant airfoil (above,
    u=0) and the constant spar cylinder (below, u=1) tangentially — no slope crease
    at the junctions (which OCC could not fillet). u is the fraction into the
    transition band, u = -z / transition_length ∈ [0, 1].
    """
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)
```

In `oml_section_polyline`, change the transition branch:
```python
    elif z >= -spec.transition_length:
        chord = spec.root_chord
        blend = -z / spec.transition_length
```
to:
```python
    elif z >= -spec.transition_length:
        chord = spec.root_chord
        blend = _transition_blend(-z / spec.transition_length)
```

In `build_wing_solid`, change the fairing loop:
```python
    for j in range(1, spec.n_transition_sections + 2):
        blend = j / (spec.n_transition_sections + 1)
        z = -spec.transition_length * blend
```
to (z-placement unchanged — even spacing — only the morph is eased):
```python
    for j in range(1, spec.n_transition_sections + 2):
        u = j / (spec.n_transition_sections + 1)
        z = -spec.transition_length * u
        blend = _transition_blend(u)
```

- [ ] **Step 4: Run the new test + full suite**

Run: `uv run pytest tests/geometry/test_transition.py -v` → both PASS.
Run: `uv run pytest` → all green. (The fairing shape shifts slightly; existing geometry/mesh/FEA tests are topology/relative and should still pass. If any absolute-geometry test fails, report it — don't weaken without flagging.)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/geometry/wing.py tests/geometry/test_transition.py
git commit -m "feat(geometry): eased (smoothstep) transition morph — no junction creases"
# end with the Co-Authored-By trailer
```

---

## Task 2: Retire `apply_wing_fillets`

**Files:**
- Modify: `src/wing_design/beams/build.py`, `src/wing_design/beams/__init__.py`, `tests/beams/test_sized_build.py`, `examples/23_sized_geometry.py`

- [ ] **Step 1: Remove the fillet test** — delete `test_apply_wing_fillets_returns_valid_solid` from `tests/beams/test_sized_build.py` (and any now-unused import of `apply_wing_fillets`/`build_wing_solid` in that file — keep imports still used by the remaining tests).

- [ ] **Step 2: Remove the function + imports** — in `src/wing_design/beams/build.py`, delete the entire `apply_wing_fillets` function and remove `Axis` and `fillet` from the `from build123d import (...)` block (they were used only by that function — verify with a grep before removing).

- [ ] **Step 3: Remove the export** — in `src/wing_design/beams/__init__.py`, remove `apply_wing_fillets` from both the `from .build import ...` line and `__all__`.

- [ ] **Step 4: De-fillet the example** — in `examples/23_sized_geometry.py`, remove the `apply_wing_fillets` import and replace:
```python
    wing = apply_wing_fillets(build_wing_solid(spec), spec)
    wing.label = "wing"
```
with:
```python
    wing = build_wing_solid(spec)   # eased transition → manufacturable junctions, no fillet needed
    wing.label = "wing"
```

- [ ] **Step 5: Verify** — `uv run pytest` (all green); `uv run python -c "import wing_design.beams; assert not hasattr(wing_design.beams, 'apply_wing_fillets'); print('removed')"`.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/build.py src/wing_design/beams/__init__.py tests/beams/test_sized_build.py examples/23_sized_geometry.py
git commit -m "refactor(beams): retire apply_wing_fillets (eased transition removes the crease)"
# trailer
```

---

## Task 3: Radius resampling (decouple geometry resolution from sizing)

**Files:**
- Modify: `src/wing_design/beams/build.py`
- Test: `tests/beams/test_sized_build.py`

- [ ] **Step 1: Write the failing test** (append to `tests/beams/test_sized_build.py`)

```python
from wing_design.beams.build import resample_segment_radii


def test_resample_uniform_radii_stays_uniform():
    out = resample_segment_radii(np.full(4 * 3, 0.02), n_beams=4, n_from=4, n_to=9)
    assert out.shape == (4 * 8,)          # n_beams * (n_to - 1)
    assert np.allclose(out, 0.02)


def test_resample_preserves_per_beam_trend():
    # beam 0 ramps 1→3 over its 2 segments; resampled to 4 segments stays monotone in [1,3]
    n_beams, n_from = 2, 3
    long_radii = np.array([1.0, 3.0, 5.0, 7.0])  # beam0:[1,3], beam1:[5,7]
    out = resample_segment_radii(long_radii, n_beams=n_beams, n_from=n_from, n_to=5)
    seg_to = 4
    b0 = out[:seg_to]
    b1 = out[seg_to:]
    assert np.all(np.diff(b0) >= 0) and b0.min() >= 1.0 - 1e-9 and b0.max() <= 3.0 + 1e-9
    assert np.all(np.diff(b1) >= 0) and b1.min() >= 5.0 - 1e-9 and b1.max() <= 7.0 + 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_sized_build.py::test_resample_uniform_radii_stays_uniform -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement** — append to `src/wing_design/beams/build.py`:

```python
def resample_segment_radii(
    long_radii: np.ndarray,
    *,
    n_beams: int,
    n_from: int,
    n_to: int,
) -> np.ndarray:
    """Resample per-segment longitudinal radii from ``n_from`` to ``n_to`` levels.

    Beam sizing runs at a coarse level count (SLSQP speed); the geometry is built
    at a finer one. For each beam, its ``n_from-1`` segment radii are interpolated
    (by normalized position along the beam) onto ``n_to-1`` segments. Returns a
    ``(n_beams*(n_to-1),)`` array in the same beam-major/level-minor layout.
    """
    _check_long_radii(long_radii, n_beams, n_from)
    seg_from = n_from - 1
    seg_to = n_to - 1
    x_from = (np.arange(seg_from) + 0.5) / seg_from
    x_to = (np.arange(seg_to) + 0.5) / seg_to
    src = np.asarray(long_radii, dtype=float)
    out = np.empty(n_beams * seg_to)
    for b in range(n_beams):
        rb = src[b * seg_from:(b + 1) * seg_from]
        out[b * seg_to:(b + 1) * seg_to] = np.interp(x_to, x_from, rb)
    return out
```

- [ ] **Step 4: Run the tests** — `uv run pytest tests/beams/test_sized_build.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/build.py tests/beams/test_sized_build.py
git commit -m "feat(beams): resample_segment_radii to build geometry above sizing resolution"
# trailer
```

---

## Task 4: Wire the example to high-resolution geometry + close the items

**Files:**
- Modify: `src/wing_design/beams/__init__.py`, `examples/23_sized_geometry.py`, `docs/plan.md`

- [ ] **Step 1: Export the helper** — in `src/wing_design/beams/__init__.py`, add `resample_segment_radii` to the `from .build import ...` line and `__all__`.

- [ ] **Step 2: Build geometry at high resolution in the example.** In `examples/23_sized_geometry.py`, add `n_geo = 60` next to `n_beams, n_levels = 16, 8`, import `resample_segment_radii`, and change the sizing→build handoff. Replace the build block:
```python
    try:
        beams = build_sized_lens_beams(spec, sized.radii, n_beams=n_beams, n_levels=n_levels)
        section_kind = "lens"
    except Exception as exc:
        print(f"  lens build failed ({type(exc).__name__}: {exc}); falling back to circular.")
        beams = build_sized_circular_beams(spec, sized.radii, n_beams=n_beams, n_levels=n_levels)
        section_kind = "circular"
    print(f"  built {len(beams)} {section_kind} beams")
```
with:
```python
    # Sizing ran at n_levels (SLSQP speed); build the geometry at the finer n_geo
    # by interpolating the sized radii — decouples fidelity from sizing cost.
    geo_radii = resample_segment_radii(sized.radii, n_beams=n_beams, n_from=n_levels, n_to=n_geo)
    err_geo = spline_surface_error(spec, n_beams=n_beams, n_levels=n_geo)
    print(f"  geometry built at {n_geo} levels; on-surface fidelity worst {err_geo*1e3:.0f} mm "
          f"(vs {err_full*1e3:.0f} mm at the {n_levels}-level sizing resolution)")
    try:
        beams = build_sized_lens_beams(spec, geo_radii, n_beams=n_beams, n_levels=n_geo)
        section_kind = "lens"
    except Exception as exc:
        print(f"  lens build failed ({type(exc).__name__}: {exc}); falling back to circular.")
        beams = build_sized_circular_beams(spec, geo_radii, n_beams=n_beams, n_levels=n_geo)
        section_kind = "circular"
    print(f"  built {len(beams)} {section_kind} beams")
```

- [ ] **Step 3: Run the example** — `uv run python examples/23_sized_geometry.py`. Expect: sizing completes (~1–2 min), geometry builds at 60 levels with worst fidelity ≈ 52 mm (down from 286 mm), 16 lens beams, STEP written, sane bbox, **no fillet lines**. Paste the full output.

- [ ] **Step 4: Update `docs/plan.md`.** In the Phase-D status note, replace the two open-findings bullets with their resolutions:
  - Fillets: eased smoothstep transition removes the junction creases by construction (no fillet needed); `apply_wing_fillets` retired.
  - Fidelity: decoupled geometry resolution from sizing — geometry built at 60 levels (worst on-surface error ≈ 52 mm, down from 286 mm at the 8-level sizing resolution) via `resample_segment_radii`.
  Keep the wording factual and matching the actual numbers observed in Step 3.

- [ ] **Step 5: Full suite** — `uv run pytest` → all green.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/__init__.py examples/23_sized_geometry.py docs/plan.md
git commit -m "feat(beams): build sized geometry at high resolution; close Phase-D deferred items"
# trailer
```

---

## Self-Review

- **Spec coverage:** smooth-transition construction → Task 1 (eased morph in both the OML used by beams and the wing solid); fillet retirement → Task 2; fidelity decouple-and-densify → Tasks 3–4. Both deferred items closed.
- **Placeholder scan:** none.
- **Type consistency:** `_transition_blend` used in both `oml_section_polyline` and `build_wing_solid`; `resample_segment_radii` returns the `n_beams*(n_to-1)` layout that `build_sized_*_beams(..., n_levels=n_to)` consume; `_check_long_radii` reused for input validation.
- **Risk:** Task 1 changes the wing solid shape slightly (eased fairing) — existing tests are relative/topological and should hold; full suite is run. Task 2 removes a confirmed-nonfunctional API (no behavior lost). Task 4's high-res build is fast (geometry build is cheap; only sizing is slow) and the fidelity gain is verified by the probe.
- **Note:** the eased transition may *also* reduce the spline overshoot (smoother path), so the reported high-res fidelity could be even better than the 52 mm uniform-morph probe figure — report the actual number.
