# Phase D — Deepen Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the form-beam geometry manufacturable and faithful to the structural model: derive the spar radius from the airfoil, add an on-surface spline-fidelity metric, generate inward-arc ("lens") beam cross-sections sized to the Phase-C areas with true surface normals, build the sized lens beams in build123d, and fillet the wing solid — with honest fallbacks where build123d 0.10.0 can't deliver cleanly.

**Architecture:** Four threads, ordered low-risk → high-risk. (1) `WingSpec.spar_radius`/`spar_diameter` become derived properties (max inscribed circle at the pivot, floored to cm). (2) A pure-numpy spline-fidelity metric compares each beam spline to the true OML path. (3) Pure-numpy primitives — OML outward normals + an area-exact lens cross-section generator. (4) A build123d builder lofts sized lens sections along each beam (circular fallback if lofting custom faces fails), plus a best-effort filleter on the wing solid. The deterministic numpy work (Tasks 1–3) is fully unit-tested; the build123d work (Tasks 4–5) is smoke-tested and verified by running the example, with fallbacks reported rather than hidden.

**Tech Stack:** numpy (geometry primitives), build123d 0.10.0 (`BuildPart/BuildSketch/BuildLine/Polyline/make_face/loft/Plane/fillet`), the Phase-C `beams.sizing` output, scipy splines (existing).

---

## Background facts (verified in the codebase)

- `WingSpec` (geometry/wing.py:36) is a frozen dataclass; `spar_diameter: float = 0.15` is a plain field. Read at wing.py:183/210/220/227 (as `spec.spar_diameter`, passed as `diameter`), mesh.py:65 (`0.25 * spec.spar_diameter`), examples/00 (print), and asserted self-consistently at tests/beams/test_cross_section.py:23 (`r == spec.spar_diameter / 2.0`).
- `geometry/airfoil.py`: `naca_00xx_coords(t, n=200, closed_te=True) -> (M,2)` (closed contour, unit chord); `naca_00xx_thickness(x, t, closed_te=True) -> (N,)` (half-thickness). wing.py already imports `naca_00xx_coords`.
- `beams/cross_section.py`: `beam_section_points(spec, z, n_beams, n_pts=None) -> (n_beams,2)` arc-spaced OML points (0=TE, n_beams//2=LE); `resample_closed_polyline(poly, n) -> (n,2)`.
- `beams/splines.py`: `default_z_levels(spec, n=20) -> (n,)` (descends tip→keel); `form_beam_grid(spec, z, n_beams) -> (n_beams,n_levels,3)`; `fit_beam_splines(grid, smoothing=0.0) -> list[tck]`; `sample_spline(tck, n=50) -> (n,3)`.
- `beams/build.py` `build_form_beams` insets a fixed-radius circle along the radial-from-pivot direction and lofts (Plane.XY.offset(z) sketches). No true surface normal exists anywhere.
- Phase-C `beams.sizing.size_beams(frame, load_arrays, config, *, rho, ...) -> SizingResult`; `SizingResult.radii` is `(n_beams*(n_levels-1),)` longitudinal radii, beam-major/level-minor: beam b, segment k at index `b*(n_levels-1)+k`. `n_longitudinal(frame)`, `frame_mass(...)` available.
- build123d 0.10.0: `fillet`, `Polyline`, `loft`, `make_face`, `BuildLine`, `BuildSketch`, `BuildPart`, `Plane` available. 3D solid `offset`/`shell` are unreliable on this morphing geometry (documented in build.py) — do NOT use them.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/geometry/wing.py` (modify) | Replace `spar_diameter` field with derived `spar_radius` + `spar_diameter` properties (max inscribed circle at pivot, floored to cm). |
| `src/wing_design/beams/fidelity.py` (create) | `beam_path_at_z`, `spline_surface_error` — on-surface fidelity metric. |
| `src/wing_design/beams/sections.py` (create) | `oml_outward_normals`, `lens_section_polyline` — pure-numpy normals + area-exact inward-arc cross-section. |
| `src/wing_design/beams/build.py` (modify) | Add `build_sized_lens_beams` (+ `build_sized_circular_beams` fallback) and `apply_wing_fillets`. |
| `src/wing_design/beams/__init__.py` (modify) | Export the new builders + metric + section primitives. |
| `examples/23_sized_geometry.py` (create) | Derive spar radius → size beams (Phase C) → build sized lens beams + filleted wing → assembly → STEP to `exports/` → print fidelity + areas. |
| `tests/geometry/test_spar_radius.py` (create) | Derived spar-radius value + property relations. |
| `tests/beams/test_fidelity.py` (create) | Surface error decreases with more z-levels. |
| `tests/beams/test_sections.py` (create) | Lens area exact, geometry; normals unit + outward. |
| `tests/beams/test_sized_build.py` (create) | Smoke: sized lens beams + fillet build valid solids. |
| `docs/plan.md` (modify) | Mark Phase D done; record build123d outcomes/fallbacks. |

---

## Task 1: Derived spar radius

**Files:**
- Modify: `src/wing_design/geometry/wing.py`
- Test: `tests/geometry/test_spar_radius.py`

- [ ] **Step 1: Write the failing test** (`tests/geometry/test_spar_radius.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec


def test_spar_radius_is_floored_inscribed_circle():
    spec = WingSpec()  # NACA0018, root_chord=1, pivot 0.25c
    # Max inscribed circle centered at the pivot ≈ local half-thickness ≈ 0.089 m,
    # floored to the nearest cm → 0.08 m.
    assert abs(spec.spar_radius - 0.08) < 1e-9
    # It is floored, so it never exceeds the true min distance from pivot to surface.
    from wing_design.geometry.airfoil import naca_00xx_coords
    coords = (naca_00xx_coords(spec.thickness, n=spec.n_airfoil_points)
              - np.array([spec.pivot_frac, 0.0])) * spec.root_chord
    true_min = float(np.min(np.hypot(coords[:, 0], coords[:, 1])))
    assert spec.spar_radius <= true_min + 1e-12


def test_spar_diameter_is_twice_radius():
    spec = WingSpec()
    assert abs(spec.spar_diameter - 2.0 * spec.spar_radius) < 1e-12
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/geometry/test_spar_radius.py -v`
Expected: FAIL (currently `spar_diameter` is 0.15, `spar_radius` doesn't exist).

- [ ] **Step 3: Implement** — in `src/wing_design/geometry/wing.py`, remove the `spar_diameter` field and add two derived properties.

Delete this line from the `WingSpec` field block (wing.py:44):
```python
    spar_diameter: float = 0.15        # m, diameter of the cylindrical spar / fairing base
```

Add these properties to `WingSpec` (e.g., just before `chord_at_z`):
```python
    @property
    def spar_radius(self) -> float:
        """Max circle inscribed in the root airfoil, centered at the pivot, floored to cm.

        The cylindrical spar is the largest circle that fits inside the airfoil at
        the rotation axis: its radius is the minimum distance from the pivot point
        to the airfoil surface. Floored to the nearest centimetre for a tidy,
        manufacturable stock size. (~0.08 m for NACA0018, 1 m chord, 25% pivot.)
        """
        coords = (naca_00xx_coords(self.thickness, n=self.n_airfoil_points)
                  - np.array([self.pivot_frac, 0.0])) * self.root_chord
        r_min = float(np.min(np.hypot(coords[:, 0], coords[:, 1])))
        return np.floor(r_min * 100.0) / 100.0

    @property
    def spar_diameter(self) -> float:
        """Derived spar/fairing-base circle diameter = 2 * spar_radius."""
        return 2.0 * self.spar_radius
```

(`np` and `naca_00xx_coords` are already imported in wing.py. All existing `spec.spar_diameter` reads keep working unchanged via the property.)

- [ ] **Step 4: Run the new test + full suite**

Run: `uv run pytest tests/geometry/test_spar_radius.py -v` → PASS.
Run: `uv run pytest` → all green. (The spar diameter shifts 0.15→0.16, which only changes geometry slightly; `test_cross_section.py:23` is self-consistent and stays valid. If any test asserts an absolute spar dimension and fails, report it — do not weaken it without flagging.)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/geometry/wing.py tests/geometry/test_spar_radius.py
git commit -m "feat(geometry): derive spar_radius/spar_diameter from airfoil"
# end with the Co-Authored-By trailer
```

---

## Task 2: Spline-fit fidelity metric

**Files:**
- Create: `src/wing_design/beams/fidelity.py`
- Test: `tests/beams/test_fidelity.py`

- [ ] **Step 1: Write the failing test** (`tests/beams/test_fidelity.py`)

```python
from wing_design.geometry import WingSpec
from wing_design.beams.fidelity import spline_surface_error


def test_surface_error_decreases_with_more_levels():
    spec = WingSpec()
    coarse = spline_surface_error(spec, n_beams=16, n_levels=6)
    fine = spline_surface_error(spec, n_beams=16, n_levels=24)
    assert fine < coarse           # more levels → closer to the true OML
    assert fine >= 0.0


def test_surface_error_is_small_when_fine():
    spec = WingSpec()
    err = spline_surface_error(spec, n_beams=16, n_levels=40)
    assert err < 0.01              # < 1 cm worst-case on a 5 m wing
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_fidelity.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/wing_design/beams/fidelity.py`

```python
"""On-surface fidelity metric for the form-beam splines.

Each beam spline interpolates the OML at the chosen z-levels; between levels it
may drift off the true surface. We measure that drift: for many z between tip and
keel, compare each beam spline's (x, y) to the true OML beam position at that z
(`beam_section_points`), and report the worst distance over all beams. Lower is
better; it shrinks as the z-level count rises.
"""
from __future__ import annotations

import numpy as np

from ..geometry.wing import WingSpec
from .cross_section import beam_section_points
from .splines import default_z_levels, fit_beam_splines, form_beam_grid, sample_spline


def beam_path_at_z(spec: WingSpec, z: float, n_beams: int) -> np.ndarray:
    """True on-OML (x, y, z) of every beam at height ``z`` (the arc-spaced position)."""
    xy = beam_section_points(spec, z, n_beams)
    return np.column_stack([xy, np.full(n_beams, z)])


def spline_surface_error(
    spec: WingSpec,
    *,
    n_beams: int,
    n_levels: int,
    smoothing: float = 0.0,
    n_samples: int = 400,
) -> float:
    """Worst-beam max distance [m] between the fitted splines and the true OML path.

    Fits the beam splines through ``n_levels`` z-levels, then samples each spline
    densely and, at each sample's own z, compares its (x, y) to the true OML beam
    position. Returns the maximum over all samples and beams.
    """
    z_levels = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z_levels, n_beams)
    splines = fit_beam_splines(grid, smoothing=smoothing)
    worst = 0.0
    for b in range(n_beams):
        pts = sample_spline(splines[b], n=n_samples)  # (n_samples, 3)
        for px, py, pz in pts:
            true_xy = beam_section_points(spec, float(pz), n_beams)[b]
            worst = max(worst, float(np.hypot(px - true_xy[0], py - true_xy[1])))
    return worst
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/beams/test_fidelity.py -v`
Expected: PASS. (If `test_surface_error_is_small_when_fine` fails because the worst error at 40 levels is ≥1 cm, that's a real fidelity finding — report it with the actual number rather than loosening the threshold silently.)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/fidelity.py tests/beams/test_fidelity.py
git commit -m "feat(beams): on-surface spline fidelity metric"
# trailer
```

---

## Task 3: OML normals + lens cross-section (pure numpy)

**Files:**
- Create: `src/wing_design/beams/sections.py`
- Test: `tests/beams/test_sections.py`

- [ ] **Step 1: Write the failing test** (`tests/beams/test_sections.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.sections import oml_outward_normals, lens_section_polyline


def test_normals_unit_and_outward():
    spec = WingSpec()
    P = __import__("wing_design.beams.cross_section", fromlist=["beam_section_points"]).beam_section_points(spec, 1.0, 16)
    n = oml_outward_normals(spec, 1.0, 16)
    assert n.shape == (16, 2)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)
    # outward: each normal points away from the pivot (positive dot with the radial dir)
    radial = P / np.linalg.norm(P, axis=1, keepdims=True)
    assert np.all(np.sum(n * radial, axis=1) > 0.0)


def test_lens_area_is_exact():
    center = np.array([0.3, 0.05])
    normal = np.array([0.0, 1.0])
    area = 1.2e-3
    poly = lens_section_polyline(center, normal, area, n_arc=64)
    # shoelace area of the closed polygon
    x, y = poly[:, 0], poly[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert abs(shoelace - area) / area < 1e-3


def test_lens_bulges_inward():
    center = np.array([0.3, 0.05])
    normal = np.array([0.0, 1.0])   # outward = +y, so the lens must bulge toward -y
    area = 1.2e-3
    poly = lens_section_polyline(center, normal, area, n_arc=64)
    # deepest point is opposite the normal (smallest y), below the chord through center
    assert poly[:, 1].min() < center[1]
    assert abs(poly[:, 1].max() - center[1]) < 1e-9   # flat (outer) edge sits on the surface
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_sections.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/wing_design/beams/sections.py`

```python
"""Pure-numpy beam cross-section primitives for Phase-D sized geometry.

`oml_outward_normals` gives the true in-plane outward surface normal at each beam
position (replacing Phase-A's radial-from-pivot approximation). `lens_section_polyline`
builds an area-exact inward-arc ("lens") section: a semicircular segment whose flat
(outer) edge sits on the OML surface and whose arc bulges inward along -normal. The
flat edge approximates the locally-flat OML; the area equals the Phase-C sized area.
"""
from __future__ import annotations

import numpy as np

from ..geometry.wing import WingSpec
from .cross_section import beam_section_points


def oml_outward_normals(spec: WingSpec, z: float, n_beams: int) -> np.ndarray:
    """(n_beams, 2) unit outward normals at the arc-spaced OML beam positions at ``z``."""
    P = beam_section_points(spec, z, n_beams)
    tang = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)   # central difference around the loop
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.column_stack([tang[:, 1], -tang[:, 0]])         # rotate tangent -90°
    radial = P / np.linalg.norm(P, axis=1, keepdims=True)    # outward reference (from pivot)
    sign = np.sign(np.sum(nrm * radial, axis=1))
    sign[sign == 0] = 1.0
    return nrm * sign[:, None]


def lens_section_polyline(
    center: np.ndarray,
    normal: np.ndarray,
    area: float,
    n_arc: int = 24,
) -> np.ndarray:
    """Closed (n_arc+1, 2) inward-arc lens of exactly ``area`` at ``center``.

    Semicircular segment: flat diameter along the surface tangent through ``center``
    (the outer edge, on the OML), arc bulging inward along ``-normal``. Half-disk
    area = π ρ² / 2, so ρ = sqrt(2 area / π). The polygon's closing edge
    (last→first vertex) is the flat outer edge.
    """
    rho = float(np.sqrt(2.0 * area / np.pi))
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    t = np.array([-n[1], n[0]])                              # tangent ⟂ normal
    c = np.asarray(center, dtype=float)
    theta = np.linspace(0.0, np.pi, n_arc + 1)
    # θ=0 → c+ρt ; θ=π → c−ρt ; bulge along −n
    return c + rho * (np.cos(theta)[:, None] * t - np.sin(theta)[:, None] * n)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/beams/test_sections.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/sections.py tests/beams/test_sections.py
git commit -m "feat(beams): OML outward normals + area-exact lens section"
# trailer
```

---

## Task 4: Build sized lens beams (build123d, with circular fallback)

**Files:**
- Modify: `src/wing_design/beams/build.py`
- Test: `tests/beams/test_sized_build.py`

- [ ] **Step 1: Write the failing smoke test** (`tests/beams/test_sized_build.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.build import build_sized_lens_beams, build_sized_circular_beams


def _radii(n_beams, n_levels):
    # one radius per longitudinal segment, beam-major/level-minor
    return np.full(n_beams * (n_levels - 1), 0.02)


def test_sized_circular_beams_build():
    spec = WingSpec()
    beams = build_sized_circular_beams(spec, _radii(8, 5), n_beams=8, n_levels=5)
    assert len(beams) == 8
    for b in beams:
        assert b.volume > 0.0


def test_sized_lens_beams_build():
    spec = WingSpec()
    beams = build_sized_lens_beams(spec, _radii(8, 5), n_beams=8, n_levels=5)
    assert len(beams) == 8
    for b in beams:
        assert b.volume > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_sized_build.py -v`
Expected: FAIL — functions don't exist.

- [ ] **Step 3: Implement** — append to `src/wing_design/beams/build.py`. Add the imports it needs at the top (extend the existing `from build123d import ...`): add `Polyline`, `BuildLine`, `make_face`. Add `from .sections import lens_section_polyline, oml_outward_normals` and `from .cross_section import beam_section_points`.

```python
def _segment_station_radii(long_radii: np.ndarray, b: int, n_levels: int) -> np.ndarray:
    """Per-station radii for beam ``b`` from its (n_levels-1) segment radii.

    Station k uses the radius of segment min(k, n_seg-1), so the loft is a
    piecewise-prismatic stack of the sized segments.
    """
    seg = n_levels - 1
    radii_b = np.asarray(long_radii)[b * seg:(b + 1) * seg]
    return np.array([radii_b[min(k, seg - 1)] for k in range(n_levels)])


def build_sized_circular_beams(
    spec: WingSpec,
    long_radii: np.ndarray,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
) -> list:
    """Loft circular beams whose per-station radius comes from the sized segment radii.

    Circular sections inset inward along the true OML normal — the FEA-faithful
    fallback when lens lofting is not desired/available.
    """
    z_levels = default_z_levels(spec, n_levels)
    beams = []
    for b in range(n_beams):
        st_r = _segment_station_radii(long_radii, b, n_levels)
        with BuildPart() as bp:
            for k in range(n_levels):
                z = float(z_levels[k])
                P = beam_section_points(spec, z, n_beams)[b]
                n_out = oml_outward_normals(spec, z, n_beams)[b]
                c = P - st_r[k] * n_out          # inset inward by the radius
                with BuildSketch(Plane.XY.offset(z)):
                    with Locations((float(c[0]), float(c[1]))):
                        Circle(float(st_r[k]))
            loft(ruled=True)
        beams.append(bp.part)
    return beams


def build_sized_lens_beams(
    spec: WingSpec,
    long_radii: np.ndarray,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    n_arc: int = 24,
) -> list:
    """Loft inward-arc 'lens' beams sized to the Phase-C areas (π r²) per segment.

    Each station's lens has area = π·(station radius)², its flat edge on the OML and
    its arc bulging inward. Raises if build123d cannot loft the custom faces — the
    caller may fall back to `build_sized_circular_beams`.
    """
    z_levels = default_z_levels(spec, n_levels)
    beams = []
    for b in range(n_beams):
        st_r = _segment_station_radii(long_radii, b, n_levels)
        with BuildPart() as bp:
            for k in range(n_levels):
                z = float(z_levels[k])
                P = beam_section_points(spec, z, n_beams)[b]
                n_out = oml_outward_normals(spec, z, n_beams)[b]
                area = float(np.pi * st_r[k] ** 2)
                poly = lens_section_polyline(P, n_out, area, n_arc=n_arc)
                with BuildSketch(Plane.XY.offset(z)):
                    with BuildLine():
                        Polyline([(float(x), float(y)) for x, y in poly], close=True)
                    make_face()
            loft(ruled=True)
        beams.append(bp.part)
    return beams
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/beams/test_sized_build.py -v`
Expected: both PASS. **If `test_sized_lens_beams_build` fails because build123d 0.10.0 cannot `make_face` the polyline or `loft` the custom sections, STOP and report DONE_WITH_CONCERNS** with the exact error. Do not delete the lens builder — the circular builder is the fallback, and the controller will decide whether to (a) tweak the lens face construction (e.g., close the polyline explicitly / use `Spline`) or (b) accept the documented fallback. Try one reasonable fix (e.g., dropping the duplicate closing vertex, or `Polyline(..., close=True)` vs explicit closing point) before reporting.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/build.py tests/beams/test_sized_build.py
git commit -m "feat(beams): build sized lens + circular beams from Phase-C radii"
# trailer
```

---

## Task 5: Best-effort wing fillets

**Files:**
- Modify: `src/wing_design/beams/build.py`
- Test: `tests/beams/test_sized_build.py`

- [ ] **Step 1: Write the failing smoke test** (append to `tests/beams/test_sized_build.py`)

```python
from wing_design.beams.build import apply_wing_fillets, build_wing_solid


def test_apply_wing_fillets_returns_valid_solid():
    spec = WingSpec()
    base = build_wing_solid(spec)
    out = apply_wing_fillets(base, spec)
    # Whatever fillets succeed, the result must be a valid, non-degenerate solid.
    assert out.volume > 0.0
    # Filleting only rounds edges, so volume changes modestly (never collapses/explodes).
    assert 0.5 * base.volume < out.volume < 1.5 * base.volume
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_sized_build.py::test_apply_wing_fillets_returns_valid_solid -v`
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Implement** — append to `src/wing_design/beams/build.py`. Add `fillet` to the build123d import.

```python
def apply_wing_fillets(solid, spec: WingSpec):
    """Best-effort fillets at the spar/transition and root/transition rings.

    build123d 0.10.0 fillet selection on this lofted, morphing solid is fragile;
    each fillet is attempted independently and skipped (with the original solid
    retained) if it raises. The trailing-edge fillet is intentionally NOT attempted
    here — the sharp TE is a continuous-winding feature handled in manufacturing,
    not a solid-edge round. Returns the (possibly partially) filleted solid.

    Targets:
      * spar/transition ring  (z = -transition_length): 30 mm
      * root/transition ring  (z = 0):                   50 mm
    """
    targets = [
        (-spec.transition_length, 0.030),
        (0.0, 0.050),
    ]
    result = solid
    for z_ring, radius in targets:
        try:
            edges = result.edges().filter_by_position(Axis.Z, z_ring - 1e-3, z_ring + 1e-3)
            if edges:
                result = fillet(edges, radius=radius)
        except Exception as exc:  # build123d raises a variety of OCP errors
            print(f"  [fillet] skipped z={z_ring:.3f} r={radius}: {type(exc).__name__}: {exc}")
    return result
```

Add `Axis` to the build123d import line as well.

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/beams/test_sized_build.py -v`
Expected: PASS. The fillets may or may not apply; the test only requires a valid solid. **Report in your summary how many fillets actually succeeded** (watch the `[fillet] skipped ...` prints). If ALL fillets are skipped, that's an honest build123d-limitation finding — note it; the test still passes because the original solid is returned.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/build.py tests/beams/test_sized_build.py
git commit -m "feat(beams): best-effort wing fillets (spar/root transition)"
# trailer
```

---

## Task 6: Export API + deliverable example + plan update

**Files:**
- Modify: `src/wing_design/beams/__init__.py`
- Create: `examples/23_sized_geometry.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Export the new API** — in `src/wing_design/beams/__init__.py` add to the imports and `__all__`:
  - from `.fidelity`: `spline_surface_error`
  - from `.sections`: `lens_section_polyline`, `oml_outward_normals`
  - from `.build`: `apply_wing_fillets`, `build_sized_circular_beams`, `build_sized_lens_beams`

- [ ] **Step 2: Create `examples/23_sized_geometry.py`**

```python
"""Phase-D spike: manufacturable, sized form-beam geometry.

Derives the spar radius, sizes the beams via the Phase-C FEA-in-the-loop loop,
builds inward-arc 'lens' beams sized to those areas (circular fallback if build123d
can't loft the lens faces), fillets the wing solid, assembles everything, and
exports a STEP to exports/. Also prints the spline on-surface fidelity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from build123d import Compound, export_step

from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    SizingConfig,
    apply_wing_fillets,
    build_beam_frame,
    build_sized_circular_beams,
    build_sized_lens_beams,
    project_panels_to_beam_nodes,
    size_beams,
    spline_surface_error,
)
from wing_design.geometry import build_wing_solid


def main() -> None:
    P = default_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    print(f"Derived spar radius = {spec.spar_radius*1e3:.0f} mm "
          f"(diameter {spec.spar_diameter*1e3:.0f} mm)")
    err = spline_surface_error(spec, n_beams=n_beams, n_levels=n_levels)
    print(f"Spline on-surface fidelity at {n_levels} levels: worst {err*1e3:.1f} mm")

    # --- Size the beams (Phase C) ---
    frame = build_beam_frame(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
    )
    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane, P.load_cases, method="lifting_line",
        spanwise_resolution=P.aero.spanwise_resolution,
    )
    load_arrays = [
        project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
        for ar in envelope
        if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0
    ]
    cfg = SizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0,
    )
    print("\nSizing beams (SLSQP, ~1-2 min)...")
    sized = size_beams(frame, load_arrays, cfg, rho=P.rho_kgm3, maxiter=60)
    print(f"  sized mass={sized.mass_kg:.2f} kg, "
          f"radii {sized.radii.min()*1e3:.1f}-{sized.radii.max()*1e3:.1f} mm")

    # --- Build sized geometry (lens, with circular fallback) ---
    try:
        beams = build_sized_lens_beams(spec, sized.radii, n_beams=n_beams, n_levels=n_levels)
        section_kind = "lens"
    except Exception as exc:
        print(f"  lens build failed ({type(exc).__name__}: {exc}); falling back to circular.")
        beams = build_sized_circular_beams(spec, sized.radii, n_beams=n_beams, n_levels=n_levels)
        section_kind = "circular"
    print(f"  built {len(beams)} {section_kind} beams")

    wing = apply_wing_fillets(build_wing_solid(spec), spec)
    wing.label = "wing"
    for i, b in enumerate(beams):
        b.label = f"beam_{i:02d}"
    asm = Compound(label="wingsail_sized", children=[wing, *beams])

    out = Path(__file__).resolve().parent.parent / "exports"
    out.mkdir(exist_ok=True)
    export_step(asm, str(out / "sized_geometry_v0.step"))
    print(f"\nWrote {out}/sized_geometry_v0.step  ({section_kind} sections)")
    print(f"Assembly bbox: {asm.bounding_box()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the example end-to-end**

Run: `uv run python examples/23_sized_geometry.py`
Expected: prints derived spar radius (~80 mm), a fidelity number, the sized mass, builds N beams (lens or circular fallback — report which), writes a STEP, prints a sane bbox. **Paste the full output** and report: which section kind built, how many fillets applied, and whether anything fell back. A fallback or skipped fillets is an acceptable, reportable outcome — not a failure.

- [ ] **Step 4: Update `docs/plan.md`** — in **Phase D — Deepen geometry**, append a status note recording what landed and the build123d outcomes (lens vs circular, fillet success count), and add a Decisions-log row:

```markdown
| Phase-D geometry | Derived spar_radius (max inscribed circle at pivot, floored to cm); inward-arc 'lens' beam sections sized to Phase-C areas with true OML normals (circular fallback if build123d can't loft the faces); best-effort transition fillets; on-surface spline-fidelity metric. |
```

(Fill the status note with the ACTUAL outcome observed in Step 3 — e.g., "lens sections built successfully" or "fell back to circular because …"; "N/2 fillets applied".)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/__init__.py examples/23_sized_geometry.py docs/plan.md
git commit -m "feat(beams): Phase-D sized-geometry example + export API; mark phase done"
# trailer
```

---

## Self-Review

- **Spec coverage:** derived spar radius → Task 1; spline fidelity → Task 2; lens sections + true normals → Task 3; sized lens geometry (beams hit sized areas, by construction of `lens_section_polyline`) → Task 4; fillets → Task 5; integrated deliverable → Task 6. All four chosen threads covered.
- **Placeholder scan:** none — all code concrete.
- **Type consistency:** `beam_section_points`/`default_z_levels`/`form_beam_grid`/`sample_spline` used per their verified signatures; `SizingResult.radii` indexing (beam-major/level-minor) matches `_segment_station_radii`; lens area `π r²` ties Task 4 build to Task 3 generator and Phase-C radii; `build_wing_solid` imported from `wing_design.geometry`.
- **Risk + fallbacks (intentional):** Task 4 lens loft and Task 5 fillets are the build123d-fragile parts; both have fallbacks (circular beams; skip-and-retain) and explicit "report the real outcome, don't fake it" instructions. The deterministic numpy core (Tasks 1–3) is fully unit-tested and unaffected by build123d behavior.
- **Known spike limitations:** the lens is a semicircular segment with a locally-flat outer edge (approximates the curved OML; exact only in the small-beam limit); per-station radius is piecewise-constant from segment radii; fillet edge-selection is position-based and may miss/over-select on the morphing solid (hence best-effort).
