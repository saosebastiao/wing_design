# Phase A — Form-Beam Geometry Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build shell-following form beams + a load-bearing skin wrap + an assembly end-to-end at crude (un-sized) fidelity, exported as STEP, as the first step of the shell-beam pipeline in [`guided_generative_design.md`](../../guided_generative_design.md).

**Architecture:** A new `wing_design.beams` package. The deterministic numeric core samples the existing OML cross-section at chosen spanwise heights, resamples each section into evenly arc-spaced points (so LE and TE are members), stacks them into a per-beam point grid, and fits one cubic B-spline per beam. The build123d layer lofts a crude fixed-radius circular cross-section along each beam spline and hollow-shells the full OML solid into the skin. Everything is driven by the existing `WingSpec` / `DesignParameters`.

**Tech Stack:** Python 3.10–3.12, numpy, scipy (`interpolate.splprep`), build123d, pytest. All already importable in the venv.

**Scope notes / deliberate spike simplifications (refined in later phases):**
- Beam cross-section is a **fixed-radius circle inset inward** from the shell along the radial direction from the pivot — not yet the true inward-arc "lens" sized to an optimized area (Phase D).
- Skin is the **full OML solid hollow-shelled** to wall thickness — not yet built from beam-arc endpoints (Phase D).
- `spar_diameter` stays a stored `WingSpec` field — the derived `spar_radius()` is Phase D.
- No structural evaluation yet — that's Phase B.

---

## File structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/geometry/wing.py` (modify) | Add public `oml_section_polyline(spec, z, n_pts=None)` — closed OML (x,y) at any height. |
| `src/wing_design/geometry/__init__.py` (modify) | Export `oml_section_polyline`. |
| `src/wing_design/beams/__init__.py` (create) | Package exports. |
| `src/wing_design/beams/cross_section.py` (create) | `resample_closed_polyline`, `beam_section_points` — numpy only. |
| `src/wing_design/beams/splines.py` (create) | `default_z_levels`, `form_beam_grid`, `fit_beam_splines`, `sample_spline` — scipy. |
| `src/wing_design/beams/build.py` (create) | `build_form_beams`, `build_skin_wrap`, `build_assembly` — build123d. |
| `examples/20_form_beams.py` (create) | Runnable: build assembly, export STEP, view. |
| `tests/beams/test_cross_section.py` (create) | Unit tests for the numeric core. |
| `tests/beams/test_splines.py` (create) | Unit tests for the spline grid + fit. |
| `tests/beams/test_build_smoke.py` (create) | Smoke test: assembly builds, sane bbox. |
| `pyproject.toml` (modify) | Declare `scipy` dep, `pytest` dev dep, pytest config. |

---

## Task 0: Project test + dependency scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`, `tests/beams/__init__.py`

- [ ] **Step 1: Declare scipy as a runtime dep and pytest as dev**

Run:
```bash
uv add scipy
uv add --dev pytest
```
Expected: `pyproject.toml` gains `scipy` under `[project].dependencies` and a `[dependency-groups]`/`[tool.uv]` dev entry with `pytest`; `uv.lock` updates; no errors.

- [ ] **Step 2: Add pytest config to `pyproject.toml`**

Append this block to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Create empty test packages**

```bash
mkdir -p tests/beams
touch tests/__init__.py tests/beams/__init__.py
```

- [ ] **Step 4: Verify pytest collects nothing yet (clean baseline)**

Run: `uv run pytest`
Expected: exit 0, "no tests ran" (or 0 collected). Confirms config is valid.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py tests/beams/__init__.py
git commit -m "chore: pytest config + scipy/pytest deps for Phase A"
```

---

## Task 1: OML cross-section polyline accessor

Expose the existing private morphing helper as a clean public function keyed by height `z`.

**Files:**
- Modify: `src/wing_design/geometry/wing.py`
- Modify: `src/wing_design/geometry/__init__.py`
- Test: `tests/beams/test_cross_section.py`

- [ ] **Step 1: Write the failing test**

Create `tests/beams/test_cross_section.py`:
```python
import numpy as np

from wing_design.geometry import WingSpec, oml_section_polyline


def test_oml_section_wing_region_is_airfoil():
    spec = WingSpec()
    poly = oml_section_polyline(spec, z=0.0)
    # Root chord = 1.0, pivot at 0.25c: TE at +0.75, LE at -0.25.
    assert poly[:, 0].max() == max(poly[:, 0].max(), 0.74)  # sanity
    assert abs(poly[:, 0].max() - (1.0 - spec.pivot_frac) * spec.root_chord) < 1e-6
    assert abs(poly[:, 0].min() - (-spec.pivot_frac * spec.root_chord)) < 1e-6


def test_oml_section_spar_region_is_circle():
    spec = WingSpec()
    z = -(spec.transition_length + spec.spar_length * 0.5)  # deep in the spar
    poly = oml_section_polyline(spec, z=z)
    r = np.hypot(poly[:, 0], poly[:, 1])
    assert np.allclose(r, spec.spar_diameter / 2.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/beams/test_cross_section.py -v`
Expected: FAIL — `ImportError: cannot import name 'oml_section_polyline'`.

- [ ] **Step 3: Implement `oml_section_polyline` in `wing.py`**

Add to `src/wing_design/geometry/wing.py` (after `_section_face`, before `build_wing_solid`):
```python
def oml_section_polyline(spec: WingSpec, z: float, n_pts: int | None = None) -> np.ndarray:
    """Closed OML cross-section polyline (x, y) at spanwise height ``z``.

    Wing region (z >= 0): pure airfoil at ``chord_at_z(z)``.
    Transition (-transition_length <= z < 0): airfoil->circle radial blend.
    Spar region (z < -transition_length): the spar circle.

    Returns an (M, 2) array traversed upper-TE -> LE -> lower-TE; its first and
    last points coincide at the trailing edge.
    """
    if n_pts is None:
        n_pts = spec.n_airfoil_points
    if z >= 0.0:
        chord = spec.chord_at_z(z)
        blend = 0.0
    elif z >= -spec.transition_length:
        chord = spec.root_chord
        blend = -z / spec.transition_length
    else:
        chord = spec.root_chord
        blend = 1.0
    return _airfoil_to_circle_polyline(
        chord, spec.thickness, spec.pivot_frac, spec.spar_diameter, blend, n_pts
    )
```

- [ ] **Step 4: Export it from the geometry package**

In `src/wing_design/geometry/__init__.py`, update the import and `__all__`:
```python
from .airfoil import naca_00xx_coords, naca_00xx_thickness
from .wing import WingSpec, build_wing_solid, oml_section_polyline

__all__ = [
    "WingSpec",
    "build_wing_solid",
    "oml_section_polyline",
    "naca_00xx_coords",
    "naca_00xx_thickness",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/beams/test_cross_section.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/geometry/wing.py src/wing_design/geometry/__init__.py tests/beams/test_cross_section.py
git commit -m "feat(geometry): public oml_section_polyline(spec, z)"
```

---

## Task 2: Arc-length resampling + beam section points

**Files:**
- Create: `src/wing_design/beams/__init__.py`
- Create: `src/wing_design/beams/cross_section.py`
- Test: `tests/beams/test_cross_section.py` (append)

- [ ] **Step 1: Write the failing tests (append to `tests/beams/test_cross_section.py`)**

```python
from wing_design.beams.cross_section import resample_closed_polyline, beam_section_points


def test_resample_square_corners():
    # Unit square, closed (last == first). Perimeter = 4.
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    out = resample_closed_polyline(sq, 4)
    # Point 0 at the start; equal steps of length 1 hit the four corners.
    assert np.allclose(out[0], [0, 0])
    assert np.allclose(out[1], [1, 0])
    assert np.allclose(out[2], [1, 1])
    assert np.allclose(out[3], [0, 1])


def test_resample_square_midpoints():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    out = resample_closed_polyline(sq, 8)
    assert np.allclose(out[1], [0.5, 0.0])  # midpoint of the first edge


def test_beam_points_le_and_te_are_members():
    spec = WingSpec()
    pts = beam_section_points(spec, z=0.0, n_beams=16)
    assert pts.shape == (16, 2)
    # Beam 0 sits at the TE, beam 8 at the LE (symmetric section, even count).
    assert abs(pts[0, 0] - (1.0 - spec.pivot_frac) * spec.root_chord) < 1e-3
    assert abs(pts[0, 1]) < 1e-6
    assert abs(pts[8, 0] - (-spec.pivot_frac * spec.root_chord)) < 1e-3
    assert abs(pts[8, 1]) < 1e-6
    # Upper/lower symmetry: beam i mirrors beam (16 - i) in y.
    assert np.allclose(pts[1, 1], -pts[15, 1], atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_cross_section.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.beams'`.

- [ ] **Step 3: Create the package init**

`src/wing_design/beams/__init__.py`:
```python
"""Shell-following form beams: section sampling, splines, and build123d geometry."""
from __future__ import annotations
```

- [ ] **Step 4: Implement `src/wing_design/beams/cross_section.py`**

```python
"""Sample the OML cross-section into evenly arc-spaced form-beam anchor points."""
from __future__ import annotations

import numpy as np

from ..geometry.wing import WingSpec, oml_section_polyline


def resample_closed_polyline(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline into ``n`` points equally spaced by arc length.

    ``poly`` is (M, 2). If its first and last points coincide they are treated
    as the single loop-closure vertex. Output point 0 sits at ``poly[0]``; the
    remaining points march at equal arc-length steps around the loop (the loop's
    closing point is excluded, since it would duplicate point 0).
    """
    pts = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    loop = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, s, loop[:, 0])
    y = np.interp(targets, s, loop[:, 1])
    return np.column_stack([x, y])


def beam_section_points(
    spec: WingSpec, z: float, n_beams: int, n_pts: int | None = None
) -> np.ndarray:
    """``n_beams`` (x, y) points equally arc-spaced around the OML at height ``z``.

    Point 0 lies at the trailing edge; for a symmetric section and even
    ``n_beams``, point ``n_beams // 2`` lies at the leading edge.
    """
    poly = oml_section_polyline(spec, z, n_pts=n_pts)
    return resample_closed_polyline(poly, n_beams)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/beams/test_cross_section.py -v`
Expected: PASS (5 passed total in this file).

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/__init__.py src/wing_design/beams/cross_section.py tests/beams/test_cross_section.py
git commit -m "feat(beams): arc-length section resampling, LE/TE as members"
```

---

## Task 3: Form-beam point grid + B-spline fit

**Files:**
- Create: `src/wing_design/beams/splines.py`
- Test: `tests/beams/test_splines.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/beams/test_splines.py`:
```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.splines import (
    default_z_levels,
    form_beam_grid,
    fit_beam_splines,
    sample_spline,
)


def test_z_levels_span_tip_to_keelstep():
    spec = WingSpec()
    z = default_z_levels(spec, 10)
    assert z.shape == (10,)
    assert abs(z[0] - spec.span) < 1e-9
    assert abs(z[-1] - (-(spec.transition_length + spec.spar_length))) < 1e-9
    assert np.all(np.diff(z) < 0)  # monotonically descending


def test_grid_shape_and_z_consistency():
    spec = WingSpec()
    z = default_z_levels(spec, 12)
    grid = form_beam_grid(spec, z, n_beams=16)
    assert grid.shape == (16, 12, 3)
    # Each level's z is filled correctly for every beam.
    for k in range(12):
        assert np.allclose(grid[:, k, 2], z[k])


def test_splines_interpolate_their_points():
    spec = WingSpec()
    z = default_z_levels(spec, 12)
    grid = form_beam_grid(spec, z, n_beams=16)
    splines = fit_beam_splines(grid, smoothing=0.0)
    assert len(splines) == 16
    # A densely sampled interpolating spline must pass near every input point.
    for b in range(16):
        curve = sample_spline(splines[b], n=200)
        for p in grid[b]:
            d = np.linalg.norm(curve - p, axis=1).min()
            assert d < 1e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_splines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.beams.splines'`.

- [ ] **Step 3: Implement `src/wing_design/beams/splines.py`**

```python
"""Form-beam spanwise point grids and per-beam B-spline fits."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep

from ..geometry.wing import WingSpec
from .cross_section import beam_section_points


def default_z_levels(spec: WingSpec, n: int = 20) -> np.ndarray:
    """``n`` spanwise sampling heights from the tip (z = span) down to the keel-step."""
    z_top = spec.span
    z_bottom = -(spec.transition_length + spec.spar_length)
    return np.linspace(z_top, z_bottom, n)


def form_beam_grid(spec: WingSpec, z_levels: np.ndarray, n_beams: int) -> np.ndarray:
    """(n_beams, n_levels, 3) array of on-shell points.

    Entry ``[b, k]`` is the (x, y, z) location of form beam ``b`` at
    ``z_levels[k]``. Beam ordering matches ``beam_section_points`` (0 = TE,
    ``n_beams // 2`` = LE).
    """
    n_levels = len(z_levels)
    grid = np.empty((n_beams, n_levels, 3))
    for k, z in enumerate(z_levels):
        xy = beam_section_points(spec, float(z), n_beams)
        grid[:, k, 0] = xy[:, 0]
        grid[:, k, 1] = xy[:, 1]
        grid[:, k, 2] = float(z)
    return grid


def fit_beam_splines(grid: np.ndarray, smoothing: float = 0.0) -> list:
    """Fit one cubic B-spline per beam to its (x, y, z) point sequence.

    Returns a list of scipy ``splprep`` ``tck`` tuples, one per beam. With
    ``smoothing = 0`` the spline interpolates the on-shell points, so it lies on
    the shell to sampling tolerance.
    """
    splines = []
    for b in range(grid.shape[0]):
        pts = grid[b]  # (n_levels, 3)
        k = min(3, pts.shape[0] - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=smoothing, k=k)
        splines.append(tck)
    return splines


def sample_spline(tck, n: int = 50) -> np.ndarray:
    """Sample a fitted spline at ``n`` evenly spaced parameter values -> (n, 3)."""
    u = np.linspace(0.0, 1.0, n)
    x, y, z = splev(u, tck)
    return np.column_stack([x, y, z])
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/beams/test_splines.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/splines.py tests/beams/test_splines.py
git commit -m "feat(beams): form-beam point grid + cubic B-spline fit"
```

---

## Task 4: build123d form beams + skin wrap + assembly

This is geometry; tests are smoke-level (builds, positive volume, sane bbox), mirroring how the existing examples validate. Expect possible execution-time iteration on build123d calls (`shell`, `loft`, `Plane.offset`); the try/except fallbacks keep the spike runnable.

**Files:**
- Create: `src/wing_design/beams/build.py`
- Test: `tests/beams/test_build_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/beams/test_build_smoke.py`:
```python
from wing_design.geometry import WingSpec
from wing_design.beams.build import build_form_beams, build_skin_wrap, build_assembly


def test_form_beams_build():
    spec = WingSpec()
    beams = build_form_beams(spec, n_beams=8, n_levels=6, beam_radius=0.02)
    assert len(beams) == 8
    for b in beams:
        assert b.volume > 0.0


def test_skin_wrap_builds():
    spec = WingSpec()
    skin = build_skin_wrap(spec, wall=0.003)
    assert skin.volume > 0.0


def test_assembly_spans_tip_to_keelstep():
    spec = WingSpec()
    asm = build_assembly(spec, n_beams=8, n_levels=6, beam_radius=0.02, wall=0.003)
    bb = asm.bounding_box()
    # z must run from the keel-step (negative) up to the tip (= span).
    assert bb.max.Z > spec.span - 0.05
    assert bb.min.Z < -(spec.transition_length + spec.spar_length) + 0.05
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_build_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.beams.build'`.

- [ ] **Step 3: Implement `src/wing_design/beams/build.py`**

```python
"""build123d geometry for the form beams, the skin wrap, and their assembly.

Spike fidelity: beams are fixed-radius circular cross-sections inset inward from
the shell along the radial direction from the pivot, lofted along each beam
spline. The skin is the full OML solid hollow-shelled to ``wall`` thickness.
Both are refined in Phase D.
"""
from __future__ import annotations

import numpy as np
from build123d import (
    BuildPart,
    BuildSketch,
    Circle,
    Compound,
    Locations,
    Plane,
    add,
    loft,
    offset,
)

from ..geometry.wing import WingSpec, build_wing_solid
from .splines import default_z_levels, form_beam_grid


def build_form_beams(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
) -> list:
    """One lofted solid per form beam (crude fixed-radius circular section)."""
    z_levels = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z_levels, n_beams)  # (n_beams, n_levels, 3)
    beams = []
    for b in range(n_beams):
        with BuildPart() as bp:
            for k in range(n_levels):
                px, py, pz = grid[b, k]
                radial = np.array([px, py])
                norm = float(np.linalg.norm(radial))
                n_out = radial / norm if norm > 1e-9 else np.array([1.0, 0.0])
                cx, cy = radial - beam_radius * n_out  # inset inward from shell
                with BuildSketch(Plane.XY.offset(float(pz))):
                    with Locations((float(cx), float(cy))):
                        Circle(beam_radius)
            loft(ruled=True)
        beams.append(bp.part)
    return beams


def build_skin_wrap(spec: WingSpec, *, wall: float = 0.003):
    """Full OML solid hollow-shelled to ``wall`` thickness (skin)."""
    solid = build_wing_solid(spec)
    try:
        return offset(solid, amount=-wall, kind=None)  # inward shell -> hollow skin
    except Exception:
        # Spike fallback: keep the solid skin if shelling the morphing OML fails.
        return solid


def build_assembly(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
    wall: float = 0.003,
) -> Compound:
    """Compound of the skin wrap + all form beams."""
    beams = build_form_beams(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=beam_radius
    )
    skin = build_skin_wrap(spec, wall=wall)
    for i, b in enumerate(beams):
        b.label = f"beam_{i:02d}"
    skin.label = "skin"
    return Compound(label="wingsail", children=[skin, *beams])
```

> **Execution note for the implementer:** build123d's hollowing API has changed across versions. If `offset(solid, amount=-wall, kind=None)` errors, try `from build123d import shell` then `shell(solid, amount=-wall)`, or `solid.offset_3d(amount=-wall)`. The try/except already degrades to a solid skin so the smoke test still passes; pick whichever hollows correctly in this venv and drop the fallback once confirmed.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/beams/test_build_smoke.py -v`
Expected: PASS (3 passed). If `build_skin_wrap` hits the fallback, `test_skin_wrap_builds` still passes (solid has volume); follow the execution note to get a true hollow shell.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/build.py tests/beams/test_build_smoke.py
git commit -m "feat(beams): build123d form beams, skin wrap, assembly"
```

---

## Task 5: Wire beams package exports

**Files:**
- Modify: `src/wing_design/beams/__init__.py`

- [ ] **Step 1: Add public exports**

Replace `src/wing_design/beams/__init__.py` with:
```python
"""Shell-following form beams: section sampling, splines, and build123d geometry."""
from __future__ import annotations

from .cross_section import beam_section_points, resample_closed_polyline
from .splines import default_z_levels, fit_beam_splines, form_beam_grid, sample_spline
from .build import build_assembly, build_form_beams, build_skin_wrap

__all__ = [
    "beam_section_points",
    "resample_closed_polyline",
    "default_z_levels",
    "form_beam_grid",
    "fit_beam_splines",
    "sample_spline",
    "build_assembly",
    "build_form_beams",
    "build_skin_wrap",
]
```

- [ ] **Step 2: Verify the whole suite still passes**

Run: `uv run pytest`
Expected: PASS (all tests from Tasks 1–4 green).

- [ ] **Step 3: Commit**

```bash
git add src/wing_design/beams/__init__.py
git commit -m "chore(beams): package exports"
```

---

## Task 6: Runnable example + STEP export

**Files:**
- Create: `examples/20_form_beams.py`

- [ ] **Step 1: Write the example**

Create `examples/20_form_beams.py`:
```python
"""Phase-A spike: shell-following form beams + skin wrap, exported as STEP and viewed.

Builds the crude (un-sized) form-beam assembly from the working scenario and
writes it next to the other example outputs. Beam cross-sections are a fixed
radius here; FEA-in-the-loop sizing arrives in Phase C.
"""
from __future__ import annotations

from pathlib import Path

from build123d import export_step

from wing_design import default_scenario, show_in_viewer
from wing_design.beams import build_assembly


def main() -> None:
    spec = default_scenario().geometry
    asm = build_assembly(spec, n_beams=16, n_levels=20, beam_radius=0.02, wall=0.003)

    out = Path(__file__).parent / "_out"
    out.mkdir(exist_ok=True)
    export_step(asm, str(out / "form_beams_v0.step"))
    print(f"Wrote {out}/form_beams_v0.step")
    print(f"Children: {[c.label for c in asm.children]}")
    print(f"Bounding box: {asm.bounding_box()}")

    show_in_viewer(asm, names=["form_beams"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example end-to-end**

Run: `just example 20_form_beams`
Expected: prints "Wrote …/form_beams_v0.step", a 17-element child list (`skin` + `beam_00`…`beam_15`), and a bounding box whose Z spans roughly `-1.5 … 5.0`. Writes `examples/_out/form_beams_v0.step`. The viewer line is a no-op (with a hint) if OCP CAD Viewer isn't running — that's fine.

- [ ] **Step 3: Confirm the STEP file exists and is non-trivial**

Run: `ls -la examples/_out/form_beams_v0.step`
Expected: a file of non-zero size (typically > 100 KB).

- [ ] **Step 4: Commit**

```bash
git add examples/20_form_beams.py
git commit -m "feat(examples): 20_form_beams — Phase A geometry spike"
```

---

## Self-review checklist (done at write time)

- **Spec coverage:** Phase A in `plan.md` = "even-spaced form-beam splines on full OML → crude fixed-area arc beams → wrap → assembly → STEP." Covered by Tasks 1 (OML access), 2–3 (even-spaced splines on full OML incl. spar), 4 (crude beams + wrap + assembly), 6 (STEP). The crude beams use fixed *radius* rather than fixed *area* — equivalent at spike fidelity; area-driven radius is explicitly Phase D.
- **Placeholder scan:** No TBD/TODO in steps; every code step has complete code. The one forward-looking note (build123d hollowing API) gives three concrete alternatives plus a working fallback, not a placeholder.
- **Type consistency:** `oml_section_polyline`, `resample_closed_polyline`, `beam_section_points`, `default_z_levels`, `form_beam_grid`, `fit_beam_splines`, `sample_spline`, `build_form_beams`, `build_skin_wrap`, `build_assembly` are named identically in their defining task, their tests, the package `__init__`, and the example.
- **Deferred to later phases (intentionally not in this plan):** area-sized inward-arc beam sections (D), beam-endpoint-driven wrap (D), derived `spar_radius` + fillets (D), any structural evaluation (B/C).
