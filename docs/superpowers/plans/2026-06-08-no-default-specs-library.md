# No-default Specs + Named Wingsail/Scenario Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `WingSpec` and the six `scenario.py` parameter dataclasses pure no-default schemas (like `UDPly`), add named `small/medium/large` wingsails + scenarios, migrate tests to small and examples to medium, and give every example a purpose docstring.

**Architecture:** Additive-first sequencing. Tasks 1–2 ADD the new named instances/builders without removing any defaults (suite stays green). Tasks 3–5 migrate all call sites (tests → small, examples → medium) while the old defaults still exist. Task 6 finally removes the field defaults and `default_scenario()` — by then nothing depends on them. Each task ends green.

**Tech Stack:** Python dataclasses, pytest, `uv run pytest`. Mirrors the existing `UDPly`(no-default class)/`T700_EPOXY`(named instance) pattern in `materials/unidir.py`.

**Domain notes for the implementer (read once):**
- Existing pattern to mirror: `materials/unidir.py` — `UDPly` is `@dataclass(frozen=True)` with NO field defaults; concrete values live in module-level instances `T700_EPOXY`, `T800_EPOXY`.
- `WingSpec` is in `geometry/wing.py`; the scenario param dataclasses + `default_scenario()` are in `scenario.py`; the package re-exports `default_scenario` from `wing_design/__init__.py`.
- Only TWO `src/` sites use a `WingSpec()` default arg: `geometry/wing.py:218` (`build_wing_solid(spec: WingSpec = WingSpec())`) and `structural/mesh.py:49` (`spec: WingSpec = WingSpec()`). Everything else passes `spec` explicitly.
- Commit trailer (every commit): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Full suite is currently green (~6 min with the CLT sizing tests). Keep tests on the small wingsail so runtime is unchanged.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/wing_design/geometry/wingsails.py` | `small/medium/large_wingsail` instances | **new** (Task 1) |
| `src/wing_design/geometry/__init__.py` | export the three wingsails | modify (Task 1) |
| `src/wing_design/scenario.py` | `STANDARD_*` instances + 3 scenario builders; later drop defaults + `default_scenario` | modify (Tasks 2, 6) |
| `src/wing_design/__init__.py` | re-export scenario builders; later drop `default_scenario` | modify (Tasks 2, 6) |
| `src/wing_design/geometry/wing.py` | `WingSpec` no defaults; `build_wing_solid` spec required | modify (Task 6) |
| `src/wing_design/structural/mesh.py` | drop `WingSpec()` default arg | modify (Task 6) |
| `tests/**` (~16 files) | `WingSpec()`→`small_wingsail`, `default_scenario()`→`small_scenario()` | modify (Task 3) |
| `examples/00..20` | migrate to medium + purpose docstrings | modify (Task 4) |
| `examples/21..34` | migrate to medium + purpose docstrings | modify (Task 5) |
| `tests/geometry/test_wingsails.py`, `tests/test_scenarios.py` | new tests | **new** (Tasks 1, 2) |

---

## Task 1: Named wingsails (additive)

**Files:**
- Create: `src/wing_design/geometry/wingsails.py`
- Modify: `src/wing_design/geometry/__init__.py`
- Test: `tests/geometry/test_wingsails.py`

- [ ] **Step 1: Write the failing test** — create `tests/geometry/test_wingsails.py`:

```python
import numpy as np
import pytest

from wing_design.geometry import small_wingsail, medium_wingsail, large_wingsail
from wing_design.geometry.wing import WingSpec


def _ar(w):
    return w.span / ((w.root_chord + w.tip_chord) / 2.0)


def test_spans():
    assert small_wingsail.span == 5.0
    assert medium_wingsail.span == 22.0
    assert large_wingsail.span == 45.0


def test_similar_aspect_ratio():
    ars = [_ar(small_wingsail), _ar(medium_wingsail), _ar(large_wingsail)]
    assert max(ars) - min(ars) <= 0.05 * max(ars)  # within 5%


def test_dimensionless_fields_shared():
    for w in (medium_wingsail, large_wingsail):
        assert w.thickness == small_wingsail.thickness
        assert w.pivot_frac == small_wingsail.pivot_frac
        assert w.n_sections == small_wingsail.n_sections
        assert w.n_airfoil_points == small_wingsail.n_airfoil_points
        assert w.taper_profile == small_wingsail.taper_profile


def test_small_matches_legacy_default():
    # small_wingsail reproduces the historical WingSpec() default geometry
    assert (small_wingsail.span, small_wingsail.root_chord, small_wingsail.tip_chord) == (5.0, 1.0, 0.6)
    assert small_wingsail.spar_length == 0.75
    assert small_wingsail.transition_length == 0.20


def test_are_wingspec_instances():
    for w in (small_wingsail, medium_wingsail, large_wingsail):
        assert isinstance(w, WingSpec)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/geometry/test_wingsails.py -v`
Expected: FAIL with `ImportError: cannot import name 'small_wingsail'`

- [ ] **Step 3: Implement** — create `src/wing_design/geometry/wingsails.py`:

```python
"""Named, fully-instantiated wingsail geometries (a size family).

`WingSpec` is a pure schema (no field defaults); concrete designs live here as
module-level instances, mirroring how `materials.unidir` defines `UDPly` plus named
plies (`T700_EPOXY`). Three sizes share the same airfoil/taper shape (identical
dimensionless fields) and a similar aspect ratio (~6.1-6.25 = span / mean-chord); the
lengths scale with span.

- `small_wingsail`  - 5 m demo (the historical default); used by the test suite.
- `medium_wingsail` - 22 m wingsail; used by the examples.
- `large_wingsail`  - 45 m wingsail.
"""
from __future__ import annotations

from .wing import WingSpec

small_wingsail = WingSpec(
    span=5.0, root_chord=1.0, tip_chord=0.6, thickness=0.18, pivot_frac=0.25,
    spar_length=0.75, transition_length=0.20,
    n_transition_sections=4, n_sections=5, n_airfoil_points=160, taper_profile=None,
)

medium_wingsail = WingSpec(
    span=22.0, root_chord=4.5, tip_chord=2.7, thickness=0.18, pivot_frac=0.25,
    spar_length=3.3, transition_length=0.9,
    n_transition_sections=4, n_sections=5, n_airfoil_points=160, taper_profile=None,
)

large_wingsail = WingSpec(
    span=45.0, root_chord=9.0, tip_chord=5.4, thickness=0.18, pivot_frac=0.25,
    spar_length=6.75, transition_length=1.8,
    n_transition_sections=4, n_sections=5, n_airfoil_points=160, taper_profile=None,
)
```

NOTE: at this point `WingSpec` still HAS defaults (removed in Task 6), so passing every
field explicitly is fine and forward-compatible.

- [ ] **Step 4: Export from `src/wing_design/geometry/__init__.py`**

Add `small_wingsail, medium_wingsail, large_wingsail` to the imports and `__all__`. The
file currently has `from .wing import WingSpec, build_wing_solid, oml_section_polyline`
and an `__all__`. Add:
```python
from .wingsails import small_wingsail, medium_wingsail, large_wingsail
```
and append the three names to `__all__`.

- [ ] **Step 5: Run to verify pass + full suite**

Run: `uv run pytest tests/geometry/test_wingsails.py -v`  → PASS (5 tests)
Run: `uv run pytest -q`  → still green (purely additive).

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/geometry/wingsails.py src/wing_design/geometry/__init__.py tests/geometry/test_wingsails.py
git commit -m "$(cat <<'EOF'
Specs library: add small/medium/large_wingsail instances

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Scenario builders + shared param instances (additive)

**Files:**
- Modify: `src/wing_design/scenario.py`
- Modify: `src/wing_design/__init__.py`
- Test: `tests/test_scenarios.py`

**Context:** Add `STANDARD_*` named param instances and `small/medium/large_scenario()`
builders WITHOUT removing any defaults yet (so `default_scenario()` and existing call
sites keep working). The builders construct `DesignParameters` explicitly, which is
forward-compatible with Task 6's default removal.

- [ ] **Step 1: Write the failing test** — create `tests/test_scenarios.py`:

```python
import math

from wing_design.scenario import small_scenario, medium_scenario, large_scenario
from wing_design import DesignParameters


def test_builders_return_design_parameters():
    for f in (small_scenario, medium_scenario, large_scenario):
        assert isinstance(f(), DesignParameters)


def test_scenario_spans():
    assert small_scenario().geometry.span == 5.0
    assert medium_scenario().geometry.span == 22.0
    assert large_scenario().geometry.span == 45.0


def test_derived_properties_resolve():
    P = medium_scenario()
    for v in (P.E_iso_Pa, P.nu_iso, P.G_iso_Pa, P.sigma_allow_Pa, P.rho_kgm3):
        assert math.isfinite(v) and v > 0.0


def test_shared_params_equal_across_sizes():
    a, b = small_scenario(), large_scenario()
    assert a.material_iso == b.material_iso
    assert a.aero == b.aero
    assert a.skin_sizing == b.skin_sizing
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_scenarios.py -v`
Expected: FAIL with `ImportError: cannot import name 'small_scenario'`

- [ ] **Step 3: Implement in `src/wing_design/scenario.py`**

Add the wingsails import near the top (with the other imports):
```python
from .geometry.wingsails import small_wingsail, medium_wingsail, large_wingsail
```
Then, AFTER the `DesignParameters` class definition (and after `default_scenario`, or
replacing nothing yet), add the shared instances and builders:
```python
# ---------------------------------------------------------------------------
# Named shared parameter sets (resolution / material knobs, size-independent)
# ---------------------------------------------------------------------------

STANDARD_MATERIAL_ISO = MaterialParameters(
    skin_E_knockdown=0.5, nu_isotropic=0.32, sigma_allow_safety_factor=2.0)
STANDARD_MESH = MeshParameters(target_element_size_m=0.05)
STANDARD_AERO = AeroParameters(spanwise_resolution=16)
STANDARD_FRAME_FIELD = Phase5FrameFieldParameters(
    n_levels=24, sigma_floor_fraction=0.05, sigma_augment_fraction=1.0)
STANDARD_SKIN = SkinParameters(t_baseline_m=0.003)


def _scenario(geometry: WingSpec) -> DesignParameters:
    return DesignParameters(
        geometry=geometry,
        material=T700_EPOXY,
        material_iso=STANDARD_MATERIAL_ISO,
        load_cases=DESIGN_CASES,
        mesh=STANDARD_MESH,
        aero=STANDARD_AERO,
        frame_field=STANDARD_FRAME_FIELD,
        skin_sizing=STANDARD_SKIN,
    )


def small_scenario() -> DesignParameters:
    """Project working scenario at the 5 m demo wingsail (used by the test suite)."""
    return _scenario(small_wingsail)


def medium_scenario() -> DesignParameters:
    """22 m wingsail scenario (used by the examples)."""
    return _scenario(medium_wingsail)


def large_scenario() -> DesignParameters:
    """45 m wingsail scenario."""
    return _scenario(large_wingsail)
```
(Leave the existing `default_scenario()` in place for now — Task 6 removes it.)

- [ ] **Step 4: Re-export builders from `src/wing_design/__init__.py`**

Extend the import + `__all__` to add the three builders (KEEP `default_scenario` for
now):
```python
from .scenario import (
    DesignParameters, default_scenario,
    small_scenario, medium_scenario, large_scenario,
)
```
and add `"small_scenario", "medium_scenario", "large_scenario"` to `__all__`.

- [ ] **Step 5: Run to verify pass + full suite**

Run: `uv run pytest tests/test_scenarios.py -v`  → PASS (4 tests)
Run: `uv run pytest -q`  → still green.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/scenario.py src/wing_design/__init__.py tests/test_scenarios.py
git commit -m "$(cat <<'EOF'
Specs library: add small/medium/large scenario builders + STANDARD_* params

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Migrate tests to small (mechanical)

**Files:**
- Modify: all test files using `WingSpec()` or `default_scenario()` (~16 files; see list
  below).

**Context:** Replace every bare `WingSpec()` with `small_wingsail` and every
`default_scenario()` with `small_scenario()`, adding the imports. The named instances
already exist (Tasks 1–2) and equal the old defaults, so behavior and runtime are
unchanged. This must leave the suite green.

Affected test files (from `grep -rln "WingSpec()\|default_scenario" tests/`):
`tests/beams/test_sized_build.py, test_tip_coupling.py, test_banded_skin.py,
test_splines.py, test_sizing.py, test_sections.py, test_build_smoke.py,
test_shell_model.py, test_shell_sizing.py, test_laminate_sizing.py, test_fea_model.py,
test_cross_section.py, test_tsai_wu_sizing.py, test_fidelity.py,
tests/geometry/test_transition.py, test_spar_radius.py` (re-run the grep to be sure).

- [ ] **Step 1: Find every site**

Run: `grep -rn "WingSpec()\|default_scenario(" tests/`
Record the list.

- [ ] **Step 2: Edit each file**

In each file:
- If it uses `WingSpec()`: ensure an import `from wing_design.geometry import small_wingsail`
  (or extend an existing `from wing_design.geometry import ...`), then replace each
  `WingSpec()` with `small_wingsail`. If the file imports `WingSpec` only to call
  `WingSpec()`, you may keep or drop the `WingSpec` import as needed (drop if now unused).
- If it uses `default_scenario()`: replace the import `default_scenario` →
  `small_scenario` (from `wing_design` or `wing_design.scenario`, matching the existing
  import path in that file) and each call `default_scenario()` → `small_scenario()`.
- Watch for any direct `DesignParameters(...)` construction in tests — if a test builds
  one positionally/partially relying on defaults, switch it to `small_scenario()` (or
  `dataclasses.replace(small_scenario(), ...)`); `grep -rn "DesignParameters(" tests/`.

Do NOT change any numeric expectations — `small_wingsail` is the old default, so values
are identical.

- [ ] **Step 3: Verify no stragglers + full suite green**

Run: `grep -rn "WingSpec()\|default_scenario(" tests/`  → expect NO matches.
Run: `uv run pytest -q`  → all green (same count, same runtime).

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
Specs library: migrate tests to small_wingsail / small_scenario

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Migrate + docstring examples 00-20

**Files:**
- Modify: `examples/00_design_parameters.py, 01_wing_solid.py, 02_aero_envelope.py,
  03_spar_sizing.py, 04_volumetric_fea.py, 05_stress_lines.py, 06_frame_field.py,
  14_shell_wing.py, 15_shell_stress_lines.py, 20_form_beams.py`

**Context:** Point these examples at the MEDIUM wingsail and ensure each has a purpose
docstring. Examples are not run by pytest; verify by byte-compiling and grepping.

- [ ] **Step 1: For EACH file, migrate the scenario/geometry**
- Replace `default_scenario()` → `medium_scenario()` (and fix the import:
  `from wing_design import medium_scenario` or `from wing_design.scenario import medium_scenario`,
  matching the file's existing import path).
- Replace any bare `WingSpec()` → `medium_wingsail` (import from `wing_design.geometry`).
- Replace any direct `build_wing_solid()` (no arg) → `build_wing_solid(medium_wingsail)`.
- `grep -rn "default_scenario(\|WingSpec()" examples/0*.py examples/14*.py examples/15*.py examples/20*.py` to find them.

- [ ] **Step 2: For EACH file, ensure a purpose docstring**

Each example must open with a module docstring whose FIRST line states what it
demonstrates and why (one sentence), optionally followed by 1-3 lines of detail. Read
the example's code to write an accurate purpose. Examples that already have a clear
purpose docstring (check each) need only minor tightening, not a rewrite. Do NOT change
the example's logic. Example shape:
```python
"""Build the wing OML solid from a WingSpec and export it to STEP.

Demonstrates geometry.build_wing_solid on the medium (22 m) wingsail: lofted airfoil
sections + faired transition + cylindrical rotation spar.
"""
```

- [ ] **Step 3: Verify each file imports/parses**

Run: `for f in examples/00_design_parameters.py examples/01_wing_solid.py examples/02_aero_envelope.py examples/03_spar_sizing.py examples/04_volumetric_fea.py examples/05_stress_lines.py examples/06_frame_field.py examples/14_shell_wing.py examples/15_shell_stress_lines.py examples/20_form_beams.py; do uv run python -m py_compile "$f" && echo "ok $f"; done`
Expected: `ok` for every file (compiles).
Run: `grep -rn "default_scenario(" examples/00_design_parameters.py examples/01_wing_solid.py examples/02_aero_envelope.py examples/03_spar_sizing.py examples/04_volumetric_fea.py examples/05_stress_lines.py examples/06_frame_field.py examples/14_shell_wing.py examples/15_shell_stress_lines.py examples/20_form_beams.py`
Expected: NO matches.

(Do NOT run the examples end-to-end — they are slow. Compile + grep is the gate.)

- [ ] **Step 4: Commit**

```bash
git add examples/00_design_parameters.py examples/01_wing_solid.py examples/02_aero_envelope.py examples/03_spar_sizing.py examples/04_volumetric_fea.py examples/05_stress_lines.py examples/06_frame_field.py examples/14_shell_wing.py examples/15_shell_stress_lines.py examples/20_form_beams.py
git commit -m "$(cat <<'EOF'
Specs library: migrate examples 00-20 to medium + purpose docstrings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Migrate + docstring examples 21-34

**Files:**
- Modify: `examples/21_beam_fea.py, 22_sized_beams.py, 23_sized_geometry.py,
  24_skin_coupling.py, 25_resize_with_skin.py, 26_clt_skin.py, 27_buckling.py,
  28_ply_datum.py, 29_stock_catalog.py, 30_nonuniform_spacing.py, 31_tip_coupling.py,
  32_final_design.py, 33_banded_skin.py, 34_tsai_wu_skin.py`

**Context:** Same as Task 4 for the second batch. Note 32/33/34 already have good
purpose docstrings — only migrate their scenario call and verify the docstring's first
line states the purpose.

- [ ] **Step 1: Migrate each file's scenario/geometry** (same rules as Task 4 Step 1):
`default_scenario()` → `medium_scenario()`, bare `WingSpec()` → `medium_wingsail`,
fix imports. `grep -rn "default_scenario(\|WingSpec()" examples/2*.py examples/3*.py`.

- [ ] **Step 2: Ensure each file has a purpose docstring** (same standard as Task 4
Step 2). Read each example to write an accurate one-sentence purpose. Do not change logic.

- [ ] **Step 3: Verify each file compiles + no stragglers**

Run: `for f in examples/21_beam_fea.py examples/22_sized_beams.py examples/23_sized_geometry.py examples/24_skin_coupling.py examples/25_resize_with_skin.py examples/26_clt_skin.py examples/27_buckling.py examples/28_ply_datum.py examples/29_stock_catalog.py examples/30_nonuniform_spacing.py examples/31_tip_coupling.py examples/32_final_design.py examples/33_banded_skin.py examples/34_tsai_wu_skin.py; do uv run python -m py_compile "$f" && echo "ok $f"; done`
Expected: `ok` for every file.
Run: `grep -rn "default_scenario(" examples/2*.py examples/3*.py`  → NO matches.

- [ ] **Step 4: Commit**

```bash
git add examples/21_beam_fea.py examples/22_sized_beams.py examples/23_sized_geometry.py examples/24_skin_coupling.py examples/25_resize_with_skin.py examples/26_clt_skin.py examples/27_buckling.py examples/28_ply_datum.py examples/29_stock_catalog.py examples/30_nonuniform_spacing.py examples/31_tip_coupling.py examples/32_final_design.py examples/33_banded_skin.py examples/34_tsai_wu_skin.py
git commit -m "$(cat <<'EOF'
Specs library: migrate examples 21-34 to medium + purpose docstrings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Remove all field defaults + drop default_scenario

**Files:**
- Modify: `src/wing_design/geometry/wing.py`, `src/wing_design/structural/mesh.py`,
  `src/wing_design/scenario.py`, `src/wing_design/__init__.py`
- Test: `tests/geometry/test_wingsails.py` (append one assertion)

**Context:** All call sites are migrated (Tasks 3–5), so now strip the defaults. After
this, `WingSpec()` and `DesignParameters()` with no args raise `TypeError`.

- [ ] **Step 1: Write the failing test** — append to `tests/geometry/test_wingsails.py`:

```python
def test_wingspec_has_no_defaults():
    with pytest.raises(TypeError):
        WingSpec()
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/geometry/test_wingsails.py -k no_defaults -v`
Expected: FAIL (currently `WingSpec()` still works — no exception).

- [ ] **Step 3: Remove `WingSpec` field defaults** in `geometry/wing.py`

Strip the `= ...` from every field of `WingSpec` (keep names, types, ORDER, and the
explanatory comments). Result (types as today):
```python
@dataclass(frozen=True)
class WingSpec:
    span: float
    root_chord: float
    tip_chord: float
    thickness: float
    pivot_frac: float
    spar_length: float
    transition_length: float
    n_transition_sections: int
    n_sections: int
    n_airfoil_points: int
    taper_profile: tuple[tuple[float, float], ...] | None
    # ... (keep the existing docstring/comment block for taper_profile)
    # properties + methods unchanged
```
Then change `build_wing_solid(spec: WingSpec = WingSpec())` →
`build_wing_solid(spec: WingSpec)` (drop the default).

- [ ] **Step 4: Drop the `WingSpec()` default in `structural/mesh.py`**

Change the function signature at `mesh.py:49` from `spec: WingSpec = WingSpec()` to
`spec: WingSpec` (required). Check its callers pass `spec` (they do — the truss/mesh
pipeline is explicit); if any bare call exists, pass `small_wingsail`.

- [ ] **Step 5: Remove scenario param-class defaults + `default_scenario` in `scenario.py`**

- Strip every field default from `MaterialParameters`, `MeshParameters`,
  `AeroParameters`, `Phase5FrameFieldParameters`, `SkinParameters`, and
  `DesignParameters` (remove `= value` and `field(default_factory=...)`; keep field
  names/types/order and `DesignParameters`'s `@property` methods). The `STANDARD_*`
  instances (Task 2) now supply every value explicitly, so they still construct.
- Delete the `default_scenario()` function and update the module docstring to reference
  `small_scenario()/medium_scenario()/large_scenario()` instead of `default_scenario()`.

- [ ] **Step 6: Drop `default_scenario` from `src/wing_design/__init__.py`**

Remove `default_scenario` from the `from .scenario import (...)` line and from
`__all__`; update the module docstring's `default_scenario()` mention to
`small_scenario()`. Keep the three builders exported.

- [ ] **Step 7: Verify no remaining references + full suite green**

Run: `grep -rn "default_scenario" src/ tests/ examples/`  → NO matches.
Run: `grep -rn "WingSpec()" src/ tests/ examples/`  → NO matches.
Run: `uv run pytest -q`  → ALL green (including the new `test_wingspec_has_no_defaults`).

- [ ] **Step 8: Commit**

```bash
git add src/wing_design/geometry/wing.py src/wing_design/structural/mesh.py src/wing_design/scenario.py src/wing_design/__init__.py tests/geometry/test_wingsails.py
git commit -m "$(cat <<'EOF'
Specs library: make WingSpec + scenario param dataclasses no-default

Drop field defaults (pure schemas, mirroring UDPly), drop build_wing_solid /
mesh WingSpec() default args, and remove default_scenario() now that all call
sites use small/medium/large_scenario.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (after all tasks)

- [ ] `grep -rn "default_scenario\|WingSpec()" src/ tests/ examples/` — only the
  `default_scenario` mentions left should be in `docs/` (if any); none in code.
- [ ] `uv run pytest -q` — all green, runtime unchanged (tests on small wingsail).
- [ ] `uv run python -m py_compile examples/*.py` — every example compiles.
- [ ] Dispatch a final code review over the whole diff, then use
  superpowers:finishing-a-development-branch.

## Notes / risks for the implementer

- **Ordering is the safety net.** Do NOT remove any default before Tasks 3–5 migrate the
  call sites — the suite must be green at every task boundary. Task 6 is the only task
  that removes defaults, and by then nothing references them.
- **`small_wingsail` == the old default**, so Task 3 must not change any numeric test
  expectation. If a geometry test value changes, something is wrong with `small_wingsail`.
- **Examples are not in pytest** — gate them with `py_compile` + grep, not by running
  (running medium-scale sizing examples is minutes each).
- **Don't strip `LoadCase` or the optimizer `Config` defaults** — out of scope.
- **Import-path matching:** when editing a file, follow its existing import style
  (`from wing_design import ...` vs `from wing_design.scenario import ...`).
