# No-default specs + named wingsail/scenario library: design

**Date:** 2026-06-08
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Make the design-parameter dataclasses pure schemas (no field defaults), mirroring the
existing `UDPly` → `T700_EPOXY` pattern, and provide named, fully-instantiated
`small`/`medium`/`large` wingsails and scenarios. Tests run on the small wingsail (fast);
examples run on the medium wingsail and each explains its purpose.

## Motivation

Today `WingSpec` and the `scenario.py` parameter dataclasses carry inline field
defaults, and `default_scenario()` returns the implicit 5 m demo. Values are entangled
with the schema, there's only one implicit size, and there's no clean way to run the
pipeline at a different scale. The repo already demonstrates the target pattern: `UDPly`
is a no-default dataclass whose concrete values live in named instances (`T700_EPOXY`,
`T800_EPOXY`) in the same module. This refactor applies that pattern consistently and
adds a small/medium/large size family.

## Decisions (from brainstorming)

- **Scaling:** scale + round chords. Lengths scale roughly with span; chords rounded to
  tidy numbers. Dimensionless fields and section counts identical across sizes.
- **Module layout:** split by domain — wingsail instances in a new
  `geometry/wingsails.py`; scenario param instances + scenario builders in `scenario.py`.
- **No-default scope:** all six `scenario.py` param dataclasses become no-default (plus
  `WingSpec`); each gets one named instance.
- **Scenario API:** replace `default_scenario()` with `small_scenario()` /
  `medium_scenario()` / `large_scenario()`. Tests → small; examples → medium.

## Design

### 1. `WingSpec` becomes no-default — `geometry/wing.py`

Remove every field default from `WingSpec` (it becomes a pure schema). All current
fields stay, in the same order, just without `= ...`:
`span, root_chord, tip_chord, thickness, pivot_frac, spar_length, transition_length,
n_transition_sections, n_sections, n_airfoil_points, taper_profile`. Properties and
methods (`spar_radius`, `chord_at_z`, `section_z_fractions`, …) are unchanged.
`build_wing_solid(spec)` drops its `spec=WingSpec()` default — `spec` is now required.

### 2. Named wingsails — `geometry/wingsails.py` (new)

Module-level `WingSpec` instances. Values (lengths in metres):

| field | `small_wingsail` | `medium_wingsail` | `large_wingsail` |
|---|---|---|---|
| span | 5.0 | 22.0 | 45.0 |
| root_chord | 1.0 | 4.5 | 9.0 |
| tip_chord | 0.6 | 2.7 | 5.4 |
| thickness | 0.18 | 0.18 | 0.18 |
| pivot_frac | 0.25 | 0.25 | 0.25 |
| spar_length | 0.75 | 3.3 | 6.75 |
| transition_length | 0.20 | 0.9 | 1.8 |
| n_transition_sections | 4 | 4 | 4 |
| n_sections | 5 | 5 | 5 |
| n_airfoil_points | 160 | 160 | 160 |
| taper_profile | None | None | None |

`small_wingsail` is byte-for-byte the current `WingSpec()` default. Aspect ratio
(span / mean-chord): small 6.25, medium ≈ 6.11, large 6.25 — "similar" as requested.
Exported from `geometry/__init__.py` alongside `WingSpec`.

### 3. Scenario param classes no-default + named instances + builders — `scenario.py`

- Remove field defaults from `MaterialParameters`, `MeshParameters`, `AeroParameters`,
  `Phase5FrameFieldParameters`, `SkinParameters`, and `DesignParameters`
  (drop the `field(default_factory=...)` / `= value` on every field; keep `DesignParameters`'s
  derived `@property` methods unchanged).
- Named shared instances holding today's values (scenario-independent resolution/material
  knobs, reused by all three sizes):
  ```python
  STANDARD_MATERIAL_ISO = MaterialParameters(skin_E_knockdown=0.5, nu_isotropic=0.32, sigma_allow_safety_factor=2.0)
  STANDARD_MESH = MeshParameters(target_element_size_m=0.05)
  STANDARD_AERO = AeroParameters(spanwise_resolution=16)
  STANDARD_FRAME_FIELD = Phase5FrameFieldParameters(n_levels=24, sigma_floor_fraction=0.05, sigma_augment_fraction=1.0)
  STANDARD_SKIN = SkinParameters(t_baseline_m=0.003)
  ```
- Scenario builders compose a wingsail with the shared params:
  ```python
  def _scenario(geometry: WingSpec) -> DesignParameters:
      return DesignParameters(
          geometry=geometry, material=T700_EPOXY, material_iso=STANDARD_MATERIAL_ISO,
          load_cases=DESIGN_CASES, mesh=STANDARD_MESH, aero=STANDARD_AERO,
          frame_field=STANDARD_FRAME_FIELD, skin_sizing=STANDARD_SKIN)

  def small_scenario() -> DesignParameters:  return _scenario(small_wingsail)
  def medium_scenario() -> DesignParameters: return _scenario(medium_wingsail)
  def large_scenario() -> DesignParameters:  return _scenario(large_wingsail)
  ```
- `default_scenario()` is removed. Update the package `__init__` (it re-exports
  `default_scenario`) to export `small_scenario`/`medium_scenario`/`large_scenario`
  instead.

### 4. Call-site migration

- **Tests** — every `WingSpec()` → `small_wingsail` (import from `wing_design.geometry`),
  every `default_scenario()` → `small_scenario()`. ~16 test files; runtime unchanged
  (still the 5 m wingsail). Any bare `build_wing_solid()` call gets `small_wingsail`.
- **Examples** — every `default_scenario()` → `medium_scenario()`; any direct
  `WingSpec()` / geometry reference → `medium_wingsail`.

### 5. Example purpose docstrings

Each `examples/*.py` (24 files) has a module docstring whose first line states what it
demonstrates and why. Examples 32–34 already meet this; bring 21–31 (and any earlier)
to the same standard. Keep it to a short, accurate purpose statement — do not rewrite
the example logic.

### 6. Scope boundaries

- **In scope (become no-default + named instance):** `WingSpec`, `MaterialParameters`,
  `MeshParameters`, `AeroParameters`, `Phase5FrameFieldParameters`, `SkinParameters`,
  `DesignParameters`.
- **Out of scope (keep their field defaults):**
  - `LoadCase` (`aero/cases.py`) — its `altitude_m=0.0`, `safety_factor=1.0`,
    `description=""` are per-case field conveniences; the canonical instances already
    exist as `DESIGN_CASES`.
  - Optimizer `Config` dataclasses (`LaminateSizingConfig`, `BeamShellSizingConfig`,
    `SizingConfig`) — per-run tuning knobs (`r_min`, `r_max`, `t_min`, …) that call
    sites override; not scenario design parameters.

## Testing

- `tests/geometry/test_wingsails.py` (new): the three wingsails instantiate; spans are
  5.0 / 22.0 / 45.0; AR (span / ((root+tip)/2)) for all three within ±5% of 6.25;
  dimensionless fields (`thickness`, `pivot_frac`, `n_sections`, `n_airfoil_points`,
  `taper_profile`) equal across the three; `small_wingsail` equals the documented
  current values.
- `tests/test_scenarios.py` (new): `small/medium/large_scenario()` return
  `DesignParameters` whose `.geometry.span` is 5/22/45; derived properties `E_iso_Pa`,
  `nu_iso`, `G_iso_Pa`, `sigma_allow_Pa`, `rho_kgm3` resolve to finite positive values;
  the shared params are the same object/values across sizes.
- `WingSpec` no longer constructible without args — a test asserting
  `pytest.raises(TypeError): WingSpec()` documents the schema-only contract.
- Full suite green after migration (the gate that the 53 + 32 call-site edits are
  correct). Tests remain on the small wingsail, so total runtime is unchanged.

## Components / file map

| File | Responsibility | Change |
|---|---|---|
| `geometry/wing.py` | `WingSpec` schema (no defaults); `build_wing_solid` spec required | modify |
| `geometry/wingsails.py` | `small_/medium_/large_wingsail` instances | **new** |
| `geometry/__init__.py` | export the three wingsails | modify |
| `scenario.py` | six dataclasses no-default; `STANDARD_*` instances; 3 scenario builders; drop `default_scenario` | modify |
| `wing_design/__init__.py` | re-export the new scenario builders (drop `default_scenario`) | modify |
| `tests/**` (~16 files) | `WingSpec()`→`small_wingsail`, `default_scenario()`→`small_scenario()` | modify |
| `examples/*.py` (24) | `default_scenario()`→`medium_scenario()`; purpose docstrings | modify |
| `tests/geometry/test_wingsails.py`, `tests/test_scenarios.py` | new tests | **new** |

## Success criteria

- `WingSpec` and the six scenario param dataclasses have no field defaults; `WingSpec()`
  with no args raises `TypeError`.
- `small_wingsail` reproduces the current default geometry exactly (existing geometry
  tests still pass once pointed at it).
- `small/medium/large_wingsail` and `small/medium/large_scenario()` exist and are
  exported; spans 5/22/45 with similar AR.
- Tests use small (runtime unchanged); examples use medium and each has a purpose
  docstring.
- Full test suite green.

## Out of scope (deferred)

- Scaling resolution/seed knobs (mesh element size, skin `t_baseline`, solver
  resolutions) with wingsail size — kept shared/constant for now.
- A registry/lookup (`scenario_by_name("medium")`) — three explicit builders suffice.
- Touching `LoadCase` / optimizer `Config` defaults (see scope boundaries).
