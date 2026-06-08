# Per-spanwise-band skin thickness: design

**Date:** 2026-06-07
**Phase:** E refinement (item #3, part 1 of 2 — per-band skin thickness; per-ply Tsai-Wu deferred)
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Replace the single uniform skin-thickness design variable in the CLT co-sizer with
**B contiguous spanwise bands**, each carrying its own thickness, so the optimizer can
taper the skin (thick root → thin tip) to follow the bending-moment / buckling demand.
The skin is ~2/3 of total mass and is currently uniform, so spanwise tapering is the
highest-leverage remaining mass lever.

## Scope

- Per-band skin thickness only. Per-ply Tsai-Wu failure is a separate follow-up
  (item #3 part 2), explicitly out of scope here.
- FEA-only; no CAD.
- **Opt-in**: a config field `n_skin_bands` defaults to 1, which reproduces today's
  uniform-thickness design exactly (same design vector, same result numerically).

## Motivation

F.1 (beam re-spacing) and F.2 (diagonal lattice) both confirmed the structure is
**skin/buckling-dominated** — beam-layout levers have low leverage. The skin itself is
uniform thickness, sized by the worst (root) panel, so the tip carries far more
material than its local buckling/stress demand requires. Banding lets the tip thin out.

## Approach

Chosen (from brainstorming): per-band thickness only; configurable band count B
(default 4) splitting contiguous spanwise levels into near-equal groups; opt-in via
`n_skin_bands` (default 1 = current behavior); `t_skin` reused as the area-weighted
mean so existing callers keep working. Alternatives rejected: per-level thickness
(max freedom but unmanufacturable many-step skin and more SLSQP noise); fixed 3 bands
(too coarse, leaves mass on the table vs a tunable B).

### 1. Band mapping — `beams/shell_sizing.py`

Skin triangle ordering (from `shell_model.skin_triangles`) is level-major: triangle `e`
sits at spanwise level `level(e) = (e // 2) // n_beams`, with `n_levels - 1` levels
total.

```
skin_band_map(model: BeamShellModel, n_bands: int) -> np.ndarray   # (M,) band index in [0, n_bands)
skin_band_areas(model: BeamShellModel, band_of_tri: np.ndarray, n_bands: int) -> np.ndarray  # (n_bands,)
```

- `skin_band_map` splits the `n_levels - 1` levels into `n_bands` contiguous,
  near-equal groups (use `np.array_split(range(n_levels-1), n_bands)` to assign a band
  to each level, then map each triangle through its level). Band index increases with
  level index; note `default_z_levels` runs tip→keel, so **band 0 is the tip end and
  the last band the keel/root**. Band ordering is by level index — the example labels
  ends by inspecting node z, the sizer/result do not assume a physical orientation.
  `n_bands == 1` → all zeros.
  `n_bands` must satisfy `1 <= n_bands <= n_levels - 1` (raise `ValueError` otherwise).
- `skin_band_areas` sums `skin_areas(model)` per band — used for the analytic mass
  gradient. Bands with no triangles get area 0 (cannot happen given the validation
  above, but the sum handles it).

### 2. Sizer changes — `beams/laminate_sizing.py`

- `LaminateSizingConfig` gains `n_skin_bands: int = 1`.
- Let `B = config.n_skin_bands`, `band_of_tri = skin_band_map(model, B)` (M,),
  `band_area = skin_band_areas(model, band_of_tri, B)` (B,).
- **Design vector:** `x = [long_radii(0..n-1), t_band(0..B-1), f0, f45]`, length
  `nx = n + B + 2`. Index of `f0` is `n + B`, `f45` is `n + B + 1`. When `B == 1` this
  is exactly the current `[radii, t, f0, f45]` layout.
- `x0`: radii = `r_max`, every `t_band = t_max`, `f0 = f45 = 1/3`.
- **Per-triangle thickness:** `t_tri = t_band_vec[band_of_tri]` (M,) where
  `t_band_vec = x[n : n+B]`.
- **Laminate stiffness construction (per evaluate):**
  - *No datum* (`config.ply_angle_datum is None`): build `B` laminates
    `laminate_stiffness(ply, f0, f45, f90, thickness=t_band_vec[k])` for `k in 0..B-1`,
    then index to per-triangle arrays: `A_arg = A_band[band_of_tri]` (M,3,3), same for
    `D_arg`, `C_arg`. `D11 = D_arg[:, 0, 0]` (M,).
  - *Datum* (`ply_angle_datum` set): per-triangle as today, but with per-triangle
    thickness: `laminate_stiffness_offset(ply, f0, f45, f90, thickness=t_tri[e],
    offset_deg=datum_offsets_deg[e])` for each triangle e → stack to (M,3,3).
    `D11 = D_arg[:, 0, 0]` (M,).
  - Both feed `solve_beam_shell_laminate(..., A_skin=A_arg, D_skin=D_arg, ...)`
    (already accepts (M,3,3)).
- **Buckling:** `panel_buckling_utilization(skin_s, Atri, D11=D11, t=t_tri, kc=..., 
  safety_factor=...)` — `t` is now the per-triangle array; the function already
  broadcasts (`σcr = kc·π²·D11/(b²·t)`), no change to `buckling.py`. Beam Euler buckling
  is unchanged.
- **Mass objective:** `skin_mass_term = rho * np.sum(t_tri * Atri)` =
  `rho * np.sum(t_band_vec * band_area)`. Full objective
  `(rho*Σ(π r² L) + rho*Σ(t_band·band_area)) / m_ref`.
- **Mass gradient (analytic):** `∂mass/∂radii` unchanged;
  `∂mass/∂t_band[k] = rho * band_area[k] / m_ref` for each band.
- **`m_ref`:** beam mass at `r_max` + `rho * t_max * total_skin_area` (same total mass
  scale as today; `Σ band_area = total_skin_area`).
- **Constraints:** beam vm, skin vm, deflection, twist, fraction simplex, and (when
  enabled) beam/panel buckling — identical structure to today, just reading the
  banded `evaluate`. The fraction-simplex indices update to `n+B`, `n+B+1`.

### 3. Result + backward compatibility

- `LaminateSizingResult` gains `t_bands: np.ndarray` (the B band thicknesses,
  by level index, band 0 = tip end).
- `t_skin: float` is **kept**, redefined as the **area-weighted mean thickness**
  `Σ(t_tri·Atri)/Σ(Atri)`. For `B == 1` this equals the single thickness, so
  `examples/32_final_design.py` (which prints `r.t_skin`) is unaffected.
- **Hard requirement:** with `n_skin_bands == 1`, the optimization and reported result
  must match the current single-thickness sizer numerically (allclose). The general
  per-triangle construction with all-equal thickness is mathematically identical to the
  current single-laminate assembly, so this holds; it is verified by a regression test.

### 4. Example — `examples/33_banded_skin.py`

Same scenario/aero/config as `examples/32_final_design.py` (n_beams=16, n_levels=8,
span datum, buckling SF 1.5, maxiter 200). Size **uniform (n_skin_bands=1)** and
**banded (n_skin_bands=4)** under identical constraints. Print, for each: total mass +
beam/skin split, the per-band thicknesses (root→tip), tip deflection, tip twist, beam &
panel buckling utilization. End with the Δmass (kg and %) and a one-line verdict.
Success metric: the banded design is lighter than uniform at equal constraints; report
the magnitude honestly (and the band taper that achieves it).

### 5. Testing — `tests/beams/test_banded_skin.py`

- **`skin_band_map`**: `n_bands=1` → all zeros; for `n_bands>1`, indices in `[0,B)`,
  every triangle assigned, bands are contiguous in level and near-equal in level-count;
  triangles at the same level share a band; out-of-range `n_bands` raises `ValueError`.
- **`skin_band_areas`**: sums to the total skin area; length B.
- **Backward-compat**: `size_beam_shell_laminate` with `n_skin_bands=1` matches the
  prior single-t behavior — on a small problem, assert `r.t_bands.shape == (1,)`,
  `r.t_skin ≈ r.t_bands[0]`, and that mass/radii are allclose to a reference run
  (the reference is the same call; the test pins internal consistency:
  `skin_mass_kg ≈ rho * t_skin * total_area`).
- **Banded sizer**: `n_skin_bands=4` on a small problem → `t_bands.shape == (4,)`,
  converged/feasible, all `t_bands ∈ [t_min, t_max]`,
  `skin_mass_kg ≈ rho * Σ(t_tri·Atri)`, and banded mass ≤ uniform mass + tol.
- **Panel buckling with array `t`**: direct unit test that
  `panel_buckling_utilization(..., t=array)` returns per-element utilization equal to
  the scalar-`t` result when the array is constant, and scales as `1/t` element-wise.

## Components / file map

| File | Responsibility | Change |
| --- | --- | --- |
| `beams/shell_sizing.py` | `skin_band_map`, `skin_band_areas` | modify |
| `beams/laminate_sizing.py` | `n_skin_bands` config; banded design vector / laminate / mass / buckling; result `t_bands` | modify |
| `beams/__init__.py` | export `skin_band_map`, `skin_band_areas` | modify |
| `examples/33_banded_skin.py` | uniform-vs-banded comparison | **new** |
| `tests/beams/test_banded_skin.py` | tests | **new** |
| `docs/plan.md` | finding + Decisions-log row (at close) | modify |

## Success criteria

- Opt-in per-band skin thickness in the CLT sizer; `n_skin_bands=1` reproduces today's
  design numerically (regression-tested).
- A documented uniform-vs-banded mass comparison at equal constraints showing the
  band taper and the mass saved.
- All existing tests still green.

## Out of scope (deferred)

- Per-ply Tsai-Wu (or max-strain) skin failure criterion (item #3 part 2).
- Per-band *layup* (f0/f45/f90) variation — layup stays global; only thickness bands.
- MILP/discrete thickness catalog for the bands.
- Chordwise (around-section) thickness variation — bands are spanwise only.
