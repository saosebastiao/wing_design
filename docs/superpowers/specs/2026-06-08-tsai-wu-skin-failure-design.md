# Per-ply Tsai-Wu skin failure: design

**Date:** 2026-06-08
**Phase:** E refinement (item #3, part 2 of 2 — per-ply Tsai-Wu skin failure)
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Replace the laminate-average von-Mises skin failure proxy in the CLT co-sizer with a
proper **per-ply Tsai-Wu** failure check, so the skin constraint reflects real
fibre/matrix/shear failure in each ply orientation rather than a single equivalent
stress on the smeared laminate. Opt-in; the default keeps today's von-Mises proxy.

## Motivation

The current skin constraint compares the laminate-average membrane von Mises against a
single isotropic allowable (`sigma_allow_Pa = Xt/2`). That ignores anisotropy: a CFRP
ply is ~25× stronger along the fibre than transverse, and von-Mises-on-the-average
neither sees the per-ply stress state nor distinguishes fibre from matrix failure.
Tsai-Wu in each ply's material axes is the standard composite criterion and gives a
defensible skin margin.

## Decisions (from brainstorming)

- **F12 interaction term:** `F12 = -0.5·√(F11·F22)` (standard normalized F12* = -0.5).
- **Safety factor:** material SF = 2.0, reusing the scenario's
  `sigma_allow_safety_factor`, applied via a **strength-ratio** formulation: R is the
  factor by which the current ply stress can scale before reaching the Tsai-Wu envelope;
  the constraint is `R ≥ SF`. Loads are already load-factored separately (the load
  arrays carry per-case safety factors), so this SF is the material side only.
- **Opt-in:** a `skin_failure` config flag (default `"von_mises"` = current behavior).
- **Present orientations only:** Tsai-Wu is evaluated for the ply orientations actually
  in the layup (0 if f0>0, ±45 if f45>0, 90 if f90>0) — absent plies do not constrain.
- **Example comparison** on a **uniform** skin for a clean von-Mises-vs-Tsai-Wu read.

## Approach

Ply stresses come from the per-triangle element-local membrane strain ε (= Bm·u). All
plies share that laminate membrane strain (CLT), so a ply's stress depends only on the
strain and the ply's orientation — independent of the area fractions. For each present
orientation we transform ε into the ply material axes, compute σ123 = Q·ε123 with the
material-axis reduced stiffness Q, and evaluate the Tsai-Wu strength ratio R. The
governing skin margin is the minimum R over all present plies, all triangles, all load
cases; feasibility is `min_R ≥ SF`.

### 1. Failure math — `materials/failure.py` (new)

```
tsai_wu_coefficients(ply) -> (F1, F2, F11, F22, F66, F12)
tsai_wu_index(sigma123, ply) -> float            # FI; failure when FI >= 1
tsai_wu_strength_ratio(sigma123, ply) -> float   # R; failure when R < 1
ply_strength_ratio(ply, eps_local, angle_deg) -> float
laminate_min_strength_ratio(ply, eps_local, *, f0, f45, f90, offset_deg) -> float
```

- Coefficients (Xc, Yc stored as positive magnitudes in `UDPly`):
  `F1 = 1/Xt - 1/Xc`, `F2 = 1/Yt - 1/Yc`, `F11 = 1/(Xt·Xc)`, `F22 = 1/(Yt·Yc)`,
  `F66 = 1/S12²`, `F12 = -0.5·√(F11·F22)`.
- `tsai_wu_index([σ1,σ2,τ12], ply)` =
  `F1σ1 + F2σ2 + F11σ1² + F22σ2² + F66τ12² + 2F12σ1σ2` (σ in material axes).
- `tsai_wu_strength_ratio`: with `a = F11σ1²+F22σ2²+F66τ12²+2F12σ1σ2`,
  `b = F1σ1+F2σ2`, solve `a·R²+b·R−1=0` → `R = (−b+√(b²+4a))/(2a)`. Guard `a≈0`
  (then `R = 1/b` if `b>0`, else `+inf` — no failure under pure benign loading).
  Returns a large finite value (e.g. `1e9`) for the zero-stress case.
- `ply_strength_ratio(ply, eps_local, angle_deg)`: transform the element-local
  engineering strain `eps_local = [εxx, εyy, γxy]` into ply material axes at
  `angle_deg`:
  `ε1 = εx c² + εy s² + γxy s c`,
  `ε2 = εx s² + εy c² − γxy s c`,
  `γ12 = −2εx s c + 2εy s c + γxy(c²−s²)`  (c=cos, s=sin of `angle_deg`).
  Then `σ123 = reduced_stiffness_Q(ply) · [ε1, ε2, γ12]`, return
  `tsai_wu_strength_ratio(σ123, ply)`.
- `laminate_min_strength_ratio(...)`: evaluate `ply_strength_ratio` at the present
  orientations — local angles `{0, +45, −45, 90} + offset_deg` gated by `f0>eps`,
  `f45>eps`, `f45>eps`, `f90>eps` (eps = 1e-9). Return the minimum R; if no orientation
  is present (degenerate), return the zero-stress large value.

### 2. Strain recovery — `structural/shell.py`

```
recover_membrane_strain(nodes, triangles, displacements) -> np.ndarray   # (M, 3)
```

The element-local membrane strain ε = Bm·u (engineering shear γxy), one row per
triangle. Mechanically identical to `recover_membrane_stress_C` with `C = I3`; factor
the shared Bm/u_local assembly so the two functions do not duplicate the kinematics, or
implement `recover_membrane_strain` directly and have it return ε. Either way ε must
equal `recover_membrane_stress_C(C=I)`.

### 3. Sizer integration — `beams/laminate_sizing.py`

- `LaminateSizingConfig` gains `skin_failure: str = "von_mises"` and
  `tsai_wu_safety_factor: float = 2.0`.
- `evaluate(x)` additionally (only when `skin_failure == "tsai_wu"`): for each load case
  recover per-triangle strain (`recover_membrane_strain`), and for each triangle compute
  `laminate_min_strength_ratio(ply, eps[e], f0=f0, f45=f45, f90=f90,
  offset_deg=datum_offsets_deg[e] if datum else 0.0)`; track `worst_R = min(worst_R,
  min over triangles)` across load cases. Return it alongside the existing tuple.
- The skin constraint:
  - `"von_mises"` (default): unchanged — `1 - ws/sigma_allow_Pa ≥ 0`.
  - `"tsai_wu"`: `worst_R / config.tsai_wu_safety_factor - 1 ≥ 0` (replaces the
    von-Mises skin constraint). The von-Mises stress `ws` is still computed and reported
    in the result for reference.
- `LaminateSizingResult` gains `min_skin_strength_ratio: float | None` (the governing R;
  `None` in the von-Mises path).
- The default-path design vector, constraints, objective, and result values are
  unchanged (only the additive `min_skin_strength_ratio=None` field differs).

### 4. Example — `examples/34_tsai_wu_skin.py`

Same scenario/aero/config as `examples/32` (span datum, buckling SF 1.5, n_beams=16,
n_levels=8, **uniform skin** for a clean comparison). Size twice:
`skin_failure="von_mises"` and `skin_failure="tsai_wu"` (SF 2.0). Print for each: total
mass + split, layup, skin thickness, the governing skin margin (von-Mises σ vs Tsai-Wu
min R), tip defl/twist, buckling utils. End with Δmass and a one-line verdict on whether
the per-ply criterion changes the design and which skin mode governs.

### 5. Testing

`tests/materials/test_failure.py`:
- Tsai-Wu calibration: `tsai_wu_index([Xt,0,0]) == 1`, `tsai_wu_strength_ratio([Xt,0,0])
  == 1`; `tsai_wu_strength_ratio([-Xc,0,0]) == 1`; `tsai_wu_strength_ratio([0,0,S12]) ==
  1`; `tsai_wu_strength_ratio([0,Yt,0]) == 1`, `[0,-Yc,0] == 1` (all within tol).
- Strength-ratio scaling: halving σ doubles R (away from the linear-term regime — test
  with a dominant shear term where the criterion is ~quadratic).
- Zero stress → large finite R (no failure).
- `ply_strength_ratio`: a fibre-aligned uniaxial element-local strain at `angle_deg=0`
  produces σ1 ≈ Q11·ε and the expected R; the same strain at `angle_deg=90` (fibre
  transverse) gives a much smaller R (transverse-weak).
- `laminate_min_strength_ratio`: returns the min over present orientations; an absent
  orientation (fraction 0) does not lower the result.

`tests/structural/test_shell.py` (or the banded/laminate test module):
- `recover_membrane_strain` equals `recover_membrane_stress_C(C=I3)` on a small mesh,
  and `Qeff · strain` ≈ `recover_membrane_stress_C(C=Qeff)`.

`tests/beams/test_tsai_wu_sizing.py`:
- `skin_failure="tsai_wu"` on a small problem converges feasible, returns
  `min_skin_strength_ratio` ≳ SF (active, i.e. close to SF at the optimum), within tol.
- Default `skin_failure="von_mises"` result is unchanged (allclose mass/radii; new field
  `min_skin_strength_ratio is None`).

## Components / file map

| File | Responsibility | Change |
| --- | --- | --- |
| `materials/failure.py` | Tsai-Wu coefficients/index/strength-ratio + per-ply & laminate-min helpers | **new** |
| `structural/shell.py` | `recover_membrane_strain` | modify |
| `beams/laminate_sizing.py` | `skin_failure` + `tsai_wu_safety_factor` config; Tsai-Wu skin constraint; result `min_skin_strength_ratio` | modify |
| `materials/__init__.py` | export failure helpers | modify |
| `examples/34_tsai_wu_skin.py` | von-Mises-vs-Tsai-Wu comparison | **new** |
| `tests/materials/test_failure.py` | failure-math tests | **new** |
| `tests/beams/test_tsai_wu_sizing.py` | sizer tests | **new** |
| `docs/plan.md` | finding + Decisions-log row (at close) | modify |

## Success criteria

- Opt-in per-ply Tsai-Wu skin failure in the CLT sizer; `skin_failure="von_mises"`
  (default) reproduces today's design numerically (regression-tested).
- Tsai-Wu math calibrated against the uniaxial/shear strength points (unit-tested).
- A documented von-Mises-vs-Tsai-Wu comparison showing whether the per-ply criterion
  changes the design and which skin failure mode governs.
- All existing tests still green.

## Out of scope (deferred)

- First-ply-failure vs last-ply/progressive failure (we use first-ply: min R).
- Stacking-sequence / interlaminar (delamination) effects — the laminate is smeared
  (fractions-only), consistent with the rest of the CLT model.
- Per-band *layup* variation; combining Tsai-Wu with the diagonal study.
- Using Tsai-Wu for the beams (beams stay isotropic-equivalent von Mises).
