# Per-band skin layup: design

**Date:** 2026-06-08
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Let each spanwise skin band carry its own layup `(f0_b, f45_b, f90_b)` (opt-in), so the
optimizer can tune the fibre mix per band. Composes with both the von-Mises and Tsai-Wu
skin checks. Default off = today's single global layup (backward-compatible).

## Motivation & key physics

The skin is ~2/3 of mass and per-band thickness already cut it (−8.1%). A single global
layup forces the whole skin to one fibre mix. **Mass is fraction-independent** (uniform
density → skin mass depends only on thickness), so per-band layup does NOT reduce mass
directly. It helps **indirectly**: a band tuned to the local load (e.g. more spanwise
fibre where bending dominates, more ±45 where shear/twist dominates) can satisfy the
buckling / twist / stress constraints at a **thinner** thickness — and the thickness
bands are the mass DVs. The example and finding must frame the win this way.

## Decisions (from brainstorming)

- **Opt-in flag**, reusing the same `n_skin_bands` partition for layup as for thickness.
- **Compose with Tsai-Wu**: extend the (already vectorized) batch to accept per-triangle
  `(M,)` fractions, gating each orientation element-wise.

## Design

### 1. Config — `LaminateSizingConfig`

Add `per_band_layup: bool = False`. When True, each of the `n_skin_bands` (= B) bands
gets its own `(f0_b, f45_b)`; `f90_b = 1 - f0_b - f45_b`. Default False = one global
layup (unchanged).

### 2. Design vector — `size_beam_shell_laminate`

Let `L = B if config.per_band_layup else 1` (layup groups). The design vector becomes
`x = [radii(n), t_band(B), f0_grp(L), f45_grp(L)]`, length `nx = n + B + 2*L`.
Index helpers: thickness `x[n:n+B]`; `f0_grp = x[n+B : n+B+L]`; `f45_grp = x[n+B+L :
n+B+2L]`. Expand groups to per-band vectors of length B:
- `per_band_layup` True: `f0_band_vec = f0_grp` (L == B, 1:1).
- False: `f0_band_vec = np.full(B, f0_grp[0])` (single global value broadcast).
Per-triangle fractions: `f0_tri = f0_band_vec[band_of_tri]` (and f45, f90). `x0`:
radii = r_max, t_band = t_max, every f0_grp = f45_grp = 1/3.

### 3. Laminate construction (in `evaluate`)

- **Non-datum:** the existing per-band loop builds B laminates, now each with its own
  `(t_b, f0_b, f45_b)` where `f0_b = f0_band_vec[b]` etc. Index to per-triangle via
  `band_of_tri` (unchanged mechanism).
- **Datum:** the existing per-triangle build uses
  `laminate_stiffness_offset(thickness=t_tri[e], f0=f0_tri[e], f45=f45_tri[e],
  f90=f90_tri[e], offset_deg=...)`.
Mass objective and analytic gradient are **unchanged** (fraction-independent).

### 4. Fraction-simplex constraint

`frac_con(x)` returns an **L-vector**: `1 - (f0_grp[g] + f45_grp[g])` for each group
`g` (feasible ≥ 0). The post-solve simplex re-projection (when `f0_grp + f45_grp > 1`)
is applied per group.

### 5. Tsai-Wu batch extension — `materials/failure.py`

Generalize `laminate_min_strength_ratio_batch(ply, eps_local, *, f0, f45, f90,
offset_deg)` so `f0/f45/f90` may each be a scalar OR a length-M array. Implementation:
- Broadcast `f0, f45, f90` to `(M,)`.
- For each of the four orientations 0/+45/-45/90, compute the strength ratio `(M,)` as
  today, then **gate element-wise**: `ratio = np.where(frac > eps, ratio, _SAFE_R)`
  where `frac` is `f0` for the 0 orientation, `f45` for both ±45, `f90` for 90.
- Stack `(4, M)` and `min(axis=0)`. (Elements whose every fraction is ~0 — shouldn't
  occur since fractions sum to 1 — yield `_SAFE_R`.)
- Scalar inputs reduce to today's behavior exactly (a scalar gate is uniformly True or
  False across M), so the existing equivalence test still passes.

The scalar `laminate_min_strength_ratio` (which delegates to the batch) is unchanged.
The sizer's Tsai-Wu call passes per-triangle fraction arrays
(`f0_band_vec[band_of_tri]`, …).

### 6. Result — `LaminateSizingResult`

Add `f0_bands: np.ndarray`, `f45_bands: np.ndarray`, `f90_bands: np.ndarray` (each
length B = `f0_band_vec` etc., so non-per-band runs report the global value broadcast
across B). Keep scalar `f0/f45/f90` redefined as **area-weighted means**
`Σ(f*_tri · Atri)/Σ Atri` (equals the single value when global), so existing readers
(examples 32/34) are unaffected.

### 7. Example — `examples/35_per_band_layup.py`

Medium wingsail, B = 4 thickness bands, span datum, buckling SF 1.5 — the banded
configuration. Size twice: `per_band_layup=False` (global layup) vs `per_band_layup=True`,
both with banded thickness, identical other constraints. Print for each: total mass +
beam/skin split, per-band thicknesses (root→tip), per-band layup `f0/f45/f90`, tip
defl/twist, buckling utils. End with Δmass and a one-line verdict that frames the win as
indirect (per-band fibre mix → thinner bands). Higher maxiter for the per-band run (more
DVs).

### 8. Testing

`tests/materials/test_failure.py` (append):
- `test_batch_array_fractions_match_per_element`: for a random `(M,3)` strain, random
  `(M,)` offsets, and random per-element `(f0,f45,f90)` (summing to 1), the batch equals
  per-element scalar `laminate_min_strength_ratio` calls (allclose). Existing scalar
  equivalence test still passes.

`tests/beams/test_per_band_layup.py` (new):
- `per_band_layup=False` reproduces the current banded result (same mass/t_bands;
  `f0_bands` shape (B,) all equal to the scalar `f0`).
- `per_band_layup=True` on a small problem: `f0_bands/f45_bands/f90_bands` shape (B,),
  converged/feasible, per-band fractions within [0,1] and `f0_b+f45_b+f90_b ≈ 1`,
  total mass ≤ global-layup mass + tol.
- `per_band_layup=True` + `skin_failure="tsai_wu"` converges feasible (the batch
  array-fraction path).

The default path is covered by the existing banded/laminate tests (must stay green).

## Components / file map

| File | Responsibility | Change |
|---|---|---|
| `beams/laminate_sizing.py` | `per_band_layup` config; L-group design vector; per-band fractions; L fraction constraints; `f*_bands` result | modify |
| `materials/failure.py` | batch accepts `(M,)` fractions (element-wise orientation gating) | modify |
| `examples/35_per_band_layup.py` | global-vs-per-band-layup comparison | **new** |
| `tests/materials/test_failure.py` | array-fraction batch test | modify |
| `tests/beams/test_per_band_layup.py` | sizer tests | **new** |
| `docs/plan.md` | finding + Decisions-log row (at close) | modify |

## Success criteria

- Opt-in per-band layup; `per_band_layup=False` reproduces today's design numerically.
- Tsai-Wu batch accepts per-triangle fractions; scalar behavior unchanged
  (equivalence test passes); composes with `per_band_layup=True`.
- A documented global-vs-per-band-layup comparison showing whether the per-band fibre
  taper buys a thinner/lighter skin, framed as an indirect (thickness-mediated) effect.
- Full suite green.

## Out of scope (deferred)

- Independent layup band count (reuses the thickness bands).
- Per-band ply-angle *datum* (datum stays a single global direction).
- Multi-start sizing (separate follow-up).
- Discrete/stacking-sequence layup (laminate stays smeared, fractions-only).
