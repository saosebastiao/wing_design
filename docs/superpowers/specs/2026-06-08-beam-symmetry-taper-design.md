# Beam symmetry + monotonic taper constraints (+ sized mesh export): design

**Date:** 2026-06-08
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Make the CLT beam sizer (`size_beam_shell_laminate`) enforce, **by default**:
1. **Mirror symmetry** — mirror-paired beams share one radius design variable; chord-line
   beams (TE, LE) are independent. (Fewer DVs → faster.)
2. **Monotonic taper** — each beam's radius is non-increasing from the keel-step up to the
   tip (no reason to grow toward the tip).

Plus: a full sizing run on the medium wingsail with the resulting structural geometry
exported as a **mesh (STL)**.

## Motivation

The radii dominate the SLSQP design vector (`n_beams·(n_levels-1)` ≈ 112 of ~115 DVs);
SLSQP's finite-difference Jacobian cost scales with DV count. Mirror symmetry (a free-
rotating wingsail flies on both tacks, so the load envelope is symmetric → symmetric
sizing is correct) roughly halves the radius DVs. Monotonic taper removes physically
nonsensical tip-heavy solutions and tightens the feasible set. Together: faster, more
sensible solves.

## Decisions (from brainstorming)

- **Always-on** (change the sizer default), scoped to `size_beam_shell_laminate` (the
  active sizer). Legacy `size_beam_shell` / `size_beams` untouched.
- **Symmetry = shared DVs**; **monotonic taper = algebraic inequality constraints**.
- Symmetry **auto-detects** mirror-symmetric placement; falls back to independent radii
  if placement is asymmetric (so always-on never hard-fails).

## Design

### 1. Symmetry grouping — `beams/shell_sizing.py`

```
beam_radius_groups(model: BeamShellModel) -> tuple[np.ndarray, int]   # (group_of_element (n,), n_groups)
```
- Beam `b` mirrors `m(b) = (n_beams - b) % n_beams`; representative `rep(b) = min(b, m(b))`.
  Self-mirror (chord-line) beams: `b == 0`, and `b == n_beams//2` when `n_beams` even.
- **Symmetry check:** for every `b`, the node coordinates of beam `b` and beam `m(b)`
  must be mirror images across the chord plane (y → −y) within a tolerance (positions
  match in x and z, opposite in y). If ALL pairs match → build groups: element index
  `e = b*(n_levels-1) + k` maps to `group = unique_rank(rep(b))*(n_levels-1) + k`, where
  `unique_rank` enumerates the representative beams 0..U-1. `n_groups = U*(n_levels-1)`.
  Mirror-paired beams thus share radii; chord-line beams are their own group.
- If the symmetry check fails (e.g. non-uniform `arc_fractions`): fall back to
  `group_of_element = np.arange(n)`, `n_groups = n` (today's independent radii). This
  keeps the always-on default safe for asymmetric layouts.
- Even `n_beams`: `U = n_beams//2 + 1`, so `n_groups = (n_beams//2 + 1)*(n_levels-1)`
  (e.g. 16 beams, 8 levels → 9·7 = 63 vs 112 — ~44% fewer radius DVs).

Also add `beam_level_z(model) -> np.ndarray` helper (the z of each spanwise level, from
the node coordinates) if needed to order levels keel→tip for the taper constraint; or
derive ordering inline from `model.nodes`.

### 2. Sizer radii reparameterization (always-on) — `beams/laminate_sizing.py`

- Compute `group_of_element, G = beam_radius_groups(model)` at the top of
  `size_beam_shell_laminate`. The radii block of the design vector becomes the **G group
  radii** (was `n` per-element). Layout: `x = [r_group(G), t_band(B), f0_grp(L), f45_grp(L)]`,
  `nx = G + B + 2L`. Shift the thickness/layup index bases accordingly (`t at [G:G+B]`,
  `f0_lo = G+B`, `f45_lo = G+B+L`).
- Inside `evaluate`, expand `radii = r_group[group_of_element]` (length n) and use it
  exactly as today (sections, von Mises, Euler buckling).
- `x0`: group radii = `r_max`; `bounds`: `[(r_min, r_max)]*G + ...`.
- **Mass** value unchanged: `rho·Σ_e(π·radii_e²·L_e) + skin`. `mass_grad` w.r.t. group
  radius `g`: `Σ_{e: group_of_element[e]==g} rho·2π·r_group[g]·L_e`. Implement by scattering
  the per-element gradient `rho·2π·radii·Lb` into G bins via `np.add.at` (or a precomputed
  `(G, n)` reduction). The skin/layup gradient terms are unchanged.
- `result.radii` is **expanded back to the full per-element length n** (`r_group[group_of_element]`),
  so geometry/stock-catalog consumers see the familiar `n_beams·(n_levels-1)` layout.

### 3. Monotonic taper constraints

- Determine, per spanwise level, the z ordering from `model.nodes` (keel = min z, tip =
  max z). For each unique beam group and each consecutive level pair, require
  `r[lower-z level] − r[higher-z level] ≥ 0` (radius non-increasing toward the tip).
- Implement as one vectorized constraint `monotonic_con(x)` returning all the per-group
  consecutive-level differences as a flat array (feasible ≥ 0). It reads only the group
  radii in `x` — **no FEA**, so it's cheap in the FD Jacobian.
- Precompute the index pairs `(hi_z_dv, lo_z_dv)` over the group-radius DVs once (outside
  `evaluate`), so `monotonic_con` is `x[lo] - x[hi]` vectorized.

### 4. Default-change handling

This changes the sizer's default. `result.radii` keeps shape `(n,)`, so shape/feasibility
tests still pass; update any test that asserts *specific* radius values (most assert
shape/feasibility). Recorded headlines (30.15 kg etc.) assumed independent radii and
become historical — add a one-line note in `docs/plan.md`; re-running the expensive
examples to refresh every headline is deferred (the explicit full run below produces a
fresh symmetric/monotonic headline for the medium wingsail).

### 5. Sized mesh export — `examples/37_sized_export.py` (new)

Sizes the medium wingsail with the new default constraints, then exports the resulting
structural geometry as STL meshes:
- Build model (n_beams=16, n_levels=8), aero envelope, load arrays (as in example 32).
- `r = size_beam_shell_laminate(model, loads, cfg, ...)` with span datum + buckling
  (the same config family as example 32); symmetry + taper now apply by default.
- Densify radii to a finer geometry resolution: `resample_segment_radii(r.radii,
  n_beams=16, n_from=8, n_to=N_GEOM)` (e.g. N_GEOM=40).
- Build beams: `build_sized_lens_beams(spec, dense_radii, n_beams=16, n_levels=N_GEOM)`,
  falling back to `build_sized_circular_beams` on a lofting error (try/except, mirroring
  example 23).
- Build skin: `build_wing_solid(spec)` (the OML).
- `export_stl` each part and/or the assembled `Compound` to `exports/`:
  `wingsail_sized_skin.stl`, `wingsail_sized_beams.stl` (a Compound of the beams), and
  `wingsail_sized_assembly.stl`. Print the sized headline (mass, radii range, layup,
  twist/buckling) and the written paths.

### 6. The full run (executed after the code lands)

After the branch is implemented and merged, run `examples/37_sized_export.py` at full
settings, **timed**, in the background, then surface the STL artifact(s). Expected to be
faster than the prior ~2 h runs thanks to the ~halved radius DV count (measure and
report the actual wall-clock — no estimate). This is the deliverable the user asked for
("a full run with export of the resulting geometries in mesh form").

## Testing — `tests/beams/test_beam_symmetry.py`

- `beam_radius_groups` (default symmetric placement, n_beams=8, n_levels=5):
  `n_groups == (8//2 + 1)*(5-1) == 20`; mirror beams b and 8−b share group ids per level;
  chord-line beams 0 and 4 are singletons (their group ids belong to no other beam).
- Asymmetric placement (build model with a deliberately asymmetric `arc_fractions`):
  `n_groups == n` (fallback to independent).
- Sizer symmetry (small problem): `result.radii` reshaped `(n_beams, n_levels-1)` has
  row `b` == row `(n_beams-b)%n_beams` (allclose); chord rows free.
- Sizer monotonicity: each beam's radii are non-increasing keel→tip (within a small tol).
- Feasibility/shape: small sizing converges feasible; `result.radii.shape == (n,)`.
- Existing laminate/banded/tsai-wu/per-band/multistart tests stay green (fix any
  exact-radius-value assertions; shape/feasibility ones hold).

## Components / file map

| File | Change |
|---|---|
| `beams/shell_sizing.py` | `beam_radius_groups` (+ symmetry detection, level-z ordering) | modify |
| `beams/laminate_sizing.py` | group-radius DVs + expansion + mass-grad scatter + monotonic constraints (always-on) | modify |
| `beams/__init__.py` | export `beam_radius_groups` | modify |
| `examples/37_sized_export.py` | size (new defaults) + STL mesh export | **new** |
| `tests/beams/test_beam_symmetry.py` | grouping + symmetry + monotonicity tests | **new** |
| `docs/plan.md` | finding + headline-shift note (at close, with the full-run numbers) | modify |

## Success criteria

- `size_beam_shell_laminate` enforces mirror-symmetric radii (shared DVs) and monotonic
  keel→tip taper by default; `result.radii` stays full length `n`.
- Symmetry auto-detects placement and falls back gracefully when asymmetric.
- Full suite green (exact-value tests updated; shape/feasibility hold).
- A full medium-wingsail run completes (faster, measured) and exports STL meshes of the
  sized skin + beams + assembly.

## Out of scope (deferred)

- Applying the constraints to the legacy `size_beam_shell` / `size_beams` sizers.
- Reparameterized (decrement-based) taper — using inequality constraints instead.
- Refreshing every historical headline in the plan (only the medium full run is rerun).
- Diagonal/tip-coupling families (discarded / investigation-only).
