# Analytic-Jacobian caching optimization: design

**Date:** 2026-06-09
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Make the analytic adjoint Jacobian deliver its theoretical speedup by precomputing the
design-dependent `∂K/∂x` element matrices **once per design point** and reusing them
across all constraints and adjoint solves — instead of rebuilding them on every adjoint
call. **Exact:** gradients are bit-for-bit identical (same math, not recomputed), so all
existing FD-validation tests pass unchanged. No constraint-formulation change.

## Motivation

The analytic Jacobian is correct and FD-validated but measured only ~1.58× faster (vs the
~n_DV× ceiling). The bottleneck (confirmed in code): `lambdaT_dK_x` rebuilds every beam
element's `∂kg/∂r` and every triangle's `∂ke/∂t`, `∂ke/∂f0`, `∂ke/∂f45` (via `dkloc_dr`,
`dke_dAD`, `dQeff_df` — trig + Q-transforms) on **every** call. The sizer's `beam_con`
Jacobian is vector-valued (one row per beam element), so it calls `lambdaT_dK_x` ~n times
per gradient evaluation → the per-element assembly is redone ~n×. But those `∂K` matrices
depend only on the **design** (radii/thickness/layup), not on the adjoint vector `λ` or
the load case — so they should be built once and reused.

## Decision (from brainstorming)

- **Caching only (exact).** No aggregation of `beam_con`; gradients unchanged. Remaining
  cost after caching is the cheap adjoint back-substitutions (`lu.solve`). Aggregation
  (KS soft-max → 1 adjoint) deferred unless caching proves insufficient.

## Design

### 1. `prepare_sensitivity(ds, factored) -> SensCache` — `beams/sensitivity.py`

Build, once per design point:
- **Per beam element e:** `dkg_e = T_eᵀ · dkloc_dr(E_beam, G_beam, r_e, L_e) · T_e` (12×12,
  global frame), the element's 12 global DOF indices, and its radius-group index
  `group_of_element[e]`.
- **Per triangle t:** `dke_t = dke_dAD(p1,p2,p3, *dAD_dt(Qeff_t, t_t))`,
  `dke_f0_t = dke_dAD(p1,p2,p3, *dAD_df(dQeff_df(ply,'f0',off_t), t_t))`, and the `f45`
  analogue (each 18×18), the triangle's 18 DOF indices, its thickness-band index, and its
  layup-group index.
`SensCache` is a frozen dataclass holding these arrays/lists (design-only; no `λ`, no `u`,
load-case-independent). This is the expensive step (the trig/Q-transform builds), done
exactly once.

### 2. `lambdaT_dK_x_cached(cache, lam, u) -> (nx,)` — `beams/sensitivity.py`

Same contraction as `lambdaT_dK_x`, using the cached matrices: for each element/triangle,
`out[dv_index] += lam[dofs] · (dk · u[dofs])`. No rebuilds. (Implementation may further
precompute `dk · u` per load case to reduce to pure dot products, but the matrix already
being cached is the dominant win.) Returns the identical `(nx,)` vector as the current
`lambdaT_dK_x` for the same `(ds, lam, u)`.

### 3. `grad_*` functions accept an optional `cache`

Each constraint gradient (`grad_tip_defl`, `grad_tip_twist`, `grad_beam_vm`,
`grad_beam_buckling`, `grad_skin_vm`, `grad_panel_buckling`, `grad_skin_tsai_wu`) gains a
keyword `cache: SensCache | None = None`. If `None`, it lazily builds one
(`prepare_sensitivity(ds, factored)`) — so existing direct callers (the FD-validation
tests) are unchanged and still exact. When a cache is passed, it calls
`lambdaT_dK_x_cached(cache, lam, u)` instead of `lambdaT_dK_x`. (`lambdaT_dK_x` is kept as
the reference/source-of-truth used inside `prepare_sensitivity`'s validation and for the
lazy path's equivalence.)

### 4. Sizer wiring — `beams/laminate_sizing.py`

`evaluate_jac(x)` (the analytic path, behind `use_analytic_jacobian`) builds **one**
`SensCache` per design point (after the factored solves) and threads it into every
constraint-gradient call — so `beam_con`'s ~n rows and all other constraints reuse the
single assembly. The cache is design-only, so it's also reused across the per-load-case
factored solves (only `u`/`λ` differ). No change when `use_analytic_jacobian=False`.

## Correctness & testing

- **Exactness (primary gate):** `tests/beams/test_sensitivity.py` (all the FD-validation
  checks) and `tests/beams/test_analytic_sizing.py` pass **unchanged** — the cached path
  produces identical gradients. Add one test asserting
  `lambdaT_dK_x_cached(prepare_sensitivity(ds,fac), lam, u)` equals `lambdaT_dK_x(fac, ds,
  lam)` to allclose(≤1e-12) on a small case.
- These tests are small/fast (single solves) and safe to run alongside the background
  deflection sizing run.
- **Deferred to when the CPU frees** (the background run finishes): the full suite
  (`uv run pytest -q`) as the merge gate, and a re-measurement of
  `examples/38_analytic_speedup.py` (FD vs analytic wall-clock) to quantify the
  improvement over the prior 1.58×.

## Components / file map

| File | Responsibility | Change |
|---|---|---|
| `beams/sensitivity.py` | `prepare_sensitivity` + `SensCache` + `lambdaT_dK_x_cached`; `cache=` kwarg on `grad_*` | modify |
| `beams/laminate_sizing.py` | build one cache per design point in `evaluate_jac`, thread into all gradient calls | modify |
| `beams/__init__.py` | export `prepare_sensitivity` (if useful for tests) | modify (optional) |
| `tests/beams/test_sensitivity.py` | cached==uncached equivalence test | modify |
| `docs/plan.md` | finding (at close, with measured speedup) | modify |

## Success criteria

- Cached gradients are bit-for-bit identical to the current analytic gradients (the
  cached==uncached test + all existing FD tests pass).
- One `∂K` assembly per design point instead of ~n (the optimization is realized in code).
- Full suite green; analytic sizing still reaches the same feasible optimum as FD.
- Measured: `examples/38` analytic run is materially faster than the prior 1.58× (recorded
  once the CPU is free).

## Out of scope (deferred)

- Aggregating `beam_con` (KS/p-norm) into a single adjoint solve.
- Mixed direct/adjoint sensitivity for the vector constraint.
- Changing the default (`use_analytic_jacobian` stays off).
- Parallelism.
