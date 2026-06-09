# Analytic (adjoint) constraint Jacobian: design

**Date:** 2026-06-08
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Give the SLSQP laminate sizer analytic gradients for the objective AND all FEA-dependent
constraints, so scipy never finite-differences the design vector. This eliminates the
~`n_DV` full FEA solves per iteration (the dominant cost; ≈60× fewer full solves) and
improves convergence. Method: the **adjoint** method, exact, validated against finite
differences. Gated by a `use_analytic_jacobian` flag.

## Motivation

The sizer's runtime is dominated by SLSQP finite-differencing the constraints: each FD
column perturbs `x` and re-runs the full beam-shell FEA solve (cached per `x`, shared
across constraints) → ~`n_DV` solves per iteration. Because every FEA-dependent
constraint shares that one solve, the FD cost is incurred unless **all** of them have
analytic gradients (partial analytic buys nothing on wall-clock). With ≈6 constraints
and ≈60 DVs, the **adjoint** method computes every gradient with ~6 back-substitutions
per iteration (one per constraint), reusing a single factorization — far cheaper than
direct sensitivity's ~60.

## Decisions (from brainstorming)

- **Full analytic**, all six FEA constraints (all-or-nothing for the speedup).
- **Adjoint** method (few constraints, many DVs).
- **Validation-first:** an FD-validation harness checks every analytic gradient; build it
  first, implement each gradient to pass it.
- **`use_analytic_jacobian` flag retained** (per user). Default **False** (opt-in) so the
  existing FD path — the correctness reference — is unchanged; sizing examples/the
  measurement set it True. (The default may flip to True in a later increment once
  trusted.)

## Background: the FEA and constraints

Linear system `K(x)·u = f` (`f` = nodal loads, independent of `x`), clamped at the keel
(`fixed_nodes`); solved on the free DOFs `K_ff u_f = f_f`. `K` is symmetric. Design
vector `x = [r_group(G), t_band(B), f0_grp(L), f45_grp(L)]`. The sizer's scalar
constraints (each ≥ 0 feasible), max/min-aggregated over elements:
- `beam_con` = 1 − max_e(beam vM_e)/σ_allow
- `skin_con` = (von-Mises) 1 − max_tri(skin vM)/σ_allow  OR  (Tsai-Wu) min_tri R/SF − 1
- `defl_con` = 1 − max_tip‖u_trans‖/defl_lim
- `twist_con` = 1 − max_tip|u_twist|/twist_lim
- `beam_buck_con` = 1 − max_e(Euler util_e)        (when buckling on)
- `panel_buck_con` = 1 − max_tri(panel util)       (when buckling on)
- `frac_con`, `monotonic_con` — algebraic in `x` (no FEA); trivial constant Jacobians.
The objective `mass(x)` is already analytic (geometric; no `u`).

## Design

### 1. Adjoint gradient

For a scalar constraint `g` whose active member is fixed at the current point:
`dg/dxᵢ = ∂g/∂xᵢ|_explicit − λᵀ (∂K/∂xᵢ) u`, with the adjoint `λ` solving `K_ff λ_f =
(∂g/∂u)_f` (reusing `K`'s factorization; `λ` zero on fixed DOFs). One factorization per
design point serves the nominal solve and all adjoint solves.

### 2. Factorization-reusing solver — `structural/beam_shell.py`

Add a sensitivity-enabled solve (e.g. `solve_beam_shell_laminate_factored(...)`) that
returns, alongside the `FrameResult`:
- the `splu` factorization of `K_ff` (so the adjoint solves are back-subs),
- `free` DOF index and `ndof`,
- per-beam-element `(kloc, T, dofs, i, j, L)` and per-triangle `(dofs, node ids, B-matrix
  data / local frame)` needed to assemble `∂K/∂xᵢ` and the constraint partials.
The existing `solve_beam_shell_laminate` stays as-is (FD path / back-compat); the new
entry shares its assembly (refactor the common assembly into a helper to avoid drift).

### 3. `∂K/∂xᵢ` via element-routine reuse — `beams/sensitivity.py` (new)

Element stiffness is **linear** in its section/material coefficients, so each derivative
reuses the existing element routine with derivative inputs:
- **radius group `r_g`:** `local_beam_stiffness` with derivative section
  `(∂A,∂I,∂J) = (2πr, πr³, 2πr³)`, summed over the group's beam elements.
- **thickness band `t_b`:** `tri_element_stiffness_laminate` with `∂A/∂t = Qeff`,
  `∂D/∂t = (t²/4)·Qeff`, over the band's triangles.
- **layup `f0/f45`:** `∂Qeff/∂f0 = Qbar0 − Qbar90`, `∂Qeff/∂f45 = ½(Qbar45+Qbar₋45) −
  Qbar90`; `∂A,∂D = (t, t³/12)·∂Qeff`; over the band's triangles.
`λᵀ(∂K/∂xᵢ)u = Σ_{affected e} λₑᵀ (∂kₑ/∂xᵢ) uₑ` — a few local 12×12 / 18×18 products.

### 4. Per-constraint partials `∂g/∂u`, `∂g/∂x|_explicit`

- **defl / twist** — functionals of `u` only: `∂g/∂u` nonzero at the active tip node's
  translation / twist DOFs; no explicit `x` term. (Simplest — implement & validate first.)
- **beam vM** — `vM = f(floc)`, `floc = kloc·T·uₑ` at the active element: `∂g/∂u =
  (∂vM/∂floc)·kloc·T` mapped to that element's DOFs; explicit `∂vM/∂r` (section area /
  modulus). Derive `∂vM/∂floc` from `von_mises_per_element`.
- **beam Euler buckling** — `util = comp·SF/Pcr`, `comp = max(0,−axial)`,
  `Pcr ∝ E·I(r)/(KL)²`: `∂/∂u` via `∂axial/∂uₑ`; explicit `∂/∂r` via `I(r)`.
- **panel buckling** — `util = comp_stress·SF/σcr`, `σcr ∝ D11(t,f)/(b²·t)`: `∂/∂u` via
  membrane stress = `C·B·u`; explicit `∂/∂(t,f)` via `D11` and the `1/t`.
- **skin vM / Tsai-Wu** — stress `= C·B·u` at the active triangle (and active ply for
  Tsai-Wu): `∂g/∂u = (∂g/∂stress)·C·B`; explicit `∂C/∂(t,f)`. Tsai-Wu reuses
  `materials.failure` (strength-ratio derivative w.r.t. material-axis stress).
Each gradient is the **active member's** gradient (a valid subgradient of the max/min
for SLSQP).

### 5. Sizer wiring + flag — `beams/laminate_sizing.py`

- Add `LaminateSizingConfig.use_analytic_jacobian: bool = False`.
- When False: today's behavior exactly (no `jac` on constraints → scipy FD; the new code
  path is dormant).
- When True: a single `evaluate_jac(x)` (reusing the cached factored solve) produces, for
  the current `x`, the gradient rows for the objective and every constraint; each scipy
  constraint dict gets a matching `jac=`. `frac_con`/`monotonic_con` get their constant
  Jacobians. The objective keeps `mass_grad`.
- Everything downstream (multi-start, tip-gusset, banding, datum, per-band layup) inherits
  the speedup when the flag is on.

### 6. Validation-first + measurement

- **FD-validation harness (built first):** `tests/beams/test_sensitivity.py` checks every
  analytic gradient against central finite differences at representative design points to
  tight relative tolerance — objective grad, each `∂K/∂x` type (via a directional check),
  and each constraint's full `dg/dx` row. A wrong gradient fails its test; this harness
  makes the large derivation self-correcting and is the spec's primary safety net.
- **Equivalence:** an opt-in analytic sizing run reaches the same feasible optimum as the
  FD run on a small problem (allclose mass/feasible), proving the wiring.
- **Measurement:** a small-problem timed comparison (analytic vs FD) reporting solves/iter
  and wall-clock — confirms the speedup. (No multi-hour full run required to prove it.)

## Components / file map

| File | Responsibility | Change |
|---|---|---|
| `structural/beam_shell.py` | factorization-returning sensitivity solve (shared assembly helper) | modify |
| `beams/sensitivity.py` | `∂K/∂x` assemblers + adjoint solves + per-constraint gradient rows | **new** |
| `beams/laminate_sizing.py` | `use_analytic_jacobian` flag; `evaluate_jac`; per-constraint `jac=` wiring | modify |
| `beams/__init__.py` | export sensitivity entry points used by tests/examples | modify |
| `tests/beams/test_sensitivity.py` | FD-validation harness (objective, ∂K/∂x, every constraint) | **new** |
| `tests/beams/test_analytic_sizing.py` | analytic-vs-FD equivalence on a small problem | **new** |
| `examples/38_analytic_speedup.py` | analytic-vs-FD timed comparison (small problem) | **new** |
| `docs/plan.md` | finding (at close) | modify |

## Success criteria

- `use_analytic_jacobian=False` is byte-identical to today (FD path untouched).
- With `use_analytic_jacobian=True`, every analytic gradient matches central FD to tight
  tolerance (validation suite green), and an analytic sizing run reaches the same feasible
  optimum as FD on a small problem.
- Measured: analytic uses ~`n_DV`× fewer full FEA solves per iteration and is materially
  faster wall-clock on the small problem.
- Full suite green.

## Out of scope (deferred)

- Flipping the default to `use_analytic_jacobian=True` (separate decision once trusted).
- Sensitivities for the legacy `size_beam_shell` / `size_beams` sizers.
- Second-order (Hessian) information; analytic Jacobian only.
- Parallelizing anything (the later parallel-multi-start item is separate).
- Sensitivity w.r.t. geometry/placement (`arc_fractions`, node positions) — only the
  radii/thickness/layup design variables.
