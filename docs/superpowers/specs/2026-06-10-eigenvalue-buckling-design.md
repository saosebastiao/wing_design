# V.3 Eigenvalue buckling check (K + Kσ): design

**Date:** 2026-06-10
**Status:** Approved (plan.md V.3; backlog Validity #1/#2 — the single
highest-leverage validity item)

## Goal

A linear (eigenvalue) buckling solve of the **as-optimized** beam+shell structure:
assemble the geometric stiffness `Kσ(σ₀)` from each load case's linear stress state
and solve `K φ = −λ Kσ φ`; the smallest positive λ is the load multiplier to
buckling. This validates — or convicts — *both* binding closed-form checks at once
(beam element-length Euler, panel `b = √area` plate formula), sees combined membrane
stress states, beam–panel interaction, and **global modes** (ovalization/crimping)
that no closed-form check can, and prices the hoop-pretension prize properly (the
prestress state simply adds into the `σ₀` entering `Kσ`).

**Post-processing check first, NOT in the sizing loop.** The verdict (λ_cr vs the
closed-form prediction) decides what the in-loop constraint should become (P.0
refactor consumes that shape); eigen-gradients are a later, separate step (clean hand
formula `φᵀ(∂K/∂x + λ ∂Kσ/∂x)φ` — backlog Toolbox #7).

## Why now (measured motivation, 2026-06-10)

- V.1 shadow prices: ALL the binding-requirement mass sits on the two closed-form
  buckling SFs (+266/+152 kg per SF unit on the headline) — their fidelity is the
  design's fidelity.
- V.2: headlines are not mesh-converged (−9–11% per refinement step, no plateau)
  *because* both closed-form checks gain capacity from refinement (shorter element
  L, smaller triangle b). The eigenvalue solve has no such mesh pathology: capacity
  comes from the physical stiffness + stress field.

## Interpretation contract

Every load case is solved at its **factored** (SF-included) level already
(`project_panels_to_beam_nodes(..., safety_factor)`); the sizing buckling checks
then require capacity ≥ 1.5× that demand. So for the optimum to be *validated*,
each load case needs **λ_cr ≥ 1.5** (with the binding closed-form check exactly at
util = 1.0, λ_cr ≈ 1.5 means the closed-form level was exact; λ_cr > 1.5 means
conservative → mass on the table; λ_cr < 1.5 means unconservative → honest baseline
moves up).

## Design

### 1. `structural/geometric_stiffness.py`

- `beam_geometric_K(N, L) -> (12,12)` — consistent Euler–Bernoulli geometric
  stiffness for axial force `N` (tension-positive), standard Przemieniecki form in
  both bending planes (36/3L/4L²/−L² pattern × N/30L), zero axial/torsion rows.
  Wagner/torsional-flexural coupling terms neglected (circular sections, documented).
  Global: `T_eᵀ Kg T_e` with the existing `_element_rotation` transforms.
- `tri_geometric_K(local_coords, area, t, sigma_local) -> (18,18)` — plate/membrane
  geometric stiffness on the **transverse deflection field only**, linear (CST-style)
  w-interpolation: `Kg_w = t·A · Gᵀ [σ] G`, where `G (2×3)` maps nodal w to the
  constant ∇w (same b/c coefficients as the membrane B-matrix) and `[σ]` is the 2×2
  local membrane stress tensor. Scattered into the 3 local-w DOFs, rotated to global
  via the triangle frame `R`. In-plane geometric terms and DKT-consistent w
  interpolation omitted (standard faceted-shell practice; documented limitation —
  benchmark accuracy against the SS-plate reference decides if it's enough).

### 2. `structural/eigen_buckling.py`

- `assemble_geometric_stiffness(nodes, beam_elements, sections, shell_tris, t_tri, result_stress) -> Kσ (sparse)`
  from a solved state: beam axial forces (existing recovery) + per-triangle local
  membrane stress (existing `recover_membrane_stress_C`, which returns local-frame
  σ; multiply by t for resultants).
- `linear_buckling(K_ff, Kσ_ff, k=6) -> (λs, modes)` — **dense**
  `scipy.linalg.eigh` on the free DOFs (medium 16×8 ≈ 800 free DOFs — dense is
  trivial and robust; sparse shift-invert deferred until a measured need). Solve the
  symmetric generalized problem on the compressive part; return the k smallest
  positive load multipliers and mode vectors. Modes exportable to `.vtu` for
  ParaView (`just shot`) — global vs local mode identification is a deliverable.
- `K_ff` reused from the existing assembly (`solve_beam_shell_laminate_factored`
  exposes `K_ff`/`free`).

### 3. Validation tests (`tests/structural/test_eigen_buckling.py`)

Reference solutions, FD-free (eigen checks validate against closed form):
- Pinned–pinned multi-element beam column: λ·N = π²EI/L² (within ~1–2% at 8
  elements); cantilever: π²EI/(2L)².
- Simply-supported square plate, uniaxial compression, triangulated: σ_cr ≈
  4π²D/(b²t) (document achieved accuracy; the w-linear Kg converges from above).
- Sanity: all-tension state → no positive λ below a large cap.

### 4. `examples/45_eigen_buckling.py`

Size (or warm-load) the medium optimum, then per load case: λ_cr + mode class
(beam-local / panel-local / global), VTU export of the worst modes. Verdict table
λ_cr vs 1.5. Then the **pretension parametric**: superpose hoop prestress σ_p ∈
{0…100 MPa} into the σ₀ of the worst case and report λ_cr(σ_p) — the eigen-grade
answer to the example-44 probe.

## Non-goals (this increment)

- In-loop eigen constraint + gradients (next increment, after the verdict).
- Imperfection knockdowns (NASA SP-8007) — applied as reported *interpretation*,
  not solver features.
- Brazier/nonlinear collapse (V#7), self-weight load cases (V.4).
