# V.4 Self-weight + inertial/heel load cases: design

**Date:** 2026-06-10
**Status:** Approved (plan.md V.4; backlog Validity #5)

## Goal

Size against gravity and rig accelerations, not aero alone. The medium design is a
~2-tonne, 22 m cantilever sized today against aero only; gravity heeled and
wave-slam/pitching accelerations (>1 g routine at a rig) are first-order at that
scale. Expected to move the honest medium baseline **up**.

## Design

### 1. Body loads — `beams/body_loads.py`

- `body_load_vector(model, radii, t_tri, rho, accel) -> (N, 6)`: lumped nodal
  forces from the design's own mass under acceleration `accel` (m/s², wing frame):
  beam element mass ρπr²L half to each end node; triangle mass ρ·t·A a third to
  each corner node. Forces only (no moment lumping — consistent with the aero
  projection). The tip gusset stays massless (recorded convention).
- `body_load_jacobian(model, group_of_element, band_of_tri, rho, accel, ...) ->
  dF (ndof, nx)`: ∂f/∂x. Radius groups: ∂m_e/∂r = 2ρπrL; thickness bands:
  ∂m_tri/∂t = ρA; layup fractions: 0 (mass fraction-independent).

### 2. Sizer integration — `laminate_sizing.py`

- `LaminateSizingConfig.accel_vectors: tuple[(ax,ay,az), ...] = ()` — empty =
  behavior unchanged (back-compat; recorded headlines unaffected).
- Load combos = cross product `load_arrays × accel_vectors` (or aero-only when
  empty): `f = aero_lc + body_load_vector(x, accel)` rebuilt **every evaluate**
  (the load depends on the design). Caller composes heel/slam into the accel
  vectors (e.g. 30° heel: g·(sinθ ŷ − cosθ ẑ) in the wing frame).

### 3. Adjoint correction — the `λᵀ·∂f/∂x` term

For g(u(x), x) with K(x)u = f(x): dg/dx = ∂g/∂x + λᵀ(∂f/∂x − ∂K/∂x·u),
Kλ = ∂g/∂u. The code today implements only −λᵀ(∂K/∂x)u. Every `grad_*` in
`sensitivity.py` (and `_beam_vm_grad_one`) gains an optional `dF=None` parameter:
when given, add `lam @ dF` to the implicit part. `_build_jac` assembles one dF per
accel vector per design point (dense ndof×nx, ~1 MB at medium scale) and threads
it through with the matching factored load case.

### 4. Tests (tiny mesh, FD-validated per convention)

- `body_load_vector`: total force = total structural mass × accel; conservation
  under node lumping.
- Every gradient with a design-dependent load FD-validates: re-solve with
  f(x ± h) including the body load (central difference, rel-err ≤ ~1e-4), at
  minimum for one beam-row constraint, deflection, and panel buckling.
- End-to-end: small sizing with gravity converges feasible; mass ≥ aero-only mass.

### 5. `examples/47_self_weight.py`

Medium 16×8 (strip mode — V.6 direction), aero-only vs aero × {upright g,
30° heel g, heel + 1 g lateral slam}: Δmass, governing constraints, wall-clock.

## Non-goals

- Dynamic/transient slam (a static equivalent acceleration only, V#12 owns the
  envelope question).
- Aeroelastic re-trim of heeled aero loads (Phase H).
