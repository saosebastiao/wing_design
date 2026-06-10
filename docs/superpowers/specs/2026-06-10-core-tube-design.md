# P.1 Hollow members — filament-wound core tube: design

**Date:** 2026-06-10
**Status:** Approved (plan.md P.1; backlog Performance #1a — "likely the biggest
single lever"; manufacturing concept: permanent structural mandrel)

## Why this lever, why now

Beams are solid rods (A=πr², I=πr⁴/4) and the V.6 baseline's binding economics
moved to them: beam-buckling SF carries **+531 kg/SF** vs +68 for panels. A
thin-walled tube has ~(R/t)× the I of a solid rod at equal mass. The build allows
hollow only on **completely straight** members — the core tube along the spar/pivot
axis is the prime candidate: wound to any radius that fits the section, radius AND
wall free to taper along span, carries spanwise bending/torsion as a closed section,
beams/webs bond to it.

## Model

### 1. Section: `BeamSection.annular(r_outer, t_wall)`

A = π(2·r·t − t²), Iy = Iz = (π/4)(r⁴ − (r−t)⁴), J = 2I, fiber distance = r_outer.
Exact annulus (no thin-wall approximation in stiffness/stress);
`von_mises_per_element` already consumes per-element sections unchanged.

### 2. Geometry: tube in `build_beam_shell_model(..., core_tube=True)`

- One tube node per z-level on the pivot axis (the rotation axis the spar already
  uses); tube elements connect consecutive levels — straight by construction.
- **Bond representation:** stiff massless connector elements from each tube node to
  the nearest few (default 4) form-beam nodes at that level — the tip-gusset
  pattern: constant stiffness, excluded from force recovery and mass, composes with
  the analytic adjoint (design-independent K). NOT rigid diaphragms at every level
  (would fake ovalization restraint): few-point local bonds.
- **Fit bound:** r_tube(z) ≤ 0.95 · min over beams of dist(node(b,z), pivot axis) —
  the inscribed-circle bound sampled from the model's own section nodes
  (conservative via the 0.95).
- Model gains: `tube_elements` (rows in `beam_elements`), `tube_segment_index`,
  per-level fit bounds. Tube elements ride in the SAME `beam_elements`/`sections`
  arrays — the solver is unchanged.

### 3. Design variables (via P.0 `DesignVector` — one append each)

`("r_tube", S)`, `("t_wall", S)` with S = n_levels − 1 segments. Bounds:
r ∈ [2 cm, fit bound per segment], t ∈ [1 mm, r_max]. Linear specs: annulus
validity r − t ≥ 0 per segment; monotonic non-increasing r keel→tip (winding
taper, same pattern as beam radii). Tube DVs are NOT in the mirror-symmetry
groups (the tube is on the symmetry axis).

### 4. Constraints (via P.0 `ConstraintSpec` — one append each)

- Tube elements join the EXISTING beam vM + element-length Euler checks (annular
  section properties; Euler check generalized to take per-element A, I instead of
  recomputing from solid-rod radii).
- **NEW wall crimping/local buckling:** axial-compression cylinder buckling
  σ_cr = γ·0.605·E·t/r with γ = 0.65 (NASA SP-8007-class knockdown, recorded);
  utilization = SF·σ_axial_comp/σ_cr per tube element. Shear/torsion buckling and
  bending-gradient relief deferred (recorded caveats).

### 5. Sensitivities (FD-validated per convention)

Annulus derivatives: dA/dr = 2πt, dA/dt = 2π(r−t), dI/dr = π(r³−(r−t)³),
dI/dt = π(r−t)³, dJ = 2dI — `local_beam_stiffness` is linear in (A, Iy, Iz, J), so
`dkloc_dr_tube` / `dkloc_dt_tube` follow the existing `dkloc_dr` pattern.
SensCache gains tube ∂K entries; mass gradient gains the two blocks; the wall-
crimping gradient follows the beam-Euler-gradient pattern (axial force adjoint +
explicit r, t terms). Body loads (V.4): tube mass lumps like beam mass; ∂f/∂x gains
the two tube columns.

## Measurement plan

`examples/50_core_tube.py`: medium V.6 config ± core tube (same loads, constraints,
1-band, strip widths, gravity); report Δmass, what binds, eigen verification of the
tube optimum (λ_cr ≥ 1.5 — the eigen solve sees the tube as beam elements
automatically). Small wingsail spot-check. Success per backlog: −10–15% plausible;
honest report either way; if the optimizer zeroes the tube (r→min everywhere),
that's a negative finding with the same rigor.

## Non-goals (this increment)

Straight in-wing hollow beam segments (P#1b — after the tube verdict); wrapped-
joint stiffness/strength at tube-beam bonds (M#5 — connectors are idealized stiff);
Brazier/ovalization interaction (V#7); winding-angle constraints on the tube layup
(tube uses the beam material E for now — recorded).
