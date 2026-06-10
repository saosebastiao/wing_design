# V.5 Distributed panel pressure + skin bending in the failure check: design

**Date:** 2026-06-10
**Status:** Approved (plan.md V.5; backlog Validity #4)

## Problem

All aero panel forces are lumped at the single nearest node and shell triangles
never see pressure; consequences (audit V#4): no panel lateral-pressure bending in
the skin failure check, no pressure–compression interaction. The 16×8 mesh has no
nodes between beam lines, so *local* panel bending under pressure is sub-mesh —
like the across-width buckling half-wave (V.3b), the honest fix is a closed-form
strip term, not mesh-resolved DKT moments.

## Design

### 1. Triangle-distributed load projection (global load path)

`project_panels_to_skin(model, panels, safety_factor)` — each aero panel force
goes a third each to the 3 nodes of the skin triangle whose centroid is nearest
(consistent CST lumping of uniform pressure), instead of snapping to one node.
Loads stay design-independent → adjoint untouched. Opt-in at the example/caller
level (recorded headlines keep the old projection until V.6).

### 2. Per-triangle pressure + strip-bending stress in the failure check

- `panel_pressure_per_tri(model, panels, safety_factor)` → (M,) q [Pa]: each aero
  panel's normal-force magnitude assigned to its nearest triangle, divided by
  triangle area (zero where no panel maps).
- Sub-mesh strip bending, long SS strip of width w under uniform q:
  M = q·w²/8 per unit length → fiber stress σ_b = 6M/t² = **0.75·q·w²/t²**.
- Skin failure metric (von-Mises path) becomes `vm_membrane + σ_b ≤ σ_allow`
  (conservative scalar superposition). Tsai-Wu mode keeps membrane-only this
  increment (strength margin is ~7×; recorded caveat).
- Sizer API: `size_beam_shell_laminate(..., panel_pressures=[q_lc, ...])` — one
  (M,) array per aero load case, caller-built; None = unchanged. With
  `accel_vectors`, combo i maps to aero case i // len(accels).

### 3. Gradient

σ_b is design-explicit only (∂σ_b/∂t = −2σ_b/t; q, w design-independent; membrane
stress thickness-independent). `grad_skin_vm` gains optional `qw2_tri`
(= 0.75·q·w²): active triangle = argmax(vm + σ_b), implicit term unchanged,
explicit −2·qw2/t³ on the active triangle's thickness-band DV. FD-validated.

### 4. Not in scope (recorded)

Pressure–compression buckling interaction (eigen machinery is the home);
DKT-moment inclusion (sub-mesh here; revisit with chordwise mesh enrichment);
Tsai-Wu bending strains.
