# P#1b Hollow straight form-beam segments: design

**Date:** 2026-06-10
**Status:** Approved (redirect of P.1 after the core-tube negative; backlog P#1b,
M#4 path classification)

## Measured basis

- P.1 corrected verdict (bonds fixed): the core tube earns −8.3% as a TORSION
  spine (twist un-binds) while staying minimal — centroidal BENDING leverage is
  confirmed nil, so the hollow-section bending lever must act on the form beams
  themselves (beam-buck SF +455 kg/SF on the new 2060.7 kg running best).
- **Straightness probe (2026-06-10): all 16 medium form-beam paths deviate 0.0 mm
  from straight over the in-wing span (z ≈ 3.3 → 22 m, 18.7 m length)** — the
  linearly-tapered planform is a ruled surface, so fixed-arc-fraction beams are
  exactly straight; windable-hollow per the manufacturing concept. The two
  below-root transition segments curve into the spar and stay solid RTM.

## Model (reuses the P.1 annular machinery)

1. **Hollow eligibility per element:** both endpoint levels in-wing (z ≥ 0 at
   node resolution). The model tags `hollow_elements` (form rows) at build time;
   straightness asserted (≤1 mm tolerance) rather than assumed.
2. **DVs:** outer radius stays the existing `r_group` blocks (mirror-pair ×
   segment, monotonic taper unchanged). NEW block `("t_hollow", H)` — one wall
   thickness per (radius-group, in-wing-segment) combo, same mirror sharing.
   Solid is recoverable continuously at t = r (annular(r, r) ≡ circular(r));
   bounds t ∈ [1 mm, r_max], trial points clamped t ≤ r like the tube.
3. **Sections:** hollow elements get `BeamSection.annular(r_e, min(t_e, r_e))`,
   transition elements stay `circular(r_e)` — same mixed-sections list the P.1
   solver path already handles.
4. **Constraints:** existing vM + I-based Euler cover hollow elements; the P.1
   wall-crimping check extends over all annular elements (vector rows = H groups,
   binding element/combo per row); annulus validity r − t ≥ 0 per hollow group
   (linear, couples the r_group and t_hollow columns).
5. **Sensitivities:** generalize the P.1 tube fields in `DesignSens`/`SensCache`
   to "annular elements with (r_col, t_col)": for hollow form elements r_col =
   the element's existing radius-group column (dkloc_annular wrt r replaces
   dkloc_dr there), t_col = its t_hollow column. Tube remains a special case of
   the same description. All FD-validated per convention.
6. **Body loads/mass:** annular areas where hollow (the V.4 dF gains the t_hollow
   columns; r_group columns switch to the annular dA/dr for hollow elements).

## Measurement

`examples/51_hollow_beams.py`: medium V.6 config ± hollow form beams,
eigen-verified, vs the 2248.0 kg baseline. Expectation (honest): beams are 812 kg
of 2261 kg with Euler binding — the (R/t) leverage now acts where the constraint
lives; report whatever happens including the splice/transition caveat (joints
idealized — M#5 records the wrapped-joint model as future work).

## Non-goals

Splice-joint stiffness/strength (M#5); ovalization of hollow beams under skin
crush (V#7); winding-angle layup of the hollow walls (tube/beam material = beam E).
