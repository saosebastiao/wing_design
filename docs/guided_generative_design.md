# Guided Generative Design — Sailing Wingsail

A free-rotating, unstayed solid carbon-fiber **wingsail** built as a **space-frame
of unidirectional CFRP beams** (straight or curved beams) wrapped and bound by a
**filament-wound CFRP skin** that simultaneously forms the airfoil surface and the
spar. The assembly is inserted into the boat hull like a keel-stepped mast and
rotates within its bearing-lined enclosure. The spar is the pivot axis for 360°+
rotation of the entire wing.

This document is the **design specification**. The incremental build sequence
that implements it lives in [`plan.md`](./plan.md).

> **Design philosophy (decided 2026-06-04).** The structural members are placed
> **on the outer shell**, not as an interior volumetric truss: filament-winding
> the skin around shell-following beams *is* what forms the structure. This
> supersedes the earlier Arora/Jiang interior-volumetric-truss method. The
> stress-aligned frame field is **retained** — initially as a diagnostic, and
> ultimately to drive non-uniform beam spacing and direction (see Phase F in the
> plan). Cross-sections are sized **FEA-in-the-loop**, and the wound skin is
> modeled as a **load-bearing shell**, not a fairing.

## Glossary of terms

| Term | Meaning |
| --- | --- |
| **wing** | The aerodynamic section, from root (`z = 0`) to tip (`z = span`). |
| **spar** | The cylindrical pivot shaft below the wing; the rotation axis and the part that steps into the hull. |
| **keel-step** | Lowest point of the spar, `z = -(spar_length + spar_transition_length)`. Bears against the hull bottom (the deepest mast-step analogy). |
| **deck-step** | Top of the cylindrical spar / deck-bearing level, `z = -spar_transition_length`. |
| **transition** | The morphing region between the root airfoil (`z = 0`) and the full spar circle (`z = -spar_transition_length`). (The code's `geometry.wing` currently calls this region the *fairing*; we unify on **transition**.) |
| **form beam** | A unidirectional-CFRP beam that runs along the OML (outer mold line) and both *defines* the aerodynamic shape and *carries* structural load. Aircraft analogy: a stringer / spar-cap. The doc previously called these "wing form beams." |
| **LE / TE spline** | The two distinguished, full-length form beams that run along the leading edge and trailing edge: tip → root → transition → spar → keel-step. |
| **skin** (a.k.a. *wrap*) | The filament-wound CFRP shell of thickness `shell_thickness` that forms the airfoil surface. **Load-bearing**, not just a fairing. |
| **beam arc** | The inward-facing circular arc — centered on a form-beam spline point, with endpoints on the OML — whose radius is solved so the resulting cross-section hits the area chosen by the sizing optimization. |
| **OML** | Outer mold line: the aerodynamic outer surface of the wing+transition+spar. |

> **Retired term — "virtual beam."** The original draft modeled the
> stretched-skin effect with infinite-tension "virtual beams" between adjacent
> form beams. Because the skin is now modeled as a real load-bearing shell in the
> FEA-in-the-loop sizing, this abstraction is no longer needed.

## Axis conventions

The wing is created in an `x, y, z` coordinate system:

- The wing **root** airfoil is drawn on the `x/y` plane at `z = 0`, with the pivot
  point at `x = 0, y = 0`.
- The wing **tip** airfoil is drawn on the `x/y` plane at `z = span`, also with
  the pivot point at `(x, y) = (0, 0)`.
- The wing **spar** is centered on `(x, y) = (0, 0)`.
  - The spar **keel-step** (lowest point) is at `z = -(spar_length + spar_transition_length)`.
  - The spar **deck-step** (highest point of the cylinder) is at `z = -spar_transition_length`.
- The wing **chord line** (root, tip, or any intermediate `z`) is always the `x` axis.

## Wing shape constraints

- **Center of pressure ahead of the pivot axis** → angle of attack passively
  decreases with increasing wind speed (passive feathering).
- **Center of gravity aft of the pivot axis** → angle of attack passively
  decreases when heeling.
- The **spar centerpoint is the pivot point** (`x = y = 0`).
- **Spar radius is derived, not stored.** It equals the radius of the maximum
  inscribed circle centered at the pivot within the root airfoil, **rounded down
  to a whole centimeter**. For the working scenario (NACA 0018, root chord 1.0 m,
  pivot at 25% chord) the controlling dimension is the airfoil half-thickness at
  the pivot station (≈ 0.089 m), so `spar_radius ≈ 0.08 m` (diameter 0.16 m).
  This becomes a method on the design dataclass, replacing the stored
  `spar_diameter`.
- **Fillets** (all sized at or above the minimum continuous-winding bend radius):
  - `spar_transition_fillet_r` — between the spar and the transition. Default **30 mm**.
  - `root_transition_fillet_r` — between the transition and the wing root. Default **50 mm**.
  - `te_fillet_r` — trailing-edge rounding that lets the filament wind continuously.
    A pointed TE would force continuous tow into a kink that destroys fiber
    strength; this is the minimum radius that preserves full fiber strength.
    Default **5 mm**.
- **Form beams always lie on the outer shell.** This is required so that
  filament-winding the skin forms the beam, it doubles as the primary structural
  support, and there must be enough of them to hold the aerodynamic shape (though
  some carry more load than others).

### Known design parameters

| Parameter | Value |
| --- | --- |
| `span` (root → tip) | 5.0 m |
| `wing_root_chord` | 1.0 m |
| `wing_root_airfoil` | NACA 0018 (symmetric, t/c = 0.18) |
| `wing_tip_chord` | 0.6 m |
| `wing_tip_airfoil` | NACA 0018 (symmetric, t/c = 0.18) |
| `spar_transition_length` | 0.5 m |
| `spar_length` | 1.0 m |
| `pivot_location` | ≈ 25% chord; tune fore/aft for a small, predictable passive-feathering moment |
| `material` (primary) | UD carbon / epoxy (E1 ≈ 130 GPa, anisotropic) |
| `spar_radius` | **derived** (see above) ≈ 0.08 m |
| `n_form_beams` | **16**, evenly spaced by arc-length around the OML perimeter, with LE and TE counted as two of them |
| `shell_thickness` | 3 mm |
| `spar_transition_fillet_r` | 30 mm |
| `root_transition_fillet_r` | 50 mm |
| `te_fillet_r` | 5 mm |

## Aerodynamic scenarios

- Full lift range of the foil (no need to exceed stall — we passively feather).
- Full range of reasonable sailing conditions (Reynolds numbers, `ncrit`).
- Full range of reasonable wind shear.
- Full range of reasonable apparent winds (wind shear + boat speed ⇒ variable AoA
  along the span).
- **Beaufort force 11 winds at −1° to +1° AoA** — the primary failure criterion.

## Structural requirements

- The structure must survive every scenario within the safety factor.
- The structure must not exceed tip / twist / shape-deflection tolerances.
- Track stress/strain and the principal stress directions at the mesh
  cell/node level.

## Performance metrics and constraints

**Aerodynamic:** moment about the pivot axis · lift-to-drag ratio · total lift.

**Structural:** tip deflection · twist deflection · airfoil-shape deflection.

## Pipeline

Steps 1–4 reuse the existing foundation (geometry, aero loads, 3D FEA, frame
field). Steps 5–10 are the shell-beam generative design.

1. **build123d — wing OML.** Build the wing outer solid for aerodynamic modeling.
   Aero only needs the wing section (root → tip), not the spar or transition.
   *(exists: `geometry.wing`)*
2. **AeroSandbox — load field.** Generate pressure fields and per-panel load
   points per load scenario; project onto the OML mesh.
   *(exists: `aero.model`, `aero.cases`, `aero.loads`)*
3. **3D linear FEA — Cauchy stress σ(x).** *(exists: `structural.mesh`,
   `structural.fea`)*
4. **Stress-aligned frame field** (à la Arora). *(exists: `truss.frame_field`,
   `truss.parametrization`)* Retained as a diagnostic now; drives beam
   spacing/direction in Phase F.
5. **build123d — form-beam splines on the full OML** (wing + transition + spar to
   keel-step):
   1. Build the full outer shell for structural modeling.
   2. Build the collection of 3D spline control points, all sampled **on** the
      OML:
      - The **LE spline** starts at the tip leading edge, proceeds to the root
        leading edge, then down the transition leading edge and the spar leading
        edge to the keel-step. The **TE spline** does the same along the trailing
        edge.
      - The remaining form beams are sampled at fixed perimeter fractions: at each
        `z` level, the beam's `(x, y)` point is evenly spaced by arc length around
        the OML cross-section at that `z`. `n_form_beams = 16` (including LE and
        TE), chosen to roughly match the aero chordwise pressure resolution.
      - Choose a set of `z` levels for the spline points (tip → root → transition
        → spar → keel-step).
   3. **Fit the splines** to the sampled points. Because every control point is
      sampled directly *on* the shell, an interpolating/least-squares cubic
      B-spline (e.g. `scipy.interpolate.splprep`) through those points lies on the
      shell to sampling tolerance. Fidelity is governed by the number of `z`
      levels and the smoothing tolerance — no separate NURBS weight optimization
      is required. (Earlier open question resolved this way.)
6. **FEA-in-the-loop cross-section sizing** *(replaces the original LP)*:
   1. Assemble a structural FEA model from the chosen spline geometry: **beam
      elements** along each form-beam spline (fixed/cantilevered at the
      keel-step) plus a **load-bearing skin shell** between adjacent beams
      (isotropic-equivalent first, anisotropic CLT later — supersedes the old
      fairing-only treatment).
   2. Optimize the per-beam cross-sectional area at each spline point to satisfy
      stress and tip/twist/shape-deflection constraints across **all** load
      cases. Start with fully-stressed-design / optimality-criteria iteration;
      promote to MILP over a discrete cross-section catalog (OR-Tools) later.
   3. Iterate until results are coherent and realistic.
   4. **Output:** an optimal cross-section area for every form-beam spline point.
7. **build123d — create the beams.**
   1. At each spline point, create an `x/y`-plane arc whose center is the spline
      point and whose endpoints lie on the OML, facing inward from the shell.
   2. Solve each arc's radius so its cross-section equals the optimized area from
      step 6.
   3. Loft from the bottom of each spline to the top, aligning the arc points, to
      form each beam.
8. **build123d — create the skin wrap.**
   1. At each `z` level, connect the beam-arc termination points to form a closed
      cross-section.
   2. Loft the per-`z` closed sections into a shell.
   3. Thicken the shell outward by `shell_thickness`.
9. **build123d — assemble** the beams and the skin wrap.
10. **Verification FEA** on the as-built assembly: solve under every load case,
    generate metrics and visualizations, and feed σ(x) back to step 4 to iterate
    toward a fixed point.

## Future optimizations & directions

Notes kept deliberately concrete so we can pick any of these up — or change
direction — without re-deriving the rationale. Roughly ordered by expected
payoff-to-effort.

### 1. Frame-field-driven beam layout (Phase F)
The stress-aligned frame field is computed but unused in the spike. Two levers:
- **Spacing:** replace arc-length-uniform perimeter spacing with a density that
  follows principal-stress magnitude — more beams over the spar/root and
  high-bending regions, fewer near the lightly loaded tip and TE. Implementation:
  integrate stress magnitude around each OML cross-section, then place beams at
  equal *cumulative-stress* intervals instead of equal arc length.
- **Direction:** add a second **helical/diagonal beam family** whose in-plane
  angle tracks the principal-stress direction of the frame field (the shear/
  torsion load path). This is the step that turns the structure from "spanwise
  stringers" into a true wound laminate and is the natural home for the retained
  Arora machinery. *Risk:* lofting non-spanwise beam arcs and keeping them
  windable is geometrically harder; prototype on a flat panel first.

### 2. Discrete cross-section catalog (MILP)
Continuous fully-stressed sizing yields arbitrary areas. Real pultruded/RTM
stock comes in discrete sizes. Promote step 6 to a MILP (OR-Tools, already a
dependency) that picks from a catalog and groups co-linear neighbors into single
stock pieces. Use FEA-derived sensitivities as the linear objective; re-solve FEA
between MILP rounds (sequential linear programming). See the `ortools-cp` skill.

### 3. Anisotropic skin (CLT) and winding-angle co-design
Replace the isotropic-equivalent skin with Classical Laminate Theory over the
actual wound fiber orientations (`materials.unidir` already has the ply data).
Then the **winding angle becomes a design variable** co-optimized with beam
areas — the skin's ±θ plies carry torsion while the beams carry bending. This is
where the structure stops being "beams + dead skin" and becomes one optimized
composite.

### 4. Skin-weight vs. beam-weight trade
With a load-bearing skin, there is a genuine mass trade between thickening the
skin and growing the beams. Add `shell_thickness` (and per-region thickness) to
the design vector and minimize total mass subject to the same constraints.
Expected outcome: thicker skin + fewer/smaller beams over torsion-dominated
spans, the reverse over bending-dominated spans.

### 5. Filament-winding path planner + manufacturability
Generate continuous winding passes that (a) form the airfoil surface and
(b) reinforce joints and buckling-prone members (IsoTruss-style node wrapping).
Output robot/G-code paths + a per-pass fiber-orientation map that feeds the CLT
of item 3. Add manufacturability constraints back into sizing: minimum bend
radius (the `te_fillet_r` rationale generalized), max fiber turn rate, no
bridging across concavities. *This closes the loop:* the winding plan's actual
fiber map becomes the FEA input, not an assumed one.

### 6. Cross-beam members, ribs, and reinforcements
The current model has only OML-following beams. Add generated internal ribs /
shear webs where buckling or panel deflection governs, and local reinforcement
plies at the spar-to-wing transition (the highest-moment region). Could be driven
by a buckling eigenvalue check in the verification FEA.

### 7. Richer load cases and physics
- **Buckling:** linear eigenvalue buckling on the as-built shell (skin panels
  between beams are the likely first mode).
- **Dynamics:** flutter / divergence (free-rotating unstayed wing is
  aeroelastically interesting), vortex shedding, gust transients.
- **Fatigue:** cyclic load spectrum from a sailing-condition distribution.
- **Impact / knockdown** beyond the static Beaufort-11 case.

### 8. Full design-loop closure & scaling
- **Fixed-point iteration:** feed verification σ(x) back into the frame field and
  re-run, to convergence (doc step 10 → step 4).
- **Outer optimization over global geometry:** today `pivot_frac`, taper profile,
  and planform are fixed inputs. Wrap the whole pipeline in an outer optimizer
  (pivot location for the passive-feathering target; taper for mass/lift) — the
  pipeline becomes the objective function.
- **Scaling:** parametrize span/chord/taper to retarget from the 5 m wingsail to
  ~1 m FPV drone wings and >100 m turbine blades (the repo's stated long-term
  goal). Validate that the shell-beam method's assumptions (winding feasibility,
  beam-count resolution) hold across scales.

### 9. Verification against the retired interior-truss method
Keep the Arora/Jiang volumetric-truss code reachable (even if off the main path)
so we can, once, compare mass and stiffness of an interior truss vs. the
shell-beam design for the same load cases — a sanity check that "beams on the
shell" is actually competitive, not just more manufacturable.

## Development conventions

1. Maintain a **single dataclass** holding all design parameters. A value that
   can be mathematically derived from another parameter is **not** stored — it is
   a method on the class (e.g. `spar_radius`). *(exists: `scenario.DesignParameters`)*
2. Maintain an instance of that class with the **default** design parameters.
   *(exists: `scenario.default_scenario()`)*
