# Wing Design — Development Plan

The design specification lives in
[`guided_generative_design.md`](./guided_generative_design.md). This file is the
**incremental build sequence** that implements it.

## Goal

A free-rotating, unstayed solid carbon-fiber **wingsail** built as a **space-frame
of unidirectional CFRP beams** wrapped and bound by a **filament-wound CFRP skin**
that simultaneously forms the airfoil surface. The structural members lie **on the
outer shell** — filament-winding the skin around shell-following beams *is* what
forms the structure. Beam cross-sections are sized **FEA-in-the-loop**, and the
wound skin is modeled as a **load-bearing shell**.

The repo is meant to evolve from a 5 m demo wingsail to a process applicable from
~1 m FPV drone wings up to >100 m wind turbine blades.

## Direction change (2026-06-04)

The original plan generated an **interior volumetric truss** (Arora stress-aligned
frame field → global parametrization → isocurve tracing → Jiang ALP). That method
is **retired as the structural approach** in favor of **shell-following form
beams** sized against the real FEA. Decisions taken:

| Decision | Choice |
| --- | --- |
| Structural concept | **Beams on the OML** (shell-following), not an interior volumetric truss. |
| Frame field | **Kept.** Diagnostic now; drives non-uniform beam spacing + direction in Phase F. |
| Beam layout (early) | **Even** arc-length spacing around the OML perimeter. |
| Cross-section sizing | **FEA-in-the-loop** (fully-stressed / optimality-criteria; MILP catalog later). |
| Skin role | **Load-bearing structural shell** (supersedes the earlier fairing-only decision). |
| Build-out strategy | **Thin end-to-end spike first**, then deepen weak links. |

Retired modules (interior-truss path): `truss.extract` isocurve tracing, ALP,
manufacturability filters. The frame-field/parametrization code survives for
Phase F. See [`guided_generative_design.md` → Future optimizations](./guided_generative_design.md#future-optimizations--directions)
item 9 for keeping the old path reachable as a comparison baseline.

## Working specification (first design point)

| Parameter | Value |
| --- | --- |
| Span | 5.0 m |
| Root chord | 1.0 m |
| Tip chord | 0.6 m |
| Airfoil | NACA 0018 (symmetric, t/c = 0.18) |
| Spar length / transition | 1.0 m / 0.5 m below root, free 360°+ rotation |
| Spar radius | **derived** (max inscribed circle at pivot, floored to cm) ≈ 0.08 m |
| Pivot location | ≈ 25% chord (passive feathering, refine later) |
| `n_form_beams` | 16 (incl. LE + TE), even arc-length spacing |
| `shell_thickness` | 3 mm |
| Material (primary) | UD carbon / epoxy (E1 ≈ 130 GPa, anisotropic) |

## Pipeline (target end state)

```
   build123d            AeroSandbox            3D linear FEA
 wing OML solid  ──►   pressure field   ──►   Cauchy stress σ(x)
                         per load case               │
                                                     ▼
                                    stress-aligned frame field R(x)
                                  (diagnostic now; layout driver later)
                                                     │
       ┌─────────────────────────────────────────────┘
       ▼
 form-beam splines ON the OML  (LE + TE + 14 arc-spaced, full wing→spar)
       │
       ▼
 FEA-in-the-loop sizing:  beam elements + load-bearing skin shell
   optimize per-beam cross-section areas vs. stress + tip/twist limits
       │
       ▼
 build123d:  inward beam arcs sized to area → loft beams
             connect arc endpoints → loft skin → thicken
       │
       ▼
 assembly  ──►  verification FEA  ──►  iterate to fixed point
```

## Phases

Each phase ends in a runnable example under `examples/` and a passing smoke test.
We spike end-to-end at low fidelity before deepening any one phase.

### Foundation — Phases 1–5 (done)

Kept as-is from the prior build; these are the inputs to the shell-beam pipeline.

- **Phase 1 — Wing geometry.** `geometry.airfoil` (NACA 4-digit symmetric),
  `geometry.wing` (lofted OML + spar/transition). `examples/01_wing_solid.py`.
- **Phase 2 — Aero loads.** `aero.model` (ASB `Wing` from `WingSpec`),
  `aero.cases` (load-case envelope), `aero.loads` (per-panel → OML traction).
  `examples/02_aero_envelope.py`.
- **Phase 3 — Coupled beam baseline.** `materials.unidir` (UD ply + CLT),
  `structural.beam` (tapered-tube `Opti` sizing). `examples/03_spar_sizing.py`.
  *Mass/stress/tip-deflection baseline every later phase must beat.*
- **Phase 4 — Volumetric FEA.** `structural.mesh` (gmsh tets),
  `structural.fea` (linear elastic σ(x)), `structural.shell`,
  `structural.projection`. `examples/04_volumetric_fea.py`.
- **Phase 5 — Stress-aligned frame field.** `truss.frame_field`,
  `truss.parametrization`, `truss.streamline`, `truss.surface_streamlines`.
  `examples/05_stress_lines.py`, `06_frame_field.py`, `15_shell_stress_lines.py`.
  **Retained as diagnostic; reused as the Phase F layout driver.**

### Phase A — Form-beam geometry spike

First step of the new pipeline: get shell-following beams + skin + assembly
built end-to-end at crude fidelity.

- `wing_design.beams.splines` — sample the **full** OML (wing + transition + spar
  to keel-step) at chosen `z` levels: LE spline, TE spline, and 14 arc-spaced
  splines (even arc-length around each cross-section). Fit cubic B-splines through
  the on-surface points (`scipy.splprep`).
- `wing_design.beams.build` — at each spline point, inward beam arc with a
  **fixed crude radius**; loft each beam bottom→top.
- `wing_design.beams.wrap` — connect beam-arc endpoints per `z` level → loft →
  thicken by `shell_thickness`.
- `wing_design.beams.assembly` — beams + skin into one assembly; export STEP.

**Deliverable: `examples/20_form_beams.py` → STEP of the shell-beam wingsail
(unsized).**

### Phase B — Structural-eval spike

- `wing_design.beams.fea_model` — assemble a structural FEA model from the
  splines: **beam elements** along each form-beam (cantilevered at the keel-step)
  + **load-bearing skin shell** between beams, isotropic-equivalent properties.
- Solve under the existing `aero.cases` load cases; report stress, tip
  deflection, twist.

**Deliverable: `examples/21_beam_fea.py` → per-load-case stress/deflection
metrics for the unsized structure.**

**Status: done (2026-06-04).** Beam elements implemented as a fresh in-house 3D
Euler–Bernoulli frame solver (`structural.frame`); the skin's transverse shear
transfer is stood in for by **ring connectors** between adjacent beams at each
z-level, not yet a coupled shell (deferred to a later deepen). Loads applied via
full panel-force projection onto the nearest beam node (`beams.fea_model`). The
unsized 20 mm-radius frame is far over-stiff (tip deflection ≈ 0.2 % span, member
σ_vm ≲ 20 MPa vs. ~400+ MPa allowable) — the baseline Phase C thins down.

### Phase C — FEA-in-the-loop sizing spike

- `wing_design.beams.sizing` — fully-stressed-design / optimality-criteria loop:
  set each beam's per-station cross-section area from its computed stress, re-solve
  Phase-B FEA, iterate to a stress/deflection-feasible structure; minimize mass.

**Deliverable: `examples/22_sized_beams.py` → sized assembly with mass that beats
the Phase-3 tube-spar baseline, all constraints satisfied.**

**Status: done (2026-06-04).** Per-element longitudinal radii sized by SLSQP
(`beams.sizing.size_beams`) minimizing beam mass s.t. per-element von Mises +
tip-deflection + tip-twist constraints, re-solving the Phase-B frame FEA each
step (rings held at a fixed minimum radius; their stress is reported, not sized).
Deliverable: `examples/22_sized_beams.py`. Caveat logged: the tube-spar baseline
is a single bending member, so its mass is reported for context but is not
directly comparable to the full form-beam frame.

**Key finding:** the sized frame is **stiffness-driven, not strength-driven** —
tip deflection sits exactly on its limit while the worst longitudinal stress is
~42 MPa against a 1100 MPa allowable (≈26× margin). So sizing UD beams to a
deflection target leaves them hugely over-strength; the mass-efficient stiffness
lever is the **load-bearing skin** (Phase D shell coupling), not thicker beams.
(SLSQP needed objective/constraint normalization to O(1) — the raw Pa-vs-metre
scale gap made the QP ill-conditioned and the solver quit early at an infeasible
point.)

### Phase D — Deepen geometry

- Promote `spar_diameter` → derived `spar_radius()` (max inscribed circle at
  pivot, floored to cm) on `WingSpec`.
- Add fillets: `spar_transition_fillet_r` (30 mm), `root_transition_fillet_r`
  (50 mm), `te_fillet_r` (5 mm, continuous-winding limit).
- Spline-fit fidelity: tune `z`-level count + smoothing tolerance against an
  on-surface error metric.
- Solve beam-arc radius for the **target area** from sizing (replace Phase-A's
  fixed crude radius).

**Deliverable: manufacturable, correctly-filleted geometry whose beams hit their
sized areas.**

**Status: done (2026-06-05).** `spar_radius` derived (80 mm / 160 mm dia at the
pivot). Beams built as inward-arc **lens** sections sized to the Phase-C radii
(`beams.build_sized_lens_beams`; circular fallback unused — lens lofts cleanly,
16/16 beams). `examples/23_sized_geometry.py` runs the full Phase-C SLSQP sizing
(mass ≈ 45 kg, radii 4–40 mm) and exports `exports/sized_geometry_v0.step`.
The TE fillet (`te_fillet_r`) is deliberately not attempted: the sharp trailing
edge is a continuous-winding feature handled in manufacturing, not a solid round.

**Deferred items resolved (2026-06-05).**
(1) *Manufacturable junctions by construction.* Post-hoc filleting proved
impossible — build123d 0.10.0's OCC kernel cannot blend the morphing airfoil→circle
loft at any radius (`max_fillet` fails outright), though `fillet` works on a plain
box. The cause is a slope crease where the linearly-morphing fairing met the
constant airfoil and the constant spar cylinder. Fixed at the source: the morph now
uses **smoothstep easing** (`_transition_blend`, zero slope at both junctions), so
the surface is crease-free and no fillet is needed. `apply_wing_fillets` was retired.
(2) *On-surface fidelity.* Decoupled geometry resolution from sizing resolution —
sizing stays at 8 levels (SLSQP speed) while the geometry is built at **60 levels**
via `resample_segment_radii`, cutting worst on-surface error from **289 mm → 113 mm**.
The residual now lives mid-transition (smoothstep is steepest there); **transition-
dense z-levels** would target it further — recorded as the remaining refinement.

### Phase E — Deepen structural

- Anisotropic skin via CLT (`materials.unidir`) replacing isotropic-equivalent.
- Discrete cross-section **catalog via MILP** (OR-Tools) with co-linear grouping;
  sequential linear programming using FEA sensitivities.
- Buckling (panel eigenvalue) and twist-deflection constraints in the sizing loop.

**Deliverable: sized structure with discrete stock cross-sections + anisotropic
skin, buckling-checked.**

**Status: E.1 done (2026-06-05).** Load-bearing skin coupled into the structural
model: DKT+CST shell panels assembled between beam nodes via `structural.solve_beam_shell`
(`beams.build_beam_shell_model`), with the **skin replacing the ring connectors**.
Isotropic-equivalent skin (3 mm). `examples/24_skin_coupling.py` measures the skin
making the structure ~2.4x stiffer (tip deflection) than the ring frame at equal
beam radius (n_levels=12) — confirming the Phase-C/D finding that the skin, not
thicker beams, is the mass-efficient stiffness lever. (The factor is resolution-
dependent: ~8.6x at n_levels=6, ~2.4x at n_levels=12 — more ring levels add ring
stiffness, so the closed-skin advantage over discrete rings narrows as the mesh
refines; the skin always wins.)

**Status: E.2 done (2026-06-06).** Co-sized per-element beam radii **and** a uniform
skin thickness against the combined beam+shell FEA (`beams.size_beam_shell`),
minimizing total beam+skin mass under beam-vm, skin-vm, tip-deflection and tip-twist
constraints (skin membrane stress recovered each solve via `structural.recover_membrane_stress`).
`examples/25_resize_with_skin.py` (n_levels=8): the load-bearing skin cuts structural
mass from the Phase-C beams-only **43.5 kg → 27.5 kg (37% lighter)** — beams 17.8 kg
(radii 4–14 mm) + skin 9.7 kg (0.71 mm), converged & feasible.

**Key finding:** with the skin carrying load the structure becomes **twist-governed**
— tip twist sits exactly on its 5° limit while tip deflection is slack (49 mm of
100) and stresses are tiny (beam 18 MPa, skin 73 MPa vs 1100 allowable). So the next
mass lever is torsional stiffness (skin shear path / fibre angle), not beam or skin
gauge. This motivates **CLT anisotropic skin** (E.4 — tailor ±45° plies for shear)
and revisiting the twist limit.

**Status: E.4 done (2026-06-06).** CLT anisotropic skin: a symmetric-balanced
laminate (smeared bending D) whose membrane `A`/bending `D` feed
`tri_element_stiffness_laminate` / `solve_beam_shell_laminate`; the layup area
fractions (0/±45/90) are co-sized as design variables with beam radii + skin
thickness (`beams.size_beam_shell_laminate`), minimizing mass under beam-vm,
skin-vm, tip-deflection and tip-twist constraints. `examples/26_clt_skin.py`
(n_levels=8): CLT cuts mass **27.5 kg (E.2 isotropic) → 25.6 kg (7% lighter)**,
converged & feasible.

**What is real:** the 7% mass gain is a valid optimum of the posed problem — CLT
anisotropy lets the optimizer drop skin stiffness in unneeded directions, shrinking
the beams (17.8→14.7 kg) while holding twist (the active constraint) with slightly
thicker skin (0.71→0.79 mm). Stiffness and stress use the same per-triangle Qeff
self-consistently, so the converged result is sound. A dedicated test confirms the
optimizer *does* drive the layup toward ±45° when twist is the sole binding lever.

**Important limitation (do NOT read the optimal layup as a manufacturable layup):**
ply angles are defined relative to **each skin triangle's local frame** (e1 along
its first edge), and the tiling's triangle frames are split ~50/50 spanwise vs
chordwise (measured mean |cos∠(local-x, span)| = 0.49). So `(f0,f45,f90)` are not
consistent global fibre directions — the converged "all-0°" is "0° per arbitrary
local edge," not a coherent spanwise layup. A fixed-geometry sweep shows the layup
effect is modest and that 90° is simply the worst orientation (highest deflection
and twist); the exact `(1,0,0)` corner is somewhat arbitrary among the non-90°
options. Per-ply Tsai-Wu failure (vs laminate-average vm) and per-band layup
would refine it further.

**E.4b done (2026-06-07) — consistent span datum; layup now manufacturable AND 26%
lighter.** `LaminateSizingConfig.ply_angle_datum=(0,0,1)` measures ply angles against
the span axis: each triangle's laminate is built with its plies offset by the
triangle's local-frame angle to the datum (`skin_datum_angles` + `laminate_stiffness_offset`;
solver + stress recovery generalized to per-triangle stiffness). The optimized layup
is now coherent (0°=spanwise). Re-running the co-sizing (`examples/28_ply_datum.py`):
the span-datum optimum is **19.0 kg** with a clean manufacturable layup — **100%
chordwise (90°) skin** — vs the incoherent per-tri-local 25.6 kg. It is **26% lighter**
because a consistent datum lets the optimizer coherently exploit anisotropy and use
both the deflection and twist budgets fully (the datum design sits at *both* limits;
the per-tri-local one left deflection slack at 66/100 mm). Physically sensible: the
longitudinal beams carry spanwise bending, so the skin's most efficient role is
**chordwise (hoop) fibres** for section-shape/shear, not spanwise. CAVEAT: this 19.0 kg
is WITHOUT buckling (apples-to-apples vs E.4); the true final design is datum + the
closed-form buckling constraints (a quick combined run) — expect it modestly heavier.

**Deferred to later E increments:** per-spanwise-band skin thickness; MILP discrete
stock catalog (E.3); per-ply Tsai-Wu skin failure.

**Buckling check done (2026-06-06).** Closed-form constraints added (optional, gated
by `buckling_safety_factor`): beam Euler (`Pcr=π²EI/(KL)²`, element-length buckling
since the skin restrains nodes) + skin panel plate-buckling (triangle-as-plate,
`b=√area`, `kc=4`), in both sizers. Helpers in `structural.buckling`;
`examples/27_buckling.py` re-sizes the CLT design with vs without (SF=1.5).

**Validity finding — buckling binds, but the mass is robust (+2%).** Without buckling
the E.4 design is 25.6 kg, twist-governed, beam-heavy (beams 14.7 / skin 10.8 @
0.79 mm) — and is in fact **buckling-infeasible**. With buckling (SF 1.5) the optimum
is **26.1 kg (+2%)**, now **buckling-governed** (beam & panel utilization both = 1.0,
twist slack at 2.8°): the optimizer rebalances material **from beams into a thicker
skin** (beams 9.7 / skin 16.4 @ 1.20 mm, layup shifts slightly toward ±45) — the
thicker skin both resists panel buckling and laterally stabilizes the beams. So the
~26 kg headline holds; the prior 25.6 kg was mildly optimistic. Approximations
(triangle-as-plate panel, element-length Euler, no eigenvalue/global modes) are
conservative-ish; eigenvalue/global buckling is the refinement if needed.

**E.3 done (2026-06-07) — MILP stock catalog (CP-SAT).** `beams.select_stock_sizes`
discretizes the continuous beam radii to a stock catalog via a CP-SAT min-mass
assignment with a cap of K distinct sizes (the co-linear-grouping / part-count knob);
`examples/29_stock_catalog.py` demonstrates + FEA-verifies. Stock catalog 4–20 mm
(2 mm steps); continuous beam mass 8.65 kg. Mass-vs-part-count Pareto (all feasible):
K=4 → **4 sizes, 9.5 kg (+10%)**; K=2 → 2 sizes, 17.4 kg (+101%); K=8 → 4 sizes (no
gain over K=4). **Finding — round-up alone is NOT buckling-safe:** the monotonic
"each beam ≥ its continuous radius stays feasible" argument holds for stress,
deflection and twist but FAILS for beam buckling — non-uniform stiffening
redistributes compression onto the pinned r_min (4 mm) beams, pushing their Euler
utilization to ~1.3. A greedy **bump-and-reverify repair** (raise over-utilized
beams to the next catalog size, re-select, re-solve — 1 iteration here) restores
feasibility. The FEA verify step is therefore essential, not optional. Full
SLP+FEA-sensitivity MILP remains deferred.

**Combined final design done (2026-06-07) — span-datum CLT skin + buckling together.**
Earlier increments measured the two refinements separately (E.4b: 19.0 kg with a
coherent span-datum layup but WITHOUT buckling; the buckling study: ~26.1 kg on the
*isotropic* skin). `examples/32_final_design.py` co-sizes BOTH at once — a
manufacturable span-datum CLT layup under beam-stress, skin-stress, tip-deflection,
tip-twist AND closed-form buckling (Euler + panel, SF 1.5). Result: **30.15 kg**
(beams 9.38 / skin 20.77 @ **1.52 mm**), **buckling-governed** (beam & panel
utilization both = 1.0), twist slack (1.30°/5°), deflection slack (28.8/100 mm),
layup 31% spanwise / 15% ±45 / 54% chordwise. This is *heavier* than either partial
number because the two constraints compound: buckling forces a thick skin (skin = 2/3
of mass) and a single coherent global datum layup is less free than per-triangle-local
angles — manufacturability and buckling each cost real mass. **This 30.15 kg is the
defensible, fully-constrained, manufacturable headline for the shell-beam wingsail.**
Caveat: SLSQP is a local optimum (per-tri-local + buckling reached 26.06 kg), so a
better basin may exist; a global/multi-start pass is the refinement if mass is
critical.

**Per-band skin thickness done (2026-06-08) — clear win.** The CLT sizer's single
uniform skin thickness is now an opt-in set of B contiguous spanwise thickness bands
(`LaminateSizingConfig.n_skin_bands`, default 1 = unchanged; `beams.skin_band_map` maps
each triangle to a band, per-triangle thickness drives CLT stiffness, panel buckling,
and mass). Since the skin is ~2/3 of mass and uniform thickness is set by the worst
(root) panel, the lightly-loaded tip carries excess material. `examples/33_banded_skin.py`
compares uniform vs. 4 bands at equal constraints (span datum + buckling SF 1.5,
n_beams=16, n_levels=8): **uniform 30.11 kg → banded 27.67 kg (−8.1%, −2.44 kg, all in
the skin: 20.46→18.02)**, both converged & feasible (beam/panel buckling util = 1.0).
The taper thins the tip and keeps the keel thick — band thicknesses (tip→keel)
**0.96 / 1.29 / 1.57 / 1.55 mm** (mean 1.32). The **governing constraint flips from
buckling (uniform) to twist (banded, tip twist pinned at 5.0°)**: banding lets the
optimizer thin the tip skin until twist — not panel buckling — becomes binding.
Banding needs more SLSQP iterations (4× the thickness DVs + tighter active set; ~354 vs
70). Highest-leverage lever realized; per-ply Tsai-Wu and per-band *layup* remain
deferred follow-ups. NB this −8.1% is on top of the 30.15 kg headline, i.e. a banded
final design lands ~27.7 kg.

**Per-ply Tsai-Wu skin failure done (2026-06-08) — confirms strength is not the binding
skin mode.** Opt-in `skin_failure="tsai_wu"` (default `"von_mises"`; `materials.failure`)
replaces the laminate-average von-Mises proxy with a per-ply Tsai-Wu strength-ratio check
(R ≥ SF, F12 = -0.5·√(F11·F22), material SF 2.0) over the present ply orientations, using
`recover_membrane_strain`. `examples/34_tsai_wu_skin.py` compares the two criteria at
equal constraints (span datum + buckling SF 1.5, uniform skin): **von-Mises 30.11 kg vs
Tsai-Wu 30.07 kg (−0.1%)**, layup essentially unchanged (34/13/52 → 35/13/52), and the
governing **Tsai-Wu min strength ratio is R = 7.01 — far above the required SF of 2.0**.
**Finding:** the skin is **buckling/stiffness-governed, not strength-governed** (both
designs sit at beam & panel buckling util = 1.0 with vM skin stress ~22 MPa ≪ 1100 MPa
allowable and ~3.5× margin even on the proper Tsai-Wu criterion), so the cruder
von-Mises proxy was already adequate here — the accurate criterion confirms it rather
than changing the design. Tsai-Wu remains the right check to keep available for
load/geometry regimes where the skin *is* strength-critical. **Cost caveat (measured):**
the Tsai-Wu run took **~2 h 11 m** wall-clock (per-triangle Python loop inside every
SLSQP `evaluate` × the FD-Jacobian's ~115 evaluates/iter); vectorize the inner loop
before using it in any larger sweep. First-ply only; smeared laminate (no
stacking/delamination).

**Tsai-Wu vectorized (2026-06-08).** The per-triangle Python loop in the Tsai-Wu check
was replaced by a vectorized `laminate_min_strength_ratio_batch` (identical results;
scalar delegates to it). **Lesson (corrected):** this gave only a *modest* speedup — a
post-vectorization run of `examples/34` was still >1 h 18 m before being killed — because
the dominant cost is the **FEA solve per `evaluate`** (`solve_beam_shell_laminate` called
~(n_design_vars+1)× per SLSQP iteration by the finite-difference Jacobian × maxiter ×
n_load_cases), NOT the per-ply inner loop. Sizing-run cost is gated by mesh size, maxiter,
and DV count; an analytic/!FD Jacobian or fewer DVs would be the real lever.

**Per-band skin layup done (2026-06-08) — built & tested; mass win unquantified.**
Opt-in `LaminateSizingConfig.per_band_layup` gives each `n_skin_bands` spanwise band its
own `(f0,f45,f90)` (default off = global layup; the Tsai-Wu batch was generalized to
per-triangle fractions so it composes). Backward-compatible (full suite green, 120
tests). Since skin mass is fraction-independent, any win is **indirect** (a band's fibre
mix can meet the constraints at a thinner thickness). **Measurement inconclusive:** a
coarse comparison (medium wingsail, n_beams=12/n_levels=6, 4 bands, datum+buckling,
maxiter 120 global / 200 per-band) took **1 h 04 m** and *neither* run converged
(both hit maxiter); the per-band run ended **beam-buckling-infeasible (util 1.70)**, so
its −6.6% lower mass is not a valid feasible win. The per-band run did produce a sensible
fibre **taper** (root band ≈48/23/29 spanwise-heavy, mid bands more ±45/chord), so the
mechanism works, but a trustworthy headline needs a converged feasible run (more
iterations / better conditioning) — deferred given cost. Consistent with the
buckling/stiffness-governed picture (Tsai-Wu showed huge strength margin), per-band layup
is expected to be a small, hard-to-realize lever. Independent layup-band-count and a
global/multi-start pass remain deferred.

### Phase F — Frame-field-driven layout

The retained Arora frame field finally drives geometry.

- Non-uniform beam spacing by **cumulative principal-stress** around each
  cross-section (replaces even arc-length spacing).
- Optional **second helical/diagonal beam family** whose winding angle follows
  the in-plane principal-stress direction.

**Deliverable: a beam layout demonstrably lighter/stiffer than even spacing for
the same load cases.**

**F.1 done (2026-06-07) — non-uniform (stress-weighted) spacing; marginal benefit,
clear lesson.** Beams re-placed at equal-cumulative-skin-stress arc positions
(`stress_weighted_targets` from a baseline `cross_section_stress_weights`;
`arc_fractions` threaded through cross_section/splines/shell_model, default even =
backward-compatible). `examples/30_nonuniform_spacing.py` (n_levels=8, with buckling):
the per-segment skin stress is only **mildly concentrated (max/mean ≈ 1.8×)**;
stress-weighted spacing comes out **~1% lighter (29.7 → 29.3 kg)** and reaches feasible
beam buckling (util 1.00) where the even-spacing sizing stalled at 1.30 within maxiter.
**Finding:** longitudinal beam re-spacing has **low leverage** on total mass for this
design because it is **skin/buckling-dominated** (beams ~8.7 kg of ~30 kg; the skin
ballooned to ~21 kg under the buckling constraint). The real layout levers are the
**second diagonal beam family (F.2)** and skin tailoring, not longitudinal re-spacing.
One-shot (not iterated); per-z-uniform arc_fractions.

**F.2 investigated (2026-06-07) — diagonal/helix beam family; strong NEGATIVE result,
clear lesson. Implementation was a throwaway measurement spike and was NOT merged —
only this finding is kept.** A balanced both-hand grid-helix lattice (diagonals as
chains on the existing structured node grid — no remesh) was tied into the combined
beam-shell solver and co-sized with ONE shared diagonal-radius DV; the grid pitch was
chosen to best track the baseline principal-stress field. A baseline-vs-diagonal
comparison was run at equal constraints (span datum + buckling, SF 1.5, n_beams=16,
n_levels=6): recommended **pitch 2** (mean principal alignment only **0.68**, 160
diagonal elements). Result: **baseline 33.1 kg → diagonal 60.6 kg (+83%)**; the 160
diagonals alone add **20.8 kg** at a buckling-forced **4.7 mm** radius, and tip twist
went *up* slightly (1.25°→1.58°). **Finding:** the diagonal lattice is strongly
mass-**counterproductive** here, for a physical reason consistent with F.1 and the
buckling study — the design is **buckling-governed with large twist slack** (both
designs sit at twist ≈1.3–1.6° vs. the 5° limit, beam & panel buckling util = 1.00).
Diagonals are long compression-loaded members whose own Euler buckling forces a fat
radius, so they bloat mass *without relieving a binding constraint* — there is no twist
budget to recover. Diagonals would only pay off if twist were binding and the lattice
carried tension. **Implication:** abandon (or radically thin/sparsify) the distributed
diagonal lattice for this load/constraint regime; twist, when it matters, is far more
cheaply killed at the tip (see the gusset investigation). Streamline-following diagonals
(F.3) are unlikely to change the verdict while buckling dominates, so F.3 is **not**
recommended unless a twist-governed variant emerges. The spike (helix topology,
principal-stress alignment / pitch recommendation, the shared-`r_diag` sizing path, and
a comparison example) was discarded after measuring; it can be reconstructed from this
note and the approach above if a twist-governed case ever justifies it.

### Investigations

**Hard tip coupling / gusset (2026-06-07).** Modeled a rigid tip joint (a clamp all
beams seat into) as a clique of stiff connector beams tying the tip nodes
(`beams.solve_beam_shell_tip_coupled`, reuses `solve_beam_shell`; tunable
`gusset_radius`), to measure beam-to-beam stress transfer at the tip.
`examples/31_tip_coupling.py` (uniform r=20 mm design, sweep gusset stiffness):
skin-only → near-rigid gusset gives **peak beam σ −1%, σ-spread 3.75→3.38, tip
deflection −14%, tip twist 0.197°→0.004° (~47× lower)**, saturating at a small gusset
(20 mm ≈ rigid). **Finding:** a hard tip joint barely redistributes *beam* stress —
the load-bearing skin already shares spanwise load between beams — but it is a powerful
**torsional** restraint, near-eliminating tip section rotation. Since the sized design
is twist-governed, a tip gusset is a candidate to relax that binding constraint and
lighten the design (re-size-with-gusset is the natural follow-up). (Earlier attempt
used ad-hoc Z penalty springs to pass a tip-Z-spread test; a probe showed the skin
already makes tip-Z-spread ~0, so that test was meaningless — replaced with the clean
model + this honest measurement.)

### Phase G — Filament-winding path planner

- `wing_design.manufacturing.winding` — continuous winding passes forming the
  airfoil + reinforcing joints/buckling-prone members; output robot/G-code paths
  + per-pass fiber-orientation map feeding the CLT of Phase E.

**Deliverable: animated winding sequence + CLT-homogenized skin stiffness map.**

### Phase H — Verification, fixed-point iteration, scaling

- Re-mesh the **as-designed** assembly; re-solve every load case; check stress,
  deflection, buckling, fatigue.
- Feed σ(x) back into Phase F; iterate to convergence.
- Parametrize span/chord/taper for FPV-drone-wing / turbine-blade retargeting;
  add dynamic (flutter, vortex shedding) and impact cases.

**Deliverable: certified-on-paper wingsail with documented load envelope and
safety margins.**

See [`guided_generative_design.md` → Future optimizations](./guided_generative_design.md#future-optimizations--directions)
for the longer backlog (skin-vs-beam mass trade, outer geometry optimization,
interior-truss comparison baseline).

## Tooling

- **Geometry viewer:** examples that produce build123d geometry call
  `wing_design.show_in_viewer(part)`, which sends to the **OCP CAD Viewer** VS Code
  extension (bernhard-42) on port 3939. If the viewer isn't running, the call is a
  no-op with a printed hint.
- **ParaView 6.x** for FEA field visualization via the `just view` / `just shot`
  recipes (`paraview/*.py`).
- **Bytecode:** `PYTHONPYCACHEPREFIX` is exported from the justfile so generated
  `.pyc` land in a single gitignored `.pycache/` instead of scattering through the
  source tree. Use `just`-based entry points to inherit it.
- **Python interpreter for VS Code:** select `.venv/bin/python` so basedpyright
  resolves `build123d`, `aerosandbox`, etc.

## Module map

```
src/wing_design/
  __init__.py
  scenario.py    # DesignParameters dataclass + default_scenario()   [done]
  geometry/      # build123d wing OML + spar                         [Phase 1 ✅]
  aero/          # AeroSandbox loads                                 [Phase 2 ✅]
  materials/     # UD ply + CLT                                      [Phase 3 ✅]
  structural/    # tube-spar baseline (P3), mesh+FEA (P4), shell     [✅]
  truss/         # frame field, parametrization, streamlines         [Phase 5 ✅]
                 #   (retained as diagnostic + Phase F layout driver)
  beams/         # form-beam splines, build, wrap, assembly,         [Phases A–F]
                 #   fea_model, sizing
  manufacturing/ # winding path planner, BOM                         [Phase G]
  viz/           # PyVista / ocp_vscode helpers
```

## Decisions log

| Decision | Choice |
| --- | --- |
| Structural concept | **Shell-following form beams**, not interior volumetric truss (2026-06-04). |
| Frame field | Kept; diagnostic now, layout driver in Phase F. |
| Cross-section sizing | FEA-in-the-loop (fully-stressed first, MILP catalog in Phase E). |
| Skin role | Load-bearing structural shell (supersedes fairing-only). |
| Beam layout (early) | Even arc-length spacing; frame-field-driven in Phase F. |
| Build-out strategy | Thin end-to-end spike (Phases A–C), then deepen (D+). |
| Phase-4 FEA backend | Roll-our-own linear-tet solver in numpy/scipy; promote to sfepy/FEniCSx when anisotropic homogenization matters. |
| Spline fitting | Dense on-surface sampling + cubic B-spline interpolation (no NURBS weight optimization). |
| Spar radius | Derived (max inscribed circle at pivot, floored to cm), not stored. |
| Phase-B FEA element | Fresh 3D Euler–Bernoulli frame (`structural.frame`); ring connectors stand in for skin shear transfer in the spike (coupled shell = later deepen). |
| Example outputs | All examples write generated artifacts to the repo-root `exports/` (gitignored); no outputs tracked in git. |
| Phase-C sizing | Continuous SLSQP NLP over per-element longitudinal radii; stress + tip-deflection + tip-twist constraints; analytic mass gradient, FD constraint Jacobian over FEA re-solves; rings fixed (stress reported). MILP stock catalog deferred to Phase E. |
| Phase-D geometry | Inward-arc **lens** beam sections sized to Phase-C radii (lens lofts cleanly, circular fallback unused); fillets best-effort and currently **no-op** (0/2 — OCC rejects the 30/50 mm transition radii, skipped not fatal); spline fidelity coarse (~286 mm full / ~282 mm aero at 8 levels) — fidelity tuning + valid fillet radii are open Phase-D items (2026-06-05). |
| Phase-E.1 skin coupling | Load-bearing skin assembled as DKT+CST shell panels between beam nodes into the combined `structural.solve_beam_shell`; skin replaces ring connectors. Isotropic-equivalent skin; re-sizing/MILP/buckling/CLT deferred to later E increments. |
| Phase-E.2 co-sizing | SLSQP co-sizes per-element beam radii + a uniform skin thickness minimizing total beam+skin mass under beam-vm, skin-vm, tip-deflection and tip-twist constraints (skin membrane stress recovered each solve). 43.5→27.5 kg (37% lighter); structure becomes twist-governed. Per-band skin / MILP / buckling / CLT deferred. |
| Phase-E.4 CLT skin | Anisotropic skin via CLT (symmetric-balanced laminate, smeared D); laminate (A,D) feed `tri_element_stiffness_laminate` / `solve_beam_shell_laminate`; co-sizes beam radii + skin thickness + layup fractions (f0,f45,f90) under beam-vm, skin-vm, deflection, twist. 27.5→25.6 kg (7% lighter, valid optimum). LIMITATION: ply angles are per-triangle-local (frames ~50/50 spanwise/chordwise), so the optimal layup is not a manufacturable fibre prescription — needs a consistent ply-angle datum (E.4b). Per-ply Tsai-Wu / per-band layup / MILP / buckling deferred. |
| Buckling check | Closed-form beam Euler (element-length) + skin panel plate-buckling (triangle-as-plate, b=√area, fixed kc) added as optional sizing constraints in both sizers (gated by `buckling_safety_factor`). Buckling binds but mass robust: 25.6→26.1 kg (+2%), governing constraint flips twist→buckling, optimizer shifts mass from beams into thicker skin. Eigenvalue/global buckling + refined panel geometry deferred. |
| Phase-E.4b ply datum | Ply angles measured against the span axis (per-triangle offset via `skin_datum_angles` + `laminate_stiffness_offset`; solver/recovery generalized to per-triangle stiffness). Opt-in via `LaminateSizingConfig.ply_angle_datum`. Coherent layup AND 26% lighter: 25.6→19.0 kg, optimum is 100% chordwise (90°) skin; design now uses both deflection + twist budgets. (19.0 kg is without buckling — datum+buckling is the real final combo.) |
| Phase-E.3 stock catalog | Beam radii discretized to a stock catalog via CP-SAT (`beams.select_stock_sizes`): min-mass assignment with a ≤K-distinct-sizes cap (co-linear grouping). Round-up is feasible for stress/defl/twist but NOT buckling (non-uniform stiffening redistributes load onto pinned r_min beams) → greedy bump-and-reverify repair restores feasibility; FEA verify is essential. K=4 → 4 sizes at +10% mass. Full SLP+sensitivity MILP deferred. |
| Phase-F.1 non-uniform spacing | Beams placed at equal-cumulative-skin-stress arc positions (`stress_weighted_targets` from `cross_section_stress_weights`); `arc_fractions` threaded through cross_section/splines/shell_model (default even, backward-compatible). One-shot; 2nd diagonal family deferred (F.2). Result: only ~1% lighter (skin stress mildly concentrated 1.8×, design is skin/buckling-dominated so beam re-spacing has low leverage) — the layout lever is F.2 / skin tailoring, not re-spacing. |
| Combined final design | Span-datum CLT layup co-sized WITH buckling (`examples/32_final_design.py`) — both refinements together, not separately. 30.15 kg (beams 9.38 / skin 20.77 @ 1.52 mm), buckling-governed (beam & panel util = 1.0), twist/defl slack, layup 31/15/54% span/±45/chord. Heavier than the partial numbers (E.4b 19.0 no-buckling; isotropic+buckling 26.1) because manufacturable-datum + buckling compound. **This is the defensible fully-constrained headline.** SLSQP local optimum (per-tri-local+buckling hit 26.06) → multi-start is the refinement if mass-critical. |
| Per-band skin thickness | Skin thickness split into B contiguous spanwise bands (opt-in `n_skin_bands`, default 1 = uniform/unchanged; `beams.skin_band_map`), each a thickness DV in the SLSQP laminate loop; per-triangle thickness drives CLT stiffness, panel buckling, and mass; `t_skin` retained as the area-weighted mean. Uniform 30.11 kg → 4-band **27.67 kg (−8.1%)**, all saved in the skin; taper 0.96/1.29/1.57/1.55 mm tip→keel; governing constraint flips buckling→twist. Needs more SLSQP iters (~354 vs 70). Highest-leverage skin lever. Per-ply Tsai-Wu + per-band layup deferred. |
| Per-ply Tsai-Wu skin failure | Opt-in `skin_failure="tsai_wu"` (default von-Mises unchanged; `materials.failure`) — per-ply Tsai-Wu strength ratio R≥SF (F12=-0.5√(F11 F22), material SF 2.0) over present orientations, from per-triangle membrane strain (`recover_membrane_strain`). vs von-Mises proxy at equal constraints: 30.11→30.07 kg (−0.1%), layup ~unchanged, **min R = 7.01 ≫ SF 2.0** → skin is buckling/stiffness-governed, not strength-governed; the proxy was already adequate. Kept for strength-critical regimes. Measured cost ~2h11m (per-triangle loop × SLSQP FD-Jacobian — vectorize before large sweeps). First-ply only; smeared laminate. |
| Tsai-Wu vectorized | Per-triangle Python loop replaced by vectorized `laminate_min_strength_ratio_batch` (identical results; scalar delegates). Modest speedup only — the dominant sizing cost is the FEA solve per `evaluate` × FD-Jacobian width × maxiter, not the inner loop. Real lever would be an analytic Jacobian or fewer DVs. |
| Per-band skin layup | Opt-in `per_band_layup` — each n_skin_bands band gets its own (f0,f45,f90) DVs (L-group design vector, L fraction constraints; Tsai-Wu batch generalized to per-triangle fractions). Built + tested (120 tests, backward-compatible). Mass is fraction-independent → win is indirect (per-band mix → thinner bands). Headline UNQUANTIFIED: coarse comparison (1h04m) didn't converge and ended beam-buckling-infeasible; sensible fibre taper emerged. Expected small lever (buckling/stiffness-governed). Converged headline + independent layup-band-count deferred. |
| Phase-F.2 diagonal beams | Balanced both-hand grid-helix lattice on existing grid nodes (`beams.helix_elements`, no remesh), co-sized with one shared diagonal-radius DV in the SLSQP laminate loop; pitch chosen by principal-stress alignment (`recommend_pitch`, best pitch 2 @ align 0.68). **Strong negative result:** baseline 33.1 kg → diagonal 60.6 kg (+83%), diagonals add 20.8 kg at a buckling-forced 4.7 mm radius, twist rose 1.25°→1.58°. The design is buckling-governed with large twist slack, so long compression diagonals bloat mass without relieving a binding constraint. Lattice abandoned for this regime; twist (when binding) is killed far cheaper at the tip. Streamline-following (F.3) not recommended while buckling dominates. Implementation was a throwaway spike, NOT merged — only the finding is kept. |
| Tip-coupling study | Hard tip joint (gusset) modeled as a stiff connector-beam clique tying the tip nodes (`beams.solve_beam_shell_tip_coupled`, tunable `gusset_radius`), reusing `solve_beam_shell` (no rigid MPC, no penalty hacks). Finding: barely redistributes BEAM stress (peak −2%, spread 3.75→3.38) — the skin already shares spanwise load — but near-eliminates **tip twist** (0.197°→0.004°, ~50×) and stiffens the tip (~14%), saturating at low gusset stiffness. Investigation only (no CAD / not in the sizing loop). Implication: the twist-governed design could be relaxed/lightened by a tip gusset (re-size-with-gusset = follow-up). |
