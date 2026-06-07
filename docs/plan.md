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

### Phase F — Frame-field-driven layout

The retained Arora frame field finally drives geometry.

- Non-uniform beam spacing by **cumulative principal-stress** around each
  cross-section (replaces even arc-length spacing).
- Optional **second helical/diagonal beam family** whose winding angle follows
  the in-plane principal-stress direction.

**Deliverable: a beam layout demonstrably lighter/stiffer than even spacing for
the same load cases.**

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
