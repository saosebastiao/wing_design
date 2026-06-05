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

### Phase E — Deepen structural

- Anisotropic skin via CLT (`materials.unidir`) replacing isotropic-equivalent.
- Discrete cross-section **catalog via MILP** (OR-Tools) with co-linear grouping;
  sequential linear programming using FEA sensitivities.
- Buckling (panel eigenvalue) and twist-deflection constraints in the sizing loop.

**Deliverable: sized structure with discrete stock cross-sections + anisotropic
skin, buckling-checked.**

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
