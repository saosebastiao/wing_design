# Wing Design — Development Plan

The design specification lives in
[`guided_generative_design.md`](./guided_generative_design.md). The build history —
phase-by-phase analyses, findings, and the decisions log from the original shell-beam
build-out (Phases 1–F, complete) — is archived in [`findings.md`](./findings.md).
The audit-driven improvement backlog behind this plan is
[`improvement_backlog.md`](./improvement_backlog.md), which also records the intended
**manufacturing concept** (RTM'd / filament-wound beams, wrapped joints, hoop-wound
prestressed skin) that several phases below build toward.

This file is the **incremental build sequence** for the next stage: validity
hardening, manufacturability, and the performance harvest.

## Goal

A free-rotating, unstayed solid carbon-fiber **wingsail** built as a **space-frame
of unidirectional CFRP beams** wrapped and bound by a **filament-wound CFRP skin**
that simultaneously forms the airfoil surface. The structural members lie **on the
outer shell** — filament-winding the skin around shell-following beams *is* what
forms the structure. Beam cross-sections are sized **FEA-in-the-loop**, and the
wound skin is modeled as a **load-bearing shell**.

The repo is meant to evolve from a 5 m demo wingsail to a process applicable from
~1 m FPV drone wings up to >100 m wind turbine blades.

## Where we are (2026-06-09)

The shell-beam pipeline is built end-to-end and sized FEA-in-the-loop. Defensible
headlines: small (5 m) **30.15 kg** fully constrained / **27.67 kg** with 4 thickness
bands; medium (22 m) **2264.6 kg** (analytic Jacobian). The design is robustly
**panel-buckling/stiffness-governed — strength never binds** (Tsai-Wu R = 7.0 vs
required 2.0), and beam-layout levers are mined out (all negative or marginal).

Both governing constraints sit at utilization 1.0 on **closed-form approximations**,
so the plan below first firms up validity (some fixes will move the honest baseline
*up*), then exploits the huge strength margin and the manufacturing concept's hollow /
prestressed options (the step-change levers change *what kind of section resists
buckling*, not where material sits). The analytic adjoint Jacobian is merged (1.98×
vs FD, exact). Full record: [`findings.md`](./findings.md).

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

## Pipeline (as built)

```
   build123d            AeroSandbox            3D linear FEA
 wing OML solid  ──►   pressure field   ──►   Cauchy stress σ(x)
                         per load case               │
                                                     ▼
                                    stress-aligned frame field R(x)
                                          (diagnostic)
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
**Findings are recorded in [`findings.md`](./findings.md)** (not here), with measured
wall-clock for every sizing run. Backlog references (V#/M#/P#) point into
[`improvement_backlog.md`](./improvement_backlog.md).

### Phase V.0 — Iteration-speed enablers

Phase V multiplies the number of sizing runs (mesh sweeps, re-baselines) and Phase P
adds parameter sweeps and multi-start; at ~1.5 h per medium sizing, compute speed is
the rate limiter on the whole plan. The FEA system is tiny (768 DOF — `splu` is
microseconds), so wall-clock is almost certainly dominated by **per-element Python
loops run every evaluate** (K assembly, SensCache build, adjoint contraction, stress
recovery). Items V.0.2–V.0.5 compose multiplicatively and are exact (no formulation
change) except KS, which is testable.

- **V.0.1 Profile first (the gate).** py-spy/cProfile a few SLSQP iterations of the
  medium problem; rank assembly vs cache build vs back-subs vs recovery. Runtime
  estimates run ~10× off — the profile decides whether V.0.2 or V.0.3 leads.
- **V.0.2 Vectorize the per-element Python work** — assembly, SensCache build, and the
  adjoint contraction as batched numpy over all elements/triangles. Helps every
  evaluate on both FD and analytic paths; payoff grows with mesh size (V.2 pushes
  n_levels up). Plausibly order-of-magnitude if the profile confirms the loops
  dominate.
- **V.0.3 KS aggregation of the vector beam constraints.** Collapses n adjoint
  back-substitutions to 1 (the known wall, backlog P#7) and shrinks the SLSQP QP from
  ~n stress rows to 1. KS is a smooth conservative approximation — re-validate optima
  against the hard-max formulation (same pattern as the analytic-vs-FD equivalence
  tests).
- **V.0.4 Process-parallel multi-start and sweeps.** The multistart wrapper is
  "serial, parallel-ready"; V.2 and P.4 are embarrassingly parallel across points.
  A process pool gives near-linear core scaling with no numerical work.
- **V.0.5 Warm-start continuation as standard sweep protocol.** Each sweep point
  starts from its neighbor's optimum; fine meshes start from resampled coarse-mesh
  solutions (the `resample_segment_radii` pattern). Attacks iteration *count*, which
  V.0.2–V.0.4 don't touch (examples 39/40 already proved warm-starting converges).
- **V.0.6 Opportunistic cleanups.** The uncached `_beam_vm_grad_one` path; sparse
  Cholesky instead of `splu` only if the profile shows factorization mattering at
  finer meshes.

V.0.1 runs now; the rest pull in as the profile directs. Don't block V.1 on this —
shadow prices is a few independent lines.

### Phase V — Validity hardening

Make the model behave like the real structure before harvesting levers. Three of
these fixes may move the honest baseline *up*; lever gains must be measured against a
trustworthy baseline, not the current one.

- **V.1 Shadow prices.** Extract the SLSQP Lagrange multipliers (computed, currently
  discarded) and report kg-per-unit for each limit (twist 5°, deflection 2% span,
  buckling SF 1.5) — tells us which *requirement* to renegotiate first. (P#8)
- **V.2 Mesh-convergence study of the optimized design.** Sweep n_levels (and spot-check
  n_beams) at fixed constraints; quantifies the element-length-Euler mesh dependence,
  which currently rewards refinement unconservatively. (V#9, V#3)
- **V.3 Eigenvalue buckling check (K + Kσ).** Linear buckling solve of the current
  optimum: curvature credit, stiffness-direction match, combined loading, global
  ovalization modes — validates or relaxes *both* binding closed-form checks. Includes
  the cheap parametric hoop-pretension post-processing probe to bound the prestress
  prize before P.2. (V#1, V#2)
- **V.4 Self-weight + inertial/heel load cases.** First-order for the 2.3-tonne medium
  cantilever. (V#5)
- **V.5 Distributed panel pressure + skin bending stress in the failure check.**
  Panels currently receive no pressure; DKT bending moments are recovered but unused.
  (V#4)
- **V.6 Re-baseline.** Re-run the small and medium headlines with V.4/V.5 active; all
  Phase-P levers are measured against these numbers.

Deferred validity items: aeroelastic load feedback + divergence/flutter → Phase H;
Brazier crush → with P.2 webs; environmental/fatigue knockdowns → with the M.4 as-built
pass; root/bearing compliance → M.5. (V#6, V#7, V#10, V#11)

### Phase M — Manufacturability

Encode the manufacturing concept's constraints so designs stay buildable. M.1–M.2 gate
any "as-built" headline claim; later items can interleave with Phase P.

- **M.1 Beam buildability classification.** At the geometry stage, tag each beam
  RTM-vs-wound (path convexity / fiber bridging) and solid-vs-hollow (straightness);
  reject or re-route infeasible sections before mold design. (M#4)
- **M.2 Ply discretization.** Round band thicknesses/layup fractions to integer ply
  counts (~0.25 mm/ply) with a mandatory first-layer hoop wind floor; FEA re-verify via
  the existing bump-and-reverify pattern. (M#1, M#2)
- **M.3 Windable-angle constraints.** Constrain layup fractions to windable angle sets
  per region; longer-term, Phase G's planner feeds the as-wound orientation map back
  into the CLT. (M#3)
- **M.4 Bondline + as-built mass.** Wrapped-joint/bondline checks (shear flow, peel —
  post-processing on the existing solution) and an as-built mass model (wound Vf,
  turnaround overlaps, joints, hardware); report ideal vs as-built mass separately.
  (M#5, M#7, V#8)
- **M.5 Root socket model.** Replace the rigid clamp with the slot-and-bond socket
  (finite stiffness + bond-area check); size the engagement length from recovered root
  reactions. (M#8, V#11)

### Phase P — Performance harvest

One lever at a time, each against the V.6 re-baselined numbers, finding recorded
before the next lever starts.

- **P.1 Hollow members.** New member type: the straight filament-wound **core tube**
  (permanent structural mandrel) with `r(z)`, `t_wall(z)` DVs + wall-buckling/crimping
  check. If the tube alone doesn't capture the win, add straight in-wing wound beam
  segments spliced to RTM'd solid curved transition segments. Likely the biggest
  single lever (beams are solid rods today). (P#1)
- **P.2 Skin prestress + compression cross-members.** Winding pretension as a
  self-equilibrated load case (the build already commits to the tensioned hoop wrap);
  spreader struts / shear webs concentrate the equilibrating compression into short,
  buckling-cheap members. New DVs: strut radii + pretension magnitude(s); guard with a
  prestress-retention knockdown and strut-foot peel checks. (P#2, P#3)
- **P.3 Sandwich (cored) skin.** Core multiplies panel D at ~5% of the monolithic mass
  cost; needs wrinkling/shear-crimping checks. Partially redundant with P.2 — measure
  which wins. (P#5)
- **P.4 n_beams sweep + light rings as panel-breakers.** Panel size is the proven
  currency; beam count has never been varied. (P#4)
- **P.5 Cheap-iteration levers.** Once gradients are fast: multi-start at scale, more
  and chordwise thickness bands, per-band layup revisit. (P#9, P#10)

### Phase G — Filament-winding path planner

- `wing_design.manufacturing.winding` — continuous winding passes forming the
  airfoil + reinforcing joints/buckling-prone members; output robot/G-code paths
  + per-pass fiber-orientation map feeding the CLT skin model (closes M.3).

**Deliverable: animated winding sequence + CLT-homogenized skin stiffness map.**

### Phase H — Verification, fixed-point iteration, scaling

- Re-mesh the **as-designed** assembly; re-solve every load case; check stress,
  deflection, buckling, fatigue.
- Aeroelastic closure: loads recomputed on the deformed/twisted shape, iterate to a
  fixed point; **divergence/flutter check for the free-rotating wing** (V#6).
- Parametrize span/chord/taper for FPV-drone-wing / turbine-blade retargeting;
  add dynamic (vortex shedding) and impact cases.

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
  geometry/      # build123d wing OML + spar                         [done]
  aero/          # AeroSandbox loads                                 [done; V.4/V.5 extend]
  materials/     # UD ply + CLT + failure                            [done]
  structural/    # frame, shell, beam_shell, buckling                [done; V.3 extends]
  truss/         # frame field, parametrization, streamlines         [diagnostic]
  beams/         # splines, build, wrap, fea_model, sizing,          [done; M/P extend]
                 #   laminate_sizing, sensitivity
  manufacturing/ # winding path planner, BOM                         [Phase G]
  viz/           # PyVista / ocp_vscode helpers
```

## Decisions log

Archived (Phases 1–F) and ongoing: see [`findings.md` → Decisions log](./findings.md#decisions-log).
New decisions are appended there.
