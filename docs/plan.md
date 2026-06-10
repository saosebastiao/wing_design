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

## Where we are (2026-06-10)

The shell-beam pipeline is built end-to-end and sized FEA-in-the-loop. Recorded
headlines: small (5 m) **30.15 kg** fully constrained / **27.67 kg** with 4 thickness
bands; medium (22 m) **2264.6 kg** (4-band, reproduced exactly 2026-06-10). The design
is **buckling/stiffness-governed — strength never binds** (Tsai-Wu R = 7.0 vs 2.0),
and beam-layout levers are mined out.

The 2026-06-10 validity sprint changed the picture: the medium gradient path is
**~23× faster** (cache fix; sizing ≈ 5–8 min, compute is no longer the rate limiter);
shadow prices put all binding-requirement mass on twist (−41 kg/deg, 4-band) and the
two closed-form buckling SFs; the **eigenvalue buckling solve (V.3) showed the
closed-form checks ≥1.6× (converged ~4–5×) conservative**, and the **width-based
panel check (V.3b) converted that to −18.8% on the medium 1-band design
(2471 → 2006 kg, eigen-verified)**; winding-pretension (P.2) measured **null** at the
optimum and is demoted. Headlines are NOT mesh-converged under the legacy checks
(V.2); the strip check + eigen verification is the honest pair going forward. Next:
V.4/V.5 load-completeness (will move mass *up*), then the V.6 re-baseline with
`panel_width_mode="strip"`, then P.0/P.1 (hollow members) against the new baseline.
Full record: [`findings.md`](./findings.md).

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
sub-millisecond), so wall-clock is almost certainly dominated by **per-element Python
loops run every evaluate** (K assembly, SensCache build, adjoint contraction, stress
recovery). Items V.0.2–V.0.5 compose multiplicatively and are exact (no formulation
change) except KS, which is testable.

- **V.0.1 Profile first (the gate). — DONE (2026-06-10).** `examples/41_profile_sizing.py`.
  Verdict: 93% of wall was the *uncached* `_beam_vm_grad_one` path (V.0.6), not general
  assembly. Fixed (SensCache threaded through `beam_con_jac`, exact): **1.13 s/iter raw
  vs ~26 s/iter recorded ≈ 23×**; medium sizing now ~6 min. See findings.md.
- **V.0.2 Vectorize the per-element Python work** — DEPRIORITIZED after the V.0.1 fix:
  post-fix profile shows no dominant wall (assembly 24%, shell recovery 17%, sensitivity
  10% of a small total). Re-profile before investing here (payoff returns if V.2 pushes
  n_levels up).
- **V.0.3 KS aggregation of the vector beam constraints.** DEPRIORITIZED after the
  V.0.1 fix: the n back-substitutions cost ~1.7 s per 10 medium iterations now.
  Reconsider only if fine meshes (V.2) or DV growth (P.1/P.2) bring the wall back.
- **V.0.4 Process-parallel multi-start and sweeps.** The multistart wrapper is
  "serial, parallel-ready"; V.2 and P.4 are embarrassingly parallel across points.
  A process pool gives near-linear core scaling with no numerical work.
- **V.0.5 Warm-start continuation as standard sweep protocol.** Each sweep point
  starts from its neighbor's optimum; fine meshes start from resampled coarse-mesh
  solutions (the `resample_segment_radii` pattern). Attacks iteration *count*, which
  V.0.2–V.0.4 don't touch (examples 39/40 already proved warm-starting converges).
- **V.0.6 Opportunistic cleanups.** ~~The uncached `_beam_vm_grad_one` path~~ (DONE
  2026-06-10 — this WAS the wall; see V.0.1); sparse Cholesky instead of `splu` only
  if the profile shows factorization mattering at finer meshes (it doesn't today:
  `splu` ≈ 0.3 s / 10 medium iterations).
- **V.0.7 Engineering guardrails. — CI LANDED (2026-06-10).** `sizing` marker on the
  19 SLSQP tests measured ≥5 s; fast subset = **149 tests / 12.2 s** on every push/PR,
  sizing job on main + nightly (`.github/workflows/ci.yml`). gmsh already declared.
  Still pending: example smoke layer (2–3 examples at maxiter≈10 — note example 14
  is currently broken: imports `meshio`, no longer in the venv) and the pairwise
  config-flag smoke matrix. (Toolbox #2, #3, #5)
- **V.0.8 Python version.** Migrated to **3.13** (2026-06-09; full suite green,
  160/160). **3.14 is blocked solely by build123d 0.10.0** (`requires-python <3.14`,
  OCP pin `<7.9`; the cp314 OCP wheels already exist and build123d's dev branch
  already supports `<3.15` + OCP 7.9) — bump to 3.14 when build123d's next release
  ships; that's where the real interpreter gains live (tail-call interpreter,
  incremental GC).

V.0.1 runs now; the rest pull in as the profile directs. Don't block V.1 on this —
shadow prices is a few independent lines.

### Phase V — Validity hardening

Make the model behave like the real structure before harvesting levers. Three of
these fixes may move the honest baseline *up*; lever gains must be measured against a
trustworthy baseline, not the current one.

- **V.1 Shadow prices. — DONE (2026-06-10).** `result.shadow_prices` (FD-validated
  0.02–0.5%; `examples/42_shadow_prices.py`). Headline (4-band): twist **−41.3 kg/deg**
  (cheapest requirement), beam/panel buckling SF +266/+152 kg/SF-unit, deflection +
  σ_allow free. Renegotiate twist first; V.3 prices the buckling SFs in fidelity terms.
  See findings.md. (P#8)
- **V.2 Mesh-convergence study of the optimized design. — DONE (2026-06-10).**
  `examples/43_mesh_convergence.py`: n_levels 6→10 gives 2807→2486→2271 kg (−9–11%
  per step, no plateau, all feasible/buckling-governed) — headlines NOT mesh-converged,
  bias unconservative as predicted; 16×12 diverged (diagnostic). 20-beam spot ≈ equal
  total mass, consistent with the V#1 ∝n mis-scaling — P.4 stays gated. Conclusion
  feeds V.3: fix the physics, not the mesh. See findings.md. (V#9, V#3)
- **V.3 Eigenvalue buckling check (K + Kσ). — DONE (2026-06-10).**
  `structural/geometric_stiffness.py` + `structural/eigen_buckling.py`
  (reference-validated), `examples/45_eigen_buckling.py`. Verdict: worst λ_cr = 2.443
  vs the closed-form's claimed 1.5 (≥1.6× conservative; λ → 5.67 on finer meshes of
  the same design → lower bound); worst mode = distributed skin-normal waving;
  hoop-pretension parametric is **null** at this optimum (λ 2.443→2.445, mode cluster)
  → P.2 demoted. Follow-up that harvests this: **V.3b width-based panel check**
  (physical strip width + per-direction D, calibrated against the eigen solve) —
  fixes the level AND the ∝n mis-scaling, unlocks P.4, then V.6 re-baseline.
  See findings.md. (V#1, V#2)
- **V.3b Width-based panel-buckling check. — DONE (2026-06-10).** Opt-in
  `panel_width_mode="strip"` (`skin_panel_widths`); analytic Jacobian exact in both
  modes (FD-validated). Calibration: implied capacity 5.14× at the old optimum vs
  converged eigen ~5.7 (conservative side, ~10%). **Harvest: medium 1-band 2471.4 →
  2006.3 kg (−18.8%)**, eigen-verified (λ_cr 2.286 ≥ 1.5). Compression-direction D
  (V#2) deliberately left open; kc=4 SS-edge caveat documented. Default flips to
  "strip" at the V.6 re-baseline; P.4 unlocked. `examples/46_width_based_panels.py`.
  (V#1)
- **V.4 Self-weight + inertial/heel load cases. — DONE (2026-06-10).**
  `accel_vectors` body loads + the λᵀ·∂f/∂x adjoint term (FD-validated).
  Measured: +15.5% on the medium strip baseline with upright + 30° heel gravity
  (2006.3 → 2316.9 kg). The 1 g lateral slam static-equivalent is envelope-dominating
  (diagnostic: → +110%, unconverged) — **deferred to the V#12 load-envelope
  decision**; V.6 standard set = upright + heel. `examples/47_self_weight.py`. (V#5)
- **V.5 Distributed panel pressure + skin bending stress in the failure check. —
  DONE (2026-06-10).** Skin-distributed projection ~neutral (−1.6%, noise floor;
  adopted as standard), strip-bending failure term σ_b = 0.75qw²/t² measures zero
  effect at medium scale (skin strength margin ~30×) — kept as the guard for
  thin-skin / high-pressure / large-scale regimes. Buckling–pressure interaction +
  Tsai-Wu bending deferred (recorded). `examples/48_panel_pressure.py`. (V#4)
- **V.6 Re-baseline. — DONE (2026-06-10).** New eigen-verified headlines (strip +
  upright/heel gravity + distributed pressure; slam deferred): **small 25.13 kg
  (4-band, λ_cr 2.00) / medium 2248.0 kg (1-band, λ_cr 2.26)**. Medium 4-band found
  only a worse basin (+4.4%) — banding under strip-mode buckling is being settled by
  the P#9 parallel multistart. Beam-buckling SF now carries the dominant shadow
  price (+531 kg/SF) → P.1 hollow members is the right next lever. Artifacts
  exported (`examples/49_rebaseline.py`). All P-levers measure against these.

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

- **P.0 Sizer extensibility refactor. — DONE (2026-06-10).** `DesignVector`
  (named-block x layout) + `ConstraintSpec` (one list entry per constraint; scipy
  dicts, Jacobian registration, and shadow-price attribution derive from it).
  Behavior-preserving verified: 189 tests green + the V.6 medium headline reproduced
  bit-exactly. IPOPT fallback unchanged (trigger: SLSQP stalls or DVs > ~150 — watch
  during P.1). (Toolbox #1, #4)
- **P.1 Hollow members.** New member type: the straight filament-wound **core tube**
  (permanent structural mandrel) with `r(z)`, `t_wall(z)` DVs + wall-buckling/crimping
  check. If the tube alone doesn't capture the win, add straight in-wing wound beam
  segments spliced to RTM'd solid curved transition segments. Likely the biggest
  single lever (beams are solid rods today). (P#1)
- **P.2 Skin prestress + compression cross-members. — DEMOTED (2026-06-10):** the
  V.3 eigen parametric shows λ_cr 2.443→2.445 under 5–20 MPa hoop pretension at the
  current optimum (the critical-mode cluster is pretension-insensitive); revisit only
  if the binding set changes after V.3b/V.6. Original scope: winding pretension as a
  self-equilibrated load case (the build already commits to the tensioned hoop wrap);
  spreader struts / shear webs concentrate the equilibrating compression into short,
  buckling-cheap members. New DVs: strut radii + pretension magnitude(s); the adjoint
  gains a `λᵀ·∂f_pre/∂x` term (prestress load depends on DVs). Use a
  direction-resolved (biaxial) buckling check — hoop pretension is transverse to the
  spanwise panel compression. Guard with a prestress-retention knockdown and
  strut-foot peel checks. (P#2, P#3)
- **P.3 Sandwich (cored) skin.** Core multiplies panel D at ~5% of the monolithic mass
  cost; needs wrinkling/shear-crimping checks. Partially redundant with P.2 — measure
  which wins. (P#5)
- **P.4 n_beams sweep.** Panel *width* is the proven currency; beam count has never
  been varied. **Gated on V.3's panel-model verdict**: the current `b = √area` check
  mis-scales with beam count (∝n vs ∝n² physically) and falsely credits length-cutting
  members like rings, so sweeping before the fix would understate the win and mis-rank
  rings vs beams. (P#4)
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
