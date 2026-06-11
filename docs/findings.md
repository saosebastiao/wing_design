# Wing Design — Findings & Decisions Archive

> Extracted from `plan.md` on 2026-06-09, when plan.md was repurposed to plan the
> improvement build-out (see [`improvement_backlog.md`](./improvement_backlog.md)).
> This is the durable record of the analyses, findings, and decisions from the original
> shell-beam build-out (Phases 1–F, complete). **Append new findings and decisions
> here**; plan.md holds only the forward plan.

## Headline results

Small wingsail (5 m span, NACA 0018, n_beams=16, n_levels=8) progression:

| Design increment | Mass | Governing constraint | Example |
| --- | --- | --- | --- |
| Phase-C beams-only (ring frame) | 43.5 kg | tip deflection | `25_resize_with_skin.py` (baseline leg) |
| + load-bearing skin co-sized (E.2) | 27.5 kg | tip twist | `25_resize_with_skin.py` |
| + CLT anisotropic skin (E.4, per-tri-local) | 25.6 kg | tip twist | `26_clt_skin.py` |
| + span ply datum, no buckling (E.4b) | 19.0 kg | deflection + twist | `28_ply_datum.py` |
| **Fully constrained:** datum + buckling SF 1.5 | **30.15 kg** | beam + panel buckling | `32_final_design.py` |
| + 4 spanwise thickness bands | **27.67 kg** (−8.1% vs its 30.11 kg uniform re-run) | tip twist | `33_banded_skin.py` |
| **V.6 re-baseline (2026-06-10):** strip-width buckling + gravity/heel + distributed pressure, eigen-verified | **25.13 kg** (4-band, λ_cr 2.00) | twist + beam + panel buckling | `49_rebaseline.py` |

Medium wingsail (22 m span; symmetric+monotonic radii, span datum, buckling SF 1.5):

| Run | Mass | Notes | Example |
| --- | --- | --- | --- |
| FD Jacobian (4-band) | 2316.6 kg | converged feasible, 1 h 22 m | `37_sized_export.py` |
| Analytic Jacobian (free tip, 4-band) | 2264.6 kg | ~2% better basin than FD; legacy `b=√area` checks | `39_tip_gusset.py` |
| **V.6 re-baseline (2026-06-10):** strip + gravity/heel + pressure, eigen-verified | **2248.0 kg** (1-band, λ_cr 2.26) | twist + beam + panel buckling | `49_rebaseline.py` |

The V.6 numbers are the **current headlines** (all Phase-P levers measure against
them); pre-V.6 numbers are historical (legacy `b=√area` buckling, aero-only loads).

## The governing-physics picture (cross-cutting findings)

1. **The design is stiffness/buckling-governed, never strength-governed.** Phase C found
   26× beam stress margin at the deflection limit; the proper per-ply Tsai-Wu check
   confirmed it (min strength ratio R = 7.0 vs required 2.0). Strength constraints have
   never bound at any increment.
2. **The load-bearing skin is the dominant lever.** Coupling it cut mass 37%; it then
   becomes ~2/3–3/4 of total mass, and the only consistently positive mass lever found
   afterwards is direct skin tailoring (per-band thickness, −8.1%).
3. **Coherent (manufacturable) anisotropy pays.** A consistent span ply datum was
   simultaneously buildable AND 26% lighter than the incoherent per-triangle-local
   optimum — chordwise (hoop) fibres are the skin's efficient role, with the
   longitudinal beams carrying spanwise bending.
4. **Buckling binds and reshapes the design** (+2% mass; material moves from beams into
   thicker skin), and constraints compound: manufacturable datum + buckling together
   (30.15 kg) is heavier than either alone (19.0 / 26.1 kg).
5. **Beam-layout levers are dead or marginal while panel buckling governs.** F.1
   stress-weighted spacing ~−1%; F.2 diagonal lattice +83%; tip gusset +8.3%;
   mirror-symmetric re-spacing +2.7%. Even spacing minimizes the largest panel, which is
   what a panel-buckling-governed skin wants. Only when the deflection budget is
   tightened to co-bind does re-spacing turn slightly positive (−0.6%).
6. **Twist, though often pinned at its limit, is not the mass driver** — the tip gusset
   eliminated twist (5.0°→0.03°) and still came out 8.3% heavier, because the rigid
   coupling redistributes load in a buckling-governed structure.
7. **Sizing cost is gated by FEA solves × Jacobian width × maxiter**, not inner-loop
   Python (the Tsai-Wu vectorization lesson). The analytic adjoint + ∂K caching got
   1.98× (exact); the remaining gap to the ~n_DV ceiling is the n adjoint
   back-substitutions of the vector beam constraint (KS aggregation is the next lever).
8. **Optimization hygiene matters:** SLSQP needs O(1) normalization (raw Pa-vs-metre
   scales make the QP ill-conditioned); it finds local optima (FD and analytic landed in
   basins ~2% apart), so same-config comparisons under ~2–3% are within noise; and
   wall-clock must be measured, not estimated (estimates ran up to ~10× off).

---

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


## Phase history & findings

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

**Multi-start sizing done (2026-06-08) — machinery only (not run at scale).** A serial,
parallel-ready `beams.size_beam_shell_laminate_multistart` runs the sizer from N initial
guesses and returns the best **feasible** result (`laminate_result_is_feasible`): start 0
is the default guess (so it never regresses), starts 1.. are seeded uniform-random within
`laminate_design_bounds` (simplex-projected). The sizer gained an opt-in `x0` param
(default None = unchanged). Per the cost realities (one sizing is hours; multi-start is
N×), **no full multi-start was executed** — the orchestration (start-0-default,
best-feasible selection, determinism, never-worse) is verified with a *stubbed* sizer,
and `examples/36_multistart.py` is a minutes-scale validation harness left for on-demand
use. The levers for a real headline are parallel execution (structured for it, not
implemented) and/or an analytic constraint Jacobian to make each sizing cheaper. This
closes the deferred-followup queue (vectorize Tsai-Wu / per-band layup / multi-start);
next is Phase G/H.

**Beam symmetry + monotonic taper done (2026-06-08) — now the sizer default.**
`size_beam_shell_laminate` enforces, by default: mirror-symmetric beam radii (mirror-
paired beams share one radius DV via `beams.beam_radius_groups`, which auto-detects
symmetric placement and falls back to independent radii if asymmetric) and monotonic
non-increasing radius from keel-step to tip (cheap algebraic constraints). This ~halves
the radius DVs (16 beams → 9 unique groups) — fewer FD-Jacobian columns — for a faster,
more physical solve. `result.radii` is still the full per-element length `n` (expanded),
so geometry/stock-catalog consumers are unaffected. **This changes the sizer default**;
earlier headlines assumed independent per-element radii and are historical. **Full
medium-wingsail run** (`examples/37_sized_export.py`; symmetric+monotonic, span datum,
4 thickness bands, buckling SF 1.5, von-Mises skin): **converged feasible at 2316.6 kg**
(beams 585 + skin 1731; twist pinned 5.0°, beam & panel buckling util = 1.0) in 187
iters, **wall-clock 1 h 22 m** — a converged, feasible headline where the prior
unconstrained per-band coarse run could not converge. The sized beams taper monotonically
**35.7 mm (keel) → 4.0 mm (tip)**, mirror-symmetric. The resulting structure was lofted
(lens-section beams + OML skin) and **exported as STL meshes**
(`exports/wingsail_sized_{skin,beams,assembly}.stl`). NB the 2316 kg is the **medium
(22 m)** wingsail — a ~2-tonne structure by surface area — not comparable to the small
(5 m) ~30 kg headlines.

**Analytic (adjoint) constraint Jacobian done (2026-06-09) — correct, modest speedup.**
Opt-in `LaminateSizingConfig.use_analytic_jacobian` (default False; FD path byte-identical)
supplies SLSQP analytic gradients for the objective + all six FEA constraints via the
**adjoint** method, reusing one `splu(K_ff)` factorization (`structural.beam_shell.
solve_beam_shell_laminate_factored` + `beams.sensitivity`). Every gradient is
**FD-validated** (element ∂K/∂x primitives, adjoint engine, and each constraint to
rel-err ≤ ~1e-4; 27 sensitivity/failure tests). TDD caught two real derivation gaps: the
recovered internal force `floc=klocal(r)·T·u` adds a `∂klocal/∂r` term (radius gradient
was 13× off without it), and `beam_con` is **vector-valued** (one row per beam element) so
its Jacobian is the full `(n,nx)` matrix. `examples/38_analytic_speedup.py` (small problem,
n_beams=12/n_levels=6, datum+buckling+4 bands): analytic and FD reach the **same feasible
optimum** (31.15 vs 31.13 kg, buckling 1.0) with analytic **1.58× faster (489 s → 309 s)**.
**The 1.58× is far below the ~n_DV× ceiling**, throttled by (a) the vector `beam_con` needing
~n adjoint solves per gradient and (b) `∂K/∂x` re-assembled in pure-Python loops each adjoint
(the design-dependent per-element derivative matrices aren't cached). The method is sound;
the realized gain is implementation-bound. **Path to the big win (follow-up):** cache the
per-element/per-triangle ∂K matrices once per design point (reused across `beam_con`'s n
rows) and/or aggregate the beam-stress constraint (KS/max → 1 adjoint), and vectorize the
assembly. Default stays FD until that lands.

**Analytic-Jacobian caching done (2026-06-09) — exact, 1.58×→1.98×.** Implemented the
first follow-up above: `beams.sensitivity.prepare_sensitivity(ds, factored) -> SensCache`
precomputes every beam element's `∂kg/∂r` and every triangle's `∂ke/∂t`,`∂ke/∂f0`,`∂ke/∂f45`
(the trig/Q-transform builds) **once per design point**; `lambdaT_dK_x_cached` reuses them as
cheap `λ[dofs]·(∂k·u[dofs])` dots. Each `grad_*` takes an optional `cache=` (lazy-builds if
None, so the FD-validation tests are unchanged), and `evaluate_jac` builds one cache per
design point and threads it through all constraints (reused across `beam_con`'s n rows and
all load cases — the cache is design-only). **Exact:** gradients are bit-for-bit identical
(same math, not recomputed) — cached==uncached to ≤1e-12, and every existing FD-grad test
passes unchanged. Re-measured `examples/38` (same small problem): FD 546.4 s → analytic
**276.2 s = 1.98× faster** (up from 1.58×), same feasible optimum (31.13 vs 31.15 kg,
buckling 1.0). The remaining gap to the ~n_DV× ceiling is now the n adjoint *back-substitutions*
for the vector `beam_con` (KS/max aggregation → 1 adjoint is the deferred next lever; one
non-cached path, `_beam_vm_grad_one`, still calls `lambdaT_dK_x` directly — a candidate
follow-up). Default stays FD.

**Tip gusset in the sizer done (2026-06-09) — NEGATIVE result (heavier).** A rigid,
massless tip-gusset (stiff connector-beam clique among the tip-ring nodes) is now opt-in
in the laminate sizer (`build_beam_shell_model(tip_gusset_radius=...)` /
`model_with_tip_gusset`; assembled into K but excluded from beam-force recovery, so it
composes with the analytic Jacobian for free — the gusset is constant stiffness). At a
fixed design it cuts tip twist ~23× (and the investigation showed ~50×). But sizing the
medium wingsail **with** the gusset, warm-started from the free-tip optimum and run with
the analytic Jacobian (both converged feasible, buckling 1.0): free-tip **2264.6 kg →
gusset 2453.1 kg (+8.3%)**, even though it eliminates tip twist (5.0°→0.03°) and stiffens
the tip (defl 156→71 mm). **Why heavier:** the design is buckling-governed; the rigid tip
coupling **redistributes load** so the free-tip optimum becomes infeasible with the gusset
(members overloaded) and the optimizer must add material. Twist, though pinned at the
limit, was not the mass driver — killing it frees nothing while the coupling costs
material. Confirms the original tip-coupling investigation (a twist/stiffness lever, not a
mass lever). **Recommendation:** don't use the tip gusset for mass; keep it opt-in for
twist/aeroelastic-stiffness purposes. (Aside: the analytic free-tip run found 2264.6 vs
the FD 2316.6 kg — exact gradients reached a ~2% better basin.)

**Mirror-symmetric non-uniform spacing done (2026-06-09) — NEGATIVE result (heavier).**
`beams.chord_symmetrize_weights` (max-of-mirror) makes the per-segment skin-stress weights
chord-symmetric so stress-weighted placement (`stress_weighted_targets` via the existing
`arc_fractions` path) yields a mirror-symmetric layout that **keeps `beam_radius_groups`
grouping** (verified: symmetric-spaced n_groups = even n_groups = 63 — the radii stay
mirror-paired). `examples/40_symmetric_spacing.py` (medium, span datum + buckling + 4
bands, analytic Jacobian, both converged feasible): even **2264.6 kg → symmetric-weighted
2325.2 kg (+2.7%)**, despite a real stress concentration (max/mean segment = 2.45).
**Why heavier:** clustering beams toward the high-stress mirror regions leaves *larger
panels* in the gaps, and the design is **panel-buckling-governed** — bigger panels force
more material (beam mass 557→626 kg; skin ~flat). Even spacing minimizes the largest
panel, which is what a buckling-governed skin wants. So even spacing is near-optimal here
and stress-weighted re-spacing is counterproductive — consistent with F.1's low leverage,
now clearly *negative* under the current symmetry+taper+banded+buckling model. (F.1's
earlier ~1%-lighter was asymmetric and predated this stack.) Even spacing stays the
default; the helper is kept (built+tested) for completeness. **Takeaway across F.1 / F.2
diagonals / tip gusset / this:** the design is robustly skin/panel-buckling-dominated, so
beam-layout levers don't reduce mass — only direct skin tailoring (per-band thickness,
−8.1%) does.

**Re-spacing under a *binding* deflection budget (2026-06-09) — small positive (−0.6%).**
Re-ran even-vs-symmetric-weighted on the medium wingsail with the tip-deflection budget
*tightened to bind* (0.5% span = 110 mm; the default 2% is slack, deflects ~156 mm only at
the unconstrained optimum). Both converged feasible at **defl 110/110 mm and buckling
1.00/1.00** (deflection and panel-buckling *co-bind*): even **2362.1 kg → symmetric-weighted
2347.9 kg = −14.3 kg (−0.6%)** (beams 479→501, skin 1883→1847). So once *stiffness* governs
rather than buckling alone, clustering beams toward the high-stress regions flips re-spacing
from negative (+2.7% at the slack budget) to a small positive — but buckling still co-binds,
so the skin keeps doing most of the work and the win stays marginal. Reinforces the
cross-cutting lesson: layout levers move the needle only at the margin; skin tailoring is
the real lever. (1 h 41 m, analytic Jacobian.)

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

### Phase V.0 — Iteration-speed enablers

**V.0.1 profile + beam-Jacobian cache fix done (2026-06-10) — ~23× faster medium
gradient path (exact, no formulation change).** `examples/41_profile_sizing.py`
cProfiles 10 SLSQP iterations of the medium 16×8 analytic-Jacobian sizing (the
example-32 config). Profile verdict: **93% of wall-clock was `beam_con_jac`** — each
of the n=112 vector-vM rows called the *uncached* `lambdaT_dK_x`, rebuilding every
triangle stiffness derivative per row (`tri_element_stiffness_laminate`: 856,576
calls / 148 s cumulative in 10 iterations); assembly+solve was ~12 s and `splu`,
SLSQP, and the closed-form buckling checks were ~0 — so the suspected per-element
assembly loops were NOT the wall, the known-uncached `_beam_vm_grad_one` path was
(plan V.0.6, flagged in the analytic-Jacobian-caching decision row). Fix: thread the
per-design-point `SensCache` (already built in `_build_jac`, previously discarded by
`beam_con_jac`) into `_beam_vm_grad_one` → `lambdaT_dK_x_cached`. Exact: all 21
sensitivity-FD + analytic-vs-FD equivalence tests pass unchanged. **Measured:**
profiled self-time 181.8 → 19.8 s for the same 10 iterations (9.2× like-for-like);
raw uninflated wall **1.13 s/iter** vs ~26 s/iter on the recorded 1h22m/187-iter
medium run (**~23×**). A full 300-iter medium sizing now projects to ~6 min.
**Why:** the SensCache contraction replaces ~3×695 18×18 element-stiffness rebuilds
per constraint row with cached dots; the rebuild cost was paid n_beam_elements times
per gradient evaluation. Post-fix profile: K assembly 24%, shell recovery 17%,
sensitivity 10% of a now-small total — no single dominator left, so **V.0.2
(vectorize assembly) and V.0.3 (KS aggregation) are deprioritized** until a measured
wall reappears (KS's n back-substitutions now cost ~1.7 s per 10 iterations).
(Profile runs: 3 m 09 s before / 23 s after, measured 2026-06-10.)

### Phase V — Validity hardening

**V.1 shadow prices done (2026-06-10) — twist is the cheapest requirement on the
headline (−41.3 kg/deg); buckling SFs carry the rest; deflection and strength are
free.** SLSQP's KKT multipliers (`res.multipliers`, scipy ≥ 1.15) are captured in
`size_beam_shell_laminate` and converted to physical dm*/dparam on
`LaminateSizingResult.shadow_prices` (for `g = 1 − v/L`: dm*/dL = −m_ref·λ̃/L;
SF-type constraints get +m_ref·λ̃/SF; beam rows share σ_allow so their multipliers
sum). Conversion FD-validated by re-optimizing a small problem at perturbed limits:
agreement **0.5% (h=5%) / 0.02% (h=2%)** (`tests/beams/test_shadow_prices.py`).
`examples/42_shadow_prices.py`, medium 16×8, both configs, converged AND feasible:
**1-band (ex-32 config)** 2471.7 kg, 244 iters, **383 s**: twist 1.23°/5° slack →
price ≈ 0; panel-buckling SF **+352.2 kg/SF-unit**, beam **+235.5** (SF 1.5→1.4 =
−59 kg, −2.4%). **4-band (the 2264.6 kg headline, reproduced exactly, cold start)**
2264.6 kg, 273 iters, **452 s**: twist binds → **−41.35 kg/deg** (5°→6° = −41 kg,
−1.8%); beam-buckling SF **+266.5**, panel **+151.7 kg/SF-unit**; deflection (156/440
mm) and σ_allow price at zero in both. **Why:** prices are the constraint-space view
of the governing-physics picture — buckling + (with banding) twist bind, nothing else
does. **Implications:** (1) the 5° twist limit is an admitted heuristic now costing
41 kg/deg — the cheapest kilograms available, renegotiate or kill twist at the tip if
requirements allow; (2) both buckling SFs guard *closed-form approximations* — V.3's
eigenvalue solve prices the same kilograms in model-fidelity currency (~42 kg per
0.1 SF combined); (3) a small-problem caveat: twist-bound small cases price near zero
because the optimizer buys twist with mass-free layup fractions — prices are
basin-local and config-dependent (1-band vs 4-band flip the binding set). Wall-clocks
measured post-cache-fix; 42's two runs shared the machine with the V.2 sweep
(contention ≤ minor). (383 s + 452 s, analytic Jacobian.)

**V.2 mesh-convergence study done (2026-06-10) — headline masses are NOT
mesh-converged; refinement removes ~9–11% per step with no plateau, in the
unconservative direction predicted by V#3/V#9.** `examples/43_mesh_convergence.py`:
medium config (16 beams, span-datum + buckling, analytic Jacobian), n_levels swept
6→12 warm-started coarse→fine (`resample_segment_radii`), n_beams=20 spot at 8
levels. Converged AND feasible points: **16×6 2807.1 kg (144 s) → 16×8 2486.5 kg
(235 s) → 16×10 2271.0 kg (390 s)** — every point buckling-governed (beam & panel
util = 1.00), twist/defl slack; mass falls −11.4% then −8.7% per refinement with no
flattening. **16×12 FAILED (diagnostic, not a result):** SLSQP quit at 83 iters
unconverged + infeasible (beam-buck util 19.4 — the optimizer had stripped beams to
156 kg before the early exit). **n_beams spot:** 20×8 = 2485.6 kg ≈ 16×8 (+0.0%
total) but with beams +31% / skin −10% — more beams *should* win big physically
(panel width: σcr ∝ n²) yet the implemented `b = √area` check credits only ∝ n
(backlog V#1), consistent with the mass-neutral outcome; do not run the P.4 sweep
before fixing the panel model. **Why:** the beam Euler check uses element length as
buckling length (refinement shortens L → raises Pcr) and the panel check uses
b = √(triangle area) (refinement shrinks b → raises σcr); both let the optimizer
legally remove real mass as the mesh refines. **Implication:** mesh-converged
headlines do not exist under the current closed-form checks — the honest fix is a
physical buckling length / V.3's eigenvalue solve, not a finer mesh; the 16×8
headline numbers stay comparable to each other but their absolute level carries an
unquantified mesh bias (bracketed +13%/−9% by the neighboring meshes). 16×8
warm-started here reached 2486.5 vs 2471.7 cold in ex-42 (+0.6%, inside the 2–3%
noise floor). (Sweep total ≈ 24 min measured, shared the machine with ex-42's runs
for its first ~14 min.)

**P#2 prestress probe done (2026-06-10) — the hoop-pretension prize is large and
invisible to the scalar buckling check.** `examples/44_prestress_probe.py`: medium
optimum (1-band config, 2471.7 kg, converged+feasible, 382 s re-size), parametric
chordwise (hoop) pretension superposed on the recovered per-panel membrane stresses
of all 4 load cases, two metrics: (a) the implemented scalar most-compressive-
principal check on the net stress, (b) a direction-resolved linear biaxial
interaction in the span/chord datum frame (util = SF·comp_span/σcr_span −
SF·tens_chord/σcr_chord + SF·comp_chord/σcr_chord, per-direction σcr from the
optimized band layup's D11/D22). **Measured:** (a) 1.000 → 0.949 (5 MPa) → 0.880
(≥50 MPa, saturated — ~12% max credit); (b) 1.174 (0 MPa) → 0.724 (5 MPa) → 0.307
(10 MPa) → 0.000 (≥20 MPa). **Why:** the hoop wrap tensions the chord direction,
transverse to the spanwise compression — a scalar principal-stress offset barely
moves (the principal circle shifts, the compressive branch stays), while a biaxial
interaction credits transverse tension directly. **Implications:** (1) ~5–10 MPa
*retained* pretension (winding at 10–20 MPa with the 50% retention knockdown of
V#10) neutralizes most of the binding panel-buckling utilization — potentially the
skin-mass lever P.2 hoped for; (2) the sizer cannot price this without a
direction-resolved panel check or the V.3 eigenvalue solve (prestress enters Kσ);
(3) **one-sided bound:** the probe superposes tension WITHOUT its equilibrating
compression (which lands in beams/spreader struts and must be bought) and the
linear interaction is a crude plate approximation; (b)'s zero-pretension level
(1.17 > 1.0) also shows the scalar and biaxial checks are not mutually calibrated —
compare within columns only. V.3's eigen solve with a prestress load case is the
decision-grade follow-up. (Probe post-processing ≈ seconds on top of the re-size;
analytic Jacobian.)

**V.3 eigenvalue buckling check done (2026-06-10) — closed-form buckling is
conservative by ≥1.6× (likely 2–4×) at the binding point; the prestress prize is
~zero at the current optimum; the in-loop fix is a width-based panel check, not a
finer mesh.** New machinery: `structural/geometric_stiffness.py` (consistent beam Kg
from axial force; w-linear triangle Kg from local membrane stress) +
`structural/eigen_buckling.py` (dense Cholesky-reduced `K φ = −λ Kσ φ`), validated
against closed form (Euler columns ≤0.2% at 8 elements, SS-plate kc=4 ≤5% at 12×12,
tension → no mode; `tests/structural/test_eigen_buckling.py`).
`examples/45_eigen_buckling.py` on the medium optimum (1-band config, 2471.7 kg,
converged+feasible, 374 s re-size; closed-form beam & panel util = 1.000): worst
**λ_cr = 2.443** (lc3; per-case 12.1 / 3.8 / 3.5 / 2.4; loads pre-factored, so the
closed-form claims capacity at exactly 1.5) — eigen modes exported to
`exports/eigen_mode_lc*.vtu` (regenerate: `just example 45_eigen_buckling`). Worst
mode is **skin-normal waving (89% normal motion) spread across both surfaces and
most of the span** (peaks z≈3.3/7.0 m), modes clustered 2.44–3.07.
**Mesh honesty of λ itself (same design resampled, eigen-only re-solves, 1–2 s each):
λ_cr = 2.443 (16×8) → 3.847 (16×12) → 5.669 (16×16)** — rising because (a) the
linear membrane stress field redistributes with mesh (the known 8.6×→2.4× skin
stiffening shift) and (b) the mesh topology has no nodes *between* beam lines, so no
member of this mesh family can represent the physical across-width panel half-wave:
treat 2.443 as a **lower bound** on the model's eigen capacity at the optimum.
**Eigen-grade pretension parametric (one-sided, worst case): λ_cr 2.443 → 2.445 at
5–20 MPa hoop pretension** — the panel mechanism lifts slightly and the critical
mode immediately flips to a pretension-insensitive global mode at the same λ:
**the P.2 prize at this optimum is bounded by the mode-cluster spacing (~15%), not
the ex-44 interaction-formula bound** (that formula models an isolated panel
mechanism the coupled structure doesn't exhibit at this design point).
**Interpretation:** even the harshest SP-8007-style shell knockdown (γ≈0.65) on the
*lower-bound* λ gives 1.59 ≥ 1.5 (and the plate-like mode character argues for a
much milder knockdown) — the optimum is not buckling-deficient; the conservatism is
real mass on the table. The audited `b=√area` level bias ((0.85/0.46)² ≈ 3.4×,
backlog V#1) is consistent with the fine-mesh λ ≈ 5.7. **Implications:** (1) the
in-loop fix is a **width-based panel check** (physical strip width + per-direction
D), calibrated against this eigen machinery — it fixes both the level and the ∝n
mis-scaling, unlocking P.4; (2) P.2 prestress is demoted until the binding set
changes; (3) beam element-length Euler (V#3) is subsumed: at λ ≥ 2.4 the beams are
not the critical mechanism at this optimum. (Sizing 374 s + eigen solves ~0.1–2 s
each; analytic Jacobian.)

**V.3b width-based panel check done (2026-06-10) — −18.8% on the medium design
(2471.4 → 2006.3 kg), eigen-verified; the strip formula lands within ~10% of the
converged eigen capacity.** `skin_panel_widths(model)` gives each skin triangle the
physical strip width (chordwise distance between its two bounding beam lines —
median **0.449 m vs 0.897 m** for the old `√area`, a ~4.3× σcr level credit);
opt-in `LaminateSizingConfig.panel_width_mode="strip"` substitutes width² for area
at both buckling call sites, so the analytic Jacobian stays exact (FD-validated;
b is design-independent). Default remains `"sqrt_area"` until the V.6 re-baseline.
`examples/46_width_based_panels.py` (medium 16×8, 1-band config, all converged AND
feasible): **(1) calibration** — the strip check at the √area optimum reads worst
util 0.292 = implied capacity 5.14×, vs the V.3 eigen 2.44 (16×8 lower bound) and
~5.7 (16×16, same design): within ~10% of the converged eigen value, conservative
side. **(2) harvest** — re-sized strip-mode optimum **2006.3 kg vs 2471.4 kg
baseline = −18.8%** (145 iters, 193 s; twist now binds at 5.0° alongside beam+panel
buckling at 1.0 — the V.1 twist shadow price now applies to the lighter design too).
**(3) eigen verification** — worst λ_cr of the new optimum **2.286 ≥ 1.5**: every
mechanism the mesh can represent (beam, global, coupled) keeps ≥52% margin; the
across-width panel half-wave the mesh cannot see is exactly what the strip formula
checks. **Why:** σcr ∝ 1/b² and the physical b is half the √area surrogate, so the
binding constraint was ~4× too strict in level — V.3's verdict converted to mass.
**Caveats:** kc=4 assumes simply-supported strip edges (flexible beams could be
softer — the eigen check guards the coupled modes it can see); D11 direction
mismatch (V#2) still open; NOT yet the official headline — the 4-band config +
V.4/V.5 load additions come first (V.6 re-baseline). CAD of the strip optimum is
exportable via the example-37 sized-export path with `panel_width_mode="strip"`.
(321 s baseline + 193 s strip re-size + eigen seconds; analytic Jacobian.)

**V.4 self-weight + inertial loads done (2026-06-10) — gravity costs +15.5% on the
medium strip baseline; a 1 g lateral slam is envelope-dominating and goes to the
V#12 requirements decision.** `beams/body_loads.py` lumps the design's own mass to
nodes under arbitrary acceleration vectors (`LaminateSizingConfig.accel_vectors`,
cross-product with the aero cases; loads rebuilt every evaluate since they depend
on the design), and the analytic adjoint gains the **`λᵀ·∂f/∂x` design-dependent-
load term** — every touched gradient FD-validated at rel-err ≤ 2e-4
(`tests/beams/test_body_loads.py`; this term is also the prerequisite the P.2
prestress machinery would have needed). `examples/47_self_weight.py`, medium 16×8
strip-mode (V.3b config): **A aero-only 2006.3 kg** (reproduces the V.3b optimum
exactly; 145 iters, 201 s) → **B + upright & 30°-heel gravity 2316.9 kg = +15.5%**
(converged AND feasible, 260 iters, 720 s, twist + both bucklings binding) →
**C + 1 g lateral slam: DIAGNOSTIC, not a result** — unconverged AND infeasible at
maxiter 400 (4219.4 kg and still beam-buckling-violated, 2127 s): the
static-equivalent slam roughly doubles structural demand. **Why:** the 2.3-tonne,
22 m cantilever's own weight under heel is a first-order bending load exactly as
backlog V#5 predicted ([↑]); a >1 g lateral acceleration doubles the lateral load
envelope on a structure whose binding constraints were set by aero bending.
**Decisions:** (1) upright + heel gravity join the standard load set (V.6
re-baseline includes them); (2) the slam static-equivalent is a **requirements
question (V#12)** — excluded from the V.6 default pending an envelope decision
(options: justified lower factor, dynamic analysis, or accepting the ~2× mass);
(3) warm-start C from B and/or IPOPT if the envelope keeps it. (201 s + 720 s +
2127 s measured, analytic Jacobian.)

**V.5 panel pressure + strip-bending done (2026-06-10) — skin-distributed
projection is ~neutral (−1.6%, inside the noise floor); the pressure-bending term
measures ZERO effect at medium scale because skin strength carries ~30× margin.**
`examples/48_panel_pressure.py`, medium 16×8 strip-mode, peak panel pressure
1.9 kPa, all three runs converged AND feasible: **A node-lumped 2006.3 kg**
(reproduces the baseline exactly) → **B skin-distributed 1973.7 kg (−1.6%)** —
within the 2–3% noise floor, so claimed as *neutral*, but adopted as the V.6
standard anyway (physically cleaner: per-triangle CST lumping, total force
conserved, no single-node snapping) → **C + strip-bending failure term: 1973.7 kg,
bit-identical trajectory to B** (same iters/wall). **Why C changes nothing here:**
the skin von-Mises sits at 34 MPa vs 1100 MPa allowable — the audited ~7× Tsai-Wu
strength margin is ~30× on this lighter strip-mode design, so augmenting a deeply
slack constraint cannot move the optimum. The σ_b = 0.75·q·w²/t² term (FD-validated
gradient) stays in as the guard for regimes where it CAN bind: thin-skin small
wings, slam pressures (V#12), and the >100 m retargeting where q·w²/t² scales up.
Pressure–compression buckling interaction and Tsai-Wu bending remain recorded
caveats (spec 2026-06-10-panel-pressure-design.md). (198 s + 232 s + 232 s
measured, analytic Jacobian.)

**V.6 re-baseline done (2026-06-10) — new headlines: small 25.13 kg / medium
2248.0 kg, all eigen-verified; the honest model is now LIGHTER than the old one
(strip-width fix outweighs the gravity penalty).** `examples/49_rebaseline.py`,
the post-sprint standard configuration: strip-width panel buckling (V.3b) +
upright/30°-heel gravity with the λᵀ·∂f/∂x term (V.4; slam deferred by user
decision) + skin-distributed projection + strip-bending guard (V.5) + span-datum
CLT + beam Euler SF 1.5, analytic Jacobian, 16×8. All four runs converged AND
feasible, each eigen-verified over every load combo: **small 1-band 25.65 kg
(λ_cr 2.14, 165 s) / small 4-band 25.13 kg (λ_cr 2.00, 239 s)** — vs the old
27.67 kg banded headline, **−9.2% net of gravity**; **medium 1-band 2248.0 kg
(λ_cr 2.26, 784 s) / medium 4-band 2345.9 kg (λ_cr 2.44, 466 s)**. **4-band
anomaly (recorded honestly):** warm-started from the 1-band optimum, the 4-band
run converged +4.4% heavier in a different basin (beam-buckling-dominated, skin
thickened to 4.6–6.2 mm, panel constraint slack, beam-buck SF shadow price
+1201 kg/SF) — since the 1-band design is admissible in the 4-band space, the
**defensible medium headline is 2248.0 kg (1-band)**; the P#9 12-start parallel
multi-start on the 4-band config is running to settle whether banding still pays
under the new constraint set (banding's −8.1% was measured under `b=√area`
panel buckling, which strip mode just removed — the lever may genuinely be gone).
Shadow prices on the new medium baseline: twist −50.7 kg/deg (binding, still the
cheapest requirement), beam-buck SF +531 kg/SF (now the dominant SF — the strip
fix moved the price from panels to beams), panel-buck SF +68 kg/SF, deflection
free. **Artifacts (milestone):** `exports/rebaseline_medium_beams.stl` (lens
beams, radii densified 8→40 levels) + `exports/rebaseline_medium_worst_mode.vtu`;
regenerate via `just example 49_rebaseline`. All Phase-P levers measure against
these numbers. (165+239+784+466 s sizing + eigen seconds, measured.)

**P#9 multi-start at scale done (2026-06-10) — banding is DEAD under the honest
model (best 4-band +0.9% vs 1-band, inside noise); measured basin scatter 2.9%;
the parallel machinery works (12 starts / 41 min).** First at-scale run of
`size_beam_shell_laminate_multistart` with the new V.0.4 process pool (12 starts,
11 workers, medium 4-band V.6 config, maxiter 500): **wall 2489 s** for 12 runs
(~2.9× over serial-estimate — stragglers at high iteration counts + core
contention bound the speedup; parallel==serial verified bit-exact by test).
8/12 starts converged feasible, spanning **2267.4–2332.5 kg (2.9% basin scatter** —
the noise floor measured directly at medium scale); 4 infeasible (random cold
starts are poor; the default start was feasible). **Best feasible 4-band 2267.4 kg
vs the 1-band headline 2248.0 kg = +0.9%** across 13 total 4-band attempts
(12 starts + ex-49's warm start). **Why banding died:** the −8.1% banding lever
(2026-06-08) was earned under the legacy `b=√area` panel check — thinning outboard
bands paid because panel buckling over-priced the skin everywhere; with the V.3b
strip widths the panel constraint is honest and the optimizer can no longer
harvest it band-by-band. **Decisions:** (1) the medium headline stays **2248.0 kg
(1-band)**; (2) **1-band becomes the standard P-phase config** (fewer DVs, faster,
nothing lost within noise); banding revisits only if a future lever re-prices the
skin (e.g. P.3 sandwich). 1-band 12-start basin check running for the same
scrutiny of the headline itself. (2489 s measured, 11 workers.)

**P#9 follow-up: 1-band headline basin check (2026-06-10) — the 2248.0 kg headline
is basin-robust.** Same 12-start parallel protocol on the medium 1-band V.6 config:
**12/12 converged feasible, wall 1404 s**; best 2242.5 kg (−0.25% vs the
default-start 2248.0 — inside noise), worst 2421.6 (+7.7% — random cold starts can
land in clearly worse basins, reinforcing warm-start discipline). The quoted
headline stays **2248.0 kg** (default start: deterministic and reproducible via
`just example 49_rebaseline`); best-of-12 confirms no materially better basin
exists under this formulation. (1404 s measured, 11 workers.)

**P.1 core tube done (2026-06-10) — NEGATIVE result at medium scale, clear lesson:
the optimizer zeroes a centroidal tube; the hollow-member lever must go to the
form beams (P#1b).** Full machinery built and FD-validated: annular sections
(`BeamSection.annular`), tube geometry on the pivot axis with per-segment fit
bounds and stiff massless bonds (`build_beam_shell_model(core_tube=True)`),
`r_tube(S)`/`t_wall(S)` DV blocks via the P.0 `DesignVector`, annulus validity +
monotonic-taper linear specs, the NEW wall-crimping check (σcr = 0.65·0.605·E·t/r,
SP-8007-class knockdown) as a vector `ConstraintSpec`, annulus ∂K
(`dkloc_annular`) through the SensCache/adjoint with tube explicit terms in the
vM/Euler gradients, tube columns in the V.4 body-load ∂f/∂x — wall-crimping
gradient FD-validated ≤2e-4 and the analytic tube optimum is FD-stationary
(`tests/beams/test_core_tube.py`). `examples/50_core_tube.py`, medium 16×8 V.6
config, both runs converged AND feasible, eigen-verified (λ 2.24/2.21):
**baseline 2261.4 kg → tube 2260.7 kg (−0.0%)**, with the optimizer pinning
r_tube = 20 mm (the lower bound; fit bounds allowed 233–378 mm) and t_wall = 1 mm
— a vestigial 5 kg tube, wall util 0.00, beams unchanged at 812 kg. **Why:** the
tube sits on the pivot axis — the section centroid — where added material has
near-zero bending leverage; the existing structure is already a monocoque of
depth 0.5–1.9 m, so a centroidal tube's I (∝ r³t) is negligible against the OML
section. The textbook (R/t)× tube advantage applies to a standalone column, not
to a member at the neutral axis of a far deeper section. **Implications:**
(1) P#1a (core tube as a mass lever) is dead at this scale — it remains a
manufacturing aid (mandrel/assembly datum) whose ~5 kg is affordable;
(2) the hollow lever redirects to **P#1b: hollow straight FORM-beam segments**
(at the OML where the leverage lives; beam-buck SF still carries +514 kg/SF on
the baseline) — gated on per-beam path-straightness verification (M#4); (3) a
second ulp-association regression (body-load product regrouping, baseline
2248.0→2261.4 cold-start shift, inside noise) was caught by the acceptance
protocol and bit-compat restored — the protocol works. (636 s + 1037 s measured,
analytic Jacobian.)

**P.1 CORRECTION (2026-06-10) — the earlier core-tube NEGATIVE was an artifact of
a missing bond assembly (user-caught); the bonded tube is a REAL lever: −8.3%
(2248.0 → 2060.7 kg), eigen-verified, acting as a TORSION SPINE, not a bending
member.** The sizer's solve calls assembled only `tip_gusset_elements`;
`tube_bond_elements` never reached the sizing FEA, so the earlier run optimized a
tube connected to the wing only at its clamped root — structurally useless by
construction (the FD-gradient tests and the eigen check passed their own solves
*with* bonds, which is why every validation was green around the bug). Fix: tip
gusset and tube bonds compose into the solver's stiff-massless path; tube+gusset
tests pass; bonded FD-vs-analytic cold runs agree at 0.4%. A third ulp-association
leak (body-load product regrouping) was caught by the acceptance protocol and
restored — **the baseline control then reproduced bit-exactly (2248.0 kg, 290
iters, shadows to the digit)**. `examples/50_core_tube.py` re-run, both converged
AND feasible: **baseline 2248.0 kg (λ_cr 2.26, 776 s) → bonded tube 2060.7 kg
(λ_cr 2.49, 1030 s) = −8.3% (−187 kg)**. **Mechanism:** the optimizer keeps the
tube MINIMAL — r_tube pinned at the 20 mm lower bound, 1 mm walls, 5.8 mm at the
root segment, 8 kg total — so the centroidal-bending argument from the invalid
finding still holds (a centroidal tube buys no bending). What it buys is
**torsion**: the closed section to the root un-binds the twist constraint (shadow
−50.7 → 0.0 kg/deg) and the skin sheds 180 kg of torsion-carrying plies
(1440 → 1260 kg); beams −16 kg. Binding set is now beam-buck (+455 kg/SF) +
panel-buck (+208 kg/SF), twist free. Wall-crimping never binds (util 0.00).
**Implications:** (1) P#1a re-verdicted POSITIVE as a torsion spine — the wingsail
wants its pivot-axis tube after all (manufacturing concept vindicated: the
"optional structural mandrel" earns 187 kg); (2) running-best medium mass is
**2060.7 kg**; (3) beam-buck SF still carries +455 kg/SF → **P#1b hollow form
beams remains the next lever**; (4) follow-ups: tube_r_min sweep (r pinned at the
bound), CAD export of the tube cylinder in the sized-export path (trivial,
pending), wrapped-joint model for the bond stiffness (M#5 — bonds are idealized
stiff). Lesson recorded: a "plausible mechanism" fitted to a measurement is not a
finding until the model's load path is verified — the bond check belongs in the
smoke tests. (776 s + 1030 s measured, analytic Jacobian.)

**P#1b hollow form beams done (2026-06-10) — −6.6% on top of the tube
(2060.7 → 1924.6 kg, eigen λ 2.54); hollow-only is NO win — the hollow and tube
levers are complementary, not independent.** Implementation: the P.1 annular
machinery generalized to any annular element via explicit DV-column arrays (a
hollow form beam's r-column IS its existing radius group; walls get a
`("t_hollow", H)` block, one per hollow group, mirror sharing preserved);
in-wing elements tagged at build with straightness asserted (measured exactly
0.0 mm — ruled planform); wall-crimping rows span the combined annular set;
solid stays continuously reachable at t = r. FD-validated (cold FD-vs-analytic
agree ≤2%; wall-crimp gradient ≤2e-4). `examples/51_hollow_beams.py` + warm
protocol: **(1) hollow-only 2274.9 kg (+1.2% vs the 2248.0 baseline, noise/worse
basin; converged+feasible, eigen λ 2.74)** — walls drop to the 1 mm floor, beams
−92 kg but the lost axial stiffness pushes +119 kg into the skin while twist
still binds: hollow beams do NOT pay without the torsion spine. **(2) tube +
hollow, cold start: DIAGNOSTIC** — 1562 kg at maxiter 500, infeasible, **eigen
REJECTED (λ 0.91)** — the verification layer caught a genuinely buckling-deficient
trajectory. **(3) tube + hollow, warm-started from the feasible tube optimum with
solid-equivalent walls (control reproduced 2060.7 exactly): 1924.6 kg, converged
AND feasible, eigen λ 2.538** — beams 808 → 674 kg (hollow, walls mostly at the
1 mm floor), skin 1302 → 1246, tube 5 kg. **Running-best medium: 1924.6 kg
(−14.4% vs the V.6 2248.0).** Binding economics: beam-buck +326 / panel-buck
+259 kg/SF, twist free. **Caveats:** 1 mm walls = 4 wound plies (buildable;
M#2 ply rounding + M#4 RTM-solid transition splices recorded); the cold
diagnostic hints at lighter basins reachable only infeasibly — continuation/
multistart later; wrapped-joint stiffness idealized (M#5). (Control 1049 s +
warm 1128 s + variants 1358/1393 s + eigen seconds; analytic Jacobian.)

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
| Multi-start sizing | Serial parallel-ready `size_beam_shell_laminate_multistart`: N seeded starts (start 0 = default → never worse), best-feasible-by-mass selection (`laminate_result_is_feasible`); sizer gained an `x0` param + `laminate_design_bounds` helper. Machinery built + unit-tested via a stubbed sizer; NOT run at scale (cost). Parallel exec + full converged headline deferred. |
| Beam symmetry + monotonic taper | Sizer DEFAULT: mirror-paired beams share one radius DV (`beam_radius_groups`, auto-detect symmetric placement + fallback), radius monotonic non-increasing keel→tip (algebraic constraints). ~Halves radius DVs → faster FD-Jacobian. Changes the default (prior headlines historical); `result.radii` still full length n. Full medium run converged feasible: 2316.6 kg (beams 585 / skin 1731), twist 5.0° / buck 1.0, 187 iters, 1 h 22 m; beams taper 35.7→4.0 mm symmetric. Sized geometry exported as STL (`exports/wingsail_sized_*.stl`). |
| Analytic (adjoint) Jacobian | Opt-in `use_analytic_jacobian` (default off; FD byte-identical): adjoint analytic gradients for objective + all 6 FEA constraints, one reused `splu` factorization (`beams.sensitivity`, `solve_beam_shell_laminate_factored`). Every gradient FD-validated (≤~1e-4); TDD caught the ∂klocal/∂r internal-force term + the vector-valued beam_con. Same feasible optimum as FD; measured **1.58× faster** (489→309 s, small problem). Speedup implementation-bound (vector beam_con = n adjoint solves; uncached Python ∂K assembly) — follow-up: cache per-element ∂K + aggregate beam constraint for the big win. |
| Analytic-Jacobian caching | First follow-up to the above, **exact** (no formulation change): `prepare_sensitivity → SensCache` precomputes every element/triangle ∂K (the trig/Q-transform builds) once per design point; `lambdaT_dK_x_cached` reuses them as cheap dots; `grad_*` take an optional `cache=` (lazy if None → FD tests unchanged); `evaluate_jac` builds one cache per design point, threaded through all constraints + load cases. Cached==uncached ≤1e-12; all FD-grad tests pass unchanged. `examples/38` re-measured: FD 546→analytic **276 s = 1.98×** (was 1.58×), same optimum. Remaining gap = n adjoint back-subs for vector beam_con (KS aggregation = deferred next lever; `_beam_vm_grad_one` still uncached). Default stays FD. |
| Re-spacing under binding deflection | Re-ran even-vs-symmetric-weighted spacing (medium) with the tip-deflection budget tightened to **bind** (0.5% span = 110 mm; default 2% is slack). Both feasible, defl & panel-buckling **co-bind** (110/110 mm, 1.00/1.00): even 2362.1 → symmetric-weighted 2347.9 kg = **−14.3 kg (−0.6%)**. So once stiffness governs (not buckling alone), re-spacing flips from +2.7% (slack) to a small positive — but buckling co-binds so the win stays marginal. Confirms layout levers help only at the margin; skin tailoring is the real lever. (1 h 41 m, analytic Jac.) |
| Tip gusset in the sizer | Opt-in rigid massless tip-node clique (`build_beam_shell_model(tip_gusset_radius=...)`/`model_with_tip_gusset`), assembled into K but excluded from force recovery → composes with the analytic Jacobian (constant stiffness). **Negative for mass:** medium free-tip 2264.6 → gusset 2453.1 kg (+8.3%), both feasible/converged, despite twist 5.0°→0.03° and tip defl 156→71 mm. Buckling-governed design + rigid coupling redistributes load → must add material. Twist not the mass driver. Keep opt-in for twist/stiffness, not mass. |
| Mirror-symmetric non-uniform spacing | `chord_symmetrize_weights` (max-of-mirror) → symmetric stress-weighted arc placement that keeps `beam_radius_groups` grouping (verified n_groups unchanged). **Negative for mass:** medium even 2264.6 → symmetric-weighted 2325.2 kg (+2.7%), both feasible; stress concentration 2.45 real, but clustering enlarges gap panels and the design is panel-buckling-governed → more material. Even spacing (minimizes max panel) is near-optimal; re-spacing counterproductive. Even stays default; helper kept. |
| Phase-F.2 diagonal beams | Balanced both-hand grid-helix lattice on existing grid nodes (`beams.helix_elements`, no remesh), co-sized with one shared diagonal-radius DV in the SLSQP laminate loop; pitch chosen by principal-stress alignment (`recommend_pitch`, best pitch 2 @ align 0.68). **Strong negative result:** baseline 33.1 kg → diagonal 60.6 kg (+83%), diagonals add 20.8 kg at a buckling-forced 4.7 mm radius, twist rose 1.25°→1.58°. The design is buckling-governed with large twist slack, so long compression diagonals bloat mass without relieving a binding constraint. Lattice abandoned for this regime; twist (when binding) is killed far cheaper at the tip. Streamline-following (F.3) not recommended while buckling dominates. Implementation was a throwaway spike, NOT merged — only the finding is kept. |
| Tip-coupling study | Hard tip joint (gusset) modeled as a stiff connector-beam clique tying the tip nodes (`beams.solve_beam_shell_tip_coupled`, tunable `gusset_radius`), reusing `solve_beam_shell` (no rigid MPC, no penalty hacks). Finding: barely redistributes BEAM stress (peak −2%, spread 3.75→3.38) — the skin already shares spanwise load — but near-eliminates **tip twist** (0.197°→0.004°, ~50×) and stiffens the tip (~14%), saturating at low gusset stiffness. Investigation only (no CAD / not in the sizing loop). Implication: the twist-governed design could be relaxed/lightened by a tip gusset (re-size-with-gusset = follow-up). |
| P#1b hollow form beams (2026-06-10) | Annular machinery generalized (r-col = existing radius group; t_hollow block per group). Hollow-only: +1.2% = NO win (axial-stiffness loss feeds the skin; twist binds without the tube). Tube+hollow warm-started: **1924.6 kg, eigen λ 2.54 — running best, −14.4% vs V.6**; beams 674 kg at ~1 mm walls (4 plies). Cold start was eigen-REJECTED at λ 0.91 (verification layer works). Levers are complementary: spine frees twist, hollow walls then harvest the beams. Next: P.4 n_beams sweep (user-requested) on this config. |
| P.1 CORRECTION (2026-06-10) | Earlier negative INVALID — tube bonds never assembled in the sizer (user-caught). Bonded re-run: **2248.0 → 2060.7 kg (−8.3%), eigen λ 2.49** — the tube stays minimal (8 kg, r at the 20 mm bound) and acts as a **torsion spine**: twist constraint un-binds (−50.7 → 0 kg/deg), skin sheds 180 kg of torsion plies. Centroidal-bending uselessness confirmed; torsion value missed by the artifact. Running-best medium 2060.7 kg; beam-buck +455 kg/SF → P#1b next. Follow-ups: tube_r_min sweep, tube CAD export, M#5 joint model. |
| P.1 core tube (2026-06-10) | Full annular-member machinery built + FD-validated (sections, fit-bounded r/t DV blocks, wall-crimping check w/ 0.65 knockdown, annulus ∂K through the adjoint). **NEGATIVE at medium scale: optimizer zeroes the tube** (r→20 mm bound, −0.0% mass, eigen 2.21 ✓) — a centroidal tube has no bending leverage inside a 0.5–1.9 m-deep monocoque. P#1a dead as a mass lever (kept as manufacturing aid); hollow lever redirects to **P#1b hollow form-beam segments** (OML leverage; beam-buck SF +514 kg/SF still dominant). Machinery reusable for P#1b. |
| P.0 sizer refactor (2026-06-10) | `DesignVector` (named blocks = single source of x-layout) + `ConstraintSpec` (name, closures, rows, shadow conversion per constraint; scipy dicts/Jacobian registration/multiplier attribution derive from one list). Behavior-preserving: 189 tests green, V.6 medium headline reproduced bit-exactly (2248.0284 kg, 290 iters, shadows to the digit). Adding a constraint or DV block is now one append — the P.1 gate is open. Bonus lesson: a 1-ulp change (sqrt(area)² → area in panel b²) flipped an FD cold-start basin 10% on a small problem — bisected, sqrt-roundtrip deliberately retained in the legacy path for bit-reproducibility; cold-start optima are ulp-sensitive, warm starts + deterministic default starts are the protocol. |
| P#9 multi-start at scale (2026-06-10) | 12 parallel starts (V.0.4 pool, 11 workers, 41 min) on the medium 4-band config: best feasible 2267.4 kg = +0.9% vs the 1-band 2248.0 → **banding dead under strip-mode buckling** (its −8.1% was a legacy-panel-model artifact); 1-band becomes the standard P-phase config. Feasible basin scatter 2.9% = the noise floor measured at medium scale. Parallel==serial bit-exact (tested); speedup ~2.9× (stragglers + contention). |
| V.6 re-baseline (2026-06-10) | New eigen-verified headlines under the honest model (strip widths + upright/heel gravity + distributed pressure): **small 25.13 kg** (4-band, λ 2.00; −9.2% vs old 27.67) / **medium 2248.0 kg** (1-band, λ 2.26; old 4-band 2264.6 not comparable — legacy checks, aero-only). Medium 4-band landed +4.4% heavier in a beam-buck-dominated basin (warm start included) → banding's value under strip-mode buckling unresolved, P#9 multistart running. Beam-buck SF is now the dominant price (+531 kg/SF) — beams are the next physics to attack (P.1 hollow members). STL + worst-mode VTU exported (`just example 49_rebaseline`). |
| Slam envelope deferred (2026-06-10) | DECISION (user): the 1 g lateral slam case (V#12; ex-47 diagnostic: → +110%, infeasible at maxiter 400) is re-introduced only AFTER the Phase-P performance harvest (hollow members, etc.) lightens/strengthens the structure. V.6 standard load set stays aero × {upright, 30° heel} gravity. When re-introduced: warm-start from the gravity optimum; IPOPT if SLSQP stalls. |
| V.5 panel pressure + bending (2026-06-10) | Skin-distributed projection (force-conserving CST lumping) −1.6% vs node-lumped = neutral (noise floor) but adopted as V.6 standard. Strip-bending failure term σ_b = 0.75qw²/t²: zero effect at medium scale (skin vM 34 vs 1100 MPa — strength margin ~30×); kept as guard for thin-skin/high-pressure/large-scale regimes. Buckling-pressure interaction + Tsai-Wu bending deferred (caveats recorded). |
| V.4 self-weight/inertial (2026-06-10) | `accel_vectors` body loads (mass lumped per evaluate) + the λᵀ·∂f/∂x adjoint term (FD-validated ≤2e-4; also the P.2 prerequisite). Medium strip baseline 2006.3 → **2316.9 kg (+15.5%)** with upright + 30° heel gravity (converged/feasible). 1 g lateral slam: diagnostic — unconverged/infeasible at maxiter 400, mass heading +110% → slam envelope deferred to the V#12 requirements decision; V.6 default = upright + heel only. |
| V.3b width-based panels (2026-06-10) | Opt-in `panel_width_mode="strip"`: b = physical chordwise beam spacing (median 0.449 vs 0.897 m √area surrogate). Calibration: implied capacity 5.14× at the old optimum ≈ converged eigen 5.7 (conservative side). Harvest: medium 1-band **2471.4 → 2006.3 kg (−18.8%)**, converged+feasible; twist now co-binds. Eigen verify of new optimum: λ_cr 2.286 ≥ 1.5. Fixes V#1 level + ∝n scaling → P.4 unlocked. Default stays sqrt_area until V.6 re-baseline (after V.4/V.5). kc=4 edge + V#2 direction caveats open. |
| V.0.7 CI split (2026-06-10) | `sizing` pytest marker assigned from measured durations (19 tests ≥5 s; top 192/166/99 s). Fast job: 149 tests / **12.2 s measured** on every push/PR; sizing job on main pushes + nightly (uv, locked sync). Full suite green 168/168 post-cache-fix. Example smoke layer + flag matrix still open; example 14 broken (meshio gone from venv) — smoke layer would have caught it. |
| V.3 eigenvalue buckling (2026-06-10) | K+Kσ linear buckling built + reference-validated (columns ≤0.2%, plate kc=4 ≤5%). Medium optimum: worst λ_cr = 2.443 vs the closed-form's claimed 1.5 → ≥1.6× conservative; λ rises to 5.67 on finer meshes of the same design (stress-field redistribution + no nodes between beam lines) → 2.44 is a lower bound. Worst mode = distributed skin-normal waving. Pretension moves λ 2.443→2.445 (null at this optimum) → P.2 demoted. Even γ=0.65 knockdown × 2.443 = 1.59 ≥ 1.5 → design not deficient; conservatism is harvestable. Next: width-based panel check calibrated by eigen (fixes level + ∝n scaling), then P.4. |
| P#2 prestress probe (2026-06-10) | Hoop pretension superposed on the medium optimum's panel stresses: scalar principal check sees ≤12% credit (saturates 0.88), direction-resolved biaxial interaction goes 1.17→0.72 (5 MPa)→0.31 (10 MPa)→0 (20 MPa) — the P.2 prize is large but requires a biaxial panel check or the V.3 eigen solve to price; probe is one-sided (no equilibrating compression, no retention knockdown applied). P.2 sizing work stays gated on V.3. |
| V.2 mesh convergence (2026-06-10) | n_levels 6→10 (feasible points): 2807→2486→2271 kg, −9–11% per refinement, no plateau — headlines NOT mesh-converged; bias is unconservative (element-length Euler + b=√area both gain capacity from refinement). 16×12 diverged (diagnostic). n_beams 20×8 ≈ 16×8 total mass, consistent with the V#1 ∝n mis-scaling muting the physical ∝n² panel win — P.4 sweep stays gated on the panel-model fix. Headline comparisons remain valid at fixed 16×8; absolute level carries mesh bias. V.3 (eigenvalue/physical buckling length) is the fix, not finer meshes. |
| V.1 shadow prices (2026-06-10) | KKT multipliers captured + converted to kg-per-unit (`shadow_prices` on the result; FD-validated 0.02–0.5%). Medium headline (4-band, 2264.6 kg reproduced exactly): twist −41.35 kg/deg (binding, cheapest requirement), beam-buck SF +266.5 / panel-buck SF +151.7 kg/SF-unit, deflection + σ_allow free. 1-band: buckling-only (panel +352.2). Renegotiation order: twist limit first, then buckling SF — which V.3 prices in model-fidelity terms. |
| V.0.1 profile → cache fix (2026-06-10) | cProfile of 10 medium SLSQP iters (examples/41): 93% of wall = uncached `_beam_vm_grad_one` rebuilding triangle ∂K per beam-vM row. Fixed by threading the existing `SensCache` through `beam_con_jac` (exact; 21 equivalence/FD tests unchanged). 1.13 s/iter raw vs ~26 s/iter recorded → ~23×; medium sizing now ~minutes. V.0.2 vectorization + V.0.3 KS deprioritized — no dominant wall remains; re-profile before investing further. |
| Python 3.13 migration (2026-06-09) | `requires-python >=3.13,<3.14` (was `<3.13`), `.python-version` 3.13, lock regenerated; **full suite green 160/160 in 19 m 30 s** (measured — suite is NOT fast; CI needs a fast/slow marker split). 3.14 blocked solely by build123d 0.10.0 (`<3.14` + OCP `<7.9`; cp314 OCP wheels exist, build123d dev branch already supports `<3.15`+7.9) — bump when its next release ships. |
