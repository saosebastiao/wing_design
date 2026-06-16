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
| Chain re-baseline (2026-06-12): + tube + hollow + sandwich + foundation + datum_ortho (V#2-coherent), IPOPT+KS, eigen-verified | 1094.2 kg (λ_cr 3.61, converged feasible; −51.3% vs V.6) | KS buckling families | `runs/chain_rebuild.py` (stage D) |
| **Multistart re-baseline (2026-06-12):** same final config, 8 starts (D-seeded start 0), best feasible basin | **1021.6 kg** (λ_cr 2.76, converged feasible; **−54.5% vs V.6**) | KS buckling families | `runs/multistart_v2.py` |

| ~~3 g slam SURVIVAL rating (2026-06-16): rings 20 mm, failure-only~~ — **INVALID at converged mesh** | ~~1575.3 kg (λ_cr 3.77)~~ → converged λ ≤ 1.25 < 1.5 (n8 value was a ~3× artifact); **true survival mass is HEAVIER, TBD** | slam buckling under-resolved at n8; closed-form non-conservative for slam | `runs/slam_survival.py` (needs converged-mesh resize) |

The multistart number (**1021.6 kg**) is the **current medium headline for the
operational (no-slam) envelope** (V.6 remains the lever-accounting baseline); the
warm chain stage-D 1094.2 kg is the attributed-lineage figure. The **1575.3 kg**
row is the **3 g-slam-survival-rated** mass (buckling-governed; a single design
that also satisfies the operational envelope). Pre-V.6 numbers are historical
(legacy `b=√area` buckling, aero-only). **Mesh note (2026-06-16, RESOLVED):** the
n=8 eigen *verification* is mesh-sensitive (it overstiffens then overshoots), but
the design is sized to the well-calibrated closed-form SF 1.5; the **converged
(n≥24) buckling margin of the 1021.6 kg headline is λ ≈ 1.5–1.6 — VALID**, sized
right to its margin (NOT optimistic; an earlier "true margin ~1.3, masses heavier"
read was the n=12 overshoot, since corrected). Buckling should be *verified* at
n≥24 (cheap, eigen-only) on the cluster floor, not at 16×8. **The 3 g survival
1575.3 kg row is INVALID** — its λ 3.77 was an n=8 artifact; converged λ ≤ 1.25
< 1.5 (no plateau by n=48), so the real survival mass is heavier and TBD (the
closed-form is non-conservative for slam; see the survival mesh-eigen finding).

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
9. **A chord-normal inertial (slam) load has no efficient load path in a pure
   spanwise-cantilever topology.** At 3 g lateral the optimizer spent +53% mass and
   still couldn't reach a stable structure (eigen λ 0.81); the beams carry spanwise
   bending and the skin braces them, but nothing carries a large transverse inertial
   load except added *lateral* bracing (cross-members / diagonals). The aero+gravity
   envelope hid this because those loads are predominantly spanwise/chordwise in the
   beams' strong planes. Survival-class slam is therefore a *topology* requirement,
   not a sizing one — and it's where the shelved P.2/F.2 transverse-member levers
   finally earn their place. Corollary: the closed-form/foundation beam-buckling
   model is non-conservative under slam (reported feasible at true λ 0.81), so the
   eigen check must be in-loop for any slam-sized result.

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
**[Annotated 2026-06-16 — the layup ratios in this and the other E-phase findings (E.4b, per-band skin layup) describe the PRE-foundation regime; the current foundation+sandwich optimum is pure ±45 (0/100/0), confirmed coherent by V#2 (2026-06-12).]**

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
**[Annotated 2026-06-16 — this +8.3% negative is AERO-regime (the gusset is a twist/stiffness aid, not a mass lever there); it is not the verdict under slam loading.]**

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
**[Annotated 2026-06-16 — AERO-regime, marginal; not evaluated under the failure-only slam envelope.]**

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
**[Annotated 2026-06-16 — AERO-regime, marginal; not evaluated under the failure-only slam envelope.]**

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
**[Annotated 2026-06-16 — this +83% negative was under the AERO-buckling regime. Under SLAM, lateral bracing has a BINDING role: ring bracing makes 3 g survivable (λ_cr 3.16 at 20 mm rings, 2026-06-15), so the diagonal/cross-member family is superseded-by-ring-bracing for slam, not abandoned.]**

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
**[Annotated 2026-06-16 — prestress itself stays null under aero, but the COMPRESSION CROSS-MEMBER half of P.2 now has a binding role under slam (covered by the merged ring-bracing work, 2026-06-15).]**

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
**[Annotated 2026-06-16 — the 1 g lateral slam figure (→ +110%) was computed against the 2%-span tip-DEFLECTION (serviceability) budget, the WRONG formulation for a survival/ultimate case (only failure should govern); the mass is deflection-overstated. Failure-only re-run pending (plan V.7).]**

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
**[Annotated 2026-06-16 — the "1924.6 kg running best" is the PRE-correction number (solid-rod beam-mass reporting bug); corrected to 1784.5 kg, current headline 1021.6 kg. The levers-complementary verdict is unchanged.]**

**P.4 n_beams sweep done (2026-06-10) — 16 beams is the measured optimum; above
it the design goes EIGEN-limited (n=24 is closed-form-feasible but eigen-REJECTED
at λ 1.22), a result the legacy panel model could never have shown.**
`examples/52_nbeams_sweep.py`: n ∈ {12, 16, 20, 24, 28} at the running-best
config (medium 16×8-levels, V.6 standard + core tube + hollow form beams), five
parallel processes, each point running the full protocol (tube-only cold →
tube+hollow warm with solid-equivalent walls → eigen verification).
**Valid points: n=16 1924.6 kg (λ 2.54, the control, bit-exact) and n=20
1972.6 kg (+2.5%, λ 1.51 — at the floor).** n=24: 2123.0 kg closed-form
converged+feasible but **eigen-rejected (λ 1.22 < 1.5)**. n=12 and n=28:
diagnostics (unconverged/infeasible at one or both stages; n=28's 1425 kg is
meaningless). **Why:** more, thinner beams do thin the skin exactly as the strip
model predicts (skin 1246 → 1060 → 970 kg across 16/20/24) but beam mass grows
faster (674 → 908 → 1148 kg) AND the eigen-visible coupled modes erode
monotonically (λ 2.54 → 1.51 → 1.22): slender beams + narrow panels couple into
global modes the closed-form per-member checks cannot see. **Implications:**
(1) **n_beams = 16 stands** — the historical default is now a measured optimum
under the honest model; P.4 closes; (2) the λ-vs-n trend says future levers that
slim members should expect the EIGEN layer to become the active constraint —
when a config walks toward λ = 1.5, the in-loop eigen constraint (clean gradient
φᵀ(∂K + λ∂Kσ)φ, Toolbox #7) becomes the right investment; (3) per-point
convergence robustness degrades away from n=16 (12/24/28 stage failures) —
multistart/continuation needed if those counts ever matter. (Sweep wall 7719 s
total = 2 h 09 m for 10 sizings + 3 eigen verifications across 5 processes;
analytic Jacobian.)
**[Annotated 2026-06-16 — the cited 1924.6 kg is the PRE-correction number; corrected to 1784.5 kg (solid-rod beam-mass bug). The "16 beams is the measured optimum" verdict is unaffected.]**

**P.3 sandwich skin — machinery done + FD-validated; prize measured LARGE but
UNCLOSED: SLSQP cannot converge the skin→core migration and the documented IPOPT
trigger is now formally met (2026-06-11).** Implementation (committed, all
FD-validated ≤2e-4, core-off path bit-compatible): `CoreMaterial` (PVC-H80-class
default), sandwich D factor φ(t,c) = 1/4 + 3(c/t+1/2)² (exactly 1 at c = 0 —
monolithic continuously recoverable), per-band `t_core` DVs, Hoff face-wrinkling +
shear-crimping checks with a smooth c/(c+1 mm) activation, core mass in the
objective and the V.4 body loads, sandwich-aware ∂D/∂t + new ∂D/∂c through the
SensCache, sandwich branch in the panel-buckling gradient
(`tests/beams/test_sandwich.py`, 5 tests). **Measurement (medium running-best
config, warm from the 1924.6 kg optimum at c = 0 + 3 continuation restarts,
5662 s total): every leg stalls unconverged+infeasible while mass falls
1924.6 → 1543 → 1342 → 1339 → 1032 kg (−46%)** with the physics behaving exactly
as designed — 5 mm core, 1.0 mm faces, skin 1246 → 270 kg + 71 kg core, wrinkle
0.39 / crimp 0.59 (not the blockers), panel-buck shadow +567 kg/SF at the last
stall. **None of these numbers is a result** (nothing converged/feasible/
eigen-verified). **Why SLSQP fails:** the cored optimum is across a ~900 kg
design migration from the start point; the active set churns through
panel/wrinkle/crimp/Euler handoffs and SLSQP (dense active-set) exits with
direction errors — at 119 DVs and persistent stalls this is exactly the
documented IPOPT trigger (Toolbox #4). **cyipopt needs the Ipopt binary** (no
macOS arm64 wheel; `brew install ipopt` or conda — a machine-level decision) or
the casadi-bundled IPOPT via a callback adapter. **Decisions:** (1) the lever
stays OPEN with its machinery merged and the prize bounded below ~1032 kg-
trajectory level (likely hundreds of kg once closed properly); (2) next session:
IPOPT integration, then re-measure with the converged+feasible+eigen protocol;
(3) running best REMAINS 1924.6 kg until then. (Leg walls 1342 s + 5662 s
measured; analytic Jacobian.)

**P.3 follow-up (2026-06-11) — IPOPT integrated and properly warm-started; STILL
no convergence: the blocker is now isolated to formulation smoothness, and KS
aggregation (P#7) becomes the gate for the sandwich lever.** The Toolbox #4
fallback was wired (`LaminateSizingConfig.optimizer="ipopt"` via cyipopt 1.7.0 /
brew Ipopt 3.14.19, identical closures + analytic Jacobians; small-problem
equivalence test vs SLSQP passes). Medium cored runs, warm from the 1924.6 kg
optimum: **(a) IPOPT default options: 2000 iters, infeasible, wandered to
2080 kg** — interior-point barrier initialization re-centers the iterate,
discarding the warm start; **(b) IPOPT with warm_start_init_point + small
mu_init: 2000 iters, structurally coherent trajectory to 1249.2 kg (−35%),
STILL unconverged/infeasible** (3861 s). Combined with SLSQP's four stalls, both
optimizer families now fail the same migration for the same root cause: **every
max-based constraint row selects its binding element/load-case by argmax, so the
constraint Jacobians jump at switching points** — SLSQP's active set churns,
IPOPT's C² convergence assumptions break. **Decision: P.3 is gated on P#7 KS
aggregation** (smooth soft-max over elements AND load cases), which now carries
two independent motivations: C^∞ Jacobians for the optimizer and collapsed
adjoint back-substitutions. KS is conservative (over-estimates the max) —
re-validate vs hard-max per the established pattern; ρ ≈ 50 on normalized
constraints per the nlp-sizing conventions. Running best remains **1924.6 kg**;
the sandwich prize (trajectory floor ~1032–1249 kg across five attempts) is
real, large, and waiting on the smooth formulation. (5254 s + 3861 s measured.)

**Beam-constraint autopsy / IsoTruss question (2026-06-11) — the structure
already has its "helicals": the bonded skin is the bracing, node-level beam
stability has 69% margin, and the +326 kg/SF closed-form price is paid to a
sub-element model whose length is a mesh artifact → fix the MODEL (V.3c
beam-on-elastic-foundation), don't add members.** Asked (user): can IsoTruss-style
helical/cross members relieve the beam-buckling price? Eigen mode autopsy of the
1924.6 kg optimum (worst combo, 8 modes + a beam-only Kσ probe): **no
eigen-visible single-beam mode exists** (all modes distributed skin-field/coupled,
beam-line localization ≤ 0.43); **beam-only Kσ λ = 2.54 — identical to the full
mode 0** — the lowest mode's destabilizing energy is almost entirely beam axial
compression, but the shape is beams bowing WITH their bonded skin
(stiffened-panel mode) at 69% margin over the 1.5 requirement. The closed-form
element-length Euler (util = 1.0, +326 kg/SF) describes a SUB-ELEMENT bow the
mesh cannot represent, with unsupported length = node spacing — a mesh artifact
(the V.2 mesh-dependence had this as a driver); the physical configuration is a
stringer continuously co-bonded to the skin → beam-on-elastic-foundation,
Pcr = 2√(k·EI), typically ≫ π²EI/L². **Verdicts:** (1) IsoTruss's real lesson
(role separation: brace vs carry) is already embodied — skin + tube bonds ARE the
bracing; F.2's +83% negative tested helicals in the wrong role (primary
compression) and is consistent with this; (2) tie members now would buy relief
from a largely artificial constraint with real mass + joints (M#5) — SHELVED;
(3) **V.3c queued: eigen-calibrated foundation-length beam check** (V.3b
playbook) to harvest the +326 kg/SF honestly; (4) ties become relevant again in
the thin-skin sandwich endgame (foundation k softens) — re-check after P.3.
(Autopsy ≈ 3 min, post-processing only.)

**P.3 breakthrough via KS + IPOPT (2026-06-11) — first valid sandwich design:
1685.0 kg, hard-feasible, eigen λ 3.49 (the largest margin of any design yet);
optimizer not yet converged → recorded as an upper bound, polish leg running.**
KS smoothing (V.0.3/P#7, opt-in `ks_rho`; committed with property + end-to-end FD
tests over every family) + IPOPT, warm from the 1924.6 kg optimum at c = 0:
2000 iterations, **feas=True on every hard check for the first time in six
attempts** — the argmax-kink diagnosis was correct. Design: t_core 4.0 mm,
faces 2.06 mm, skin 561 + core 56 kg (vs 1246 kg monolithic), beams 1063 kg,
tube 5 kg; wrinkle 0.23 / crimp 0.70; **eigen-verified λ_cr = 3.488** (the core's
D-multiplication stiffens the panel field — eigen margin nearly doubles).
**Status: 1685.0 kg is a VALID DESIGN and upper bound (−12.4% vs the 1924.6
running best, −25.0% vs the V.6 baseline) but not a certified optimum**
(max_iter; KS slack remaining). Continuation polish from this feasible point is
running. (4662 s, IPOPT + KS ρ=50, analytic Jacobian.)
**[Annotated 2026-06-16 — the 1685.0 kg figure was overstated by the same solid-rod beam-mass accounting bug and was never recomputed; superseded by the foundation chain (1095.3 kg) then multistart (1021.6 kg).]**

**P.3 sandwich skin CLOSED (2026-06-11) — NEW RUNNING BEST: 1679.3 kg, converged
+ feasible + eigen λ 3.55 (−12.7% vs 1924.6; −25.3% vs the V.6 baseline).**
The continuation polish from the feasible 1685.0 kg point converged cleanly
(IPOPT + KS ρ=50, 520 iterations, 1198 s): **t_core 4.3 mm, faces 1.91 mm —
skin 519 + core 61 kg vs 1246 kg monolithic** (the panel-D arbitrage delivered
≈ −53% skin mass at 5% core density), beams 1095 kg, tube 5 kg; wrinkle 0.27 /
crimp 0.69; eigen-verified **λ_cr = 3.55**, the project's largest stability
margin. The full campaign for this one lever: machinery (FD-validated) → six
failed closure attempts across two optimizers → kink diagnosis → KS smoothing →
breakthrough → converged close; cumulative compute ≈ 6.1 h. **Caveats:** the KS
optimum is conservative (hard utilizations carry small slack — the documented
ρ=50 cost; a hard-max polish from here could shave a little more); IPOPT local
optimum (multi-start unexplored at this config); 1.9 mm faces ≈ 8 plies and
4.3 mm core are manufacturable (M#2 rounding pending). **Binding economics
flipped back to the beams (1095 kg, 65% of structure) → V.3c
(beam-on-elastic-foundation length, +326 kg/SF evidence from the autopsy) is
now the highest-leverage open item.** Running best: **1679.3 kg**.
**[Annotated 2026-06-16 — 1679.3/1685.0 kg were overstated by the solid-rod beam-mass accounting bug and never recomputed; superseded by the foundation chain (1095.3 kg) then multistart (1021.6 kg, current headline).]**

**V.3c foundation buckling model done (2026-06-11) — NEW RUNNING BEST: 1108.5 kg,
converged + feasible, eigen λ 3.61 (−34.0% in one step; −50.7% vs the V.6
baseline); the spokes alternative loses decisively.** Implementation (committed,
FD-audited end-to-end through the live KS closures): in-wing form beams checked
as stringers on the bonded-skin elastic foundation, Pcr = 2√(k·EI) with
k = 3·D22_chord·(1/w_l³ + 1/w_r³) from the adjacent strip widths and the band
laminate's chord-direction D (sandwich φ included); transition + tube elements
keep element Euler (their restraint spacing IS the node spacing); design chains
EI(r, t_hollow) and k(t_band, f0, f45, t_core) — the optimizer can now buy beam
capacity by stiffening the skin, pricing the bracing the structure physically
has. **Measurement (running-best config, warm from 1679.3, IPOPT+KS, 773 iters,
2093 s): 1108.5 kg — beams 1095 → 559 kg, skin 469 + core 75 (t_core grew to
5.4 mm partly to feed k), tube 5 kg; wrinkle 0.29 / crimp 0.59; eigen λ 3.61.**
**Spokes variant (IsoTruss-literal: bonds to every beam at every level +
element-Euler now physically justified): NOT competitive** — wandered to
3416.8 kg unconverged in 1728 iters (4706 s): 16 rigid bonds per level act as
hub diaphragms that restructure every load path, and the variant retains the
element-length artifact the foundation model removes. **Caveat (recorded):** the
foundation formula governs a sub-element mode the current mesh cannot audit by
eigen (same epistemic class as the V.3b strip formula): conservative-leaning by
construction (continuous-mode minimum; soft-direction k only; in-plane
restraint is membrane-stiff and ignored), but a chordwise-enriched eigen audit
is the eventual validator for both. Cumulative harvest: **2248.0 → 1108.5 kg
(−50.7%), every claim converged + feasible + eigen-verified.**
(2093 s + 4706 s measured.)
**[Annotated 2026-06-16 — the 1108.5 kg headline was corrected to 1095.3 kg (solid-rod beam-mass bug), then superseded by the multistart headline 1021.6 kg.]**

**Incumbent guard + mass-accounting CORRECTION (2026-06-11) — optimizer runs can
now never return worse than a feasible seed; implementing that exposed a
reporting bug that overstated every hollow-beam mass since P#1b. Corrected
running best: 1095.3 kg.** (1) **Incumbent guard** (user-requested): `evaluate()`
now tracks the lightest HARD-feasible design at every visited point (including
line-search trials and the explicitly pre-evaluated seed); if the optimizer's
endpoint is heavier or the run fails, the incumbent is returned
(`result.used_incumbent`, converged stays False — a design, not a certified
optimum). Property test: a feasible seed can never produce a worse result —
exact, backend-agnostic (it hooks evaluation, not the optimizer). (2) **The bug
it caught:** the result's beam-mass term used the solid-rod formula for ALL form
beams while the objective correctly used annular mass — reported masses for
hollow designs were INFLATED (optimization itself was always correct; eigen and
feasibility used correct sections). **Corrected from the saved designs (no
re-sizing): P#1b 1924.6 → 1784.5 kg; V.3c 1108.5 → 1095.3 kg (the corrected
running best; −51.3% vs the V.6 baseline).** P.3's 1679.3 was overstated by the
same mechanism but its design arrays were overwritten by the spokes run
(superseded; not recomputed). In-flight runs launched before the fix print
inflated result masses — their saved designs recompute correctly. Lesson
recorded: the objective and the reported mass must share one accounting path
(they now do for new results; a regression test pins objective == reported).

**Data loss + V#2 directional-coherence fix implemented (2026-06-11) — an
unexpected machine reboot wiped /tmp, killing both in-flight runs (big-budget
push, spacing re-test) and destroying every saved design array, including the
V.3c running best and the whole warm-start chain; everything is being
regenerated under a corrected panel model so the re-run happens once.**
(1) **Loss + process fix:** code/findings/corrected masses were all committed
and safe; only regenerable artifacts died. New convention: launcher scripts,
design arrays (.npz) and logs go to the gitignored repo `runs/` dir
(reboot-proof), each chain stage saves its design immediately on completion,
and chains are resumable (existing stage file = skipped). (2) **Why V#2 now:**
the V.3c optimum had moved the layup from 32/53/15 (span/±45/chord) to
**0/100/0 — pure ±45** — directionality completely reorganized under the
skin-as-bracing role, which makes the long-open V#2 mismatch load-bearing: the
strip panel check used *triangle-local-frame* D11 (a mesh-dependent span/chord
patchwork) while the foundation k uses datum-frame chord D22. (3) **The fix
(implemented + FD-validated):** `panel_d_mode="datum_ortho"` (opt-in, KS-only,
requires `ply_angle_datum`) replaces the local D11 in the strip σcr with the
classical orthotropic long-SS-plate combination in the DATUM frame
(`sensitivity.ortho_plate_Qstar`): N_cr = (2π²/b²)(√(D11·D22) + D12 + 2·D66),
1 = span, 2 = chord — exactly kc = 4 for an isotropic laminate (unit-tested
identity), so the model is unchanged for quasi-isotropic skins and only the
directional bookkeeping is corrected. σcr keeps the same geom(t,c)/t sandwich
factor (t/c gradient chains shared with the local path); new f0/f45 chains run
through `dQstar_df` (FD-audited end-to-end through the live KS closures, with
foundation on, alongside all 8 families). Demand stays the most-compressive
principal stress (direction-agnostic, conservative-leaning — recorded caveat);
wrinkling still uses local-frame Qeff00 (V#2-residual, smaller since Hoff goes
as E_face^⅓ — recorded caveat). Plain path bit-exact (full suite green, 15:10
wall). Warm-start helper `design_vector_from_result` added to the laminate path
(was ad-hoc in the lost /tmp scripts) with a cross-config roundtrip test.
**Measurement pending:** `runs/chain_rebuild.py` regenerates the full chain
A tube+hollow → B sandwich → C foundation (V.3c reference) → D datum_ortho
(the V#2 step measurement + does 0/100/0 survive coherent bookkeeping?), each
stage warm-started, eigen-verified, and saved to `runs/chain_*.npz`.

**Kernel-panic root cause = memory oversubscription by parallel sizing (2026-06-12)
— NEGATIVE result for unbudgeted parallelism; memory-budget convention added.**
Both "unexpected restarts" were full watchdog kernel panics, and our runs caused
them. Evidence (`/Library/Logs/DiagnosticReports/`): panic #1 Jun 11 18:24:39
("userspace watchdog timeout: no successful checkins from WindowServer in 120
seconds") minutes after a 17:59–18:09 JetsamEvent storm whose top memory consumers
were our 8 parallel python3.13 workers (~1.0–1.1 M pages each); panic #2 Jun 12
02:05:54 (configd, 181 s) 45 min after the 01:02–01:19 storm — top consumers the
`runs/multistart_v2.py` pool (8 IPOPT workers at ~2.05–2.23 M pages ≈ **~9 GiB
peak each**) plus the chain (~2 GiB), i.e. ~70+ GiB of demand on the 32 GiB Mac
Studio. Kill reason in every storm: `vm-compressor-space-shortage`; jetsam killed
`configd` and starved WindowServer, whose missed watchdog checkins panic the
kernel by design. **Why:** desktop macOS jetsam does not kill runaway user
processes — it kills/starves daemons first, so memory oversubscription escalates
to a kernel panic instead of an OOM kill of the offender. **Fix:** memory-budget
convention (CLAUDE.md): sum of worker peak footprints ≤ ~½ RAM → `multistart_v2.py`
capped at `n_workers=2`; heavy runs never stacked concurrently (chain completes,
then multistart); `ru_maxrss` (self + children) now recorded in every run meta so
the ~9 GiB/worker figure stays measured, not estimated. Diagnosis ~25 min wall (log
forensics only, no sizing run). Caveat: per-worker peak is config-dependent (medium
16×8, IPOPT+KS, foundation + datum_ortho) — re-measure before raising worker counts
on other configs or scales.
**[Annotated 2026-06-16 — the n_workers=2 cap was later revised to n_workers=1 after the measured 17.8 GiB/worker peak (see the multistart-headline entry, 2026-06-12).]**

**V#2 MEASURED + chain regenerated (2026-06-12) — datum-ortho panel model lands
at the old optimum's mass (−0.1%, noise) and the pure-±45 layup SURVIVES coherent
bookkeeping; 1094.2 kg converged is the new medium headline.** `runs/chain_rebuild.py`
re-ran the full warm chain solo post-crash-fix: A0 tube-only 2060.7 kg (matches
ex-50) → A tube+hollow 1784.3 kg, λ 2.54, converged feasible (matches corrected
P#1b 1784.5/2.54 — regeneration exact) → B sandwich 1221.0 kg, λ 3.48, feasible
but UNCONVERGED at the 2000-iter cap (81.8 min; incumbent iterate, upper bound —
well under the old ~1679 P.3 number, which predates the hollow-mass correction
and a different warm path) → C foundation 1152.1 kg, λ 3.51, feasible, unconverged
early IPOPT exit at 280 iters (14.6 min; +5.2% vs the lost V.3c 1095.3 — upper
bound only) → **D datum_ortho 1094.2 kg (beams 548 + skin 465 + core 76 + tube 5),
CONVERGED, feasible, eigen λ 3.61, 841 iters, 36.9 min, peak RSS 13.5 GiB.**
**The V#2 verdict:** D vs the corrected V.3c reference 1095.3 = **−0.1%, inside
the noise floor** — correcting the strip-panel D11 from triangle-local to
datum-frame orthotropic does not move the headline, and the optimizer *kept* the
0/100/0 (pure ±45) layup under coherent directional bookkeeping — the regime
change is real physics (skin-as-bracing), not a bookkeeping artifact. **Why:** at
this optimum the binding KS buckling families price ±45 stiffness through both
the foundation k (datum D22) and now the panel N_cr (datum √(D11·D22)+D12+2D66)
consistently; the local-frame patchwork had been noise-level at the optimum, not
load-bearing. Caveats: B and C rungs are unconverged upper bounds (step
attribution B→C→D is path-consistent but their absolute levels are soft); demand
remains principal-compression; wrinkling still local-frame (V#2-residual).
Chain total ~2.5 h wall, analytic adjoint Jacobians throughout, every stage
eigen-verified ≥ 1.5 and saved to `runs/chain_*.npz`. Follow-up in flight:
`runs/multistart_v2.py` (8 starts, 2 workers, start 0 seeded from stage D via new
`x0` param — incumbent guard ⇒ can only improve on 1094.2). CAD/VTU export of the
stage-D design queued behind the multistart (memory budget) via the sized-export
path; design regenerable from `runs/chain_D_datum_ortho.npz`.

**Multistart escapes the warm-chain basin (2026-06-12) — 1021.6 kg converged
feasible, −6.6% below stage D, NEW medium headline.** `runs/multistart_v2.py`:
8 starts on the final config (sandwich + foundation + datum_ortho, IPOPT+KS,
SF 1.5), 2 workers (memory-budgeted), maxiter 3000, start 0 seeded from chain
stage D (1094.2 kg) via the new `size_beam_shell_laminate_multistart(x0=...)`
param (incumbent guard ⇒ start 0 can only improve on the seed). Result: best
feasible **1021.6 kg (start 3), CONVERGED (2252 iters), feasible, eigen λ 2.76**,
layup still pure ±45. Start masses (feas): 1088.9✓ / 1091.8✓ / 1023.9✓ /
**1021.6✓** / 1015.3✗ / 1093.3✓ / 1450.7✓ / 1090.9✓ — 7/8 feasible; the lightest
overall (1015.3, start 4) was correctly EXCLUDED as infeasible. **Why it beats
the chain:** the D-seeded start 0 stayed at 1088.9 kg (its warm basin), but three
independent cold starts converged into a distinctly lighter basin clustered at
1015–1024 kg — the warm chain had been trapped in a local optimum the cold
restarts escaped (a textbook argument for multistart over pure warm-chaining at
this problem's nonconvexity). The headline trades buckling margin for mass
honestly: eigen λ 2.76 (vs D's 3.61) still clears the 1.5 requirement with 84%
margin. Wall **9.7 h** (35,082 s) measured, 2 workers. **Memory (measured, not
estimated):** worker peak RSS **17.8 GiB** — ~2× the page-count estimate behind
the original n_workers=2 cap; two simultaneous peaks (35.6 GiB) would have
re-crashed the 32 GiB machine, and only non-coincident peaks saved this run.
**Action:** for this medium IPOPT config the safe parallelism is **n_workers=1**;
2 is over the edge. Caveat: the 1015.3 kg infeasible near-miss says there may be
~1% more if a feasibility-restoring polish from start 4's neighborhood converges
— queued. Best design saved `runs/multistart_v2_best.npz`. **Milestone artifacts
exported + independently re-verified** (`runs/export_best.py`, 13.3 s): recomputed
worst linear-buckling λ = 2.7637, matching the saved value to 0.00% (the design
reproduces from the saved arrays) → `exports/best_medium_worst_mode.vtu` +
`best_medium_beams.stl` (lens loft) + `best_medium_skin.stl`. Sized detail: beam
radii 4.0–40.0 mm, single skin band 1.154 mm faces / 5.70 mm core, pure ±45,
6 spar tubes ~1.0 mm wall + root tube 7.41 mm. Regenerate:
`PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/export_best.py`.

**V#12 slam re-introduced at 3 g lateral (survival) — STRONG NEGATIVE + a model-
validity finding: the current topology does NOT survive 3 g, and the closed-form
beam-buckling model goes non-conservative under slam (2026-06-14).** User chose
3 g lateral (survival) for the re-introduced slam envelope. `runs/slam.py 3`
added ±3 g lateral slam + gravity (4 accel cases: upright, 30° heel, ±3 g lateral,
each combined with 1 g vertical) to the medium envelope, warm-started from the
1021.6 kg headline, IPOPT+KS, single worker. **Result is a DIAGNOSTIC, not a
feasible design:** mass drove to **1567.8 kg (+53.5% vs the headline)** and still
**did not converge** (hit maxiter 3000, 6 h 16 m wall), and — decisively — the
independent linear-buckling check gives **worst λ_cr = 0.81 < 1.0**, i.e. the
structure *buckles under ~80% of the applied slam load*, far below the SF 1.5
protocol. So 3 g lateral slam is **not survivable by the 16-beam form-beam +
sandwich-skin topology** as sized. **Two findings:** (1) the survival envelope
needs a topology change — added *lateral* bracing — not just more material; the
optimizer poured +53% mass in and still couldn't reach a stable structure,
because a cantilever wing has no efficient load path for a 3 g chord-normal
inertial load without transverse members. This is the binding justification the
shelved cross-member / IsoTruss-diagonal levers (P.2 / F.2, abandoned while
buckling-under-aero governed) never had. (2) **The closed-form/foundation
beam-buckling constraint is non-conservative under combined lateral slam:** it
reported `beam_buckling` util 1.001 ("feasible") while the true eigenvalue was
0.81 — the foundation formula, calibrated under aero + gravity, misses the
slam-driven buckling mode. Until that gap is closed the closed-form model must
not be trusted for the slam case; the eigen check has to move in-loop for any
slam-sized result (Toolbox #7). Secondary observations: the layup reorganized
**0/100/0 → 0/0.62/0.38** (chordwise 90° fibres reappear — lateral slam loads the
chord), tip deflection nearly binds (0.96), twist stays slack (0.26). The +53%
mass is reported only to show the optimizer's effort; it is **not** a survival
mass (eigen-rejected). Measured 22,525 s wall, analytic Jacobian, peak RSS in
meta. Saved `runs/slam_3g.npz`.
**[Annotated 2026-06-16 — the binding tip-deflection (0.96) was the spurious serviceability constraint; the +53.5% mass is deflection-overstated. The qualitative conclusions stand (needs lateral bracing; closed-form non-conservative under slam), and 3 g IS survivable with rings (λ_cr 3.16 at 20 mm, 2026-06-15).]**

**V#12 slam at 2 g lateral (2026-06-14) — also a NEGATIVE diagnostic, and a
WORSE-converged one: the warm-from-headline IPOPT path stalls in restoration and
never reaches feasibility for the slam problem.** `runs/slam.py 2`, identical
setup to the 3 g run (same headline seed, same 4-case ±2 g envelope, IPOPT+KS,
single worker, maxiter 3000, 5 h 56 m). Outcome: **unconverged, NOT feasible
(no incumbent ever found), eigen λ_cr = 0.60**, ending at a meaningless 929.3 kg
*infeasible* iterate with closed-form buckling violated (beam util 1.40, panel
1.19) and tip deflection at 0.99. **This is NOT "2 g is lighter/worse than 3 g"**
— it is a convergence failure: from the headline seed (feasible only under the
no-slam envelope) IPOPT entered its feasibility-restoration phase and never
escaped, drifting to a lighter constraint-violating point instead of building the
structure up as the 3 g run happened to. So the 929.3 kg / λ 0.60 are an
infeasible-iterate artifact, not a 2 g survival mass. **What it confirms /
adds:** (1) the slam-loaded problem is not reliably solvable on the current
topology from the headline seed — *neither* 2 g nor 3 g produced a stable design;
(2) because this run never reached even closed-form feasibility, it is *less*
conclusive than 3 g (which proved a closed-form-feasible-but-eigen-rejected point
exists) — we therefore cannot yet say whether a 2 g-feasible design exists on this
topology, only that it wasn't found from this seed; (3) the warm-from-headline +
closed-form-foundation IPOPT recipe is the wrong solver strategy for slam (erratic
restoration trajectories) — a cleaner 2 g read needs reseeding from the 3 g
geometry (already in the lateral-loaded regime) or a feasibility-first formulation,
and ultimately the lateral bracing + in-loop eigen of task #22. Measured 21,359 s
wall, peak RSS in meta. Saved `runs/slam_2g.npz`.

**Lateral ring-frame bracing BUILT; first 3 g braced run is an INCONCLUSIVE
negative — rings are physically capable but the solver didn't engage them
(2026-06-15).** Task #22 implemented per spec/plan (`feat/lateral-bracing-slam`,
15 commits, merge-reviewed): transverse ring members (`lateral_bracing=True`),
a `brace_radius` DV driving mass + an FD-validated elastic-foundation stiffness
credit `k_eff = k_skin + k_ring(r_brace)` in-loop (in-loop FEA/body-loads use a
fixed nominal radius → clean adjoint; post-hoc eigen rebuilds rings at the sized
radius). Every new gradient FD-validated; unbraced byte-identical (200 tests).
The two-stage subagent review caught **four pre-existing braced-path crashes**
(cold-start x0 length, missing brace sections, the `beam_vm` Jacobian
`group_of_element` overrun, the `body_load_jacobian` overrun) before they could
bite a run. **First production run (`runs/slam_braced.py 3`, warm from the
unbraced 3 g diagnostic, SF-ladder + eigen gate, single worker): rung 1 (SF 1.5)
= 623.2 kg, UNCONVERGED (maxiter 3000, 6 h 16 m), INFEASIBLE, eigen λ 0.388, and
`brace_radius` driven to 2.02 mm — the floor.** The optimizer *threw the rings
away* despite starting them at 26 mm. **Diagnosis (measured):** the `k_ring`
credit is substantial — at this medium design `r_brace` 20 mm → foundation
`Pcr` ×1.84, 35 mm → ×4.8, 50 mm → ×9.7 — so rings CAN relieve beam buckling;
the failure is a solver one. Seeded from the failed (unconverged, eigen-rejected)
unbraced 3 g geometry, IPOPT entered deep feasibility-restoration in a region
dominated by *multiple* violations (tip deflection 0.99, panel buckling), where
beam buckling — the rings' only lever — is not the marginal constraint, so it
minimized mass (rings included) instead of engaging them. The SF ladder
compounded it (it bumps SF on an already-infeasible rung; higher SF is strictly
harder), so the run was stopped after rung 1 rather than burn ~18 h on rungs
2–4. **Conclusions:** (1) the bracing feature/machinery is correct and
merge-worthy — this is a solve-strategy result, not a code defect; (2) the warm
start must come from a FEASIBLE design (the 1021.6 kg headline), not a failed
diagnostic; (3) the SF ladder must only bump on feasible-but-low-eigen rungs;
(4) likely also needs a lower-g validation (1 g braced — confirm rings engage)
and/or g-continuation (ramp 0→3 g) so the design stays near-feasible and beam
buckling becomes the active lever. Next strategy is a user decision (each retry
is multi-hour). Measured 22,544 s wall, peak RSS ~12 GiB, saved
`runs/slam_braced_3g.npz`.

**RING BRACING WORKS — 3 g lateral slam IS survivable; the optimizer, not the
physics, was the blocker (2026-06-15).** After four slam runs failed the same way
(rings driven to the 2 mm floor, true eigen < 1), a direct post-hoc eigen sweep
on the braced 3 g design settled it: **raising `brace_radius` rescues the true
buckling eigenvalue dramatically.** Measured worst λ_cr over the full 3 g slam
envelope vs ring radius (on the rung-1 design): 2 mm → 0.77, 10 mm → 1.10,
12 mm → 1.42, 14 mm → 1.91, **20 mm → 3.16**, 50 mm → 11.7. So **~13–14 mm rings
clear the 1.5 gate at 3 g**, and 20 mm rings give a 2× margin (λ 3.16). The rings
are exactly the right fix and modest ones suffice. **Root cause of the failures
(definitive):** the in-loop closed-form/foundation beam-buckling model is
non-conservative under slam — at small rings it reports beam buckling satisfied
(util ≤ 1) so the optimizer sees the rings as pure mass cost and minimizes them
to the floor, even though the *true* eigenvalue of that mode is far below 1 and
the rings would fix it. The `k_ring` foundation credit (FD-validated, working) is
real but too weak relative to the model's under-stated demand to pull the rings
in. This is the same non-conservatism flagged for the unbraced 3 g case, now
proven to be the controlling defect. **Conclusion:** the bracing feature is
correct and the architecture survives 3 g slam; to make the *sizer* find it
automatically we must either (a) pin a ring-size floor (~15–20 mm — we now know
the value) and let the other DVs size for deflection/panel, or (b) put the true
eigenvalue in the loop (the deferred analytic-`dλ/dx` constraint, spec §fallback)
so the optimizer optimizes the real buckling margin. The g-continuation run
(g=1 from the feasible headline) confirmed the blindness is g-independent: even
1 g drove rings to the floor (λ 0.331, 7.5 h, unconverged). Diagnostic sweep cost
~minutes; the failed runs are recorded above. Next: a pinned-ring survival sizing
to report the actual 3 g survival mass.

**V.9 prize MEASURED — in-loop eigen would harvest only ~14 %, and the true eigen
is a cliff (2026-06-16).** `runs/eigen_harvest.py` re-sized the 1021.6 kg headline
at decreasing closed-form buckling SF (warm-chained SLSQP, ~2 min/pt) measuring
the POST-HOC true eigenvalue at each: SF 1.5 → 1021.6 kg / λ 2.76; **1.25 → 799.9
kg / λ 0.83**; 1.0 → 659.0 / 0.59; 0.85 → 619.5 / 0.60; 0.70 → 577.4 / 0.55.
Interpolating the valid crossing (true λ = 1.5) ⇒ **≈ 876 kg, so the V.9 in-loop-
eigen prize ≈ 145 kg (14 %)**. Two takeaways that REVISE the P.6 read: (1) the
prize is modest, NOT the large harvest the +1030.8 kg/SF *local* shadow price
implied; (2) the true eigen is a **cliff** — a 17 % SF cut (1.5→1.25) crashed it
2.76→0.83 (below 1, i.e. buckles), so the closed-form SF-1.5 buffer is partly
REAL protection against a steep buckling response, and sizing to true-λ-1.5 (V.9)
puts the design at the cliff edge with no margin. Net: **V.9 is a modest,
risk-tradeoff lever, demoted from "obvious next build."** The value of measuring
vs extrapolating the shadow price: the local rate (+1030.8 kg/SF) and the global
response (cliff) tell opposite stories.

**Brazier crush CHECKED on the survival tube — NOT a concern (6.58× margin)
(2026-06-16).** The 3 g survival design grew the core tube into a large root
bending spar (r 338–378 mm, ~0.7 m dia; the 666 kg cost) with a thin transition
segment (r/t ≈ 79). The sizer checks tube WALL buckling but not Brazier
ovalization, so `runs/brazier_diag.py` extracted the actual per-segment tube
bending moment from the survival FEA under the 4-case slam envelope and compared
to the classical Brazier `M_cr = (2√2/9)πEr t²/√(1−ν²)`. Worst segment = the thin
transition (seg 4, r/t 79): M_demand 66.4 kN·m vs M_cr 437 kN·m = **margin 6.58×**;
all other segments ≫ 38× (root segments carry more moment but M_cr ∝ r·t² so thick
walls dominate); driver is the ±3 g lateral slam (confirmed). So Brazier does NOT
govern this design and the sizer's omission isn't masking a failure. **Caveat:**
the isotropic formula is approximate for a filament-wound CFRP tube (hoop modulus
governs ovalization; a low-hoop-stiffness wound layup would reduce M_cr) — but
6.58× leaves headroom for a plausible orthotropic correction. (~4 s wall.)

**Mesh convergence REVISITED — the eigen verification is UNCONSERVATIVELY
mesh-biased; the 16×8 headlines are optimistic (2026-06-16).** Hypothesis (that
the foundation+strip+eigen regime fixed V.2's unconservative bias) was REFUTED.
`runs/mesh_converge_diag.py` re-evaluated the 1021.6 kg headline's worst
linear-buckling λ at finer spanwise meshes (fixed design resampled via the V.0.5
`resample_segment_radii`; band-based skin is mesh-independent): **λ = 2.76 (N=8,
the headline mesh) → 1.52 (N=10) → 1.24 (N=12) → 1.39 (N=16)**, settling ~1.2–1.4
for N ≥ 12 (N=8 reproduced 2.7637 exactly first). **The coarse N=8 λ 2.76 is the
optimistic OUTLIER** — the true converged margin is ~1.3, *below* the SF 1.5 the
sizer enforces. **Why:** the 7-spanwise-segment N=8 beam mesh cannot represent the
short-wavelength beam-column buckling mode; finer meshes resolve it at a lower
(truer) λ — the classic Euler beam-column DISCRETIZATION effect. The V.3b/V.3c
regime change fixed the *sizer's closed-form* constraint mesh-dependence but NOT
the *verifier's* discretized eigen-mode count, and V.3's "λ rises on finer meshes"
was specific to the old V.6-era design's mode, not this one. **Implication
(serious):** every eigen-verified headline (operational **1021.6 kg** and the 3 g
survival **1575.3 kg**, all 16×8) is **unconservatively biased — its true
converged buckling margin is below the 1.5 it was sized to**, so the
mesh-converged designs are HEAVIER. The "−54.5% vs V.6" and the survival mass are
partly coarse-mesh artifacts. **A mesh-converged RE-BASELINE (re-optimizing at
N ≥ 12) is now the priority validity item** — and the eigen-verification protocol
must run at a converged mesh, not 16×8. **Caveats:** this re-evaluates a
*resampled fixed* design — the bias DIRECTION (down, large) is robust, but the
absolute converged λ carries resampling-mapping noise (the non-monotonic
1.24→1.39 at N12→N16 is likely mode-switching/mapping noise; ~1.3 is the signal);
a true re-baseline will pin the magnitude and the corrected masses.

**V.10 re-baseline attempt at n=12 — BLOCKED on a methodology problem; the eigen
doesn't mesh-converge (2026-06-16).** `runs/mesh_rebaseline.py` re-sized the
operational design at n_levels=12 with an SF ladder gated on the n=12 eigen.
Two rungs (both ~3 h, both maxiter-UNCONVERGED): SF 1.5 → 687.7 kg, n=12 eigen
1.457; **SF 1.65 → 704.1 kg, n=12 eigen 1.651 (clears 1.5), feasible.** Killed
after rung 2 — three problems make this NOT a trustworthy corrected headline:
(1) **Ladder logic flaw:** acceptance required `converged`, but IPOPT never
converges these (maxiter, as on every slam run), so the ladder would have ground
3 more rungs and saved the *most over-conservative* (SF 2.3) — fix: accept
`feasible AND eigen ≥ gate` (incumbent guard + eigen gate IS the acceptance, per
the slam-run lesson). (2) **Surprising direction:** n=12 sizing gives a ~31 %
*LIGHTER* design (704 vs 1021.6 kg), not heavier — inverting the expectation. Most
likely the finer mesh's extra radius DVs (x 126→206, finer beam tapering) + better
load-path resolution let the optimizer shed material the coarse mesh forced;
i.e. the 16×8 headline was *inefficient* (coarse tapering), not merely optimistic.
But a 31 % drop from re-meshing is extraordinary and UNVERIFIED (could be the n=12
sizer's closed-form being permissive, only partly caught by the eigen gate). (3)
**THE BLOCKER — n=12 is not mesh-converged either:** the diagnostic eigen of the
fixed headline was non-monotonic (2.76 @ n8 → 1.52 → 1.24 → 1.39 @ n16, no
plateau), so gating at n=12 (λ 1.651) gives no guarantee at n≥16; there is no
clean "converged mesh" to re-baseline at. The non-monotonicity is likely
mesh-induced mode-switching (the k=1 lowest buckling mode is a *different* physical
mode at different meshes) and/or diagnostic resampling noise — must be understood
before any re-baseline number is trustworthy. **Next: a focused eigen-mesh-behavior
diagnostic** (track k>1 modes across n=8…24, separate true mode-switching from
resampling noise, find whether/where the worst λ plateaus) — the foundation for a
defensible verification mesh. Until then both "headlines are optimistic" and the
"704 kg re-baseline" are provisional.

**RESOLVED — the eigen VERIFICATION mesh was the bug, not the design; the 1021.6 kg
headline is VALID and the closed-form is well-calibrated (2026-06-16).**
`runs/eigen_mesh_behavior.py` swept the k=5 lowest buckling modes of the fixed
headline across n_levels 8→32. The non-monotonic λ1 (2.76→1.52→1.24→1.39→…) is
**under-resolution → cluster convergence**, NOT mode-switching or resampling
noise: the coarse n=8 mesh overstiffens the ENTIRE spectrum (all 5 modes 2.76–3.99
— no low mode hidden), and as refinement resolves the short-wavelength modes the
whole spectrum descends, OVERSHOOTS (λ1 bottoms at 1.24 @ n=12), then recovers to
a **converged cluster floor λ1 ≈ 1.5–1.6 at n≥24** (1.587/1.629/1.512 at
n=24/28/32; the ±7% jitter is near-degenerate cluster reordering, the band is
converged). Eigen-only cost: ~15 s (n=24) … 40 s (n=32). **Corrections to the two
findings above:** (1) the converged margin is **~1.5–1.6, not ~1.3** — the ~1.3
was the n=12 OVERSHOOT misread as converged; the **1021.6 kg headline is VALID at
converged mesh** (λ ≥ 1.5), sized right to its SF-1.5 margin, and the closed-form
foundation model is **well-calibrated** (SF 1.5 ↔ converged true eigen ~1.5), NOT
non-conservative — the apparent non-conservatism was a coarse-VERIFICATION
artifact. (2) the **704 kg "31% lighter re-baseline" is INVALID and discarded** —
it was gated to clear the n=12 overshoot, which isn't the converged truth; a
lighter design's converged eigen would fall below 1.5. **The real fix is the
VERIFICATION PROTOCOL, not a mass re-baseline:** verify buckling at **n≥24** and
judge the converged cluster floor across the lowest 2–3 modes (all cheap,
eigen-only), instead of a single coarse-mesh λ1. The operational headline stands
at 1021.6 kg. FOLLOW-UPS: re-verify the 3 g survival design (1575.3 kg, its λ 3.77
was an n=8 value — likely lower converged, quick braced eigen-only re-check at
n≥24); and P.4b n_beams can now re-sweep with n=8 SIZING (closed-form is calibrated)
+ n≥24 VERIFICATION (cheap), not finer-mesh sizing.

**3 g SURVIVAL design FAILS buckling at converged mesh — the 1575.3 kg rating is
INVALID; closed-form is non-conservative for slam (2026-06-16).**
`runs/survival_mesh_eigen.py` re-checked the braced survival design's worst
SLAM-envelope buckling λ across spanwise mesh (k=5, rings 20 mm, reproduced n=8
λ1 = 3.775 exactly): **3.78 (n8) → 3.58 (n12) → 2.42 (n24) → 1.77 (n32) → 1.59
(n36) → 1.437 (n40) → 1.247 (n48, STILL falling ~10–13 %/refine, NO plateau).**
Unlike the operational headline (which recovered to a converged floor ~1.5–1.6),
the survival design **declines MONOTONICALLY and crosses below the 1.5 gate at
n≈40** — so its converged worst-λ is ≤ 1.247 and unbounded below by the data.
**Verdict: the 3 g survival rating does NOT hold as sized — the n=8 λ 3.77 was a
~3× coarse-mesh artifact, and for the SLAM case the closed-form/foundation model
(rings credited via k_ring) IS non-conservative at converged mesh** (it modeled a
beam-column/foundation mode while the governing converged slam mode is something
else — likely local: the thin r/t≈79 tube transition, or a ring/skin mode under
lateral load — and spanwise-only refinement hasn't plateaued it, suggesting a
chordwise/local mode the n_beams=16 mesh also limits). **Implications:** (1) the
true 3 g survival mass is HEAVIER than 1575.3 kg (needs more buckling capacity —
bigger rings/beams/tube, or a different fix for the actual mode); (2) the earlier
"20 mm rings → λ 3.16 at 3 g" eigen-sweep was also an n=8 value, inflated; (3) for
slam, V.9 (in-loop true-eigen) regains real value — the closed-form is genuinely
non-conservative there, unlike the operational case. **Contrast is the key
result:** operational closed-form well-calibrated (converged λ ≈ SF 1.5);
slam/survival closed-form non-conservative (converged λ ≪ SF 1.5). NEXT: identify
the governing converged slam buckling MODE (spanwise vs chordwise/local; does it
ever plateau?) before resizing — the verification mesh itself is unsettled for the
braced slam case. (~14 min eigen-only to n=48.)

**Converged slam buckling MODE identified — a local wing-section mode (NOT the
tube/rings), and it does not plateau (2026-06-16).** `runs/survival_mode_id.py`
extracted the worst-combo buckling eigenvector at fine mesh: **100% of modal
translation is in the wing-section grid nodes, 0% in the tube**, strongly LOCAL
(participation ~8–10 of 400–950 nodes), peaked at **z ≈ 7.7–7.9 m (inner-mid
span)**, identical mode at every mesh, driven by the **+3 g lateral slam** combo.
Caveat: the form beams and skin SHARE these grid nodes, so node-energy can't cleanly
separate a form-beam column buckle from a skin-panel buckle — but it is definitively
a local inner-mid-span WING-SECTION mode, **not** the r/t≈79 tube transition, not a
ring mode, not a global sway. It **does NOT plateau** (λ1 2.42→1.44→1.12 at
n=24/40/56, ~−22%/refine, stopped at n=56). **Consequences:** (1) the fix is NOT
bigger rings or thicker tube wall (tube carries zero modal energy) — it's
stiffening the inner-mid-span wing section against this local mode; (2) the
closed-form `beam_buckling_model="foundation"` is non-conservative for it (continuous
bonded-skin Winkler restraint + element-length effective length overestimate
capacity, and shrink the closed-form Pcr on refinement while the true mode buckles
over a multi-element span); (3) the NO-PLATEAU means even the *verification* eigen
isn't mesh-settled, so "true λ ≥ 1.5" is not yet well-defined for slam — the
braced-slam buckling needs either a much finer converged mesh or a corrected
local-buckling model. **Strategic upshot: the 3 g slam survival case genuinely
needs V.9 (in-loop true-eigen) AND a settled verification mesh — it is a
research-level open problem, not a quick resize.** The operational 1021.6 kg
headline is unaffected (its mode plateaus, closed-form calibrated). (eigen-only;
n24 32 s, n40 175 s, n56 497 s.)

**P.4b n_beams re-sweep ATTEMPTED — cold-chain-per-n_beams is too fragile;
inconclusive, 16 stands, deferred (2026-06-16).** `runs/nbeams_resweep.py` (current
foundation+sandwich+datum_ortho config, per-point warm chain, converged n≥24
verification). First point n_beams=12 **cascaded infeasible through every stage** —
A0 tube-only 2169.8 kg, A tube+hollow 1963.0, B sandwich 1364.6 — all unconverged
(maxiter) + infeasible + NO feasible incumbent, despite healthy n8 eigen
(2.48–3.79). For comparison n=16's chain converged feasible at every stage. So
cold-chaining a *different* n_beams hits the same IPOPT/SLSQP restoration-stall
that plagued the slam runs — the methodology can't cleanly compare n_beams.
**Soft signal:** n=12's infeasibility is on a closed-form constraint (eigen is
fine), most likely **panel buckling** — fewer beams ⇒ larger chordwise panels ⇒
panel buckling (slack at n=16, util 0.71) becomes binding/violated, so reducing
n_beams below 16 is NOT a win. This is consistent with the original P.4 verdict
("16 optimal") and with the headline being beam-buckling-governed with only modest
panel slack. But it is NOT rigorous (unconverged). **Conclusion:** the operational
16-beam / 1021.6 kg headline stands as near-optimal; a *rigorous* re-sweep is
DEFERRED and needs a chordwise warm-start (remap the 16-beam headline to each
n_beams for a good x0 — a single IPOPT run/point instead of a fragile cold chain),
not worth building now given the headline is valid and 16 is well-supported. Killed
after the n=12 cascade (~2 h) rather than burn ~6 h more on unreliable points.

**3 g slam survival MEASURED — 1575.3 kg, converged, eigen 3.77; slam is
BUCKLING-governed, not deflection-governed (the deflection-overstated framing was
wrong) (2026-06-16).** `runs/slam_survival.py 3 20`: failure-only formulation
(strength + buckling + eigen; deflection demoted to a 5 %-span geometric-validity
guardrail, twist ≤ 20°), rings pinned at the 20 mm floor, warm from the 1021.6 kg
headline. **First slam run to CONVERGE: 1575.3 kg, feasible, eigen λ_cr 3.77,
tip deflection 0.42 m (1.9 % span), 3 h 47 m.** Breakdown: beams 568 + **tube 666**
+ skin 163 + core 95 + brace 83 kg. Binding: **beam buckling 1.001 + panel
buckling 0.992 co-bind**; twist 0.099 and deflection (1.9 % span) both slack.
**Key (corrective) finding:** relaxing deflection from 2 % to 5 % did NOT shed
mass — the buckling-optimal design deflects only 1.9 % span, i.e. *just under* the
old 2 % serviceability limit anyway. So under 3 g lateral slam the design is
**buckling-governed, not deflection-governed**; the earlier "deflection-overstated,
expect << 1.6 t" expectation (and the annotations flagging slam masses as
deflection-inflated) was WRONG — deflection was never the binding driver, buckling
was. The failure-only formulation is still correct in principle (deflection is
serviceability), it just doesn't move this particular answer. **Structural story:**
the central core tube re-tasks from a ~5 kg torsion spine (no-slam headline) to a
**666 kg bending spar** — the dominant survival cost — to carry the 3 g lateral
inertial bending of the 22 m cantilever; rings (83 kg) handle beam buckling,
beams/skin grow for panel buckling. **3 g survival rating costs +54 % mass**
(1021.6 → 1575.3 kg), and the single design satisfies BOTH envelopes (operational
deflection 1.9 % < 2 % limit → no separate envelope needed). **Conservatism:** the
closed-form buckling SF 1.5 binds (util 1.0) while the true eigen is 3.77 (~2.5×
conservative, consistent with V.3) → 1575.3 kg is a CONSERVATIVE survival mass;
in-loop eigen (V.9) would harvest it. Caveats: rings pinned at 20 mm (eigen sweep
showed ~14 mm clears the gate → ~40 kg recoverable); SF 1.5 conservative for an
ultimate case; 16×8 mesh (V.2 bias). Saved `runs/slam_survival_3g_20mm.npz`. CAD
export (form beams + skin + 20 mm rings + worst-mode VTU) is the milestone artifact.

**P.6 shadow prices on the 1021.6 kg headline — beam buckling is the SOLE lever
at +1030.8 kg/SF; the remaining harvest is closed-form conservatism (→ V.9), not
structure (2026-06-16).** `runs/shadow_prices.py` ran SLSQP warm from the IPOPT
optimum (reproduced 1021.6 kg, converged feasible, 24 iters, 104 s) to recover the
KKT multipliers (IPOPT signs unwired, so SLSQP is needed). **Only one nonzero
shadow price: `buckling_sf_beam = +1030.8 kg per SF-unit`.** Every other
constraint is exactly slack: σ_allow, tip deflection (util 0.57), tip twist
(0.26), panel buckling (0.71), tube-wall, core wrinkle/crimp — all +0.000.
Utilizations confirm: beam buckling 1.000 (binding), all else < 1. **vs the old
V.6 ranking (+531 kg/SF on the 2248 kg design):** the marginal beam-buckling price
nearly DOUBLED on the lighter foundation-model design — it is *more*
buckling-critical, not less. **Interpretation (refines the audit's "lever spent"):
beam buckling is STILL the only lever; what is spent is the cheap STRUCTURAL ways
to buy buckling capacity** (core tube, hollow walls, foundation-length model,
rings — all built). The remaining beam-buckling harvest is the **closed-form
CONSERVATISM**: the design meets SF 1.5 while the true eigen is 2.76 (1.84×; the
survival design is worse at 2.5×). That margin is recoverable only by sizing to
the *true* buckling eigenvalue in the loop → **V.9 (analytic dλ/dx) is the clear
highest-value next build**, justified by the +1030.8 kg/SF price. Prize to be
measured by a buckling-SF sweep (mass vs true eigen) before committing to the V.9
build. (SLSQP warm-settle breakdown: beams 522 + skin 314 + core 80 + tube 106 kg.)

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
| Re-spacing under binding deflection | Re-ran even-vs-symmetric-weighted spacing (medium) with the tip-deflection budget tightened to **bind** (0.5% span = 110 mm; default 2% is slack). Both feasible, defl & panel-buckling **co-bind** (110/110 mm, 1.00/1.00): even 2362.1 → symmetric-weighted 2347.9 kg = **−14.3 kg (−0.6%)**. So once stiffness governs (not buckling alone), re-spacing flips from +2.7% (slack) to a small positive — but buckling co-binds so the win stays marginal. Confirms layout levers help only at the margin; skin tailoring is the real lever. (1 h 41 m, analytic Jac.) **[Annotated 2026-06-16 — AERO-regime, marginal; not evaluated under the failure-only slam envelope.]** |
| Tip gusset in the sizer | Opt-in rigid massless tip-node clique (`build_beam_shell_model(tip_gusset_radius=...)`/`model_with_tip_gusset`), assembled into K but excluded from force recovery → composes with the analytic Jacobian (constant stiffness). **Negative for mass:** medium free-tip 2264.6 → gusset 2453.1 kg (+8.3%), both feasible/converged, despite twist 5.0°→0.03° and tip defl 156→71 mm. Buckling-governed design + rigid coupling redistributes load → must add material. Twist not the mass driver. Keep opt-in for twist/stiffness, not mass. |
| Mirror-symmetric non-uniform spacing | `chord_symmetrize_weights` (max-of-mirror) → symmetric stress-weighted arc placement that keeps `beam_radius_groups` grouping (verified n_groups unchanged). **Negative for mass:** medium even 2264.6 → symmetric-weighted 2325.2 kg (+2.7%), both feasible; stress concentration 2.45 real, but clustering enlarges gap panels and the design is panel-buckling-governed → more material. Even spacing (minimizes max panel) is near-optimal; re-spacing counterproductive. Even stays default; helper kept. **[Annotated 2026-06-16 — AERO-regime, marginal; not evaluated under the failure-only slam envelope.]** |
| Phase-F.2 diagonal beams | Balanced both-hand grid-helix lattice on existing grid nodes (`beams.helix_elements`, no remesh), co-sized with one shared diagonal-radius DV in the SLSQP laminate loop; pitch chosen by principal-stress alignment (`recommend_pitch`, best pitch 2 @ align 0.68). **Strong negative result:** baseline 33.1 kg → diagonal 60.6 kg (+83%), diagonals add 20.8 kg at a buckling-forced 4.7 mm radius, twist rose 1.25°→1.58°. The design is buckling-governed with large twist slack, so long compression diagonals bloat mass without relieving a binding constraint. Lattice abandoned for this regime; twist (when binding) is killed far cheaper at the tip. Streamline-following (F.3) not recommended while buckling dominates. Implementation was a throwaway spike, NOT merged — only the finding is kept. **[Annotated 2026-06-16 — +83% is AERO-regime. Under SLAM, lateral bracing binds: ring bracing makes 3 g survivable (λ_cr 3.16, 20 mm rings, 2026-06-15) → diagonal/cross-member family superseded-by-ring-bracing for slam, not abandoned.]** |
| Tip-coupling study | Hard tip joint (gusset) modeled as a stiff connector-beam clique tying the tip nodes (`beams.solve_beam_shell_tip_coupled`, tunable `gusset_radius`), reusing `solve_beam_shell` (no rigid MPC, no penalty hacks). Finding: barely redistributes BEAM stress (peak −2%, spread 3.75→3.38) — the skin already shares spanwise load — but near-eliminates **tip twist** (0.197°→0.004°, ~50×) and stiffens the tip (~14%), saturating at low gusset stiffness. Investigation only (no CAD / not in the sizing loop). Implication: the twist-governed design could be relaxed/lightened by a tip gusset (re-size-with-gusset = follow-up). **[Annotated 2026-06-16 — the +8.3% sizer verdict is AERO-regime (twist/stiffness aid, not a mass lever there); not the verdict under slam loading.]** |
| /tmp data loss + runs/ convention (2026-06-11) | Unexpected reboot wiped /tmp: both in-flight runs AND all saved design arrays lost (V.3c running best, warm chain). Code/ledger safe. New convention: scripts, .npz designs, logs → gitignored `runs/`; stage-level immediate saves; resumable chains (`runs/chain_rebuild.py`). Regeneration folded into the V#2 re-baseline so the chain re-runs once. |
| V#2 CLOSED — implementation (2026-06-11) | `panel_d_mode="datum_ortho"` (opt-in, KS-only): strip σcr now uses the datum-frame orthotropic long-plate N_cr = (2π²/b²)(√(D11·D22)+D12+2·D66) instead of triangle-local D11; exact kc=4 isotropic identity (unit-tested); f-chains via `dQstar_df`, FD-audited live. Motivated by the V.3c layup regime change 32/53/15 → 0/100/0 (±45) making local-frame bookkeeping load-bearing. Caveats: demand stays principal compression; wrinkling still local-frame (V#2-residual). Measurement: chain stage D (pending). |
| 3 g slam survival measured (2026-06-16) | Failure-only formulation (deflection demoted to a geometric guardrail), rings pinned 20 mm: **1575.3 kg, CONVERGED, feasible, eigen 3.77** (first clean slam result; +54 % vs the 1021.6 kg headline). Slam is **buckling-governed, not deflection-governed** — relaxing deflection didn't shed mass (optimum deflects 1.9 % span anyway), so the "deflection-overstated" framing was wrong. Cost driver: the core tube re-tasks from ~5 kg torsion spine to a **666 kg bending spar**. Conservative (closed-form SF 1.5 binds while true eigen 3.77 → V.9 in-loop eigen would harvest). |
| Ring bracing survives 3 g slam (2026-06-15) | Post-hoc eigen sweep proves ~13–14 mm rings clear λ≥1.5 at 3 g (20 mm → λ 3.16). 3 g slam IS survivable with lateral ring bracing — the four failed runs were optimizer-blindness (closed-form non-conservative under slam → rings minimized to floor), not physics. Fix: pin a ring floor (~15–20 mm, known now) OR in-loop eigen. Pinned-ring survival sizing pending for the mass. |
| Memory-budget for parallel runs (2026-06-12) | Both machine crashes were watchdog kernel panics caused by our parallel sizing oversubscribing RAM (8 IPOPT workers × ~9 GiB measured on 32 GiB → `vm-compressor-space-shortage` jetsam storms → configd/WindowServer starved → panic). Convention (CLAUDE.md): Σ worker peaks ≤ ~½ RAM (→ `n_workers=2` for medium IPOPT), never stack heavy runs concurrently, `ru_maxrss` recorded in every run meta. **[Annotated 2026-06-16 — the n_workers=2 cap was later revised to n_workers=1 after the measured 17.8 GiB/worker peak (multistart-headline entry, 2026-06-12).]** |
| V#2 MEASURED — datum_ortho verdict (2026-06-12) | Chain stage D: **1094.2 kg converged feasible, λ 3.61 (−51.3% vs V.6)**; −0.1% vs the local-frame V.3c optimum (noise) → the directional-bookkeeping fix costs nothing and the **pure-±45 layup survives** = real physics, not artifact. `datum_ortho` becomes the standard P-phase panel mode (opt-in flag unchanged). B/C rungs unconverged (upper bounds); wrinkling local-frame residual stays open. |
| Multistart headline (2026-06-12) | 8-start multistart (D-seeded start 0, `multistart x0=` param added) found a lighter basin the warm chain missed: **1021.6 kg converged feasible, λ 2.76 — new medium headline (−54.5% vs V.6, −6.6% vs chain D)**, layup ±45. Measured worker peak 17.8 GiB ⇒ **n_workers=1 is the safe cap for medium IPOPT** (2 was over the edge). Lightest start (1015.3) excluded as infeasible — feasibility-restoring polish queued. |
| V#12 slam @ 3 g lateral (2026-06-14) | User chose 3 g (survival). DIAGNOSTIC, not a design: drove to **1567.8 kg (+53.5%), unconverged (maxiter 3000), eigen-REJECTED λ 0.81 < 1.0** → current 16-beam topology does NOT survive 3 g lateral; needs **lateral bracing** (binding reason to revive P.2/F.2 cross-members). Also: closed-form beam-buckling went **non-conservative** (util 1.001 while eigen 0.81) → eigen check must move in-loop for slam-sized results. Layup 0/100/0→0/0.62/0.38, tip defl 0.96. **[Annotated 2026-06-16 — the binding tip-deflection 0.96 was the spurious serviceability constraint; +53.5% mass is deflection-overstated. Qualitative conclusions stand; 3 g IS survivable with rings (λ 3.16, 2026-06-15).]** |
| V#12 slam @ 2 g lateral (2026-06-14) | Same setup; WORSE-converged diagnostic: **unconverged, never feasible (no incumbent), eigen λ 0.60**, ended at a meaningless 929.3 kg infeasible iterate (buckling utils 1.40/1.19). NOT "lighter than 3 g" — IPOPT stalled in restoration from the no-slam headline seed and never reached feasibility. Confirms slam is unsolvable on this topology from this seed; inconclusive on whether 2 g is *physically* feasible. Next: reseed 2 g from the 3 g geometry, then task #22 (bracing + in-loop eigen). **[Annotated 2026-06-16 — the binding tip-deflection (0.99) was the spurious serviceability constraint; the masses are deflection-overstated. Qualitative conclusions stand; 3 g IS survivable with rings (λ 3.16, 2026-06-15).]** |
| Incumbent guard + mass correction (2026-06-11) | Guard: best hard-feasible visited design always kept (never-worse-than-feasible-seed, tested). It exposed solid-vs-annular result accounting: **corrected ledger P#1b 1784.5 kg, V.3c running best 1095.3 kg (−51.3% vs V.6)**; optimization/eigen/feasibility were always correct — reporting only. |
| V.3c CLOSED (2026-06-11) | Foundation model: **1108.5 kg running best, eigen λ 3.61** (−34% step; **−50.7% vs V.6**). Beams 559 kg; k-chains let skin DVs buy beam capacity (core grew to 5.4 mm partly for k). Spokes variant loses outright (3417 kg unconverged — hub diaphragms + retained element-length artifact). Sub-element caveat: foundation formula awaits a chordwise-enriched eigen audit (V.3b-class). **[Annotated 2026-06-16 — 1108.5 kg corrected to 1095.3 kg (solid-rod beam-mass bug), then superseded by the multistart headline 1021.6 kg.]** |
| P.3 CLOSED (2026-06-11) | Polish converged: **1679.3 kg, eigen λ 3.55 — new running best** (−25.3% vs V.6 baseline). t_core 4.3 mm / faces 1.91 mm; skin 519+61 kg vs 1246 monolithic. KS-conservative local optimum (small hard slack; multi-start unexplored). Beams now 65% of mass → **V.3c is the next lever**. Cumulative P.3 compute ≈ 6.1 h. **[Annotated 2026-06-16 — 1679.3 kg overstated by the solid-rod beam-mass bug, never recomputed; superseded by the foundation chain (1095.3 kg) then multistart (1021.6 kg).]** |
| P.3 KS+IPOPT breakthrough (2026-06-11) | KS smoothing merged (8 active constraints → smooth scalars; FD-audited through the live scipy closures). First feasible sandwich design in six attempts: **1685.0 kg, eigen λ 3.49** — t_core 4 mm / faces 2.1 mm, skin+core 617 kg vs 1246 monolithic. Upper bound (unconverged); polish leg running. Confirms the argmax-kink diagnosis. |
| Beam autopsy / IsoTruss (2026-06-11) | No eigen-visible beam mode; beam-only Kσ λ = 2.54 = full λ0 (stiffened-panel mode, 69% margin). The +326 kg/SF closed-form price targets a mesh-artifact sub-element length; physical model = beam-on-elastic-foundation (continuously bonded skin = the "helicals"). Ties SHELVED (F.2-consistent); **V.3c foundation-length check queued**; re-check ties at the thin-skin sandwich endgame. |
| P.3 + IPOPT follow-up (2026-06-11) | IPOPT backend merged (`optimizer="ipopt"`, cyipopt/brew Ipopt, equivalence-tested). Default options discard warm starts (barrier re-centering → 2080 kg wander); proper warm-start options give a coherent 1249 kg (−35%) trajectory but STILL no convergence at 2000 iters. Both optimizer families now fail the cored migration for the same root cause: **argmax-switching constraint Jacobians (piecewise smoothness)**. **P.3 gated on P#7 KS aggregation** (smooth max; doubly motivated: IPOPT smoothness + adjoint count). Running best stays 1924.6 kg. |
| P.3 sandwich skin (2026-06-11, OPEN) | Machinery merged + FD-validated (φ(t,c) D-factor, wrinkle/crimp checks, core gravity, gradients). Measurement: SLSQP stalls unconverged across 4 warm legs while mass falls 1924.6 → 1032 kg (−46%, skin 1246→270 kg at 1 mm faces + 5 mm core) — **prize is large but no leg is a result**. IPOPT trigger formally met (119 DVs, persistent stalls): integrate IPOPT (needs `brew install ipopt` for cyipopt, or casadi-bundled via adapter), then re-measure. Running best stays **1924.6 kg**. |
| P.4 n_beams sweep (2026-06-10) | n ∈ {12…28} on the running-best config, full per-point protocol, parallel. **16 beams is the measured optimum** (1924.6 kg, λ 2.54); n=20 +2.5% at λ 1.51; **n=24 closed-form-feasible but eigen-REJECTED (λ 1.22)** — skin thins ∝n as predicted but beam mass grows faster and coupled eigen modes erode monotonically (2.54→1.51→1.22). P.4 closes; the historical 16 is now measured. When future levers slim members toward λ=1.5, move the eigen check in-loop (Toolbox #7 gradient). **[Annotated 2026-06-16 — cited 1924.6 kg is PRE-correction; corrected to 1784.5 kg. "16 beams optimal" verdict unaffected.]** |
| P#1b hollow form beams (2026-06-10) | Annular machinery generalized (r-col = existing radius group; t_hollow block per group). Hollow-only: +1.2% = NO win (axial-stiffness loss feeds the skin; twist binds without the tube). Tube+hollow warm-started: **1924.6 kg, eigen λ 2.54 — running best, −14.4% vs V.6**; beams 674 kg at ~1 mm walls (4 plies). Cold start was eigen-REJECTED at λ 0.91 (verification layer works). Levers are complementary: spine frees twist, hollow walls then harvest the beams. Next: P.4 n_beams sweep (user-requested) on this config. **[Annotated 2026-06-16 — 1924.6 kg is the PRE-correction number (solid-rod beam-mass bug); corrected to 1784.5 kg; current headline 1021.6 kg. Verdict unchanged.]** |
| P.1 CORRECTION (2026-06-10) | Earlier negative INVALID — tube bonds never assembled in the sizer (user-caught). Bonded re-run: **2248.0 → 2060.7 kg (−8.3%), eigen λ 2.49** — the tube stays minimal (8 kg, r at the 20 mm bound) and acts as a **torsion spine**: twist constraint un-binds (−50.7 → 0 kg/deg), skin sheds 180 kg of torsion plies. Centroidal-bending uselessness confirmed; torsion value missed by the artifact. Running-best medium 2060.7 kg; beam-buck +455 kg/SF → P#1b next. Follow-ups: tube_r_min sweep, tube CAD export, M#5 joint model. |
| P.1 core tube (2026-06-10) | Full annular-member machinery built + FD-validated (sections, fit-bounded r/t DV blocks, wall-crimping check w/ 0.65 knockdown, annulus ∂K through the adjoint). **NEGATIVE at medium scale: optimizer zeroes the tube** (r→20 mm bound, −0.0% mass, eigen 2.21 ✓) — a centroidal tube has no bending leverage inside a 0.5–1.9 m-deep monocoque. P#1a dead as a mass lever (kept as manufacturing aid); hollow lever redirects to **P#1b hollow form-beam segments** (OML leverage; beam-buck SF +514 kg/SF still dominant). Machinery reusable for P#1b. |
| P.0 sizer refactor (2026-06-10) | `DesignVector` (named blocks = single source of x-layout) + `ConstraintSpec` (name, closures, rows, shadow conversion per constraint; scipy dicts/Jacobian registration/multiplier attribution derive from one list). Behavior-preserving: 189 tests green, V.6 medium headline reproduced bit-exactly (2248.0284 kg, 290 iters, shadows to the digit). Adding a constraint or DV block is now one append — the P.1 gate is open. Bonus lesson: a 1-ulp change (sqrt(area)² → area in panel b²) flipped an FD cold-start basin 10% on a small problem — bisected, sqrt-roundtrip deliberately retained in the legacy path for bit-reproducibility; cold-start optima are ulp-sensitive, warm starts + deterministic default starts are the protocol. |
| P#9 multi-start at scale (2026-06-10) | 12 parallel starts (V.0.4 pool, 11 workers, 41 min) on the medium 4-band config: best feasible 2267.4 kg = +0.9% vs the 1-band 2248.0 → **banding dead under strip-mode buckling** (its −8.1% was a legacy-panel-model artifact); 1-band becomes the standard P-phase config. Feasible basin scatter 2.9% = the noise floor measured at medium scale. Parallel==serial bit-exact (tested); speedup ~2.9× (stragglers + contention). |
| V.6 re-baseline (2026-06-10) | New eigen-verified headlines under the honest model (strip widths + upright/heel gravity + distributed pressure): **small 25.13 kg** (4-band, λ 2.00; −9.2% vs old 27.67) / **medium 2248.0 kg** (1-band, λ 2.26; old 4-band 2264.6 not comparable — legacy checks, aero-only). Medium 4-band landed +4.4% heavier in a beam-buck-dominated basin (warm start included) → banding's value under strip-mode buckling unresolved, P#9 multistart running. Beam-buck SF is now the dominant price (+531 kg/SF) — beams are the next physics to attack (P.1 hollow members). STL + worst-mode VTU exported (`just example 49_rebaseline`). |
| Slam envelope deferred (2026-06-10) | DECISION (user): the 1 g lateral slam case (V#12; ex-47 diagnostic: → +110%, infeasible at maxiter 400) is re-introduced only AFTER the Phase-P performance harvest (hollow members, etc.) lightens/strengthens the structure. V.6 standard load set stays aero × {upright, 30° heel} gravity. When re-introduced: warm-start from the gravity optimum; IPOPT if SLSQP stalls. **[Annotated 2026-06-16 — the +110% slam figure was computed against the 2%-span tip-DEFLECTION (serviceability) budget, the wrong formulation for a survival/ultimate case (only failure governs); mass is deflection-overstated. Failure-only re-run pending (plan V.7).]** |
| V.5 panel pressure + bending (2026-06-10) | Skin-distributed projection (force-conserving CST lumping) −1.6% vs node-lumped = neutral (noise floor) but adopted as V.6 standard. Strip-bending failure term σ_b = 0.75qw²/t²: zero effect at medium scale (skin vM 34 vs 1100 MPa — strength margin ~30×); kept as guard for thin-skin/high-pressure/large-scale regimes. Buckling-pressure interaction + Tsai-Wu bending deferred (caveats recorded). |
| V.4 self-weight/inertial (2026-06-10) | `accel_vectors` body loads (mass lumped per evaluate) + the λᵀ·∂f/∂x adjoint term (FD-validated ≤2e-4; also the P.2 prerequisite). Medium strip baseline 2006.3 → **2316.9 kg (+15.5%)** with upright + 30° heel gravity (converged/feasible). 1 g lateral slam: diagnostic — unconverged/infeasible at maxiter 400, mass heading +110% → slam envelope deferred to the V#12 requirements decision; V.6 default = upright + heel only. |
| V.3b width-based panels (2026-06-10) | Opt-in `panel_width_mode="strip"`: b = physical chordwise beam spacing (median 0.449 vs 0.897 m √area surrogate). Calibration: implied capacity 5.14× at the old optimum ≈ converged eigen 5.7 (conservative side). Harvest: medium 1-band **2471.4 → 2006.3 kg (−18.8%)**, converged+feasible; twist now co-binds. Eigen verify of new optimum: λ_cr 2.286 ≥ 1.5. Fixes V#1 level + ∝n scaling → P.4 unlocked. Default stays sqrt_area until V.6 re-baseline (after V.4/V.5). kc=4 edge + V#2 direction caveats open. |
| V.0.7 CI split (2026-06-10) | `sizing` pytest marker assigned from measured durations (19 tests ≥5 s; top 192/166/99 s). Fast job: 149 tests / **12.2 s measured** on every push/PR; sizing job on main pushes + nightly (uv, locked sync). Full suite green 168/168 post-cache-fix. Example smoke layer + flag matrix still open; example 14 broken (meshio gone from venv) — smoke layer would have caught it. |
| V.3 eigenvalue buckling (2026-06-10) | K+Kσ linear buckling built + reference-validated (columns ≤0.2%, plate kc=4 ≤5%). Medium optimum: worst λ_cr = 2.443 vs the closed-form's claimed 1.5 → ≥1.6× conservative; λ rises to 5.67 on finer meshes of the same design (stress-field redistribution + no nodes between beam lines) → 2.44 is a lower bound. Worst mode = distributed skin-normal waving. Pretension moves λ 2.443→2.445 (null at this optimum) → P.2 demoted. Even γ=0.65 knockdown × 2.443 = 1.59 ≥ 1.5 → design not deficient; conservatism is harvestable. Next: width-based panel check calibrated by eigen (fixes level + ∝n scaling), then P.4. |
| P#2 prestress probe (2026-06-10) | Hoop pretension superposed on the medium optimum's panel stresses: scalar principal check sees ≤12% credit (saturates 0.88), direction-resolved biaxial interaction goes 1.17→0.72 (5 MPa)→0.31 (10 MPa)→0 (20 MPa) — the P.2 prize is large but requires a biaxial panel check or the V.3 eigen solve to price; probe is one-sided (no equilibrating compression, no retention knockdown applied). P.2 sizing work stays gated on V.3. **[Annotated 2026-06-16 — prestress stays null under aero, but the COMPRESSION CROSS-MEMBER half of P.2 now binds under slam (merged ring-bracing work, 2026-06-15).]** |
| V.2 mesh convergence (2026-06-10) | n_levels 6→10 (feasible points): 2807→2486→2271 kg, −9–11% per refinement, no plateau — headlines NOT mesh-converged; bias is unconservative (element-length Euler + b=√area both gain capacity from refinement). 16×12 diverged (diagnostic). n_beams 20×8 ≈ 16×8 total mass, consistent with the V#1 ∝n mis-scaling muting the physical ∝n² panel win — P.4 sweep stays gated on the panel-model fix. Headline comparisons remain valid at fixed 16×8; absolute level carries mesh bias. V.3 (eigenvalue/physical buckling length) is the fix, not finer meshes. |
| V.1 shadow prices (2026-06-10) | KKT multipliers captured + converted to kg-per-unit (`shadow_prices` on the result; FD-validated 0.02–0.5%). Medium headline (4-band, 2264.6 kg reproduced exactly): twist −41.35 kg/deg (binding, cheapest requirement), beam-buck SF +266.5 / panel-buck SF +151.7 kg/SF-unit, deflection + σ_allow free. 1-band: buckling-only (panel +352.2). Renegotiation order: twist limit first, then buckling SF — which V.3 prices in model-fidelity terms. |
| V.0.1 profile → cache fix (2026-06-10) | cProfile of 10 medium SLSQP iters (examples/41): 93% of wall = uncached `_beam_vm_grad_one` rebuilding triangle ∂K per beam-vM row. Fixed by threading the existing `SensCache` through `beam_con_jac` (exact; 21 equivalence/FD tests unchanged). 1.13 s/iter raw vs ~26 s/iter recorded → ~23×; medium sizing now ~minutes. V.0.2 vectorization + V.0.3 KS deprioritized — no dominant wall remains; re-profile before investing further. |
| Python 3.13 migration (2026-06-09) | `requires-python >=3.13,<3.14` (was `<3.13`), `.python-version` 3.13, lock regenerated; **full suite green 160/160 in 19 m 30 s** (measured — suite is NOT fast; CI needs a fast/slow marker split). 3.14 blocked solely by build123d 0.10.0 (`<3.14` + OCP `<7.9`; cp314 OCP wheels exist, build123d dev branch already supports `<3.15`+7.9) — bump when its next release ships. |
