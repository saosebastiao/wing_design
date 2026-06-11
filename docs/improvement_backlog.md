# Improvement backlog — model audit + design discussion (2026-06-09)

Source: a code-level audit of the structural model, load application, mass accounting,
and optimization setup (file:line citations below), plus design discussion of the
skin-to-beam bondline and skin-compression/prestress mechanics.

**The context that frames everything here:** the design is robustly
**panel-buckling/stiffness-governed** (per-band thickness was the only positive mass
lever; every beam-layout lever was negative), and **both governing constraints sit at
utilization = 1.0 on closed-form approximations** (`structural/buckling.py:33` beam
Euler, `:58` panel plate-buckling). The entire design — and every comparative finding —
rides on the fidelity of those two formulas. Many items below either firm them up
(validity) or exploit the huge margins elsewhere (performance: skin strength margin is
~7× while buckling binds).

Each item is tagged with the expected effect on the headline mass once addressed:
**[↑]** honest design likely gets heavier, **[↓]** likely lighter, **[?]** could go
either way, **[~]** negligible mass / gating check. Magnitudes given are
order-of-magnitude guesses to be measured, not predictions.

---

## Manufacturing concept (reference)

The manufacturability items assume this intended build process (recorded from design
discussion, 2026-06-09):

- **Outer form beams** — either **resin transfer molded (RTM)**: a dry
  unidirectional-tow **preform** laid into a closed matched-die (female cavity) mold by
  CNC tow placement, mold closed, resin injected, cured — implies extractable
  (draft-positive) beam cross-sections; or **filament wound** over a **washout
  (sacrificial) mandrel** carrying female mold channels, on a winder whose spindle axis
  spans the wing (the chord must fit the swing envelope).
- **Cross-member beams** — either process; if wound, **assembly-sequence access
  ordering** applies (inner members must be placed before outer members). Joined to
  beams by **wrapped joints** — local filament overwinds tying two members together at
  a junction (**co-bonded** when wound wet over cured members), not only a wet adhesive
  bond.
- **Shear webs** — CNC-cut pre-cured CFRP panels (possibly sandwich-cored), slotted
  into place and **secondary-bonded** to the beams.
- **Optional tube core** — a large filament-wound tube left in as a **permanent
  structural mandrel**; beams/webs bond to it, and sacrificial mold channels on it
  locate wound beams.
- **Skin** — a filament-wound overwrap. **The first layer is a hoop (circumferential)
  wind — LE→TE→back around the section — deliberately retaining winding tension as a
  compressive binding force around the beams** (i.e. skin prestress is an intended
  manufacturing feature, not a hypothetical). Wound wet over the cured frame, so the
  skin–beam joints are **co-bonded**. Subsequent layers at any windable orientation;
  **local UD doubler plies** applied last for targeted reinforcement.
- **Root/spar section** — weight non-critical (low and inboard); beams slot into a
  solid socket — e.g. **compression-molded chopped-fiber CFRP ("forged" carbon)** —
  and are adhesively bonded.
- **Shape constraints (added 2026-06-09):** hollow sections are only buildable on
  **completely straight** members — e.g. the core tube can be wound to any radius that
  fits inside the wing, with radius *and* wall thickness varying along the span, but it
  cannot easily be curved. A wound member must be **convex about its winding axis**
  along its whole path; on non-convex paths the tow **bridges** the concave regions
  (fiber bridging), so such beams (e.g. the LE/TE beams where they curve out from the
  rotational spar through the transition) cannot be filament wound and must be RTM'd.
- **Assembly sequence** — (1) make beams → (2) assemble truss in the winder →
  (3) place/bond/wrap webs and cross-members → (4) wind the skin (winding tension binds
  the beams) → (5) targeted reinforcement plies.

---

## 1. Validity improvements

*Goal: the model behaves as close as possible to the real structure.*

1. **Eigenvalue buckling (K + Kσ linear buckling solve). [?; DONE 2026-06-10 — see
   findings; one premise corrected]** The single
   highest-leverage validity item: replaces/validates *both* binding closed-form checks
   at once. ~~Captures panel curvature~~ **(corrected 2026-06-10: it does NOT
   capture inter-beam panel curvature — the skin is flat facets with no mid-panel
   nodes; see Performance #6 for the still-open curvature credit; flat-plate kc=4
   remains conservative, and any future curvature credit still needs an NASA
   SP-8007-style imperfection knockdown)**. Captures combined
   membrane stress states, beam–panel mode interaction, and **global modes** (section
   ovalization/crimping) that no closed-form check sees. Also the natural home for
   prestress effects (see Performance #2). **Panel-geometry mischaracterization
   (audited 2026-06-09):** the implemented `b = √(triangle area)` ignores that the
   physical panels are *long* (h/w ≈ 7 on the 16×8 medium mesh) — physically σcr is
   set by the panel **width** w ≈ 0.46 m, not b ≈ 0.85 m, so the check is ~3×
   conservative in level, **mis-scales with beam count** (σcr ∝ n_beams in-model vs
   ∝ n² physically), and **falsely credits length-cutting members** (rings/webs)
   that barely raise the real compression σcr. Edge restraint (beams are flexible,
   not rigid supports) cuts the other way and is uncharacterized. Fix (or eigenvalue-
   solve) this **before** running Performance #3/#4, which it directly biases.
2. **Panel-buckling stiffness direction mismatch. [↑?]** The check compares the
   most-compressive *principal* stress (any direction) against a single `D11`
   (`buckling.py:37–59`). With chordwise-heavy banded layups, bending stiffness in the
   actual compression direction can differ several-fold from the D11 used. Use D in the
   principal-compression direction (or the eigenvalue solve makes this moot).
3. **Beam Euler buckling length is the FEA element length — optimum is mesh-dependent
   in the unconservative direction. [↑]** `Pcr = π²EI/(KL)²` with L = element length,
   K=1 (`buckling.py:15–34`), justified by "the skin restrains the nodes" — a restraint
   stiffness never verified, and one the skin-crush mechanisms (item 7) act *against*.
   Refining the mesh shortens L, raises Pcr, and lets the optimizer legally remove beam
   mass. Verify restraint stiffness or switch to a physical buckling length.
4. **Apply aerodynamic pressure to the skin panels.** [↑] All panel forces are lumped
   at the nearest *beam node* (`fea_model.py:86–108`); shell triangles never see
   distributed pressure. Consequences: no panel lateral-pressure bending, and no
   pressure–compression buckling interaction. Related: DKT bending moments are
   recovered but **excluded from the skin failure check** (membrane von Mises only,
   `shell_sizing.py:200–203`) — include bending stress once panels are actually loaded.
5. **Self-weight, inertial, and heel loads. [↑]** The medium design is a 2.3-tonne,
   22 m cantilever sized against aero loads only (`cases.py:26–32`). Gravity (upright
   and heeled) and wave-slam/pitching accelerations (>1 g at the rig is routine) are
   first-order at that scale. The 5 m / 30 kg design plausibly shrugs this off; the
   medium headline does not.
6. **Aeroelastic feedback and stability.** [?] Loads are computed once on the
   undeformed, zero-twist shape (`aero/model.py:38`), yet the optimizer pins tip twist
   at exactly 5°, which redistributes real loading. No divergence/flutter check exists
   for a *free-rotating* wing whose torsional stiffness the optimizer actively
   minimizes. Note: several "negative" findings (tip gusset, diagonals) concluded
   "twist isn't worth buying" — that verdict could flip under a real aeroelastic
   constraint.
7. **Brazier ovalization / skin-crush loads. [↑ at scale]** Under global bending, a
   closed section develops transverse crushing loads (quadratic in curvature) that
   flatten it toward the neutral axis; curved compressed panels also resolve inward
   line loads onto the beams. Linear FEA shows zero of this. It is *the* classic
   driver of shear webs in wind-turbine blades — directly relevant to the >100 m
   ambition, and the physical motivation for Performance #3/#4.
8. **Bondline checks (post-processing, no FEA change needed). [~]** Skin and beams
   share nodes — the joint is implicitly fused, infinitely stiff, massless, and
   uncheckable (`beam_shell.py:1–9`). Add: per-edge skin↔beam **shear-flow** recovery
   vs a wound-joint allowable per unit length, and a **peel** check at
   buckling-critical panel edges (panels at utilization 1.0 are at onset of waving =
   peel loading). Also model the **eccentricity** between the inboard lens-beam
   centroid and the skin midsurface (currently collapsed onto one node line — it erases
   exactly the local joint moment that drives peel).
9. **Mesh convergence study of the *optimized* design. [?]** None exists; all
   headlines are n_beams=16 / n_levels=8, and known resolution sensitivity is large
   (skin stiffening factor moved 8.6×→2.4× from 6→12 levels). Combined with item 3,
   headline masses are not converged numbers. Now affordable with the analytic
   Jacobian.
10. **Environmental and fatigue knockdowns on the *governing* physics. [↑]** SF=2.0
    sits on strength, which never binds (7× margin). The binding constraints carry only
    SF=1.5 (buckling) and bare stiffness limits, with no hot/wet modulus knockdown
    (matrix-dominated D11 is exactly where hot/wet bites) and no fatigue spectrum for a
    structure that flogs and gybes. If prestress (Performance #2) is adopted, add a
    **prestress-retention knockdown** (design must remain feasible at ~50% retention).
11. **Root/bearing compliance. [?]** The keel ring is a rigid 6-DOF clamp
    (`fea_model.py:66–69`). The real structure is beams bonded into a socket riding a
    bearing — compliance and concentrated bearing reactions unmodeled. The
    manufacturing concept (solid forged-carbon socket) makes this tractable to model as
    a stiff-but-finite elastic boundary + socket bond check.
12. **Load envelope completeness. [↑?]** Five cases, one 1.5× gust factor
    (`cases.py:26–32`); survival is 25 m/s (~49 kt). No backwinded/aback gust, no
    dynamic slam case. A requirements question, but the envelope is what everything is
    sized to.

---

## 2. Manufacturability improvements

*Goal: what we design, we can actually build — per the manufacturing concept above.*

1. **Ply discreteness. ** Layup fractions (f0, f45, f90) and band thicknesses are
   continuous (`laminate_sizing.py:99–100`); a 0.96 mm band at 31/15/54 fractions is
   ~3–4 wound plies (~0.25 mm/ply) and may not be realizable. Add a post-sizing
   **ply-rounding + FEA re-verify** pass (same spirit as the CP-SAT stock catalog +
   bump-and-reverify for beams, which already exists at `beams/select_stock_sizes`).
2. **Mandatory first-layer hoop wind.** The build *requires* layer 1 to be a hoop
   (circumferential) wind LE→TE→back (the binding/prestress layer). Encode a minimum chordwise-ply thickness floor in the
   layup variables. Helpfully, the optimizer already favors chordwise fiber (E.4b found
   100% chordwise optimal), so this constraint should be nearly free — verify rather
   than assume.
3. **Windable-angle constraints on the skin layup.** Filament winding cannot place
   arbitrary angles everywhere (geodesic/friction slip limits, turnaround at LE/TE,
   swing-radius access). Constrain layup fractions to windable angle sets per region;
   longer term, Phase G's planner should feed the *as-wound* per-region orientation map
   back into the CLT (already the Phase G intent).
4. **Beam section + path manufacturability.** Sized beams must respect the build:
   RTM'd beams need extractable matched-die molds (draft-positive, "flat"-constrained
   cross-sections); wound beams must be **convex about the winding axis along their
   whole path** (fiber bridging otherwise), which rules out winding the LE/TE beams
   where they curve out from the spar — those are RTM'd. Hollow sections only on
   completely straight members (see Performance #1 for the split). Add section-shape
   and path-convexity classification to the geometry stage — each beam tagged
   RTM-vs-wound, solid-vs-hollow — rather than discovering the limits at mold design.
5. **Wrapped-joint model for member junctions.** Cross-members are attached by
   wrapped joints (local filament overwinds tying two members together). Characterize
   the joint's stiffness, strength, and **mass per joint**, and use it as the bond
   representation in Validity #8 — the model currently has no joint at all.
6. **Assembly-order constraints on internal members.** Wound cross-members have
   inner-before-outer access ordering, and shear webs must be slottable with the truss
   already in the winder. Any optimizer that places internal members (Performance
   #3/#4) needs these as layout constraints, or its output won't be buildable.
7. **As-built mass model. [↑]** Mass is ideal-laminate ρπr²L + ρ·t·A
   (`shell_sizing.py:144–151`). Add adders for: realistic wound Vf (local resin-rich
   zones), LE/TE turnaround overlaps, wrapped joints and bond fillets (~770 m of bondline
   on the medium wing — order 10–30 kg), the tip gusset (currently *massless* by
   construction), and root socket/bearing hardware. Report "ideal" and "as-built" mass
   separately so optimization findings stay comparable.
8. **Root socket as designed, not idealized.** Weight is non-critical at the root, so
   the win here is validity+manufacturability: model the slot-and-bond socket
   explicitly (finite stiffness, bond area check) instead of the rigid clamp, and size
   the socket engagement length from the recovered root reactions.
9. **Geometry fidelity for mold/fit.** Spline on-surface error is still ~113 mm
   mid-transition at geometry resolution. Fine for FEA trends; not fine for CNC mold
   channels. Transition-dense z-levels (already recorded as the open refinement) before
   any tooling is cut.

---

## 3. Performance improvements

*Goal: phenomenally lightweight, strong enough for the anticipated conditions.
Ranked by expected leverage.*

1. **Hollow sections where the build allows them. [↓, likely the biggest single
   lever]** Beams are *solid* rods — A=πr², I=πr⁴/4 (`frame.py:36–40`), 585 kg of the
   medium design, with Euler buckling binding. Buckling/stiffness-governed members are
   the textbook case for tubes: at equal mass a thin-walled tube has **(R/t)× the I of
   a solid rod** (~10× at R/t = 10), and at equal I it is severalfold lighter (~0.2×
   the mass at R/t ≈ 20; bounded by wall buckling and geometry). **The build constrains where hollow is possible** (Manufacturing concept
   shape constraints): only completely straight members, and wound members must be
   convex about the winding axis. So the lever splits into:
   (a) **the straight wound core tube** — promoted from "optional" to the prime hollow
   candidate: radius and wall thickness free to taper along the span, carrying spanwise
   bending/torsion as a proper closed section. Model as a new member type bonded to
   beams/webs (new DVs: r(z), t_wall(z)).
   (b) **straight in-wing beam segments** wound hollow: on the linearly-tapered planform
   the surface beam paths between root and tip are ruled (near-straight) lines — verify
   against the actual splines — joined by wrapped joints to
   (c) **RTM'd solid curved segments** through the spar transition, which stay solid.
   Needs wall-buckling/crimping checks for any hollow member and joint checks at the
   straight↔curved splices. Plausible −10–15% total mass on the medium design even
   after local checks, with (a) likely the cleanest path.
2. **Skin prestress via winding tension + compression cross-members (tensegrity
   section). [↓, potentially large]** The arbitrage: skin strength margin is ~7× to
   failure (2.0 required) while panel buckling binds, and tensile prestress raises
   effective panel buckling capacity ~additively *in the tensioned direction*.
   **Direction caveat (audited 2026-06-09):** the hoop wrap's pretension is
   **chordwise**, transverse to the mostly *spanwise* bending compression — transverse
   tension still stabilizes, but by a geometry-dependent factor, so the probe and any
   sizing must use a direction-resolved (biaxial) buckling check, not a scalar offset.
   The build *already intends* a tensioned first wrap. Equilibrate the
   skin tension with short through-thickness **spreader struts** (and/or webs, #3) that
   concentrate the compression into short members that are cheap to make buckling-proof
   (Euler ∝ 1/L²). Modeling path that fits the current linear framework: prestress as
   one extra self-equilibrated load case (negative-thermal-strain trick), superposed
   onto each aero case; buckling checks use net stress; strut radii + prestress
   magnitude become DVs. Composes with the analytic adjoint **with one addition**: the
   prestress nodal load `f_pre = ∫BᵀD(x)ε₀` depends on skin/strut DVs, so the adjoint
   gains a `λᵀ·∂f_pre/∂x` term (the framework currently assumes design-independent
   loads). **Cheap first experiment
   (minutes, no re-sizing):** superpose a parametric hoop pretension on the recovered
   membrane stresses of the current optimum and re-evaluate panel-buckling utilization
   — bounds the prize before building anything. Guard with the relaxation knockdown
   (Validity #10) and bond/peel checks at strut feet (Validity #8).
3. **Shear webs. [↓?]** CNC-cut sandwich webs are already in the manufacturing concept,
   resist Brazier crush (Validity #7), restrain the beams laterally, and act as the
   prestress reaction path. Distinct from the failed F.2 diagonals: webs are
   short-depth chordwise/through-thickness members, not long compression diagonals.
   Never tried in the sizer. **Corrected expectation (audited 2026-06-09):** for a
   long panel in spanwise compression, σcr is set by panel *width*, nearly independent
   of length — so cutting spanwise panel length helps shear buckling but not the
   governing compression mode. The implemented `b = √area` check would *falsely*
   reward length-cutting members (Validity #1); credit webs for crush/restraint/
   prestress, not panel buckling, until the panel model is fixed.
4. **Sweep n_beams. [↓?]** Panel *width* is the proven currency (the symmetric-spacing
   negative result), yet beam count has never been varied — no sweep exists anywhere.
   More, thinner beams cut panel width: physically σcr ∝ 1/w² (~n², long panels) while
   beam mass grows ~linearly — though the implemented `b = √area` check only credits
   ~n, so **fix the panel model (Validity #1) before/alongside the sweep** or it will
   understate the win. Rings cut panel *length*, which buys little against spanwise
   compression (see #3) — credit them for crush restraint only. Cheap with the
   analytic Jacobian: n_beams ∈ {12, 16, 20, 24, 28} at fixed constraints (even counts
   preserve mirror-symmetry grouping).
5. **Sandwich (cored) skin. [↓]** Skin is ~2/3–3/4 of mass and buckling-governed;
   σcr ∝ D, and a thin core multiplies D at ~5% of the mass of equivalent monolithic
   thickening. Build-compatible: wind inner skin → place core → wind outer skin.
   Needs core wrinkling/shear-crimping checks. Partially redundant with #2 (prestress)
   — measure which wins, probably don't need both at full strength.
6. **Curvature + panel-width credit. [↓; width half DONE 2026-06-10]** Validity
   #1 doubles as a mass lever twice over: flat-plate kc=4 ignores curvature, and the
   `b = √area` level is ~3× conservative vs a width-based check on the long medium
   panels. Every recovered Pa of σcr converts directly to thinner skin in the
   constraint that sizes most of the structure — but take curved-shell credit with an
   imperfection knockdown (NASA SP-8007-style), and remember the soft edge restraint
   (flexible beams) claws some margin back.
   **Status (2026-06-10): the WIDTH half is harvested** (V.3b strip check, −18.8%
   medium, eigen-calibrated). **The CURVATURE half is still fully open, and the
   original premise was wrong: the eigenvalue solve canNOT recover inter-beam panel
   curvature on this mesh family** — the skin is flat CST/DKT facets with nodes only
   on beam lines, so each strip is one flat facet across its width and the local
   panel arch is invisible to K, Kσ, AND the eigen modes (V.3 finding). Remaining
   path: (a) cheap closed-form upgrade in the V.3b pattern — curved-panel
   σcr ≈ kc·π²·D/(w²·t) + γ·0.605·E·t/R_panel with the per-strip radius computed
   from the section geometry like `skin_panel_widths` (design-independent →
   gradients unchanged), γ = SP-8007 knockdown; (b) chordwise mesh enrichment
   (mid-panel nodes) to let the eigen solve verify it. Same flat-facet caveat
   applies to the V.5 pressure-bending strip term (flat strip = conservative, no
   membrane shedding).
7. **KS aggregation of the vector beam constraint. [enabler]** The ∂K caching is
   merged (1.58×→1.98× measured) — still far below the ~n_DV ceiling, which confirms
   the next wall: the vector-valued `beam_con` needs ~n adjoint back-substitutions per
   gradient. A KS/soft-max aggregate collapses that to 1 adjoint, making finer meshes,
   sweeps (item 4), and multi-start (item 9) affordable. Prerequisite-grade for the
   validity re-baselining runs too.
8. **Extract SLSQP Lagrange multipliers (shadow prices). [↓, near-free]** Computed by
   the optimizer, currently discarded (`laminate_sizing.py:565–569`). One run tells you
   what a degree of twist limit, a millimetre of deflection budget, or 0.1 of buckling
   SF costs in kg — i.e. which *requirement* to renegotiate first. The 5° twist / 2%
   deflection limits are admitted heuristics; this is the cheapest kilogram available.
9. **Multi-start at scale. [↓ small]** Machinery exists, never run at scale. The
   FD-vs-analytic runs landed in basins ~2% apart, proving better basins exist. Also
   note the noise floor: with ftol=1e-4 and basin scatter, same-config comparisons
   under ~2–3% are within noise — relevant when judging small levers.
10. **More/smarter thickness bands. [↓ small]** 4 spanwise bands gave −8.1%; sweep
    band count (8, 16) and add *chordwise* banding (buckling-critical panels are
    localized around the section, not just along the span). Cheap once gradients are
    fast.
11. **OML/geometric levers (t/c, taper, planform). [↓?, Phase H]** Section depth drives
    beam bending efficiency; thicker sections fight buckling with geometry instead of
    material. Out of scope for the current fixed-OML pipeline but the largest knob at
    retargeting time.

---

## 4. Toolbox & architecture improvements

*Goal: the codebase and toolchain stay sound as Phases V/M/P repeatedly extend the
sizing core. From the toolbox audit (2026-06-09). The library choices themselves were
audited and confirmed: roll-your-own FEA is right at this scale and with custom
adjoints; AeroSandbox, build123d, CP-SAT, and scipy.sparse all fit their roles.*

1. **Constraint + design-vector extensibility refactor (do between Phase V and Phase
   P).** `size_beam_shell_laminate` is a 482-line function (`laminate_sizing.py:129–610`)
   bundling DV layout, constraint assembly, analytic-Jacobian wiring, optimizer call,
   and post-solve; **adding one constraint touches ~6 places** (evaluate loop,
   constraint closure, scipy dict, sensitivity.py gradient, Jacobian registration with
   manual idx bookkeeping, result dataclass), and the hand-indexed design vector
   `x = [r_groups | t_bands | f0 | f45]` means every new DV block touches every unpack
   site. Performance #1/#2 each add DV blocks *and* constraints. Refactor scope: a
   named-block `DesignVector` (pack/unpack once) + a small `Constraint` object (value,
   optional analytic gradient, feasibility entry) consumed by one driver loop. Schedule
   after V.3 (its new constraint shape informs the design), before P.1.
2. **CI + example smoke layer.** No `.github/` exists; the suite never runs
   automatically, and the numbered examples — the project's measurement record — have
   no guard against API drift. **Measured (2026-06-09): the full suite is 19 m 30 s
   (160 tests), NOT fast** — the SLSQP sizing tests dominate. CI therefore needs a
   pytest marker split: fast unit tests on every push, sizing tests as a separate job.
   Add a GitHub Actions workflow (uv + pytest) and a smoke suite running 2–3
   representative examples at maxiter≈10 / tiny mesh.
3. **Feature-flag interaction coverage.** `LaminateSizingConfig` has 6 opt-in flags;
   combinations like `per_band_layup × tsai_wu` and `per_band_layup × ply_angle_datum`
   are never exercised. Add a pairwise smoke matrix (tiny mesh, maxiter≈5, assert
   runs + sane shapes).
4. **Optimizer headroom: planned IPOPT fallback, not an emergency.** SLSQP (dense
   active-set) already needed O(1) normalization and has stalled on recorded runs; its
   cost grows with DV count. Notable: **casadi 3.7.2 ships in the venv via AeroSandbox
   and bundles IPOPT** — an interior-point, sparse-friendly, multiplier-native NLP
   solver one import away (or via cyipopt). Trigger to switch: convergence trouble
   returns or DVs grow past ~150 (Performance #1/#2 territory). Until then, SLSQP +
   the V.0 speed work is the plan.
5. **Dependency & tooling hygiene.** `gmsh` is imported (`structural/mesh.py:97`) but
   undeclared in pyproject (works by venv accident) — declare it. Add ruff (and the
   basedpyright config the docs already assume) to the dev group. **Python: on 3.13
   (2026-06-09, suite green 160/160); upgrade to 3.14 when build123d > 0.10.0
   releases** — it's the sole blocker (`<3.14` + OCP `<7.9`; cp314 OCP wheels exist,
   build123d dev already supports `<3.15`). See CLAUDE.md "Pending upgrade" for the
   bump recipe.
6. **Freeze the legacy sizers.** `sizing.py` and `shell_sizing.py` share ~20–40%
   structural duplication with the laminate path; they are live (phase examples/tests)
   but should be explicitly frozen as historical baselines — new features go to the
   laminate path only (already the de-facto policy; make it stated).
7. **Autodiff (JAX/casadi): no, with a trigger.** The hand adjoint is FD-validated and
   cached; a rewrite re-litigates 27 sensitivity tests for speculative benefit.
   Reconsider element-kernel-level autodiff only if the sensitivity surface keeps
   growing (e.g. `∂f_pre/∂x` prestress terms plus eigenvalue derivatives plus core/wall
   buckling gradients all landing). Eigenvalue-buckling gradients alone don't force it
   (clean hand formula: `φᵀ(∂K/∂x + λ ∂Kσ/∂x)φ`).

---

## Suggested sequence

Validity first — three of the fixes (panel pressure, self-weight, D11 direction) could
move the honest baseline *up*, and lever gains must be measured against a trustworthy
baseline:

1. ~~Merge the analytic-Jacobian caching~~ (done, `78c5d3d`, 1.98×). Extract Lagrange
   multipliers while the sizer code is warm (Performance #8 — a few lines), and
   consider KS aggregation (Performance #7) if the re-baselining runs feel slow.
2. Mesh-convergence study + eigenvalue buckling check of the current optimum
   (Validity #1, #9). Run the 5-minute prestress post-processing probe (Performance #2)
   alongside.
3. Add self-weight + panel pressure; re-baseline the small and medium headlines
   (Validity #4, #5).
4. Do the constraint/design-vector refactor (Toolbox #1) — after V.3's constraint
   shape is known, before any harvest lever adds DV blocks. CI + smoke layers
   (Toolbox #2/#3) can land any time alongside.
5. Harvest in leverage order against the re-baselined numbers: hollow members (core
   tube first, then straight beam segments) → prestress + spreaders/webs → sandwich
   skin → n_beams sweep — one lever at a time, recording findings as before.

The meta-observation: layout levers are mined out because buckling governs. The
remaining step-changes all attack *what kind of section resists buckling* — hollow
beams, prestressed tension skin, cored panels — rather than where material sits.
