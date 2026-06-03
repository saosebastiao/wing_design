# Constraint-Based Generative Wing-Truss Design

**Date:** 2026-06-02
**Branch:** `constraint-based-generative-optimization`
**Status:** Approved design, pending implementation plan

## 1. Summary

A constraint-programming approach to generating the internal carbon-fiber
space-frame of a wingsail. We move away from the Arora/Jiang alternating-LP /
generic-optimization path and instead use Google OR-Tools **CP-SAT** to
**generate, constrain, and score** candidate trusses against discrete
manufacturability, assembly, material, and design constraints, with FEM as the
final structural judge.

The core architectural decision (confirmed during brainstorming):

> **CP-SAT selects and assembles from a precomputed discrete candidate graph.
> Continuous spline geometry and FEM live outside the solver and feed results
> back as updated weights/constraints.**

CP-SAT is a discrete combinatorial solver — it cannot evaluate a spline, run an
FEM solve, or reason about a continuous stress field inside its model. So the
whole pipeline is organized around that boundary: precompute a finite menu of
candidate beams + cross-sections offline, let CP-SAT do constrained selection,
and let the continuous world (geometry + FEM) generate the menu and judge the
results.

## 2. Decisions locked during brainstorming

| Question | Decision |
| --- | --- |
| CP-SAT's role | Selects/assembles from a precomputed discrete candidate graph; geometry + FEM stay outside and feed back. |
| Beam encoding | **A now, B later** — CP-SAT selects from a library of complete pre-traced beam centerlines first; data model shaped so it can graduate to DAG path-routing without a rewrite. |
| Structural steering | **Stress-coverage proxy + FEM gate** — CP-SAT minimizes mass but must cover high-stress regions with adequately-sized stress-aligned beams; FEM confirms; failures tighten the coverage/section rule. No equilibrium LP. |
| First milestone | **Thin end-to-end slice** — one load case, small candidate library, CP-SAT select + discrete cross-sections + coverage, build123d export + one FEM re-check. No outer loop, no wraps. |
| FEM gate | **Frame solver now, volumetric later** — lightweight 1D frame solver during the loop; solid/volumetric FEM reserved for finalists. Prioritize fast iteration first, accuracy later. |

## 3. Glossary (project terminology, encoded in code)

- **root spar** — the virtual spar assembled by wrapping all beams; sits below
  and rotates within the boat hull (360°+).
- **keel-step** — base of the spar; bearing 1 in the rotating-spar model.
- **deck-step** — top of the spar where it meets the deck; bearing 2.
- **transition span** — deck-step → wing-root fairing; a structural member, not
  part of the wing proper.
- **wing root** — z = 0; lowest station with the full airfoil; aspect-ratio
  denominator chord.
- **wing tip** — tallest station; aspect-ratio numerator chord.
- **wing span** — wing root → wing tip (note: differs from aircraft "span").
- **entasis** — configurable chord change along the wing span (already supported
  via `WingSpec.taper_profile`).

## 4. Domain & data model

Convention: frozen dataclasses, immutable artifacts passed between stages
(matching the existing `scenario.py` pattern).

### 4.1 Geometry & manufacturability parameters

Extend `WingSpec` / `DesignParameters` (do not replace):

- **Landmarks / accessors:** `z_keel_step`, `z_deck_step`, `z_wing_root`,
  `z_wing_tip` (derived from existing `transition_length` / `spar_length` /
  `span`) so code speaks the glossary instead of scattering arithmetic.
- **Manufacturability:** mold-box height `box_max_height_m` (length/width derived
  from the wing), `beam_min_radius_m` (curvature floor), `n_beams_min` /
  `n_beams_max`, `cross_section_area_max_m2`, `max_area_step_ratio`
  (point-to-point area-change limit).

### 4.2 The discrete candidate menu (CP-SAT's input)

- **`CandidateNode`** — `id`, `xyz`, `kind` ∈ {keel_step, deck_step, tip,
  mesh_centroid, on_beam}, `z_layer` (int, groundwork for the future DAG), local
  principal-stress frame + magnitudes (from the background FEM).
- **`CandidateBeam`** — a complete pre-traced centerline: `id`, ordered control
  points (monotonic-z spline), `start_kind` / `end_kind` (enforcing valid
  endpoint rules: keel→tip, keel→on-beam, on-beam→tip), `length_m`,
  `min_radius_m` (precomputed; must clear the floor to enter the menu),
  `mirror_id` (reflected partner), `on_chord_plane` (built on y=0 vs. mirrored
  pair), `covers` (set of coverage-target ids it can serve).
- **`CrossSectionOption`** — discrete catalog entry: `shape` ∈ {circle,
  semicircle, voronoi}, `area_bucket` (int → area m²), derived radius. One
  assigned per beam (later: per segment).
- **`ConflictTable`** — precomputed pairwise incompatibilities: forbidden
  `(beam_i, bucket_a, beam_j, bucket_b)` tuples where centerlines pass closer
  than the sum of radii at those buckets. Legitimate shared nodes are excluded so
  beams can touch at junctions. This turns "no intersection except at shared
  nodes" into pure boolean CP-SAT data; the expensive geometry happens once,
  offline.
- **`CoverageTarget`** — a high-stress mesh region: `id`, `centroid`,
  `required_min_area_m2` (local stress magnitude × safety factor), list of beams
  able to cover it.

### 4.3 Design & verdict

- **`WingCandidate`** — CP-SAT output: selected beam ids, assigned cross-section
  buckets, (later) wrap assignments, proxy objective value (mass).
- **`GateResult`** — frame solver's verdict: `feasible` (bool),
  `max_stress_ratio`, `tip_deflection_m`, `governing_case`, `mass_kg`.

### 4.4 Wraps (slots now, inert until the wrap milestone)

- **`BeamWrap`** — covered beam-span + thickness (1 mm increments), anisotropic
  fiber along the wrap circumference.
- **`WingWrap`** — single shell thickness, airfoil surface + binds all beams,
  fiber approximately tip-to-tail (horizontal).

### 4.5 Symmetry

Encoded structurally: candidate beams are generated as either on-chord-plane
singletons or mirror pairs; CP-SAT ties each pair with a single boolean. "Symmetric
about the chord line" is a property of the **menu**, not a runtime constraint to
fight.

## 5. The candidate generator (offline, deterministic, cacheable)

Heavy FEM runs once; output is an inspectable artifact (VTU + serialized menu).

1. **Wing solid** — `build_wing_solid(spec)` (exists).
2. **Background FEM** — mesh (existing tet `mesh.py` or shell), apply load-case
   aero traction (existing `aero/loads.py` + `projection.py`), solve linear
   elastic (`fea.py` / `shell.py`). Extract per-node/cell principal-stress frame
   + magnitudes (existing `frame_field.principal_frame_from_voigt` +
   `align_signs_bfs`).
3. **Harvest candidate nodes** — fixed landmarks (keel-step, deck-step, tip)
   always included; mesh-cell centroids filtered to high-σ and downsampled by
   Poisson-disk. Assign integer `z_layer`.
4. **Trace candidate beams** — reuse `streamline.py` / `extract.py`, then conform
   each to the beam rules:
   - clip/resample to **monotonic-increasing z** (discard/split centerlines that
     curl back down — stress lines aren't naturally monotonic);
   - reject any whose **min curvature radius** is below the floor;
   - **snap endpoints** to legal kinds;
   - force the **keel→deck vertical run** (chord-aligned, pointed-up) for
     keel-rooted beams (the "wing-root sections parallel and pointed directly up"
     rule);
   - generate the **mirror partner** across y=0
     (`mirror_family_across_chord_plane`), or tag `on_chord_plane`.
5. **Cross-section catalog** — discrete area buckets up to
   `cross_section_area_max_m2`. Milestone: `shape = circle` everywhere. Semicircle
   (near wing wrap) and Voronoi (spar root / parallel-beam bundles) rules attach
   to the same `CrossSectionOption` type in a later milestone.
6. **Conflict table** — pairwise centerline min-distance; per bucket pair, mark
   forbidden when min-distance < r_a + r_b. Shared nodes excluded.
7. **Coverage targets** — cluster high-σ regions; each gets `required_min_area`
   and the list of beams within tolerance.

**Output:** `CandidateMenu` artifact — serialized (json/pickle) **and** VTU for
ParaView inspection before CP-SAT runs.

**Tractability levers** (explicit in `DesignParameters`): max node count, max
library size, Poisson-disk radius, z-layer count.

**Flagged behaviors:**
- The monotonic-z filter discards a meaningful fraction of raw stress lines —
  expected for the library approach, and the main motivation for the later DAG
  router (which enforces monotonic-z structurally).
- Conflict-table size grows as (library size)² × (buckets)². Levers bound it;
  **table size is logged, never truncated silently.**

## 6. The CP-SAT model

Follows `ortools-cp` skill conventions: snake_case API, booleans + reification
over big-M, integer-scaled objective.

### 6.1 Decision variables

- `select[b]` — boolean, beam `b` in the design.
- `sect[b, k]` — boolean, beam `b` uses bucket `k`. One-hot, tied to selection:
  `sum_k sect[b,k] == select[b]`.
- Wrap variables: stubs now, activated in the wrap milestone.

### 6.2 Constraints

1. **Beam count** — `n_beams_min ≤ sum_b select[b] ≤ n_beams_max` (mirror pair =
   two physical beams).
2. **Symmetry** — each mirror pair shares one decision: `select[b] ==
   select[mirror(b)]` and equal section.
3. **Endpoint/topology validity** — beams ending/starting on another beam require
   the host: `select[b] ≤ select[host(b)]`. Hosts always start at strictly lower
   z, so these implications form a DAG that transitively grounds every selected
   beam back to the keel-step — support connectivity is free, no cycles possible.
4. **Reach the tip** — at least one selected beam ends at a tip node.
5. **No intersection** — for every forbidden tuple: `sect[b_i, a] + sect[b_j, b]
   ≤ 1`.
6. **Stress coverage (proxy)** — for each target `t`: `sum over {(b,k) : b covers
   t and area_k ≥ required_t} of sect[b,k] ≥ 1`.

**Handled as menu filters, not solver constraints:** mold-box height and min
curvature radius (geometric properties — violating beams never enter the menu);
max area-step ratio (inert with one section per beam; activates with per-segment
sections).

**Generator contracts the CP-SAT model depends on but does NOT re-validate**
(the candidate generator, Plan 1C, must guarantee these; add a `validate_menu`
guard there that asserts them before solving):
- **Monotonic-increasing z** per beam centerline — so the tip is always a
  beam's high-z *end*. The reach-tip constraint checks `end_kind == TIP` only;
  a non-monotonic beam could otherwise escape it.
- **Acyclic host graph rooted at the keel** — every `host_id` chain terminates
  at a keel-rooted beam (`start_kind == KEEL_STEP`, `host_id is None`). This is
  what makes `select[b] ≤ select[host]` *transitively* ground every selected
  beam. A host cycle would let an ungrounded floating sub-truss pass the solver.
- **Reciprocal `mirror_id`** — if beam A's mirror is B, then B's mirror is A;
  relied on by the symmetry tie.
- **`solve_designs` ordering** is made robust to solver time-outs by a final
  sort of the harvested designs by mass (a round may return a non-optimal
  FEASIBLE design under `solver_max_time_s`).

**Additional contracts the frame gate (Plan 1B) depends on** (also for the
Plan 1C `validate_menu` guard — `build_frame` inherits node kinds and merges
junctions purely by coordinate, so these must hold exactly):
- **Landmark nodes sit at exact beam endpoints** — the keel-step, deck-step, and
  tip `CandidateNode` coordinates must coincide (bit-identical, within the 1e-6 m
  merge grid) with the corresponding beam control points. Otherwise `build_frame`
  silently drops the landmark kind and the gate loses its bearing BCs. `solve_frame`
  now hard-errors if keel/deck/tip kinds are absent, but 1C must place them
  correctly so that error never fires in normal operation.
- **Host control points at every ON_BEAM junction** — when a beam starts/ends on
  a host, the host beam must carry a control point at the exact junction
  coordinate, so the merged frame shares a node there and transfers load.
  Otherwise the junction dangles and the solve is singular (now a diagnostic
  `LinAlgError`).
- **Coordinate identity across the menu** — beam endpoints, junction points, and
  landmark node coordinates that are meant to coincide must be the *same* float
  values (e.g. derived from one shared node), not independently recomputed, since
  the merge is snap-to-grid rather than tolerance-based.

### 6.3 Objective

Minimize **mass** = `sum_{b,k} sect[b,k] × (length_b × area_k × ρ)`. Linear
integer objective, scaled to **milligrams** (integer coefficients per CP-SAT).
Penalties (stress-concentration, wrappability/prefer-bundles) attach as weighted
integer terms — wired but zero-weighted in the milestone.

### 6.4 Solve mode

Solve to proxy-optimality, then harvest the **top-N distinct near-optimal
designs** via CP-SAT's solution collector — the gate gets a batch, and the later
ranked-pool loop tightens via an added `mass ≤ best_feasible` constraint between
rounds.

## 7. The frame-solver gate

Lightweight 1D linear frame solver; staged for "accuracy later."

**Material fit:** UD carbon with fiber along the spline means the beam's axial
direction *is* the stiff fiber direction, so "anisotropic, fiber-parallel-to-
spline" collapses to a standard beam element using E1 axially — no special
anisotropic machinery at this fidelity.

### 7.1 Model

- **Nodes** = beam endpoints + junctions (a beam ending on another creates a
  shared load-transfer node).
- **Elements** = each beam spline discretized into short straight 3D beam
  elements, 6 DOF/node (3 translation + 3 rotation), 12×12 stiffness, with `A`,
  `I`, `J` from the assigned bucket and `E1`/`G` from `materials/unidir`.

### 7.2 Boundary conditions (the physically subtle bit)

**Not** a clamped cantilever. A free-rotating spar reacted by **two bearings**:
keel-step (translations fixed) and deck-step (radial/perpendicular-to-spar
translations fixed, axial free). The wing's overturning moment is carried as a
**force couple** between the bearings. A clamped base would massively over-stiffen
the root and mislead stresses — this differs from the existing
`TubeSparBendingStructure` and must be implemented carefully.

### 7.3 Loads

Lump the load case's spanwise normal-force density (existing
`AeroResult.distributed_normal_force`) onto nearest beam nodes per z-station.
Approximate but appropriate for an in-loop gate.

### 7.4 Verdict

Solve `K u = f` (sparse), recover per-element axial + bending stress, compute
`max_stress_ratio` vs UD allowables and `tip_deflection`. `feasible` = stress
ratio ≤ 1 **and** tip deflection ≤ limit.

### 7.5 Staging

- **v0 (milestone):** linear-elastic frame, single load case, axial+bending
  stress + tip deflection. Bare frame, no wraps.
- **v1:** Euler buckling per compression member (reuse ASB
  `column_buckling_critical_load`), torsion check, full load-case envelope with
  governing-case reporting.
- **later:** wraps as joint/section stiffeners; volumetric FEM on finalists.

**Honest caveat (recorded):** the bare frame (no wing wrap) underestimates
torsional stiffness — the wing wrap is a major torsion carrier. Acceptable for
the milestone (conservative on torsion, pipeline-proving); motivates the wrap
milestone and the volumetric finalist check.

## 8. Module layout & milestone-one cut

### 8.1 New package `src/wing_design/generative/`

- `menu.py` — the §4 dataclasses.
- `candidates.py` — `build_candidate_menu(spec, fem_result) -> CandidateMenu`.
- `model.py` — `build_cp_model(menu, params)`, `solve_designs(...) ->
  list[WingCandidate]`.
- `gate.py` — `solve_frame(candidate, menu, load_case) -> GateResult`.
- `build.py` — `wing_candidate_to_part(candidate, menu)` → build123d loft +
  STEP/STL/VTU export.
- `loop.py` — ranked-pool tightening loop (stubbed in milestone, built in M2).
- `scenario.py` gains a `GenerativeParameters` group (tractability levers,
  beam-count bounds, box height, coverage safety factor, gate limits).

### 8.2 Examples

- `examples/20_candidate_menu.py` — solid → background FEM → menu; export menu
  VTU.
- `examples/21_generate_truss.py` — menu → CP-SAT top-N → gate → best feasible →
  build123d STEP/STL/VTU.

### 8.3 Testing (write tests first)

- **CP-SAT model against synthetic menus** (critical de-risking) — hand-built
  tiny menus with known optimal selection; assert each constraint independently:
  symmetry tie, count bounds, no-intersection exclusion, coverage enforcement,
  support-implication DAG, objective minimality. No FEM/geometry needed.
- **Frame solver against closed form** — single straight cantilever tip
  deflection vs `PL³/3EI`; a 2-bar truss vs hand calc; verify bearing-couple BCs
  reproduce a known couple reaction.
- **End-to-end smoke test** — `21_generate_truss` runs on the default scenario,
  produces a feasible design + STEP file.

### 8.4 Milestone-one Definition of Done

`just example 21_generate_truss` runs end-to-end on the 5 m wingsail, **single
load case** (nominal trim), **circle cross-sections only**, **no wraps, no outer
loop**, emitting a **chord-symmetric** selected truss (STEP/STL/VTU) with a
`GateResult` passing the v0 stress + tip-deflection check — backed by passing
CP-SAT-constraint and frame-solver unit tests.

## 9. Deferred outer loop (designed now, built in M2)

Each round: CP-SAT solves min-mass subject to all constraints **plus** `mass ≤
best_feasible_so_far − ε`, harvests top-N; the gate evaluates the batch
(parallelizable); feasible designs enter a **pool ranked by mass**; the lightest
sets the next ceiling. Reaction rules:

- **On failure → increase area / add beams:** read the frame solver's
  overstressed elements, map them back to under-served coverage targets, raise
  those targets' `required_min_area`; add a **no-good cut** forbidding the exact
  failed (beam-set, sections). If nothing passes, raise `n_beams_min`.
- **On exceeding weight → reduce area / beams:** already enforced by the
  mass-ceiling constraint.

Termination: no feasible design beats the ceiling within the time budget, the
pool stabilizes, or a round cap is hit. **Multi-load-case:** background FEM per
case, coverage targets unioned, gate loops the envelope, design must pass the
governing case.

## 10. Milestone sequence

1. **M1** — thin end-to-end slice (this design).
2. **M2** — outer ranked-pool loop + full load-case envelope + gate v1 (buckling
   + torsion).
3. **M3** — wraps: beam wraps (parallel-bundle binding) + wing wrap (airfoil
   surface + torsion shell); activates semicircle / Voronoi cross-section rules.
4. **M4** — volumetric FEM finalist check + fixed-point re-solve with as-designed
   stiffness.
5. **M5** — DAG routing (Approach 2) as an alternative candidate engine feeding
   the same CP-SAT/gate stack.

## 11. Risk register

| Risk | Mitigation |
| --- | --- |
| Conflict-table blow-up ((size)² × (buckets)²) | Spatial-bin pruning; tractability levers; log table size, never truncate silently |
| Monotonic-z filter starves the menu → CP-SAT infeasible | Always seed a guaranteed-feasible fallback (a few fat keel→tip spar-cap beams); report library yield |
| No harvested design passes the gate | Diagnostics on the binding constraint/target; fallback design guarantees feasibility exists |
| Gate underestimates torsion (bare frame) | Flagged; wing-wrap milestone + volumetric finalist check |
| Load lumping is approximate | Acceptable for in-loop gate; volumetric check on finalists |
| CP-SAT scaling with library size | Time-box solver, solution pool, tune workers (per `ortools-cp` skill) |
| Proxy/gate disagreement | Loop's failure→tighten mechanism; log proxy-vs-gate agreement to calibrate the coverage rule |

## 12. Reuse map (existing code leveraged)

| Need | Existing module |
| --- | --- |
| Wing solid | `geometry/wing.py` (`build_wing_solid`, `WingSpec`) |
| Background mesh | `structural/mesh.py`, `structural/shell.py` |
| Background FEM | `structural/fea.py`, `structural/shell.py` |
| Aero loads | `aero/{model,cases,loads}.py`, `structural/projection.py` |
| Principal directions | `truss/frame_field.py` |
| Stress-line tracing | `truss/{streamline,surface_streamlines,extract}.py` |
| Mirroring | `truss/extract.py` (`mirror_family_across_chord_plane`) |
| Materials / allowables | `materials/unidir.py` |
| Buckling helpers (M2) | AeroSandbox `column_buckling_critical_load` |
| Scenario / params | `scenario.py` (`DesignParameters`) |
| Viz | `paraview/`, `viz/` |
