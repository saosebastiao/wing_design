# M2-Core: Deflection-Driven Generation Loop

**Date:** 2026-06-03
**Branch:** `constraint-based-generative-optimization`
**Status:** Implemented and verified. See "Outcome & honest limitations" below.

## 0. Outcome & honest limitations

**Delivered & verified.** Generation is now genuinely deflection-driven.
`examples/21_generate_truss.py` over the full envelope produces:

```
mass=  7.69 kg  ratio=0.485  tip=1876.4 mm  feasible=False  governing=survival
mass= 15.37 kg  ratio=0.171  tip= 469.1 mm  feasible=False  governing=survival
mass= 23.06 kg  ratio=0.093  tip= 208.5 mm  feasible=True   governing=survival
```

The lightest designs fail tip deflection under `survival`; the loop selects the
lightest survivor. The final review confirmed: the worst-case-governs logic is
provably correct (an infeasible case always has higher severity, so it always
governs); load capture went from 0 N (M1 coarse frame) to ~98.4% of the analytic
integral after subdivision; tip deflection scales as 1/mass² (textbook for a
solid circle, confirming the lumping is correctly calibrated). 59 tests pass; all
52 M1 tests remain green.

**Validated engineering conclusion:** this wingsail is **stiffness-driven**, not
strength-driven — stress ratio peaks at 0.485 (lightest) and is 0.093 at the
chosen design, so tip deflection is the binding constraint. Buckling/torsion
(gate v1) are correctly deferred — they would not change any verdict here.

**The binding limitation is now library richness, not the loop.** CP-SAT only
ever enumerates the spar at the six section buckets (mass 1:2:3:4:5:6) because the
placeholder library has just 3 landmark nodes and 5 coincident-endpoint beams —
there is no genuine truss to build, so the lightest survivor is necessarily a
"fat spar," not a clever structure. The loop is doing its job correctly on a
degenerate library. **Making this produce real trusses requires the curved
stress-line library (spec §5 tracer) — that is the highest-value next deepening.**

**Carry forward:** (1) curved stress-line library (the headline item — the loop
is ready for it); (2) make `solve_designs` lazy (generator) so enumeration stops
at the first feasible design once the library is large; (3) gate v1
(buckling/torsion) only if a future load case makes strength bind; (4) a ~1.6%
tip-truncation in lumping is benign now but worth revisiting if a case has
significant tip loading.
**Builds on:** `2026-06-02-constraint-based-generation-design.md` (Milestone 1: 1A CP-SAT
core, 1B frame gate, 1C generator + end-to-end). See that spec's §0 for the M1
known-limitations this increment addresses.

## 1. Motivation (grounded in the M1 diagnostic)

Milestone 1 delivered a working end-to-end pipeline, but a diagnostic showed its
"generative intelligence" was inert:

- The example sized against `nominal_trim` (10 m/s, SF 1.0) → only 152 N lift →
  background von Mises ≈ 14 MPa vs 1100 MPa allowable (ratio 0.013). A mild
  operating point, not a sizing case.
- The **load lumping lost ~43% of the load** (86 N of 152 N reached the frame):
  the selected designs' frames had no spar node between the deck (z=−0.2) and the
  first bow midpoint (z≈2.4), so the entire inboard half-span of lift had nowhere
  to land.
- **`n_beams_min=4` governed** the design entirely; the coverage proxy collapsed
  to the smallest section bucket for every target (so it never constrained) and
  the conflict table was empty (all beams share the deck/tip nodes).
- **Engineering insight:** even scaled to the worst envelope load, the stress
  ratio only reaches ~0.28 — a solid carbon spar is very strong for these loads.
  **Tip deflection is the binding constraint** (74 mm at nominal; ~340 mm at the
  survival load, past the 250 mm limit). This wingsail is *stiffness-driven*, not
  strength-driven. The real optimization is "minimize mass s.t. tip deflection ≤
  limit (and stress ratio ≤ 1) across the load envelope."

M2-core turns the inert plumbing into genuine deflection-driven generation.

## 2. Decisions (from brainstorming)

| Question | Decision |
| --- | --- |
| Loop mechanism | **Enumerate by mass + gate each** → return the lightest design that survives. The frame gate is cheap (tiny 1D solve; the 40 s FEA runs once in the menu build), so exhaustive ascending-mass enumeration is trivially correct and minimal. Targeted failure-reaction is deferred until the library is large. |
| Load scope | **Full envelope, worst governs.** Gate each design against every (non-feathered) design case; it must pass the worst. `GateResult.governing_case` reports which. Affordable because gating is cheap. |
| Coverage proxy | **Drop the CP-SAT constraint** (inert and conceptually shaky: shell-skin stress ≠ beam sizing), behind a flag so M1's tests stay valid. Keep coverage *targets* as diagnostics in the menu/VTU. **Relax `n_beams_min` to 1** so CP-SAT proposes from a bare spar upward; the gate guarantees adequacy. |
| Gate fidelity | **Keep v0** (stress + tip deflection). Defer Euler buckling + torsion (gate v1) — strength is far from binding, so they would not change any verdict here. |

## 3. Changes by component

### 3.1 Load-lumping fix — `gate.py` `build_frame`
Add `max_element_length_m` (default ~0.3 m). Each consecutive control-point pair
is subdivided into `ceil(segment_length / max_element_length)` straight
sub-elements with interior nodes, so the inboard half-span has nodes to receive
lift. Element stiffness / transform / solve are unchanged — this is purely a
meshing refinement. Interior nodes get `kind=None`; only landmark coordinates
keep their kind, so bearing BCs and tip detection are unaffected.

**Acceptance:** after subdivision, the per-case lumped nodal total integrates to
≈ the case's `factored_normal_force_N` (was 86/152 N; target ~100%).

### 3.2 CP-SAT model — `model.py`
`build_cp_model(menu, params, *, enforce_coverage=True)`: when `False`, skip the
`_add_coverage` call. `solve_designs` passes the flag through. Default `True`
preserves every M1 test and behavior. The loop calls with
`enforce_coverage=False`.

`GenerativeParameters.n_beams_min` default lowered to **1** (the spar is always
grounded by reach-tip + host implications, so "1" means "spar minimum"). M1 unit
tests override `n_beams_min` explicitly, so they are unaffected.

### 3.3 Selection loop — new `loop.py`
Two layers, for testability:

- **Pure core:** `select_lightest_feasible(designs, gate_fn) -> Selected | None`
  — iterate `designs` (already ascending by mass), return the first whose
  `gate_fn(design)` is feasible, else `None`. Unit-tested with a fake `gate_fn`
  (no FEA / geometry).
- **Integration:** `generate_truss(menu, params, aero_by_case, *, max_candidates)
  -> TrussResult` — enumerate `solve_designs(menu, params.generative,
  top_n=max_candidates, enforce_coverage=False)`; for each design build the frame,
  lump each case's load, gate against all cases (worst `GateResult` governs);
  return the lightest feasible design + its governing verdict + the full **gated
  frontier** (per-design: mass, stress ratio, tip deflection, feasible, governing
  case) for reporting. If none pass (catalog ceiling can't satisfy the envelope),
  return a clear "no feasible design in menu" result — honest signal to enlarge
  the catalog or add beams.

`aero_by_case`: `{case: AeroResult}` precomputed once per case by the caller
(`run_case_lifting_line` per `DESIGN_CASES`, excluding `feathered`).

`TrussResult` (dataclass): `chosen: WingCandidate | None`,
`verdict: GateResult | None`, `frontier: tuple[GatedDesign, ...]` where
`GatedDesign` carries the design + its worst `GateResult`.

### 3.4 Envelope gating helper
`generate_truss` gates a frame against the envelope by lumping each case's
spanwise normal force (via the existing `lump_spanwise_force_to_nodes`, +Y) and
taking the worst `GateResult` (least feasible / highest ratio / highest
deflection), tagging it with the governing case name.

## 4. Examples & testing

### 4.1 Examples
Update `examples/21_generate_truss.py` to call `generate_truss` over the envelope,
print the gated frontier and the chosen design. Expected result: a
deflection-driven pick (a bigger spar and/or more beams than M1's trivial
four-at-minimum), governed by `survival`.

### 4.2 Tests (write first)
- `select_lightest_feasible` with a fake gate: lightest design fails, a heavier
  one passes → returns the heavier; all-fail → `None`.
- `build_frame` subdivision: a long segment yields the expected interior-node
  count; landmark nodes keep their kinds; degenerate guard still holds.
- Lumping conservation post-subdivision: a frame with a finely-discretized spar
  captures ≈ 100% of a known uniform line load over the span.
- `enforce_coverage=False`: a menu/design that would violate coverage is now
  permitted by `build_cp_model`; `enforce_coverage=True` still rejects it.
- Envelope gating: a design feasible under a mild case but not under a severe one
  is reported infeasible with the severe case as `governing_case`.
- The existing 52 tests stay green.

## 5. Done when
- `examples/21_generate_truss.py` runs over the full envelope and selects the
  lightest design that survives it (deflection-governed), printing the frontier;
  exports STEP/STL.
- The lumped load reaching the frame ≈ the case's factored normal force.
- `uv run pytest tests/generative/` is green (M1's 52 + the new M2 tests).
- Generation is genuinely driven by the frame gate over the load envelope, not by
  a beam-count floor.

## 6. Carried forward (later increments)
Curved stress-line library (spec §5 tracer), gate v1 (buckling/torsion), wraps
(M3), volumetric finalist check (M4), targeted failure-reaction for large
libraries, and revisiting the coverage proxy only if a cheap *structural* (not
skin-stress) warm-start proves worthwhile.
