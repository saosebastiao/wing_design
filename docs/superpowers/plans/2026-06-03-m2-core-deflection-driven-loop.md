# M2-Core Deflection-Driven Generation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make wing-truss generation genuinely deflection-driven: enumerate CP-SAT designs by ascending mass and gate each against the full load-case envelope, returning the lightest design that survives.

**Architecture:** Four targeted changes reusing the Milestone 1 stack. (1) `build_frame` gains opt-in element subdivision so the inboard half-span has nodes to receive lift (fixes the ~43% load loss). (2) `build_cp_model` gains an `enforce_coverage` flag and the beam-count floor is relaxed, so the cheap frame gate — not a count floor or an inert stress proxy — decides adequacy. (3) A new `loop.py` enumerates designs and gates each over the envelope (worst case governs), returning the lightest feasible plus the gated frontier. (4) The end-to-end example is rewired to use it.

**Tech Stack:** Python 3.10–3.12; reuses `generative/{menu,model,gate,loads,candidates,build}`, `aero/{cases,loads,model}`; NumPy; `pytest`; `uv`. No new dependencies.

**Scope note:** This is the M2-core increment. Design spec: `docs/superpowers/specs/2026-06-03-m2-core-deflection-driven-loop-design.md`. Deferred to later increments (per spec §6): curved stress-line library, gate v1 (buckling/torsion), wraps, volumetric finalist check, targeted failure-reaction. Backward-compatibility is a hard requirement: **all 52 existing `tests/generative/` tests must stay green** — subdivision is opt-in (default off) precisely so M1 frames are unchanged.

**Verified APIs this plan relies on:** `build_frame(candidate, menu) -> FrameModel(coords (N,3), elements list[(i,j,area)], node_kinds list, mass_kg)`; `solve_frame(frame, params, nodal_loads, governing_case) -> GateResult(feasible, max_stress_ratio, tip_deflection_m, governing_case, mass_kg)`; `solve_designs(menu, gen_params, top_n) -> list[WingCandidate]`; `build_cp_model(menu, gen_params) -> (model, select, sect)`; `lump_spanwise_force_to_nodes(frame, density_fn, *, z_min, z_max, direction) -> {node_idx:(fx,fy,fz)}`; `build_beam_library(params) -> (nodes, beams, cross_sections)`; `WingSpec.z_wing_root` (0.0) / `z_wing_tip` (span). `GenerativeParameters` carries `n_beams_min`, `tip_deflection_limit_m`, `solver_max_time_s`, `top_n_designs`.

---

## File Structure

- Modify: `src/wing_design/scenario.py` — add `frame_max_element_length_m`; lower `n_beams_min` default to 1.
- Modify: `src/wing_design/generative/gate.py` — `build_frame` opt-in subdivision.
- Modify: `src/wing_design/generative/model.py` — `enforce_coverage` flag on `build_cp_model` + `solve_designs`.
- Create: `src/wing_design/generative/loop.py` — `select_lightest_feasible`, `generate_truss`, `TrussResult`, `GatedDesign`.
- Modify: `src/wing_design/generative/__init__.py` — export the loop API.
- Modify: `examples/21_generate_truss.py` — use `generate_truss` over the envelope.
- Modify/Create tests: `tests/generative/test_gate.py`, `tests/generative/test_model.py`, `tests/generative/test_loop.py`.

---

## Task M1: Scenario parameters

**Files:**
- Modify: `src/wing_design/scenario.py`

- [ ] **Step 1: Confirm no committed test asserts the old `n_beams_min` default**

Run:
```bash
grep -rn "n_beams_min" tests/
```
Expected: only occurrences are `_params(n_beams_min=...)` explicit overrides in `tests/generative/test_model.py` (which are unaffected by a default change). If any test asserts `default_scenario().generative.n_beams_min == 4` or similar, note it — none should exist.

- [ ] **Step 2: Edit `GenerativeParameters`**

In `src/wing_design/scenario.py`, in the `GenerativeParameters` dataclass: change the `n_beams_min` line and add a new field after `n_z_layers`.

Change:
```python
    n_beams_min: int = 4
```
to:
```python
    n_beams_min: int = 1  # M2: the gate guarantees adequacy; floor no longer governs
```

And after the `n_z_layers: int = 12` line, add:
```python
    # Frame discretization for the gate: subdivide beam segments to this length so
    # the inboard span has nodes to receive distributed load (M2). None = no split.
    frame_max_element_length_m: float = 0.3
```

- [ ] **Step 3: Verify**

Run:
```bash
uv run python -c "from wing_design.scenario import default_scenario as d; g=d().generative; print(g.n_beams_min, g.frame_max_element_length_m)"
```
Expected: `1 0.3`

- [ ] **Step 4: Confirm the suite is still green**

Run: `uv run pytest tests/generative/ -q`
Expected: `52 passed`

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/scenario.py
git commit -m "feat: relax n_beams_min to 1 and add frame_max_element_length_m"
```

---

## Task M2: `build_frame` opt-in subdivision

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.loads import lump_spanwise_force_to_nodes


def test_build_frame_subdivides_long_segments():
    menu = _single_beam_menu()  # spar keel(-0.95)->deck(-0.20)->tip(5.0)
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=1.0)
    # No subdivision: 3 nodes / 2 elements (the M1 behavior).
    coarse = build_frame(candidate, menu, max_element_length_m=None)
    assert coarse.coords.shape == (3, 3)
    assert len(coarse.elements) == 2
    # Subdivided: each ~0.3 m -> many interior nodes; landmarks keep their kinds.
    fine = build_frame(candidate, menu, max_element_length_m=0.3)
    assert fine.coords.shape[0] > 3
    assert len(fine.elements) > 2
    assert NodeKind.KEEL_STEP in fine.node_kinds
    assert NodeKind.DECK_STEP in fine.node_kinds
    assert NodeKind.TIP in fine.node_kinds
    # interior nodes are unclassified
    assert fine.node_kinds.count(None) >= 1


def test_subdivision_improves_distributed_load_capture():
    # Non-uniform line load density(z)=z^2 over the wing span [0, span]; analytic
    # integral = span^3 / 3. A coarse spar frame (only the tip node is in-span)
    # badly under/over-integrates; a finely subdivided one converges to it.
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=1.0)
    span = 5.0
    analytic = span**3 / 3.0

    def density(z):
        return z * z

    coarse = build_frame(candidate, menu, max_element_length_m=None)
    fine = build_frame(candidate, menu, max_element_length_m=0.1)

    def total(frame):
        loads = lump_spanwise_force_to_nodes(frame, density, z_min=0.0, z_max=span,
                                             direction=(0.0, 1.0, 0.0))
        return sum(fy for (_fx, fy, _fz) in loads.values())

    coarse_err = abs(total(coarse) - analytic) / analytic
    fine_err = abs(total(fine) - analytic) / analytic
    assert fine_err < 0.02          # fine frame integrates the load accurately
    assert fine_err < coarse_err    # and is strictly better than the coarse frame
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/generative/test_gate.py -k "subdivide or subdivision" -v`
Expected: FAIL — `build_frame()` got an unexpected keyword argument `max_element_length_m`.

- [ ] **Step 3: Replace `build_frame`**

In `src/wing_design/generative/gate.py`, replace the entire `build_frame` function with:
```python
def build_frame(candidate, menu, *, max_element_length_m=None):
    """Discretize a WingCandidate into a FrameModel.

    Each selected beam's consecutive control points become beam elements;
    coincident coordinates (shared endpoints / junctions) merge into one node so
    beams that touch transfer load. Node kinds are inherited from the menu's
    landmark nodes by coordinate match.

    If `max_element_length_m` is given, each control-point segment is split into
    equal sub-elements no longer than that, adding interior nodes (kind=None) so a
    distributed load has somewhere to land along the span. None keeps one element
    per control-point segment.
    """
    kind_by_key = {_node_key(n.xyz): n.kind for n in menu.nodes}
    coords = []
    node_kinds = []
    index_by_key = {}

    def node_index(point):
        key = _node_key(point)
        if key not in index_by_key:
            index_by_key[key] = len(coords)
            coords.append((float(point[0]), float(point[1]), float(point[2])))
            node_kinds.append(kind_by_key.get(key))
        return index_by_key[key]

    def n_sub(p_a, p_b):
        if max_element_length_m is None:
            return 1
        length = math.dist(p_a, p_b)
        return max(1, math.ceil(length / max_element_length_m))

    elements = []
    for beam_id, bucket in candidate.beam_sections:
        beam = menu.beam_by_id(beam_id)
        cs = next((c for c in menu.cross_sections if c.bucket == bucket), None)
        if cs is None:
            raise KeyError(f"cross-section bucket {bucket} not in menu")
        pts = beam.control_points
        for p_a, p_b in zip(pts[:-1], pts[1:]):
            steps = n_sub(p_a, p_b)
            prev = node_index(p_a)
            for k in range(1, steps + 1):
                t = k / steps
                mid = tuple(p_a[d] + t * (p_b[d] - p_a[d]) for d in range(3))
                cur = node_index(mid)
                elements.append((prev, cur, cs.area_m2))
                prev = cur

    return FrameModel(
        coords=np.array(coords, dtype=float),
        elements=elements,
        node_kinds=node_kinds,
        mass_kg=candidate.mass_kg,
    )
```

(`math` is already imported in `gate.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_gate.py -v`
Expected: PASS (new tests + all prior gate tests, since default `max_element_length_m=None` reproduces the old behavior exactly).

- [ ] **Step 5: Confirm the whole suite is green**

Run: `uv run pytest tests/generative/ -q`
Expected: `54 passed`

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: opt-in beam-segment subdivision in build_frame (fix load capture)"
```

---

## Task M3: `enforce_coverage` flag

**Files:**
- Modify: `src/wing_design/generative/model.py`
- Modify: `tests/generative/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_model.py`:
```python
def test_enforce_coverage_flag_toggles_constraint():
    # A target needs area >= 5e-3 but the catalog tops out at 2e-3, so coverage
    # is unsatisfiable. With enforce_coverage=True the model is INFEASIBLE; with
    # enforce_coverage=False the constraint is dropped and the model is feasible.
    m = menu(
        [beam(0, covers=(5,))],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        coverage=[target(5, required_min_area_m2=5.0e-3, candidate_beams=[0])],
    )
    params = _params(n_beams_min=0, n_beams_max=1)

    model_on, _s, _x = build_cp_model(m, params, enforce_coverage=True)
    solver_on, status_on = _solve(model_on)
    assert status_on == cp_model.INFEASIBLE

    model_off, _s2, _x2 = build_cp_model(m, params, enforce_coverage=False)
    solver_off, status_off = _solve(model_off)
    assert status_off in (cp_model.OPTIMAL, cp_model.FEASIBLE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_model.py::test_enforce_coverage_flag_toggles_constraint -v`
Expected: FAIL — `build_cp_model()` got an unexpected keyword argument `enforce_coverage`.

- [ ] **Step 3: Add the flag**

In `src/wing_design/generative/model.py`, change the `build_cp_model` signature and the coverage call. Replace:
```python
def build_cp_model(menu: CandidateMenu, params: GenerativeParameters):
    """Build the CP-SAT model. Returns (model, select, sect)."""
    model = cp_model.CpModel()
    select, sect = _add_variables(model, menu)
    _add_count(model, menu, params, select)
    _add_symmetry(model, menu, select, sect)
    _add_topology(model, menu, select)
    _add_reach_tip(model, menu, select)
    _add_no_intersection(model, menu, sect)
    _add_coverage(model, menu, sect)
    return model, select, sect
```
with:
```python
def build_cp_model(menu: CandidateMenu, params: GenerativeParameters, *,
                   enforce_coverage: bool = True):
    """Build the CP-SAT model. Returns (model, select, sect).

    `enforce_coverage=False` drops the stress-coverage constraint, leaving the
    frame gate as the sole judge of structural adequacy (M2). Default True
    preserves the Milestone 1 behavior.
    """
    model = cp_model.CpModel()
    select, sect = _add_variables(model, menu)
    _add_count(model, menu, params, select)
    _add_symmetry(model, menu, select, sect)
    _add_topology(model, menu, select)
    _add_reach_tip(model, menu, select)
    _add_no_intersection(model, menu, sect)
    if enforce_coverage:
        _add_coverage(model, menu, sect)
    return model, select, sect
```

Then update `solve_designs` to pass the flag through. Replace its first body line:
```python
    model, select, sect = build_cp_model(menu, params)
```
with:
```python
    model, select, sect = build_cp_model(menu, params, enforce_coverage=enforce_coverage)
```
and change the `solve_designs` signature line from:
```python
def solve_designs(menu, params, top_n=1):
```
to:
```python
def solve_designs(menu, params, top_n=1, *, enforce_coverage=True):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_model.py -v`
Expected: PASS (new test + all prior model tests).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/model.py tests/generative/test_model.py
git commit -m "feat: enforce_coverage flag to let the gate be sole judge"
```

---

## Task M4: Loop core + result types

**Files:**
- Create: `src/wing_design/generative/loop.py`
- Create: `tests/generative/test_loop.py`

- [ ] **Step 1: Write the failing test**

Create `tests/generative/test_loop.py`:
```python
from wing_design.generative.loop import (
    GatedDesign,
    TrussResult,
    select_lightest_feasible,
)
from wing_design.generative.menu import GateResult, WingCandidate


def _design(mass):
    return WingCandidate(beam_sections=((0, 0),), mass_kg=mass)


def _verdict(feasible, mass):
    return GateResult(feasible=feasible, max_stress_ratio=0.5,
                      tip_deflection_m=0.1, governing_case="x", mass_kg=mass)


def test_select_returns_first_feasible():
    # designs ascending by mass; the lightest fails, the next passes.
    a, b, c = _design(1.0), _design(2.0), _design(3.0)
    verdicts = {1.0: _verdict(False, 1.0), 2.0: _verdict(True, 2.0),
                3.0: _verdict(True, 3.0)}
    result = select_lightest_feasible([a, b, c], lambda d: verdicts[d.mass_kg])
    assert result is not None
    chosen, verdict = result
    assert chosen is b
    assert verdict.feasible


def test_select_returns_none_when_all_fail():
    a, b = _design(1.0), _design(2.0)
    result = select_lightest_feasible([a, b], lambda d: _verdict(False, d.mass_kg))
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.generative.loop'`

- [ ] **Step 3: Create loop.py core**

Create `src/wing_design/generative/loop.py`:
```python
"""Deflection-driven selection loop (M2-core).

The frame gate is cheap (a tiny 1D solve; the slow FEA runs once in the menu
build), so the loop enumerates CP-SAT designs in ascending mass and gates each
against the load envelope, returning the lightest design that survives. See
docs/superpowers/specs/2026-06-03-m2-core-deflection-driven-loop-design.md.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .gate import build_frame, solve_frame
from .loads import lump_spanwise_force_to_nodes
from .menu import GateResult, WingCandidate
from .model import solve_designs


@dataclass(frozen=True)
class GatedDesign:
    """One enumerated design and its worst-case gate verdict."""
    design: WingCandidate
    verdict: GateResult


@dataclass(frozen=True)
class TrussResult:
    """Outcome of the selection loop.

    `chosen`/`verdict` are None when no enumerated design survived the envelope.
    `frontier` is every design gated up to (and including) the chosen one, in the
    order tried (ascending mass).
    """
    chosen: WingCandidate | None
    verdict: GateResult | None
    frontier: tuple[GatedDesign, ...]


def select_lightest_feasible(designs, gate_fn):
    """Return (design, verdict) for the first design whose `gate_fn` is feasible.

    `designs` is an iterable already in ascending mass; `gate_fn(design)` returns
    a GateResult. Returns None if no design is feasible.
    """
    for d in designs:
        verdict = gate_fn(d)
        if verdict.feasible:
            return d, verdict
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_loop.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/loop.py tests/generative/test_loop.py
git commit -m "feat: loop core select_lightest_feasible + result types"
```

---

## Task M5: `generate_truss` envelope integration

**Files:**
- Modify: `src/wing_design/generative/loop.py`
- Modify: `tests/generative/test_loop.py`
- Modify: `src/wing_design/generative/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_loop.py`:
```python
from wing_design.generative.candidates import build_beam_library
from wing_design.generative.loop import generate_truss
from wing_design.generative.menu import CandidateMenu, ConflictTable
from wing_design.scenario import default_scenario


def _library_menu(params):
    nodes, beams, cross_sections = build_beam_library(params)
    return CandidateMenu(
        nodes=tuple(nodes), beams=tuple(beams), cross_sections=tuple(cross_sections),
        conflicts=ConflictTable(forbidden=()), coverage_targets=(),
        rho_kgm3=params.material.rho_kgm3,
    )


def test_generate_truss_picks_lightest_feasible_under_envelope():
    # No FEA: a real beam library + simple analytic per-case load densities.
    params = default_scenario()
    menu = _library_menu(params)
    # A gentle case everything passes, and a severe case that sizes the design.
    cases = {
        "gentle": (lambda z: 5.0),
        "severe": (lambda z: 4.0e3),
    }
    result = generate_truss(menu, params, cases, max_candidates=200)
    assert result.chosen is not None
    assert result.verdict.feasible
    # the chosen design survives the worst case
    assert result.verdict.governing_case in cases
    # frontier records what was tried, ascending in mass
    masses = [g.design.mass_kg for g in result.frontier]
    assert masses == sorted(masses)
    assert result.frontier[-1].design is result.chosen


def test_generate_truss_returns_none_when_envelope_unsurvivable():
    params = default_scenario()
    menu = _library_menu(params)
    # An absurd load no catalog section can survive.
    cases = {"impossible": (lambda z: 1.0e9)}
    result = generate_truss(menu, params, cases, max_candidates=50)
    assert result.chosen is None
    assert result.verdict is None
    assert len(result.frontier) > 0  # designs were tried and all failed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_loop.py -k generate_truss -v`
Expected: FAIL — `ImportError: cannot import name 'generate_truss'`

- [ ] **Step 3: Implement generate_truss**

Append to `src/wing_design/generative/loop.py`:
```python
def _severity(verdict, tip_limit_m):
    """Scalar demand of a verdict: max of stress ratio and normalized deflection.
    The case with the highest severity is the governing (worst) case."""
    tip_term = verdict.tip_deflection_m / tip_limit_m if tip_limit_m > 0 else 0.0
    return max(verdict.max_stress_ratio, tip_term)


def _worst_over_cases(frame, params, case_load_fns, load_direction):
    """Gate a frame against every case; return the highest-severity GateResult."""
    spec = params.geometry
    tip_limit = params.generative.tip_deflection_limit_m
    worst = None
    for case_name, density_fn in case_load_fns.items():
        loads = lump_spanwise_force_to_nodes(
            frame, density_fn, z_min=spec.z_wing_root, z_max=spec.z_wing_tip,
            direction=load_direction,
        )
        verdict = solve_frame(frame, params, loads, governing_case=case_name)
        if worst is None or _severity(verdict, tip_limit) > _severity(worst, tip_limit):
            worst = verdict
    return worst


def generate_truss(menu, params, case_load_fns, *,
                   load_direction=(0.0, 1.0, 0.0), max_candidates=200):
    """Enumerate designs by mass and return the lightest that survives the envelope.

    `case_load_fns` maps a case name to a spanwise normal-force density function
    density(z) -> N/m (the caller builds these from the aero results, keeping this
    loop independent of AeroSandbox). Returns a TrussResult; `chosen` is None if no
    enumerated design survives every case.
    """
    designs = solve_designs(menu, params.generative, top_n=max_candidates,
                            enforce_coverage=False)
    frontier = []

    def gate_fn(design):
        frame = build_frame(
            design, menu,
            max_element_length_m=params.generative.frame_max_element_length_m,
        )
        verdict = _worst_over_cases(frame, params, case_load_fns, load_direction)
        frontier.append(GatedDesign(design=design, verdict=verdict))
        return verdict

    result = select_lightest_feasible(designs, gate_fn)
    chosen, verdict = result if result is not None else (None, None)
    return TrussResult(chosen=chosen, verdict=verdict, frontier=tuple(frontier))
```

- [ ] **Step 4: Export the loop API**

In `src/wing_design/generative/__init__.py`, add after the `from .loads import ...` line:
```python
from .loop import GatedDesign, TrussResult, generate_truss, select_lightest_feasible
```
and add `"GatedDesign"`, `"TrussResult"`, `"generate_truss"`, `"select_lightest_feasible"` to `__all__`.

- [ ] **Step 5: Run tests + import check**

Run: `uv run pytest tests/generative/test_loop.py -v`
Expected: PASS (4 tests).

Run:
```bash
uv run python -c "from wing_design.generative import generate_truss, TrussResult; print('ok')"
```
Expected: `ok`

- [ ] **Step 6: Confirm the whole suite is green**

Run: `uv run pytest tests/generative/ -q`
Expected: all pass (M1's 52 + the M2 additions).

- [ ] **Step 7: Commit**

```bash
git add src/wing_design/generative/loop.py tests/generative/test_loop.py src/wing_design/generative/__init__.py
git commit -m "feat: generate_truss envelope loop (lightest design that survives)"
```

---

## Task M6: Rewire example 21 to the envelope loop

**Files:**
- Modify: `examples/21_generate_truss.py`

- [ ] **Step 1: Replace the example**

Replace the entire contents of `examples/21_generate_truss.py` with:
```python
"""M2-core: end-to-end deflection-driven generation. Build the candidate menu,
then enumerate CP-SAT designs by mass and gate each against the full load-case
envelope (worst case governs), exporting the lightest design that survives.

Run: just example 21_generate_truss
"""
from pathlib import Path

from build123d import export_step, export_stl

from wing_design import default_scenario
from wing_design.aero.cases import DESIGN_CASES
from wing_design.aero.loads import run_case_lifting_line
from wing_design.aero.model import build_airplane
from wing_design.generative import (
    build_candidate_menu,
    generate_truss,
    wing_candidate_to_part,
)

EXPORT = Path("exports")


def _density_fn(aero, span):
    """Spanwise normal-force density for a case, clamped to the wing span."""
    return lambda z: float(aero.distributed_normal_force(min(max(z, 0.0), span)))


def main() -> None:
    params = default_scenario()
    spec = params.geometry
    menu = build_candidate_menu(params)

    # Aero per sizing case (skip 'feathered' — it carries ~no load).
    airplane = build_airplane(spec)
    case_load_fns = {}
    for case in DESIGN_CASES:
        if case.name == "feathered":
            continue
        aero = run_case_lifting_line(airplane, case,
                                     spanwise_resolution=params.aero.spanwise_resolution)
        case_load_fns[case.name] = _density_fn(aero, spec.span)

    result = generate_truss(menu, params, case_load_fns,
                            max_candidates=params.generative.top_n_designs * 8)

    print("gated frontier (ascending mass):")
    for g in result.frontier:
        v = g.verdict
        print(f"  mass={g.design.mass_kg:6.2f} kg  ratio={v.max_stress_ratio:.3f}  "
              f"tip={v.tip_deflection_m*1000:6.1f} mm  feasible={v.feasible}  "
              f"governing={v.governing_case}")

    if result.chosen is None:
        print("\nNo design in the menu survives the load envelope — enlarge the "
              "cross-section catalog or add beams.")
        return

    part = wing_candidate_to_part(result.chosen, menu)
    EXPORT.mkdir(exist_ok=True)
    export_step(part, str(EXPORT / "generated_truss.step"))
    export_stl(part, str(EXPORT / "generated_truss.stl"))
    print(f"\nchosen: mass={result.chosen.mass_kg:.2f} kg  "
          f"governed by {result.verdict.governing_case}  "
          f"(ratio={result.verdict.max_stress_ratio:.3f}, "
          f"tip={result.verdict.tip_deflection_m*1000:.1f} mm)")
    print(f"wrote {EXPORT/'generated_truss.step'} and .stl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example**

Run: `uv run python examples/21_generate_truss.py`
Expected: prints the gated frontier (lightest designs failing on deflection under `survival`, heavier ones passing) and the chosen deflection-governed design; writes `exports/generated_truss.step` + `.stl`. Takes ~1–2 min (one menu FEA build + several per-case aero solves; the gating itself is fast).

If every design is reported infeasible, that is a real result — but first sanity-check that the densest design (all beams at the largest bucket) is being reached within `max_candidates`; if not, raise `max_candidates`. Do NOT loosen the deflection limit. Report BLOCKED with the frontier if the outcome is surprising (e.g., even the densest design fails, or the lightest trivially passes with no failures above it — which would suggest the loads are still understated).

- [ ] **Step 3: Commit**

```bash
git add examples/21_generate_truss.py
git commit -m "feat: example 21 selects the lightest design surviving the load envelope"
```

---

## Self-Review

**Spec coverage (spec §3 components, §4 examples/tests, §5 done-when):**
- §3.1 lumping fix (opt-in subdivision, interior nodes kind=None, landmarks keep kind, load capture acceptance) → Task M2 ✓ (the `test_subdivision_improves_distributed_load_capture` test is the acceptance check)
- §3.2 `enforce_coverage` flag (default True preserves M1) + `n_beams_min`→1 → Tasks M3, M1 ✓
- §3.3 loop two layers (`select_lightest_feasible` pure + `generate_truss` integration), `TrussResult`/`GatedDesign`, frontier, no-feasible result → Tasks M4, M5 ✓
- §3.4 envelope gating (worst governs via severity) → Task M5 `_worst_over_cases` ✓
- §4.1 example over envelope → Task M6 ✓
- §4.2 tests: fake-gate selection (M4), subdivision + conservation (M2), enforce_coverage toggle (M3), envelope governing-case (M5 `test_generate_truss_picks_lightest_feasible_under_envelope` exercises multi-case worst-governs), suite stays green (every task re-runs it) ✓
- §5 done-when → Task M6 run + suite green ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. The example's "if surprising, report BLOCKED" is execution guidance, not a code placeholder.

**Type consistency:** `build_frame(candidate, menu, *, max_element_length_m=None)` — the new kwarg is used consistently in M2 tests and `generate_truss` (M5). `build_cp_model(..., *, enforce_coverage=True)` and `solve_designs(..., *, enforce_coverage=True)` consistent across M3 and M5. `GateResult` fields (`feasible`, `max_stress_ratio`, `tip_deflection_m`, `governing_case`, `mass_kg`) used consistently in `_severity`, the loop tests, and `_worst_over_cases`. `TrussResult(chosen, verdict, frontier)` / `GatedDesign(design, verdict)` constructed identically in `generate_truss` and asserted in M5 tests. `lump_spanwise_force_to_nodes(frame, density_fn, *, z_min, z_max, direction)` call in `_worst_over_cases` matches its 1C signature. `case_load_fns` is `{name: density(z)->N/m}` consistently in M5 and the M6 example.

---

## Done When

- `uv run pytest tests/generative/` is green (M1's 52 + M2's new tests).
- `examples/21_generate_truss.py` enumerates over the full envelope and exports the **lightest design that survives it** — a deflection-governed pick (heavier than M1's trivial minimum), with the frontier showing lighter designs failing first.
- The distributed load reaching the frame integrates to ≈ the analytic total (subdivision acceptance test passes).
- Generation is driven by the frame gate over the load envelope, not by a beam-count floor or the inert coverage proxy.
