# Generative CP-SAT Core (Milestone 1A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data model and the CP-SAT selection engine that, given a precomputed discrete candidate menu, generates chord-symmetric, manufacturable, stress-covering wing-truss designs ranked by mass.

**Architecture:** Frozen dataclasses describe an immutable `CandidateMenu` (beams, cross-sections, conflict table, coverage targets). A CP-SAT model selects a subset of candidate beams and assigns each a discrete cross-section bucket, subject to beam-count bounds, chord-symmetry ties, support-topology implications, tip reachability, pairwise no-intersection, and stress-coverage constraints, minimizing total mass (integer milligrams). A solver wrapper harvests the top-N distinct near-optimal designs. This subsystem has **no FEM or geometry dependency** and is fully unit-tested against hand-built synthetic menus.

**Tech Stack:** Python 3.10–3.12, `ortools` CP-SAT (already a dependency), `pytest` (added here as a dev dependency), `uv` for env management.

**Scope note:** This is plan **1A of 3** for Milestone 1 (the thin end-to-end slice). Plan 1B (frame-solver gate) and Plan 1C (candidate generator + build123d export + examples) follow. See `docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md` §6 and §8.

**Reference conventions:** Follow the `ortools-cp` skill — snake_case CP-SAT API (`new_bool_var`, `add`, `minimize`, `solver.solve`, `solver.value`), booleans + reification over big-M, integer-scaled objective.

---

## File Structure

- Create: `src/wing_design/generative/__init__.py` — package exports.
- Create: `src/wing_design/generative/menu.py` — all frozen dataclasses + enums (the data model).
- Create: `src/wing_design/generative/model.py` — `build_cp_model` + `solve_designs` (the CP-SAT engine).
- Modify: `src/wing_design/scenario.py` — add `GenerativeParameters` and attach to `DesignParameters`.
- Modify: `src/wing_design/geometry/wing.py` — add landmark z-accessors to `WingSpec`.
- Create: `tests/generative/__init__.py` — test package marker.
- Create: `tests/generative/_menu_factory.py` — helpers to build synthetic menus.
- Create: `tests/generative/test_menu.py` — data-model unit tests.
- Create: `tests/generative/test_model.py` — CP-SAT constraint + solve unit tests.
- Modify: `pyproject.toml` — add `pytest` dev dependency (via `uv add --dev`).

---

## Task 1: Dev tooling — pytest + test package layout

**Files:**
- Modify: `pyproject.toml` (via `uv add --dev pytest`)
- Create: `tests/generative/__init__.py`

- [ ] **Step 1: Add pytest as a dev dependency**

Run:
```bash
uv add --dev pytest
```
Expected: `pyproject.toml` gains a `[dependency-groups]` (or `[tool.uv]` dev) entry for `pytest`; `uv.lock` updates; venv syncs.

- [ ] **Step 2: Create the test package marker**

Create `tests/generative/__init__.py` with exactly:
```python
```
(empty file — marks the directory as a package)

- [ ] **Step 3: Verify pytest runs**

Run:
```bash
uv run pytest tests/ -q
```
Expected: pytest collects 0 tests and exits 0 (`no tests ran`). Confirms pytest is installed and the path resolves.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/generative/__init__.py
git commit -m "chore: add pytest dev dependency and tests/generative package"
```

---

## Task 2: `GenerativeParameters` in scenario.py

**Files:**
- Modify: `src/wing_design/scenario.py`

- [ ] **Step 1: Add the dataclass**

In `src/wing_design/scenario.py`, after the `SkinParameters` dataclass (before the "Umbrella container" section), add:
```python
@dataclass(frozen=True)
class GenerativeParameters:
    """Knobs for the constraint-based generative truss stack (Milestone 1).

    Tractability levers keep the precomputed candidate menu and the CP-SAT
    model small; the structural/manufacturability bounds feed the constraints
    and the coverage proxy. See
    docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md.
    """
    # Beam-count bounds (a mirror pair counts as its two physical beams)
    n_beams_min: int = 4
    n_beams_max: int = 40
    # Manufacturability
    box_max_height_m: float = 1.5
    beam_min_radius_m: float = 0.3
    cross_section_area_max_m2: float = 5.0e-3
    n_area_buckets: int = 6
    # Coverage proxy
    coverage_safety_factor: float = 2.0
    # Candidate-menu tractability levers
    max_node_count: int = 400
    max_library_size: int = 200
    poisson_disk_radius_m: float = 0.15
    n_z_layers: int = 12
    # Gate limits (used by Plan 1B; defined here so the scenario is complete)
    tip_deflection_limit_m: float = 0.25
    # CP-SAT solve controls
    solver_max_time_s: float = 30.0
    top_n_designs: int = 8
```

- [ ] **Step 2: Attach it to `DesignParameters`**

In the `DesignParameters` dataclass, after the `skin_sizing` field, add:
```python
    generative: GenerativeParameters = field(default_factory=GenerativeParameters)
```

- [ ] **Step 3: Verify it imports and constructs**

Run:
```bash
uv run python -c "from wing_design.scenario import default_scenario; p = default_scenario(); print(p.generative.n_beams_min, p.generative.top_n_designs)"
```
Expected: prints `4 8`

- [ ] **Step 4: Commit**

```bash
git add src/wing_design/scenario.py
git commit -m "feat: add GenerativeParameters to the design scenario"
```

---

## Task 3: Landmark z-accessors on `WingSpec`

**Files:**
- Modify: `src/wing_design/geometry/wing.py`

- [ ] **Step 1: Add the accessors**

In `src/wing_design/geometry/wing.py`, inside the `WingSpec` dataclass, after the `chord_at_z` method, add:
```python
    @property
    def z_wing_root(self) -> float:
        """Z of the wing root (lowest full-airfoil station)."""
        return 0.0

    @property
    def z_wing_tip(self) -> float:
        """Z of the wing tip."""
        return self.span

    @property
    def z_deck_step(self) -> float:
        """Z of the deck-step (top of the spar / bottom of the transition span)."""
        return -self.transition_length

    @property
    def z_keel_step(self) -> float:
        """Z of the keel-step (base of the cylindrical spar)."""
        return -(self.transition_length + self.spar_length)
```

- [ ] **Step 2: Verify the values match the loft layout**

Run:
```bash
uv run python -c "from wing_design.geometry.wing import WingSpec; s=WingSpec(); print(s.z_keel_step, s.z_deck_step, s.z_wing_root, s.z_wing_tip)"
```
Expected: prints `-0.95 -0.2 0.0 5.0`

- [ ] **Step 3: Commit**

```bash
git add src/wing_design/geometry/wing.py
git commit -m "feat: add keel/deck/root/tip z-accessors to WingSpec"
```

---

## Task 4: Cross-section enums + `CrossSectionOption`

**Files:**
- Create: `src/wing_design/generative/__init__.py`
- Create: `src/wing_design/generative/menu.py`
- Create: `tests/generative/test_menu.py`

- [ ] **Step 1: Write the failing test**

Create `tests/generative/test_menu.py` with:
```python
import math

from wing_design.generative.menu import CrossSectionOption, CrossSectionShape


def test_cross_section_radius_is_equivalent_circle():
    cs = CrossSectionOption(bucket=0, shape=CrossSectionShape.CIRCLE, area_m2=math.pi)
    assert math.isclose(cs.radius_m, 1.0, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_menu.py::test_cross_section_radius_is_equivalent_circle -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.generative'`

- [ ] **Step 3: Create the package and the enum + option**

Create `src/wing_design/generative/__init__.py` with:
```python
"""Constraint-based generative wing-truss stack (Milestone 1).

CP-SAT selects/assembles internal beams from a precomputed candidate menu;
spline geometry and FEM live outside the solver. See
docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md.
"""
```

Create `src/wing_design/generative/menu.py` with:
```python
"""Immutable data model for the generative truss stack.

A `CandidateMenu` is the precomputed discrete input to the CP-SAT model: a
library of candidate beams, a cross-section catalog, a pairwise conflict table,
and stress-coverage targets. CP-SAT selects a subset of beams and assigns each
a cross-section bucket; the result is a `WingCandidate`, judged by the frame
gate into a `GateResult`.

All dataclasses are frozen so a menu can't be mutated mid-solve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class NodeKind(str, Enum):
    KEEL_STEP = "keel_step"
    DECK_STEP = "deck_step"
    TIP = "tip"
    MESH_CENTROID = "mesh_centroid"
    ON_BEAM = "on_beam"


class CrossSectionShape(str, Enum):
    CIRCLE = "circle"
    SEMICIRCLE = "semicircle"
    VORONOI = "voronoi"


@dataclass(frozen=True)
class CrossSectionOption:
    """One discrete cross-section choice in the catalog."""
    bucket: int
    shape: CrossSectionShape
    area_m2: float

    @property
    def radius_m(self) -> float:
        """Equivalent-circle radius for the cross-sectional area."""
        return math.sqrt(self.area_m2 / math.pi)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/generative/test_menu.py::test_cross_section_radius_is_equivalent_circle -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/__init__.py src/wing_design/generative/menu.py tests/generative/test_menu.py
git commit -m "feat: add generative package with CrossSectionOption"
```

---

## Task 5: Remaining data-model dataclasses

**Files:**
- Modify: `src/wing_design/generative/menu.py`
- Modify: `tests/generative/test_menu.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_menu.py`:
```python
from wing_design.generative.menu import (
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    ConflictTable,
    CoverageTarget,
    GateResult,
    WingCandidate,
)


def test_candidate_menu_lookup_and_wing_candidate_ids():
    nodes = (
        CandidateNode(id=0, xyz=(0.0, 0.0, -0.95), kind=NodeKind.KEEL_STEP, z_layer=0),
        CandidateNode(id=1, xyz=(0.0, 0.0, 5.0), kind=NodeKind.TIP, z_layer=11),
    )
    beam = CandidateBeam(
        id=7,
        control_points=((0.0, 0.0, -0.95), (0.0, 0.0, 5.0)),
        start_kind=NodeKind.KEEL_STEP,
        end_kind=NodeKind.TIP,
        start_node=0,
        end_node=1,
        length_m=5.95,
        min_radius_m=10.0,
        on_chord_plane=True,
        mirror_id=None,
        host_id=None,
        covers=(2,),
    )
    cs = (CrossSectionOption(bucket=0, shape=CrossSectionShape.CIRCLE, area_m2=1e-3),)
    menu = CandidateMenu(
        nodes=nodes,
        beams=(beam,),
        cross_sections=cs,
        conflicts=ConflictTable(forbidden=()),
        coverage_targets=(
            CoverageTarget(id=2, centroid=(0.0, 0.0, 2.5),
                           required_min_area_m2=1e-3, candidate_beams=(7,)),
        ),
        rho_kgm3=1550.0,
    )
    assert menu.beam_by_id(7) is beam
    cand = WingCandidate(beam_sections=((7, 0),), mass_kg=9.22)
    assert cand.beam_ids == (7,)
    verdict = GateResult(feasible=True, max_stress_ratio=0.8,
                         tip_deflection_m=0.1, governing_case="nominal", mass_kg=9.22)
    assert verdict.feasible
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_menu.py::test_candidate_menu_lookup_and_wing_candidate_ids -v
```
Expected: FAIL — `ImportError: cannot import name 'CandidateBeam'`

- [ ] **Step 3: Add the dataclasses**

Append to `src/wing_design/generative/menu.py`:
```python
@dataclass(frozen=True)
class CandidateNode:
    """A discrete point CP-SAT may build beams from."""
    id: int
    xyz: tuple[float, float, float]
    kind: NodeKind
    z_layer: int
    # Background-FEM principal frame (filled by the candidate generator, Plan 1C).
    principal_dirs: tuple[tuple[float, float, float], ...] = ()
    principal_mags: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CandidateBeam:
    """A complete pre-traced beam centerline (monotonic-increasing z).

    `host_id` is the beam this one starts or ends *on* (for an ON_BEAM
    endpoint), or None. Per the valid endpoint rules (keel->tip, keel->on-beam,
    on-beam->tip) at most one endpoint is ever ON_BEAM, so a single host
    suffices. `mirror_id` is the reflected partner across the chord plane, or
    None when the beam lies on the chord plane (`on_chord_plane=True`).
    """
    id: int
    control_points: tuple[tuple[float, float, float], ...]
    start_kind: NodeKind
    end_kind: NodeKind
    start_node: int
    end_node: int
    length_m: float
    min_radius_m: float
    on_chord_plane: bool
    mirror_id: int | None
    host_id: int | None
    covers: tuple[int, ...]


@dataclass(frozen=True)
class ConflictTable:
    """Precomputed pairwise incompatibilities.

    Each entry `(beam_i, bucket_a, beam_j, bucket_b)` forbids selecting beam_i
    at bucket_a together with beam_j at bucket_b (their centerlines pass closer
    than the sum of radii at those buckets). Legitimate shared nodes are
    excluded when the table is built, so beams may touch at junctions.
    """
    forbidden: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class CoverageTarget:
    """A high-stress region that must be served by an adequately-sized beam."""
    id: int
    centroid: tuple[float, float, float]
    required_min_area_m2: float
    candidate_beams: tuple[int, ...]


@dataclass(frozen=True)
class BeamWrap:
    """Filament-wound shell binding a span of beams (inert until the wrap milestone)."""
    beam_ids: tuple[int, ...]
    thickness_m: float


@dataclass(frozen=True)
class WingWrap:
    """Single airfoil-forming shell (inert until the wrap milestone)."""
    thickness_m: float


@dataclass(frozen=True)
class CandidateMenu:
    """The precomputed discrete input to the CP-SAT model."""
    nodes: tuple[CandidateNode, ...]
    beams: tuple[CandidateBeam, ...]
    cross_sections: tuple[CrossSectionOption, ...]
    conflicts: ConflictTable
    coverage_targets: tuple[CoverageTarget, ...]
    rho_kgm3: float

    def beam_by_id(self, beam_id: int) -> CandidateBeam:
        for b in self.beams:
            if b.id == beam_id:
                return b
        raise KeyError(f"no candidate beam with id {beam_id}")


@dataclass(frozen=True)
class WingCandidate:
    """A CP-SAT-selected design: (beam_id, cross-section bucket) pairs + mass."""
    beam_sections: tuple[tuple[int, int], ...]
    mass_kg: float

    @property
    def beam_ids(self) -> tuple[int, ...]:
        return tuple(bid for bid, _ in self.beam_sections)


@dataclass(frozen=True)
class GateResult:
    """The frame solver's verdict on a WingCandidate (filled by Plan 1B)."""
    feasible: bool
    max_stress_ratio: float
    tip_deflection_m: float
    governing_case: str
    mass_kg: float
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run pytest tests/generative/test_menu.py -v
```
Expected: PASS (both tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/menu.py tests/generative/test_menu.py
git commit -m "feat: add candidate-menu and design data model"
```

---

## Task 6: Synthetic menu factory for tests

**Files:**
- Create: `tests/generative/_menu_factory.py`

- [ ] **Step 1: Write the factory (a test helper, not production code)**

Create `tests/generative/_menu_factory.py` with:
```python
"""Builders for small synthetic CandidateMenus used in CP-SAT unit tests.

Every beam built here defaults to a globally-valid configuration (ends at the
tip, on the chord plane, no host, no coverage), so a menu stays feasible under
the *full* CP-SAT model even when an individual test only exercises one
constraint.
"""
from __future__ import annotations

from wing_design.generative.menu import (
    CandidateBeam,
    CandidateMenu,
    ConflictTable,
    CoverageTarget,
    CrossSectionOption,
    CrossSectionShape,
    NodeKind,
)


def cs_catalog(areas):
    """Catalog of circular cross-sections, one bucket per area (m^2)."""
    return tuple(
        CrossSectionOption(bucket=i, shape=CrossSectionShape.CIRCLE, area_m2=a)
        for i, a in enumerate(areas)
    )


def beam(
    beam_id,
    length=1.0,
    start_kind=NodeKind.KEEL_STEP,
    end_kind=NodeKind.TIP,
    on_chord_plane=True,
    mirror_id=None,
    host_id=None,
    covers=(),
):
    """A globally-valid CandidateBeam with sensible defaults."""
    return CandidateBeam(
        id=beam_id,
        control_points=((0.0, 0.0, 0.0), (0.0, 0.0, length)),
        start_kind=start_kind,
        end_kind=end_kind,
        start_node=0,
        end_node=1,
        length_m=length,
        min_radius_m=10.0,
        on_chord_plane=on_chord_plane,
        mirror_id=mirror_id,
        host_id=host_id,
        covers=covers,
    )


def menu(beams, cross_sections=None, conflicts=(), coverage=(), rho=1550.0):
    if cross_sections is None:
        cross_sections = cs_catalog([1.0e-3, 2.0e-3])
    return CandidateMenu(
        nodes=(),
        beams=tuple(beams),
        cross_sections=tuple(cross_sections),
        conflicts=ConflictTable(forbidden=tuple(conflicts)),
        coverage_targets=tuple(coverage),
        rho_kgm3=rho,
    )


def target(target_id, required_min_area_m2, candidate_beams):
    return CoverageTarget(
        id=target_id,
        centroid=(0.0, 0.0, 0.0),
        required_min_area_m2=required_min_area_m2,
        candidate_beams=tuple(candidate_beams),
    )
```

- [ ] **Step 2: Verify the factory imports**

Run:
```bash
uv run python -c "import sys; sys.path.insert(0,'tests'); from generative._menu_factory import menu, beam; m=menu([beam(0)]); print(len(m.beams), len(m.cross_sections))"
```
Expected: prints `1 2`

- [ ] **Step 3: Commit**

```bash
git add tests/generative/_menu_factory.py
git commit -m "test: add synthetic candidate-menu factory"
```

---

## Task 7: CP-SAT model — variables, one-hot tie, beam count

**Files:**
- Create: `src/wing_design/generative/model.py`
- Create: `tests/generative/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/generative/test_model.py` with:
```python
import dataclasses

from ortools.sat.python import cp_model

from wing_design.generative.model import build_cp_model
from wing_design.scenario import GenerativeParameters

from _menu_factory import beam, cs_catalog, menu, target


def _params(**overrides):
    return dataclasses.replace(GenerativeParameters(), **overrides)


def _solve(model):
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    return solver, status


def test_count_forces_exact_number_selected():
    m = menu([beam(0), beam(1), beam(2)])
    params = _params(n_beams_min=2, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    n_selected = sum(solver.value(select[b.id]) for b in m.beams)
    assert n_selected == 2


def test_one_hot_section_matches_selection():
    m = menu([beam(0), beam(1), beam(2)])
    params = _params(n_beams_min=2, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    for b in m.beams:
        n_sect = sum(solver.value(sect[(b.id, cs.bucket)]) for cs in m.cross_sections)
        assert n_sect == solver.value(select[b.id])
```

Note: tests import `_menu_factory` directly, so pytest must run with `tests/generative` on the path. We configure that in Step 4 below before first running.

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_model.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.generative.model'`

- [ ] **Step 3: Create model.py with variables, one-hot tie, and count**

Create `src/wing_design/generative/model.py` with:
```python
"""CP-SAT model that selects a wing-truss design from a CandidateMenu.

Conventions follow the ortools-cp skill: snake_case API, booleans + reification
over big-M, integer-scaled objective. The model exposes its variable
dictionaries so tests (and the outer loop) can add assumptions/cuts.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from ..scenario import GenerativeParameters
from .menu import CandidateMenu, WingCandidate

# kg -> milligrams: keeps the mass objective in integer coefficients.
MASS_SCALE = 1_000_000


def _add_variables(model, menu):
    """select[beam_id] and sect[(beam_id, bucket)] booleans, one-hot-tied."""
    select = {b.id: model.new_bool_var(f"select_{b.id}") for b in menu.beams}
    sect = {}
    for b in menu.beams:
        for cs in menu.cross_sections:
            sect[(b.id, cs.bucket)] = model.new_bool_var(f"sect_{b.id}_{cs.bucket}")
    # Exactly one section iff the beam is selected.
    for b in menu.beams:
        model.add(
            sum(sect[(b.id, cs.bucket)] for cs in menu.cross_sections) == select[b.id]
        )
    return select, sect


def _add_count(model, menu, params, select):
    """n_beams_min <= number of selected beams <= n_beams_max."""
    total = sum(select.values())
    model.add(total >= params.n_beams_min)
    model.add(total <= params.n_beams_max)


def build_cp_model(menu: CandidateMenu, params: GenerativeParameters):
    """Build the CP-SAT model. Returns (model, select, sect)."""
    model = cp_model.CpModel()
    select, sect = _add_variables(model, menu)
    _add_count(model, menu, params, select)
    return model, select, sect
```

- [ ] **Step 4: Configure pytest to find the factory, then run**

Append to `pyproject.toml` (at the end of the file):
```toml
[tool.pytest.ini_options]
pythonpath = ["tests/generative"]
```

Run:
```bash
uv run pytest tests/generative/test_model.py -v
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/model.py tests/generative/test_model.py pyproject.toml
git commit -m "feat: CP-SAT variables, one-hot section tie, beam-count bounds"
```

---

## Task 8: CP-SAT model — chord symmetry

**Files:**
- Modify: `src/wing_design/generative/model.py`
- Modify: `tests/generative/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_model.py`:
```python
def test_symmetry_ties_mirror_pair():
    # Beams 0 and 1 are a mirror pair; selecting one must select the other.
    b0 = beam(0, on_chord_plane=False, mirror_id=1)
    b1 = beam(1, on_chord_plane=False, mirror_id=0)
    m = menu([b0, b1])
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    model.add(select[0] == 1)  # assume one half is selected
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(select[1]) == 1


def test_symmetry_ties_mirror_pair_sections():
    b0 = beam(0, on_chord_plane=False, mirror_id=1)
    b1 = beam(1, on_chord_plane=False, mirror_id=0)
    m = menu([b0, b1], cross_sections=cs_catalog([1.0e-3, 2.0e-3]))
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    model.add(sect[(0, 1)] == 1)  # beam 0 uses bucket 1
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(sect[(1, 1)]) == 1  # mirror uses the same bucket
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_model.py::test_symmetry_ties_mirror_pair -v
```
Expected: FAIL — the mirror is not forced, so `select[1]` may be 0 (assertion fails).

- [ ] **Step 3: Add the symmetry helper and call it**

In `src/wing_design/generative/model.py`, add this function after `_add_count`:
```python
def _add_symmetry(model, menu, select, sect):
    """Tie each mirror pair: same selection and same cross-section bucket."""
    seen = set()
    for b in menu.beams:
        if b.mirror_id is None:
            continue
        key = frozenset((b.id, b.mirror_id))
        if key in seen:
            continue
        seen.add(key)
        model.add(select[b.id] == select[b.mirror_id])
        for cs in menu.cross_sections:
            model.add(sect[(b.id, cs.bucket)] == sect[(b.mirror_id, cs.bucket)])
```

Then, in `build_cp_model`, add the call after `_add_count(...)`:
```python
    _add_symmetry(model, menu, select, sect)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/generative/test_model.py -v
```
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/model.py tests/generative/test_model.py
git commit -m "feat: CP-SAT chord-symmetry mirror-pair ties"
```

---

## Task 9: CP-SAT model — support topology (host implication)

**Files:**
- Modify: `src/wing_design/generative/model.py`
- Modify: `tests/generative/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_model.py`:
```python
from wing_design.generative.menu import NodeKind


def test_host_implication_requires_host_selected():
    # Beam 1 starts on beam 0; selecting beam 1 forces beam 0.
    host = beam(0, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.TIP)
    dependent = beam(1, start_kind=NodeKind.ON_BEAM, end_kind=NodeKind.TIP, host_id=0)
    m = menu([host, dependent])
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    model.add(select[1] == 1)  # select the dependent beam
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(select[0]) == 1  # host pulled in


def test_host_can_exist_without_dependent():
    host = beam(0, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.TIP)
    dependent = beam(1, start_kind=NodeKind.ON_BEAM, end_kind=NodeKind.TIP, host_id=0)
    m = menu([host, dependent])
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    model.add(select[0] == 1)
    model.add(select[1] == 0)  # host selected, dependent not — must be feasible
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_model.py::test_host_implication_requires_host_selected -v
```
Expected: FAIL — host not pulled in, `select[0]` may be 0.

- [ ] **Step 3: Add the topology helper and call it**

In `src/wing_design/generative/model.py`, add after `_add_symmetry`:
```python
def _add_topology(model, menu, select):
    """A beam that starts/ends on a host requires that host: select_b <= select_host.

    Hosts always start at strictly lower z, so these implications form a DAG that
    transitively grounds every selected beam back to the keel-step.
    """
    for b in menu.beams:
        if b.host_id is not None:
            model.add(select[b.id] <= select[b.host_id])
```

Then add the call in `build_cp_model` after `_add_symmetry(...)`:
```python
    _add_topology(model, menu, select)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/generative/test_model.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/model.py tests/generative/test_model.py
git commit -m "feat: CP-SAT support-topology host implications"
```

---

## Task 10: CP-SAT model — tip reachability

**Files:**
- Modify: `src/wing_design/generative/model.py`
- Modify: `tests/generative/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_model.py`:
```python
def test_reach_tip_forces_a_tip_beam():
    # Only beam 0 reaches the tip; beam 1 ends on beam 0. With min=0, the model
    # must still select at least one tip-reaching beam (beam 0).
    tip_beam = beam(0, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.TIP)
    inner = beam(1, start_kind=NodeKind.KEEL_STEP, end_kind=NodeKind.ON_BEAM, host_id=0)
    m = menu([tip_beam, inner])
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(select[0]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_model.py::test_reach_tip_forces_a_tip_beam -v
```
Expected: FAIL — with no tip constraint and min=0, the solver may select nothing, so `select[0]` is 0.

- [ ] **Step 3: Add the reach-tip helper and call it**

In `src/wing_design/generative/model.py`, add after `_add_topology`:
```python
def _add_reach_tip(model, menu, select):
    """At least one selected beam must terminate at the wing tip."""
    from .menu import NodeKind

    tip_beams = [
        select[b.id]
        for b in menu.beams
        if b.start_kind == NodeKind.TIP or b.end_kind == NodeKind.TIP
    ]
    if tip_beams:
        model.add(sum(tip_beams) >= 1)
```

Then add the call in `build_cp_model` after `_add_topology(...)`:
```python
    _add_reach_tip(model, menu, select)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/generative/test_model.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/model.py tests/generative/test_model.py
git commit -m "feat: CP-SAT tip-reachability constraint"
```

---

## Task 11: CP-SAT model — no intersection (conflict table)

**Files:**
- Modify: `src/wing_design/generative/model.py`
- Modify: `tests/generative/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_model.py`:
```python
def test_conflict_makes_both_at_forbidden_buckets_infeasible():
    # Single-bucket catalog; beams 0 and 1 conflict at (bucket 0, bucket 0).
    # Forcing both selected (min=max=2) must be infeasible.
    m = menu(
        [beam(0), beam(1)],
        cross_sections=cs_catalog([1.0e-3]),
        conflicts=[(0, 0, 1, 0)],
    )
    params = _params(n_beams_min=2, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status == cp_model.INFEASIBLE


def test_conflict_allows_different_buckets():
    # Two buckets; the conflict is only at (0,0,1,0). Both beams can be selected
    # if at least one uses bucket 1.
    m = menu(
        [beam(0), beam(1)],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        conflicts=[(0, 0, 1, 0)],
    )
    params = _params(n_beams_min=2, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_model.py::test_conflict_makes_both_at_forbidden_buckets_infeasible -v
```
Expected: FAIL — without the conflict constraint the model is feasible, not INFEASIBLE.

- [ ] **Step 3: Add the no-intersection helper and call it**

In `src/wing_design/generative/model.py`, add after `_add_reach_tip`:
```python
def _add_no_intersection(model, menu, sect):
    """For each forbidden (beam_i, bucket_a, beam_j, bucket_b): not both."""
    for (bi, a, bj, b) in menu.conflicts.forbidden:
        model.add(sect[(bi, a)] + sect[(bj, b)] <= 1)
```

Then add the call in `build_cp_model` after `_add_reach_tip(...)`:
```python
    _add_no_intersection(model, menu, sect)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/generative/test_model.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/model.py tests/generative/test_model.py
git commit -m "feat: CP-SAT no-intersection from conflict table"
```

---

## Task 12: CP-SAT model — stress coverage proxy

**Files:**
- Modify: `src/wing_design/generative/model.py`
- Modify: `tests/generative/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_model.py`:
```python
def test_coverage_forces_adequate_section():
    # Target needs area >= 2e-3 and only beam 0 can cover it. The model must
    # select beam 0 at a bucket whose area >= 2e-3 (bucket 1 in the catalog).
    m = menu(
        [beam(0, covers=(5,)), beam(1)],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        coverage=[target(5, required_min_area_m2=2.0e-3, candidate_beams=[0])],
    )
    params = _params(n_beams_min=0, n_beams_max=2)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.value(sect[(0, 1)]) == 1  # adequate bucket chosen
    assert solver.value(select[0]) == 1


def test_coverage_infeasible_when_no_bucket_is_big_enough():
    # Target needs 5e-3 but the catalog tops out at 2e-3 -> infeasible.
    m = menu(
        [beam(0, covers=(5,))],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        coverage=[target(5, required_min_area_m2=5.0e-3, candidate_beams=[0])],
    )
    params = _params(n_beams_min=0, n_beams_max=1)
    model, select, sect = build_cp_model(m, params)
    solver, status = _solve(model)
    assert status == cp_model.INFEASIBLE
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_model.py::test_coverage_forces_adequate_section -v
```
Expected: FAIL — without coverage, beam 0 need not be selected, so `sect[(0,1)]` is 0.

- [ ] **Step 3: Add the coverage helper and call it**

In `src/wing_design/generative/model.py`, add after `_add_no_intersection`:
```python
def _add_coverage(model, menu, sect):
    """Each high-stress target must be served by a selected beam whose assigned
    section is large enough. Because sect implies select (one-hot tie), coverage
    also forces selection of a covering beam.
    """
    for tgt in menu.coverage_targets:
        adequate = [
            sect[(bid, cs.bucket)]
            for bid in tgt.candidate_beams
            for cs in menu.cross_sections
            if cs.area_m2 >= tgt.required_min_area_m2
        ]
        # If no (beam, bucket) can satisfy the target, this is an empty sum == 0
        # >= 1, i.e. genuinely infeasible — which is the correct verdict.
        model.add(sum(adequate) >= 1)
```

Then add the call in `build_cp_model` after `_add_no_intersection(...)`:
```python
    _add_coverage(model, menu, sect)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/generative/test_model.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/model.py tests/generative/test_model.py
git commit -m "feat: CP-SAT stress-coverage proxy constraint"
```

---

## Task 13: Mass objective + `solve_designs` (single optimal)

**Files:**
- Modify: `src/wing_design/generative/model.py`
- Modify: `tests/generative/test_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_model.py`:
```python
import math

from wing_design.generative.model import solve_designs
from wing_design.generative.menu import WingCandidate


def test_solve_picks_minimum_mass_design():
    # Both beams can cover the target at bucket 0 (area 1e-3). Beam 0 is short
    # (length 1), beam 1 is long (length 10). Minimum mass picks beam 0.
    m = menu(
        [beam(0, length=1.0, covers=(5,)), beam(1, length=10.0, covers=(5,))],
        cross_sections=cs_catalog([1.0e-3, 2.0e-3]),
        coverage=[target(5, required_min_area_m2=1.0e-3, candidate_beams=[0, 1])],
        rho=1550.0,
    )
    params = _params(n_beams_min=0, n_beams_max=2)
    designs = solve_designs(m, params, top_n=1)
    assert len(designs) == 1
    best = designs[0]
    assert isinstance(best, WingCandidate)
    assert best.beam_sections == ((0, 0),)
    # mass = length * area * rho = 1.0 * 1e-3 * 1550 = 1.55 kg
    assert math.isclose(best.mass_kg, 1.55, rel_tol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/generative/test_model.py::test_solve_picks_minimum_mass_design -v
```
Expected: FAIL — `ImportError: cannot import name 'solve_designs'`

- [ ] **Step 3: Add the objective helper, extractor, and solve_designs**

In `src/wing_design/generative/model.py`, add after `_add_coverage`:
```python
def _beam_bucket_mass_kg(menu, beam_id, bucket):
    b = menu.beam_by_id(beam_id)
    cs = next(c for c in menu.cross_sections if c.bucket == bucket)
    return b.length_m * cs.area_m2 * menu.rho_kgm3


def _set_mass_objective(model, menu, sect):
    """Minimize total mass, scaled to integer milligrams."""
    terms = []
    for (beam_id, bucket), var in sect.items():
        mass_mg = round(_beam_bucket_mass_kg(menu, beam_id, bucket) * MASS_SCALE)
        terms.append(mass_mg * var)
    model.minimize(sum(terms))


def _extract_design(solver, menu, select, sect):
    chosen = []
    mass_kg = 0.0
    for b in menu.beams:
        if solver.value(select[b.id]) != 1:
            continue
        for cs in menu.cross_sections:
            if solver.value(sect[(b.id, cs.bucket)]) == 1:
                chosen.append((b.id, cs.bucket))
                mass_kg += _beam_bucket_mass_kg(menu, b.id, cs.bucket)
    return WingCandidate(beam_sections=tuple(chosen), mass_kg=mass_kg)


def solve_designs(menu, params, top_n=1):
    """Solve the menu and return up to `top_n` distinct designs by ascending mass.

    Each round minimizes mass, records the design, then adds a no-good cut
    forbidding that exact (beam, bucket) set before re-solving — so successive
    designs are distinct and non-decreasing in mass.
    """
    model, select, sect = build_cp_model(menu, params)
    _set_mass_objective(model, menu, sect)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params.solver_max_time_s

    designs = []
    for _ in range(top_n):
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        designs.append(_extract_design(solver, menu, select, sect))
        chosen_vars = [
            var for key, var in sect.items() if solver.value(var) == 1
        ]
        # Forbid this exact selection on the next round.
        model.add(sum(chosen_vars) <= len(chosen_vars) - 1)
    return designs
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
uv run pytest tests/generative/test_model.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/model.py tests/generative/test_model.py
git commit -m "feat: mass objective and solve_designs (minimum-mass design)"
```

---

## Task 14: Top-N harvesting (distinct, non-decreasing mass)

**Files:**
- Modify: `tests/generative/test_model.py`
- Modify: `src/wing_design/generative/__init__.py`

(`solve_designs` already supports `top_n`; this task locks the behavior with a test and exports the public API.)

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_model.py`:
```python
def test_top_n_returns_distinct_non_decreasing_designs():
    # Three independent tip beams, each can cover its own target at bucket 0.
    # Different lengths -> three distinct single-beam-ish designs by mass.
    m = menu(
        [
            beam(0, length=1.0, covers=(10,)),
            beam(1, length=2.0, covers=(11,)),
            beam(2, length=3.0, covers=(12,)),
        ],
        cross_sections=cs_catalog([1.0e-3]),
        coverage=[
            target(10, 1.0e-3, [0]),
            target(11, 1.0e-3, [1]),
            target(12, 1.0e-3, [2]),
        ],
        rho=1550.0,
    )
    params = _params(n_beams_min=3, n_beams_max=3)
    designs = solve_designs(m, params, top_n=3)
    # With all three targets, the only feasible selection is all three beams;
    # so only one distinct design exists and top_n must not fabricate more.
    assert len(designs) == 1
    assert set(designs[0].beam_ids) == {0, 1, 2}


def test_top_n_enumerates_multiple_when_choices_exist():
    # One target coverable by either beam 0 or beam 1 (different masses).
    # min=1 lets the solver pick exactly one; top_n=2 should surface both,
    # lightest first.
    m = menu(
        [beam(0, length=1.0, covers=(7,)), beam(1, length=2.0, covers=(7,))],
        cross_sections=cs_catalog([1.0e-3]),
        coverage=[target(7, 1.0e-3, [0, 1])],
        rho=1550.0,
    )
    params = _params(n_beams_min=1, n_beams_max=1)
    designs = solve_designs(m, params, top_n=2)
    assert len(designs) == 2
    masses = [d.mass_kg for d in designs]
    assert masses == sorted(masses)  # non-decreasing
    assert designs[0].beam_ids == (0,)  # lightest first
    assert designs[1].beam_ids == (1,)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run:
```bash
uv run pytest tests/generative/test_model.py::test_top_n_enumerates_multiple_when_choices_exist tests/generative/test_model.py::test_top_n_returns_distinct_non_decreasing_designs -v
```
Expected: PASS (the behavior is already implemented in Task 13). If either fails, fix `solve_designs` before continuing — these lock the harvesting contract.

- [ ] **Step 3: Export the public API**

Replace the contents of `src/wing_design/generative/__init__.py` with:
```python
"""Constraint-based generative wing-truss stack (Milestone 1).

CP-SAT selects/assembles internal beams from a precomputed candidate menu;
spline geometry and FEM live outside the solver. See
docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md.
"""
from .menu import (
    BeamWrap,
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    ConflictTable,
    CoverageTarget,
    CrossSectionOption,
    CrossSectionShape,
    GateResult,
    NodeKind,
    WingCandidate,
    WingWrap,
)
from .model import build_cp_model, solve_designs

__all__ = [
    "BeamWrap",
    "CandidateBeam",
    "CandidateMenu",
    "CandidateNode",
    "ConflictTable",
    "CoverageTarget",
    "CrossSectionOption",
    "CrossSectionShape",
    "GateResult",
    "NodeKind",
    "WingCandidate",
    "WingWrap",
    "build_cp_model",
    "solve_designs",
]
```

- [ ] **Step 4: Run the full generative test suite**

Run:
```bash
uv run pytest tests/generative/ -v
```
Expected: PASS (every test in `test_menu.py` and `test_model.py`)

- [ ] **Step 5: Verify the public import surface**

Run:
```bash
uv run python -c "from wing_design.generative import solve_designs, CandidateMenu, WingCandidate; print('ok')"
```
Expected: prints `ok`

- [ ] **Step 6: Commit**

```bash
git add tests/generative/test_model.py src/wing_design/generative/__init__.py
git commit -m "feat: lock top-N harvesting contract and export generative API"
```

---

## Self-Review

**Spec coverage (against §4 and §6 of the design spec):**
- §4.1 landmark accessors → Task 3 ✓ | manufacturability/tractability params → Task 2 ✓
- §4.2 `CandidateNode`/`CandidateBeam`/`CrossSectionOption`/`ConflictTable`/`CoverageTarget`/`CandidateMenu` → Tasks 4–5 ✓
- §4.3 `WingCandidate`/`GateResult` → Task 5 ✓
- §4.4 `BeamWrap`/`WingWrap` stubs → Task 5 ✓
- §4.5 symmetry as menu property + CP-SAT tie → Tasks 5–6 (menu), Task 8 (tie) ✓
- §6.1 variables + one-hot tie → Task 7 ✓
- §6.2 count (1), symmetry (2), topology (3), reach-tip (4), no-intersection (5), coverage (6) → Tasks 7–12 ✓
- §6.3 mass objective in integer mg → Task 13 ✓
- §6.4 top-N harvesting → Tasks 13–14 ✓
- **Out of scope here (deferred to 1B/1C, correctly):** candidate generator, conflict-table construction, coverage-target construction, frame gate, build123d export, examples. The data model carries the fields these will populate.

**Placeholder scan:** No TBD/TODO/"handle edge cases"/uncoded steps — every code step shows complete code. ✓

**Type consistency:** `select` keyed by `beam_id`, `sect` keyed by `(beam_id, bucket)` throughout Tasks 7–14. `build_cp_model` returns `(model, select, sect)` consistently. `WingCandidate.beam_sections` is `((beam_id, bucket), ...)` in Task 5, produced identically in `_extract_design` (Task 13). `menu.beam_by_id` (Task 5) used by `_beam_bucket_mass_kg` (Task 13). `CandidateMenu.rho_kgm3` (Task 5) used in mass (Task 13) and supplied by the factory (Task 6). ✓

---

## Done When

- `uv run pytest tests/generative/ -v` is green (all data-model and CP-SAT tests pass).
- `from wing_design.generative import solve_designs` works.
- Given a synthetic `CandidateMenu`, `solve_designs` returns minimum-mass, chord-symmetric, coverage-satisfying, conflict-free designs, ranked ascending by mass — with **no FEM or geometry involved**.
- Plans **1B** (frame-solver gate) and **1C** (candidate generator + build123d export + examples) remain to complete Milestone 1.
