# Generative Frame-Solver Gate (Milestone 1B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight 1D linear-elastic 3D frame (beam-element) solver that judges a `WingCandidate` against stress and tip-deflection limits, returning a `GateResult`.

**Architecture:** Each beam centerline is discretized into 2-node 3D Euler–Bernoulli beam elements (6 DOF/node). UD carbon's fiber-along-spline anisotropy collapses to a standard beam using `E1_Pa` axially and `G12_Pa` in torsion. The structure is reacted by a **bearing couple** (keel-step + deck-step), not a clamped base. The solver assembles a global stiffness matrix, applies bearing-couple boundary conditions, solves `K u = f` for supplied nodal loads, recovers per-element axial+bending stress and lateral tip deflection, and returns feasibility. The numerics are validated against closed-form cantilever / axial / torsion solutions.

**Tech Stack:** Python 3.10–3.12, NumPy (dense linear algebra), `pytest`, `uv`. No new dependencies.

**Scope note:** This is plan **1B of 3** for Milestone 1. Plan 1A (CP-SAT core) is done; Plan 1C (candidate generator + build123d export + end-to-end examples, including aero→nodal load lumping and a `validate_menu` guard) follows. This plan consumes the data model from 1A (`CandidateMenu`, `WingCandidate`, `GateResult`, `NodeKind` in `src/wing_design/generative/menu.py`) and `DesignParameters` from `scenario.py`. See spec `docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md` §7.

**Load interface (deliberate boundary):** `solve_frame` accepts **explicit nodal loads** (`dict[node_index, (fx, fy, fz)]`). Mapping the aero pressure field onto frame nodes is Plan 1C's job (spec §7.3); keeping it out here makes the solver self-contained and analytically testable.

**Conventions:** snake_case; per-node DOF order is `[ux, uy, uz, θx, θy, θz]`; axes are the project's (chord +X, normal +Y, span +Z), so most beams run roughly along +Z (vertical) — the transform code handles that as the common case.

---

## File Structure

- Create: `src/wing_design/generative/gate.py` — the whole frame solver + gate (one focused module).
- Create: `tests/generative/test_gate.py` — analytic + integration tests.
- Modify: `src/wing_design/generative/__init__.py` — export `build_frame` and `solve_frame`.

---

## Task B1: Cross-section properties

**Files:**
- Create: `src/wing_design/generative/gate.py`
- Create: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/generative/test_gate.py`:
```python
import math

import numpy as np

from wing_design.generative.gate import section_properties


def test_section_properties_circle():
    area = 4.0e-3
    A, I, J = section_properties(area)
    assert math.isclose(A, area, rel_tol=1e-12)
    # circle: I = A^2 / (4 pi), J = 2 I
    assert math.isclose(I, area**2 / (4.0 * math.pi), rel_tol=1e-12)
    assert math.isclose(J, 2.0 * I, rel_tol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_gate.py::test_section_properties_circle -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.generative.gate'`

- [ ] **Step 3: Create gate.py with section_properties**

Create `src/wing_design/generative/gate.py`:
```python
"""1D linear-elastic 3D frame (beam-element) solver — the truss gate.

Judges a WingCandidate against stress and tip-deflection limits. Each beam
centerline is discretized into 2-node 3D Euler-Bernoulli beam elements
(6 DOF/node: ux, uy, uz, theta_x, theta_y, theta_z). UD carbon with fiber along
the spline is modeled with E1 axially and G12 in torsion. The structure is
reacted by a bearing couple (keel-step + deck-step), not a clamped base.

Validated against closed-form cantilever / axial / torsion solutions. See
docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md §7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .menu import GateResult, NodeKind


def section_properties(area_m2: float) -> tuple[float, float, float]:
    """Return (A, I, J) for a solid circular section of the given area.

    I is the second moment of area (equal about both transverse axes for a
    circle); J = 2 I is the polar moment (torsion constant for a circle).
    """
    A = area_m2
    radius = math.sqrt(A / math.pi)
    I = math.pi * radius**4 / 4.0
    J = 2.0 * I
    return A, I, J
```

Also add `import math` at the top of the file (with the other imports):
```python
import math
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/generative/test_gate.py::test_section_properties_circle -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: gate section_properties for circular beams"
```

---

## Task B2: Local beam-element stiffness matrix

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.gate import local_beam_stiffness


def test_local_beam_stiffness_symmetric_and_axial():
    E, G, A, I, J, L = 200e9, 80e9, 1e-3, 1e-6, 2e-6, 2.0
    k = local_beam_stiffness(E, G, A, I, J, L)
    assert k.shape == (12, 12)
    # symmetric
    assert np.allclose(k, k.T)
    # axial sub-terms: k[0,0] = EA/L, k[0,6] = -EA/L
    assert math.isclose(k[0, 0], E * A / L, rel_tol=1e-12)
    assert math.isclose(k[0, 6], -E * A / L, rel_tol=1e-12)
    # torsion: k[3,3] = GJ/L
    assert math.isclose(k[3, 3], G * J / L, rel_tol=1e-12)
    # bending: k[1,1] = 12 EI / L^3
    assert math.isclose(k[1, 1], 12.0 * E * I / L**3, rel_tol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_gate.py::test_local_beam_stiffness_symmetric_and_axial -v`
Expected: FAIL — `ImportError: cannot import name 'local_beam_stiffness'`

- [ ] **Step 3: Implement local_beam_stiffness**

Append to `src/wing_design/generative/gate.py`:
```python
def local_beam_stiffness(E, G, A, I, J, L):
    """12x12 local stiffness for a 2-node 3D Euler-Bernoulli beam element.

    DOF order per node: [u (axial, local x), v (local y), w (local z),
    theta_x (torsion), theta_y, theta_z]. Circular section, so the two
    transverse second moments are equal (passed as a single I).
    """
    k = np.zeros((12, 12))
    ea = E * A / L
    gj = G * J / L
    a = 12.0 * E * I / L**3
    b = 6.0 * E * I / L**2
    c = 4.0 * E * I / L
    d = 2.0 * E * I / L

    # Axial (u_i, u_j)
    k[0, 0] = ea
    k[0, 6] = -ea
    k[6, 0] = -ea
    k[6, 6] = ea

    # Torsion (theta_x_i, theta_x_j)
    k[3, 3] = gj
    k[3, 9] = -gj
    k[9, 3] = -gj
    k[9, 9] = gj

    # Bending in local x-y plane: v (1,7) and theta_z (5,11)
    k[1, 1] = a
    k[1, 5] = b
    k[1, 7] = -a
    k[1, 11] = b
    k[5, 1] = b
    k[5, 5] = c
    k[5, 7] = -b
    k[5, 11] = d
    k[7, 1] = -a
    k[7, 5] = -b
    k[7, 7] = a
    k[7, 11] = -b
    k[11, 1] = b
    k[11, 5] = d
    k[11, 7] = -b
    k[11, 11] = c

    # Bending in local x-z plane: w (2,8) and theta_y (4,10)
    k[2, 2] = a
    k[2, 4] = -b
    k[2, 8] = -a
    k[2, 10] = -b
    k[4, 2] = -b
    k[4, 4] = c
    k[4, 8] = b
    k[4, 10] = d
    k[8, 2] = -a
    k[8, 4] = b
    k[8, 8] = a
    k[8, 10] = b
    k[10, 2] = -b
    k[10, 4] = d
    k[10, 8] = b
    k[10, 10] = c

    return k
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/generative/test_gate.py::test_local_beam_stiffness_symmetric_and_axial -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: local 3D Euler-Bernoulli beam stiffness matrix"
```

---

## Task B3: Local→global transformation

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.gate import beam_transform


def test_beam_transform_length_and_orthonormal():
    T, L = beam_transform((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    assert math.isclose(L, 3.0, rel_tol=1e-12)
    assert T.shape == (12, 12)
    # The 3x3 direction-cosine block must be orthonormal.
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    # First local axis points along the element (global +X here).
    assert np.allclose(R[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_beam_transform_handles_vertical_beam():
    # Beam along global +Z (the common case for this wing). Must still produce
    # an orthonormal frame (no degenerate cross product).
    T, L = beam_transform((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
    assert math.isclose(L, 5.0, rel_tol=1e-12)
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(R[0], [0.0, 0.0, 1.0], atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_gate.py::test_beam_transform_handles_vertical_beam -v`
Expected: FAIL — `ImportError: cannot import name 'beam_transform'`

- [ ] **Step 3: Implement beam_transform**

Append to `src/wing_design/generative/gate.py`:
```python
def beam_transform(p0, p1):
    """Return (T, L): the 12x12 local->global transform and element length.

    The local x-axis runs from p0 to p1. A reference vector (global +Z, or
    global +Y when the element is itself nearly parallel to +Z) fixes the local
    y/z axes. Because the wing spans along +Z, most beams are near-vertical, so
    the +Z-parallel branch is the common path, not an edge case.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    L = float(np.linalg.norm(delta))
    x_local = delta / L

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(x_local, ref))) > 0.9999:
        ref = np.array([0.0, 1.0, 0.0])

    y_local = np.cross(ref, x_local)
    y_local /= np.linalg.norm(y_local)
    z_local = np.cross(x_local, y_local)

    R = np.vstack([x_local, y_local, z_local])  # rows = local axes in global coords
    T = np.zeros((12, 12))
    for blk in range(4):
        T[3 * blk:3 * blk + 3, 3 * blk:3 * blk + 3] = R
    return T, L
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/generative/test_gate.py -k beam_transform -v`
Expected: PASS (both transform tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: beam local-to-global transform (vertical-beam safe)"
```

---

## Task B4: Element global stiffness

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.gate import element_global_stiffness


def test_element_global_stiffness_symmetric_and_axisaligned():
    E, G, A, I, J = 200e9, 80e9, 1e-3, 1e-6, 2e-6
    # For an X-aligned element the local frame equals global, so global == local.
    ke = element_global_stiffness((0, 0, 0), (2.0, 0, 0), E, G, A, I, J)
    assert ke.shape == (12, 12)
    assert np.allclose(ke, ke.T, atol=1e-3)
    k_local = local_beam_stiffness(E, G, A, I, J, 2.0)
    assert np.allclose(ke, k_local, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_gate.py::test_element_global_stiffness_symmetric_and_axisaligned -v`
Expected: FAIL — `ImportError: cannot import name 'element_global_stiffness'`

- [ ] **Step 3: Implement element_global_stiffness**

Append to `src/wing_design/generative/gate.py`:
```python
def element_global_stiffness(p0, p1, E, G, A, I, J):
    """Global-frame 12x12 stiffness for the beam from p0 to p1: T^T k_local T."""
    T, L = beam_transform(p0, p1)
    k_local = local_beam_stiffness(E, G, A, I, J, L)
    return T.T @ k_local @ T
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/generative/test_gate.py::test_element_global_stiffness_symmetric_and_axisaligned -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: element global stiffness via transform"
```

---

## Task B5: Reduced solve + closed-form validation

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing tests (the analytic validation suite)**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.gate import solve_displacements

# Shared beam properties for the analytic single-element checks.
_E, _G, _A, _I, _J, _L = 200e9, 80e9, 1e-3, 1e-6, 2e-6, 2.0
_CLAMP = [0, 1, 2, 3, 4, 5]  # all 6 DOF at node i


def test_axial_extension_matches_PL_over_EA():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 0] = P  # axial (Fx) at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 0], P * _L / (_E * _A), rtol=1e-6)


def test_horizontal_cantilever_matches_PL3_over_3EI():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 1] = P  # transverse (Fy) at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 1], P * _L**3 / (3.0 * _E * _I), rtol=1e-6)


def test_vertical_cantilever_matches_PL3_over_3EI():
    # Element along +Z exercises the vertical-beam transform branch.
    ke = element_global_stiffness((0, 0, 0), (0, 0, _L), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 0] = P  # transverse (Fx) at the top node
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 0], P * _L**3 / (3.0 * _E * _I), rtol=1e-6)


def test_torsion_matches_TL_over_GJ():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    Tq = 1000.0
    load = np.zeros(12)
    load[6 + 3] = Tq  # torque (Mx) about the beam axis at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 3], Tq * _L / (_G * _J), rtol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/generative/test_gate.py -k "extension or cantilever or torsion" -v`
Expected: FAIL — `ImportError: cannot import name 'solve_displacements'`

- [ ] **Step 3: Implement solve_displacements**

Append to `src/wing_design/generative/gate.py`:
```python
def solve_displacements(K, load_vec, fixed_dofs):
    """Solve K u = f with `fixed_dofs` held at zero; return the full vector u.

    Reduces to the free DOFs, solves the dense system, and scatters back.
    """
    n = K.shape[0]
    fixed = set(fixed_dofs)
    free = [d for d in range(n) if d not in fixed]
    Kff = K[np.ix_(free, free)]
    Ff = np.asarray(load_vec, dtype=float)[free]
    uf = np.linalg.solve(Kff, Ff)
    u = np.zeros(n)
    u[free] = uf
    return u
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_gate.py -k "extension or cantilever or torsion" -v`
Expected: PASS (4 analytic tests). Then run the whole gate file: `uv run pytest tests/generative/test_gate.py -v` (all green).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: reduced linear solve, validated vs closed-form beam solutions"
```

---

## Task B6: FrameModel + build_frame

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.gate import FrameModel, build_frame
from wing_design.generative.menu import (
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    ConflictTable,
    CrossSectionOption,
    CrossSectionShape,
    NodeKind,
    WingCandidate,
)


def _single_beam_menu():
    """A keel->tip beam through the deck-step, with the three landmark nodes."""
    nodes = (
        CandidateNode(id=0, xyz=(0.0, 0.0, -0.95), kind=NodeKind.KEEL_STEP, z_layer=0),
        CandidateNode(id=1, xyz=(0.0, 0.0, -0.20), kind=NodeKind.DECK_STEP, z_layer=1),
        CandidateNode(id=2, xyz=(0.0, 0.0, 5.0), kind=NodeKind.TIP, z_layer=9),
    )
    beam = CandidateBeam(
        id=0,
        control_points=((0.0, 0.0, -0.95), (0.0, 0.0, -0.20), (0.0, 0.0, 5.0)),
        start_kind=NodeKind.KEEL_STEP,
        end_kind=NodeKind.TIP,
        start_node=0,
        end_node=2,
        length_m=5.95,
        min_radius_m=100.0,
        on_chord_plane=True,
        mirror_id=None,
        host_id=None,
        covers=(),
    )
    cs = (CrossSectionOption(bucket=0, shape=CrossSectionShape.CIRCLE, area_m2=4.0e-3),)
    menu = CandidateMenu(
        nodes=nodes, beams=(beam,), cross_sections=cs,
        conflicts=ConflictTable(forbidden=()), coverage_targets=(), rho_kgm3=1550.0,
    )
    return menu


def test_build_frame_nodes_elements_and_kinds():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    assert isinstance(frame, FrameModel)
    # 3 control points -> 3 nodes, 2 elements.
    assert frame.coords.shape == (3, 3)
    assert len(frame.elements) == 2
    # Node kinds inherited from the menu by coordinate match.
    assert frame.node_kinds[0] == NodeKind.KEEL_STEP
    assert frame.node_kinds[1] == NodeKind.DECK_STEP
    assert frame.node_kinds[2] == NodeKind.TIP
    # Element area carried through from the chosen bucket.
    for (_i, _j, area) in frame.elements:
        assert math.isclose(area, 4.0e-3, rel_tol=1e-12)
    assert math.isclose(frame.mass_kg, 12.5, rel_tol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_gate.py::test_build_frame_nodes_elements_and_kinds -v`
Expected: FAIL — `ImportError: cannot import name 'FrameModel'`

- [ ] **Step 3: Implement FrameModel + build_frame**

Append to `src/wing_design/generative/gate.py`:
```python
@dataclass
class FrameModel:
    """A discretized frame ready to solve.

    coords: (N, 3) node coordinates. elements: list of (node_i, node_j, area_m2).
    node_kinds: per-node NodeKind (or None for interior nodes), inherited from
    the menu by coordinate. mass_kg: carried from the WingCandidate for the
    GateResult.
    """
    coords: np.ndarray
    elements: list
    node_kinds: list
    mass_kg: float


def _node_key(xyz, tol=1e-6):
    """Quantized coordinate key so coincident points merge into one frame node."""
    return (round(xyz[0] / tol), round(xyz[1] / tol), round(xyz[2] / tol))


def build_frame(candidate, menu):
    """Discretize a WingCandidate into a FrameModel.

    Each selected beam's consecutive control points become beam elements;
    coincident coordinates (shared endpoints / junctions) merge into one node so
    beams that touch transfer load. Node kinds are inherited from the menu's
    landmark nodes by coordinate match.
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

    elements = []
    for beam_id, bucket in candidate.beam_sections:
        beam = menu.beam_by_id(beam_id)
        cs = next(c for c in menu.cross_sections if c.bucket == bucket)
        pts = beam.control_points
        for p_a, p_b in zip(pts[:-1], pts[1:]):
            i = node_index(p_a)
            j = node_index(p_b)
            elements.append((i, j, cs.area_m2))

    return FrameModel(
        coords=np.array(coords, dtype=float),
        elements=elements,
        node_kinds=node_kinds,
        mass_kg=candidate.mass_kg,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/generative/test_gate.py::test_build_frame_nodes_elements_and_kinds -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: FrameModel and build_frame from a WingCandidate"
```

---

## Task B7: Global assembly + bearing-couple BCs

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.gate import (
    assemble_global_K,
    bearing_couple_fixed_dofs,
    tip_node_indices,
)


def test_assemble_global_K_shape_and_symmetry():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    K = assemble_global_K(frame, E=135e9, G=4.5e9)
    n = 6 * frame.coords.shape[0]
    assert K.shape == (n, n)
    assert np.allclose(K, K.T, atol=1e-3)


def test_bearing_couple_fixed_dofs():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    fixed = bearing_couple_fixed_dofs(frame)
    # keel node 0: ux,uy,uz,theta_z -> dofs 0,1,2,5 ; deck node 1: ux,uy -> 6,7
    assert set(fixed) == {0, 1, 2, 5, 6, 7}


def test_tip_node_indices():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    assert tip_node_indices(frame) == [2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_gate.py::test_bearing_couple_fixed_dofs -v`
Expected: FAIL — `ImportError: cannot import name 'assemble_global_K'`

- [ ] **Step 3: Implement assembly + BCs + tip detection**

Append to `src/wing_design/generative/gate.py`:
```python
def assemble_global_K(frame, E, G):
    """Assemble the global stiffness matrix (6*N x 6*N) from the frame elements."""
    n = frame.coords.shape[0]
    K = np.zeros((6 * n, 6 * n))
    for (i, j, area) in frame.elements:
        A, I, J = section_properties(area)
        ke = element_global_stiffness(frame.coords[i], frame.coords[j], E, G, A, I, J)
        dofs = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        for r in range(12):
            for c in range(12):
                K[dofs[r], dofs[c]] += ke[r, c]
    return K


def bearing_couple_fixed_dofs(frame):
    """Boundary conditions for the free-rotating spar's two bearings.

    Keel-step bearing: fix the three translations and the spin about the spar
    axis (theta_z) — the latter represents the control system holding the
    commanded angle of attack, and removes the otherwise-free spar-rotation
    mechanism. Deck-step bearing: fix only the radial (x, y) translations; the
    overturning moment is carried as a force couple between the two bearings.
    """
    fixed = []
    for idx, kind in enumerate(frame.node_kinds):
        base = 6 * idx
        if kind == NodeKind.KEEL_STEP:
            fixed += [base + 0, base + 1, base + 2, base + 5]
        elif kind == NodeKind.DECK_STEP:
            fixed += [base + 0, base + 1]
    return sorted(set(fixed))


def tip_node_indices(frame):
    """Indices of frame nodes that are wing-tip landmarks."""
    return [i for i, kind in enumerate(frame.node_kinds) if kind == NodeKind.TIP]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_gate.py -k "assemble or bearing or tip_node" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: global assembly, bearing-couple BCs, tip-node detection"
```

---

## Task B8: Stress recovery + tip deflection

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.gate import recover_max_stress_ratio, tip_deflection


def test_recover_axial_stress_ratio():
    # Single X-aligned element, clamped at i, axial load P at j.
    # Axial stress = P/A; ratio = (P/A) / sigma_allow.
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 0] = P
    u = solve_displacements(ke, load, _CLAMP)
    frame = FrameModel(
        coords=np.array([[0, 0, 0], [_L, 0, 0]], dtype=float),
        elements=[(0, 1, _A)],
        node_kinds=[None, None],
        mass_kg=0.0,
    )
    sigma_allow = 1.0e9
    ratio = recover_max_stress_ratio(frame, u, _E, _G, sigma_allow)
    assert np.isclose(ratio, (P / _A) / sigma_allow, rtol=1e-4)


def test_tip_deflection_lateral():
    coords = np.array([[0, 0, -0.95], [0, 0, 5.0]], dtype=float)
    frame = FrameModel(coords=coords, elements=[(0, 1, _A)],
                       node_kinds=[NodeKind.KEEL_STEP, NodeKind.TIP], mass_kg=0.0)
    u = np.zeros(12)
    u[6 + 0] = 0.03  # ux at the tip
    u[6 + 1] = 0.04  # uy at the tip
    # lateral magnitude = sqrt(0.03^2 + 0.04^2) = 0.05
    assert np.isclose(tip_deflection(frame, u), 0.05, rtol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_gate.py -k "recover or tip_deflection_lateral" -v`
Expected: FAIL — `ImportError: cannot import name 'recover_max_stress_ratio'`

- [ ] **Step 3: Implement recovery functions**

Append to `src/wing_design/generative/gate.py`:
```python
def recover_max_stress_ratio(frame, disp, E, G, sigma_allow):
    """Max over elements of (|axial| + bending) stress / allowable.

    For each element, recover local end forces f = k_local (T u). Axial stress
    is N/A; bending stress is the resultant end moment times the section radius
    over I. The two are summed (conservative) and compared to the allowable.
    """
    max_ratio = 0.0
    for (i, j, area) in frame.elements:
        A, I, J = section_properties(area)
        radius = math.sqrt(area / math.pi)
        T, L = beam_transform(frame.coords[i], frame.coords[j])
        dofs = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        d_local = T @ disp[dofs]
        f_local = local_beam_stiffness(E, G, A, I, J, L) @ d_local
        axial = abs(f_local[6]) / A
        m_i = math.hypot(f_local[4], f_local[5])
        m_j = math.hypot(f_local[10], f_local[11])
        bending = max(m_i, m_j) * radius / I
        sigma = axial + bending
        max_ratio = max(max_ratio, sigma / sigma_allow)
    return max_ratio


def tip_deflection(frame, disp):
    """Largest lateral (perpendicular-to-span, i.e. x-y) displacement at a tip node."""
    best = 0.0
    for i in tip_node_indices(frame):
        lateral = math.hypot(disp[6 * i + 0], disp[6 * i + 1])
        best = max(best, lateral)
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_gate.py -k "recover or tip_deflection_lateral" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py
git commit -m "feat: stress-ratio and tip-deflection recovery"
```

---

## Task B9: solve_frame end-to-end + API export

**Files:**
- Modify: `src/wing_design/generative/gate.py`
- Modify: `tests/generative/test_gate.py`
- Modify: `src/wing_design/generative/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_gate.py`:
```python
from wing_design.generative.gate import solve_frame
from wing_design.generative.menu import GateResult
from wing_design.scenario import default_scenario


def test_solve_frame_feasible_for_stout_beam_small_load():
    menu = _single_beam_menu()  # 4e-3 m^2 circular section, ~6 m beam
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    params = default_scenario()
    tip = tip_node_indices(frame)[0]
    loads = {tip: (200.0, 0.0, 0.0)}  # small 200 N lateral tip load
    result = solve_frame(frame, params, loads, governing_case="nominal")
    assert isinstance(result, GateResult)
    assert result.governing_case == "nominal"
    assert math.isclose(result.mass_kg, 12.5, rel_tol=1e-12)
    assert result.tip_deflection_m > 0.0
    assert result.max_stress_ratio > 0.0
    assert result.feasible is True


def test_solve_frame_infeasible_under_huge_load():
    menu = _single_beam_menu()
    candidate = WingCandidate(beam_sections=((0, 0),), mass_kg=12.5)
    frame = build_frame(candidate, menu)
    params = default_scenario()
    tip = tip_node_indices(frame)[0]
    loads = {tip: (5.0e5, 0.0, 0.0)}  # 500 kN: overstressed and/or over-deflected
    result = solve_frame(frame, params, loads)
    assert result.feasible is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_gate.py -k solve_frame -v`
Expected: FAIL — `ImportError: cannot import name 'solve_frame'`

- [ ] **Step 3: Implement solve_frame**

Append to `src/wing_design/generative/gate.py`:
```python
def solve_frame(frame, params, nodal_loads, governing_case="nominal"):
    """Judge a frame under the given nodal loads against stress + deflection limits.

    `params` is a DesignParameters: E from the UD ply's E1, G from G12, the
    tensile allowable from `sigma_allow_Pa`, and the deflection cap from
    `generative.tip_deflection_limit_m`. `nodal_loads` maps a frame node index
    to a global (fx, fy, fz). Returns a GateResult.
    """
    E = params.material.E1_Pa
    G = params.material.G12_Pa
    sigma_allow = params.sigma_allow_Pa
    limit = params.generative.tip_deflection_limit_m

    n = frame.coords.shape[0]
    K = assemble_global_K(frame, E, G)
    load_vec = np.zeros(6 * n)
    for node_idx, (fx, fy, fz) in nodal_loads.items():
        load_vec[6 * node_idx + 0] += fx
        load_vec[6 * node_idx + 1] += fy
        load_vec[6 * node_idx + 2] += fz

    fixed = bearing_couple_fixed_dofs(frame)
    disp = solve_displacements(K, load_vec, fixed)

    max_ratio = recover_max_stress_ratio(frame, disp, E, G, sigma_allow)
    tip_def = tip_deflection(frame, disp)
    feasible = (max_ratio <= 1.0) and (tip_def <= limit)

    return GateResult(
        feasible=feasible,
        max_stress_ratio=max_ratio,
        tip_deflection_m=tip_def,
        governing_case=governing_case,
        mass_kg=frame.mass_kg,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_gate.py -k solve_frame -v`
Expected: PASS (both). Then the whole suite: `uv run pytest tests/generative/ -v` (all green).

- [ ] **Step 5: Export the public API**

Edit `src/wing_design/generative/__init__.py`. Add to the imports (after the `from .model import ...` line):
```python
from .gate import build_frame, solve_frame
```
And add `"build_frame"` and `"solve_frame"` to the `__all__` list (keep it alphabetical within the trailing function group, i.e. after `"build_cp_model"`):
```python
    "build_cp_model",
    "build_frame",
    "solve_designs",
    "solve_frame",
```

- [ ] **Step 6: Verify the public import surface**

Run:
```bash
uv run python -c "from wing_design.generative import build_frame, solve_frame; print('ok')"
```
Expected: prints `ok`

- [ ] **Step 7: Commit**

```bash
git add src/wing_design/generative/gate.py tests/generative/test_gate.py src/wing_design/generative/__init__.py
git commit -m "feat: solve_frame gate end-to-end and export API"
```

---

## Self-Review

**Spec coverage (spec §7):**
- §7.1 model (nodes = endpoints+junctions via coordinate merge; elements = discretized splines; 6 DOF/node; A/I/J from bucket; E1/G from material) → Tasks B1, B2, B6 ✓
- §7.2 bearing-couple BCs (keel translations + θz; deck radial; force couple) → Task B7 ✓ (θz at keel documented as the control-torque reaction that removes the spar-rotation mechanism — a precise, stated refinement of the spec)
- §7.3 loads (lumped nodal forces) → `solve_frame` accepts nodal loads; the aero→node lumping itself is explicitly Plan 1C ✓ (boundary stated up front)
- §7.4 verdict (solve K u = f; axial+bending stress ratio; lateral tip deflection; feasible = ratio ≤ 1 and deflection ≤ limit) → Tasks B5, B8, B9 ✓
- §7.5 staging: this plan is gate **v0** (linear, single load case, bare frame). Buckling, torsion check, full envelope, wraps, and volumetric finalist check are later milestones — correctly out of scope ✓
- The "bare frame underestimates torsion" caveat is inherent to v0 and recorded in the spec ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/uncoded steps — every code step shows complete code. ✓

**Type consistency:** `section_properties` returns `(A, I, J)` used identically in `assemble_global_K` and `recover_max_stress_ratio`. `local_beam_stiffness(E, G, A, I, J, L)` signature consistent across B2/B4/B5/B8. `beam_transform` returns `(T, L)` consistent in B4/B8. `element_global_stiffness(p0, p1, E, G, A, I, J)` consistent. `FrameModel(coords, elements, node_kinds, mass_kg)` constructed identically in `build_frame` and the B8 tests. `solve_frame(frame, params, nodal_loads, governing_case)` matches the B9 tests. `GateResult` fields match the 1A dataclass. DOF convention `[ux,uy,uz,θx,θy,θz]` used consistently. ✓

---

## Done When

- `uv run pytest tests/generative/ -v` is green (1A's 17 tests + the new gate tests).
- The element solver reproduces closed-form axial (`PL/EA`), bending (`PL³/3EI`, both horizontal and vertical beams), and torsion (`TL/GJ`) to 1e-6.
- `from wing_design.generative import build_frame, solve_frame` works.
- `solve_frame(build_frame(candidate, menu), params, nodal_loads)` returns a `GateResult` with a sensible feasibility verdict, stress ratio, and lateral tip deflection.
- Plan **1C** (candidate generator + aero→node load lumping + build123d export + end-to-end examples + `validate_menu` guard) remains to complete Milestone 1.
