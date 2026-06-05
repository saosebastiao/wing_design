# Phase B — Beam-Frame FEA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the unsized Phase-A form-beam structure under the aero load envelope by solving a 3D Euler–Bernoulli space-frame (16 longitudinal beams + transverse ring connectors, clamped at the keel-step) and reporting per-load-case tip deflection, twist, and member stress.

**Architecture:** A fresh, reusable 3D frame-element FEA solver lives in `structural/frame.py` (companion to the existing tet `fea.py` and shell `shell.py` solvers — neither supports beam elements). A new `beams/fea_model.py` turns the Phase-A `form_beam_grid` into frame nodes + elements, projects `LiftingLine` panel forces onto the nearest beam node (aero→geom frame), solves, and summarizes. Circular cross-sections (`Iy == Iz`) make element axial-roll orientation irrelevant, so the rotation only needs local-x along the member.

**Tech Stack:** numpy, scipy.sparse (`coo_matrix` assembly + `spsolve`), existing `aero.loads` LiftingLine envelope, `materials.unidir` isotropic-equivalent modulus, `beams.splines.form_beam_grid` (Phase A).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/structural/frame.py` (create) | Generic 3D Euler–Bernoulli frame FEA: `BeamSection`, `local_beam_stiffness`, `solve_frame`, `FrameResult`. No knowledge of wings — pure structural primitive. |
| `src/wing_design/structural/__init__.py` (modify) | Export `BeamSection`, `FrameResult`, `solve_frame`. |
| `src/wing_design/beams/fea_model.py` (create) | `BeamFrame` (nodes/elements/fixed/section/material), `build_beam_frame`, `project_panels_to_beam_nodes`, `solve_beam_frame`, `summarize_frame`, `FrameMetrics`. |
| `src/wing_design/beams/__init__.py` (modify) | Export the new `fea_model` public names. |
| `examples/21_beam_fea.py` (create) | Build frame → LL envelope → per-case solve → metrics table. |
| `tests/structural/test_frame.py` (create) | Analytical-cantilever validation of the solver (bending, axial, torsion). |
| `tests/beams/test_fea_model.py` (create) | Frame topology, load-conservation, solve-runs smoke. |
| `docs/plan.md` (modify) | Mark Phase B done; note the frame solver. |

---

## Task 1: Frame-element FEA solver

**Files:**
- Create: `src/wing_design/structural/frame.py`
- Test: `tests/structural/test_frame.py`

- [ ] **Step 1: Write the failing tests** (analytical cantilever — Euler–Bernoulli end-loaded beam is exact at the nodes, so 1 element matches the closed form)

```python
# tests/structural/test_frame.py
import numpy as np

from wing_design.structural.frame import BeamSection, solve_frame

E = 70e9
NU = 0.3
G = E / (2 * (1 + NU))


def _two_node(p0, p1, sec):
    nodes = np.array([p0, p1], dtype=float)
    elements = np.array([[0, 1]], dtype=int)
    return nodes, elements, [sec]


def test_cantilever_tip_bending():
    sec = BeamSection.circular(0.02)
    nodes, elements, sections = _two_node([0, 0, 0], [1, 0, 0], sec)
    loads = np.zeros((2, 6))
    P = 100.0
    loads[1, 2] = P  # transverse Fz at the tip
    res = solve_frame(nodes, elements, sections, E=E, G=G,
                      fixed_nodes=np.array([0]), loads=loads)
    expected = P * 1.0**3 / (3 * E * sec.Iy)  # PL^3 / 3EI
    assert abs(res.displacements[1, 2] - expected) / expected < 1e-6


def test_axial_bar():
    sec = BeamSection.circular(0.02)
    nodes, elements, sections = _two_node([0, 0, 0], [2, 0, 0], sec)
    loads = np.zeros((2, 6))
    P = 1000.0
    loads[1, 0] = P
    res = solve_frame(nodes, elements, sections, E=E, G=G,
                      fixed_nodes=np.array([0]), loads=loads)
    expected = P * 2.0 / (E * sec.A)  # PL / EA
    assert abs(res.displacements[1, 0] - expected) / expected < 1e-9
    assert abs(res.axial_force[0] - P) / P < 1e-9  # tension positive


def test_torsion_bar():
    sec = BeamSection.circular(0.02)
    nodes, elements, sections = _two_node([0, 0, 0], [1, 0, 0], sec)
    loads = np.zeros((2, 6))
    T = 50.0
    loads[1, 3] = T  # Mx (torque) at the tip
    res = solve_frame(nodes, elements, sections, E=E, G=G,
                      fixed_nodes=np.array([0]), loads=loads)
    expected = T * 1.0 / (G * sec.J)  # TL / GJ
    assert abs(res.displacements[1, 3] - expected) / expected < 1e-9
    assert abs(res.torsion[0] - T) / T < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/structural/test_frame.py -v`
Expected: FAIL — `ModuleNotFoundError: wing_design.structural.frame`.

- [ ] **Step 3: Write the solver**

```python
# src/wing_design/structural/frame.py
"""3D Euler–Bernoulli frame FEA: beam elements with 6 DOF per node.

Companion to the volumetric (`fea.py`) and shell (`shell.py`) solvers; this one
models a space-frame of 1-D beam members — the form-beam structure of Phase B,
cantilevered at the keel-step. Each node carries 3 translations + 3 rotations;
each element is a prismatic Euler–Bernoulli beam with axial, two-plane bending,
and St-Venant torsion stiffness.

Per-node DOF order: [ux, uy, uz, rx, ry, rz] in the global frame. Loads are
(n_nodes, 6) arrays [Fx, Fy, Fz, Mx, My, Mz]. For circular sections Iy == Iz so
the element's roll about its own axis does not affect stiffness — the local
frame only needs local-x along the member.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class BeamSection:
    """Prismatic cross-section properties of a frame member."""

    A: float        # area [m^2]
    Iy: float       # second moment about local y [m^4]
    Iz: float       # second moment about local z [m^4]
    J: float        # torsion constant [m^4]
    r: float        # outer radius, for stress recovery [m]

    @classmethod
    def circular(cls, radius: float) -> "BeamSection":
        area = np.pi * radius**2
        second = 0.25 * np.pi * radius**4
        polar = 0.5 * np.pi * radius**4
        return cls(A=area, Iy=second, Iz=second, J=polar, r=radius)


@dataclass(frozen=True)
class FrameResult:
    displacements: np.ndarray       # (n_nodes, 6)
    axial_force: np.ndarray         # (n_elem,) tension positive [N]
    bending_moment: np.ndarray      # (n_elem,) max |M| over the two ends [N·m]
    torsion: np.ndarray             # (n_elem,) [N·m]

    @property
    def max_translation_m(self) -> float:
        return float(np.linalg.norm(self.displacements[:, :3], axis=1).max())


def local_beam_stiffness(E: float, G: float, sec: BeamSection, L: float) -> np.ndarray:
    """12x12 local stiffness of a prismatic Euler–Bernoulli beam element."""
    a = E * sec.A / L
    jt = G * sec.J / L
    eiz = E * sec.Iz
    eiy = E * sec.Iy
    k = np.zeros((12, 12))
    # axial: ux at 0, 6
    k[0, 0] = a
    k[0, 6] = -a
    k[6, 6] = a
    # torsion: rx at 3, 9
    k[3, 3] = jt
    k[3, 9] = -jt
    k[9, 9] = jt
    # bending in x-y plane: uy(1), rz(5), uy2(7), rz2(11) -> Iz
    k[1, 1] = 12 * eiz / L**3
    k[1, 5] = 6 * eiz / L**2
    k[1, 7] = -12 * eiz / L**3
    k[1, 11] = 6 * eiz / L**2
    k[5, 5] = 4 * eiz / L
    k[5, 7] = -6 * eiz / L**2
    k[5, 11] = 2 * eiz / L
    k[7, 7] = 12 * eiz / L**3
    k[7, 11] = -6 * eiz / L**2
    k[11, 11] = 4 * eiz / L
    # bending in x-z plane: uz(2), ry(4), uz2(8), ry2(10) -> Iy (sign-flipped coupling)
    k[2, 2] = 12 * eiy / L**3
    k[2, 4] = -6 * eiy / L**2
    k[2, 8] = -12 * eiy / L**3
    k[2, 10] = -6 * eiy / L**2
    k[4, 4] = 4 * eiy / L
    k[4, 8] = 6 * eiy / L**2
    k[4, 10] = 2 * eiy / L
    k[8, 8] = 12 * eiy / L**3
    k[8, 10] = 6 * eiy / L**2
    k[10, 10] = 4 * eiy / L
    # we filled the upper triangle; mirror to the lower
    return k + k.T - np.diag(np.diag(k))


def _element_rotation(pi: np.ndarray, pj: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (R, L): R rows are the element's local axes in global coords."""
    d = pj - pi
    L = float(np.linalg.norm(d))
    ex = d / L
    up = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    ey = np.cross(up, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return np.vstack([ex, ey, ez]), L


def solve_frame(
    nodes: np.ndarray,
    elements: np.ndarray,
    sections: list[BeamSection],
    *,
    E: float,
    G: float,
    fixed_nodes: np.ndarray,
    loads: np.ndarray,
) -> FrameResult:
    """Solve K u = f for a clamped 3D frame; recover per-element internal forces.

    `fixed_nodes` are fully clamped (all 6 DOF). `loads` is (n_nodes, 6).
    """
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]
    ndof = 6 * n_nodes

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    transforms: list[np.ndarray] = []
    klocals: list[np.ndarray] = []

    for e in range(n_elem):
        i, j = int(elements[e, 0]), int(elements[e, 1])
        R, L = _element_rotation(nodes[i], nodes[j])
        kloc = local_beam_stiffness(E, G, sections[e], L)
        T = np.zeros((12, 12))
        for b in range(4):
            T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
        kg = T.T @ kloc @ T
        transforms.append(T)
        klocals.append(kloc)
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        for a_ in range(12):
            for b_ in range(12):
                rows.append(int(dofs[a_]))
                cols.append(int(dofs[b_]))
                vals.append(kg[a_, b_])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    f = loads.reshape(-1).astype(float)
    if len(fixed_nodes):
        fixed_dofs = np.concatenate([6 * int(fn) + np.arange(6) for fn in fixed_nodes])
    else:
        fixed_dofs = np.array([], dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)

    u = np.zeros(ndof)
    Kff = K[free][:, free].tocsc()
    u[free] = spla.spsolve(Kff, f[free])
    disp = u.reshape(n_nodes, 6)

    axial = np.zeros(n_elem)
    bending = np.zeros(n_elem)
    torsion = np.zeros(n_elem)
    for e in range(n_elem):
        i, j = int(elements[e, 0]), int(elements[e, 1])
        ue = np.r_[u[6 * i:6 * i + 6], u[6 * j:6 * j + 6]]
        floc = klocals[e] @ (transforms[e] @ ue)
        axial[e] = floc[6]  # force at node j along local x; tension positive
        bending[e] = max(np.hypot(floc[4], floc[5]), np.hypot(floc[10], floc[11]))
        torsion[e] = floc[9]

    return FrameResult(displacements=disp, axial_force=axial,
                       bending_moment=bending, torsion=torsion)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/structural/test_frame.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/structural/frame.py tests/structural/test_frame.py
git commit -m "feat(structural): 3D Euler-Bernoulli frame FEA solver"
```

---

## Task 2: Export the frame solver

**Files:**
- Modify: `src/wing_design/structural/__init__.py`

- [ ] **Step 1: Add the imports + `__all__` entries**

Add to the import block:

```python
from .frame import BeamSection, FrameResult, solve_frame
```

Add to `__all__` (keep it alphabetized with the rest):

```python
    "BeamSection",
    "FrameResult",
    "solve_frame",
```

- [ ] **Step 2: Verify the import resolves**

Run: `uv run python -c "from wing_design.structural import BeamSection, FrameResult, solve_frame; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/wing_design/structural/__init__.py
git commit -m "feat(structural): export frame FEA from package"
```

---

## Task 3: Beam-frame assembly from the Phase-A grid

**Files:**
- Create: `src/wing_design/beams/fea_model.py`
- Test: `tests/beams/test_fea_model.py`

- [ ] **Step 1: Write the failing topology test**

```python
# tests/beams/test_fea_model.py
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.fea_model import build_beam_frame


def test_frame_topology():
    spec = WingSpec()
    nb, nl = 16, 10
    frame = build_beam_frame(spec, n_beams=nb, n_levels=nl)
    assert frame.nodes.shape == (nb * nl, 3)
    # longitudinal: nb*(nl-1); rings: nl*nb
    assert frame.elements.shape[0] == nb * (nl - 1) + nl * nb
    assert frame.fixed_nodes.shape[0] == nb
    # the clamped ring is the most-negative-z (keel-step) level
    z_fixed = frame.nodes[frame.fixed_nodes, 2]
    assert np.allclose(z_fixed, frame.nodes[:, 2].min())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_fea_model.py::test_frame_topology -v`
Expected: FAIL — `ModuleNotFoundError: wing_design.beams.fea_model`.

- [ ] **Step 3: Write the module skeleton + `build_beam_frame`**

```python
# src/wing_design/beams/fea_model.py
"""Assemble + solve a Phase-B beam-frame FEA from the Phase-A form-beam grid.

The form-beam splines become longitudinal beam members; transverse ring members
connect adjacent beams at every z-level, producing a cantilevered space-frame
clamped at the keel-step. Aero panel forces are projected onto the nearest beam
node (aero→geom frame) and the frame is solved per load case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..aero.loads import PanelLoads
from ..geometry.wing import WingSpec
from ..materials.unidir import T700_EPOXY, UDPly
from ..structural.frame import BeamSection, FrameResult, solve_frame
from ..structural.projection import R_GEOM_FROM_AERO
from .splines import default_z_levels, form_beam_grid


@dataclass(frozen=True)
class BeamFrame:
    nodes: np.ndarray          # (n_beams*n_levels, 3) geom frame
    elements: np.ndarray       # (n_elem, 2) int
    n_beams: int
    n_levels: int
    fixed_nodes: np.ndarray    # (n_beams,) keel-step ring
    section: BeamSection
    E: float
    G: float


def build_beam_frame(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
    material: UDPly = T700_EPOXY,
    knockdown: float = 0.5,
    nu: float = 0.32,
) -> BeamFrame:
    z = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z, n_beams)          # (n_beams, n_levels, 3)
    nodes = grid.reshape(-1, 3)

    elems: list[tuple[int, int]] = []
    # longitudinal members: consecutive z-levels of each beam
    for b in range(n_beams):
        for k in range(n_levels - 1):
            elems.append((b * n_levels + k, b * n_levels + (k + 1)))
    # ring members: adjacent beams at each level, wrapping around
    for k in range(n_levels):
        for b in range(n_beams):
            bn = (b + 1) % n_beams
            elems.append((b * n_levels + k, bn * n_levels + k))
    elements = np.asarray(elems, dtype=int)

    keel_k = n_levels - 1  # default_z_levels descends tip -> keel-step
    fixed_nodes = np.array([b * n_levels + keel_k for b in range(n_beams)], dtype=int)

    E = material.isotropic_equivalent_modulus(knockdown=knockdown)
    G = E / (2.0 * (1.0 + nu))
    return BeamFrame(
        nodes=nodes,
        elements=elements,
        n_beams=n_beams,
        n_levels=n_levels,
        fixed_nodes=fixed_nodes,
        section=BeamSection.circular(beam_radius),
        E=E,
        G=G,
    )
```

- [ ] **Step 4: Run to verify the topology test passes**

Run: `uv run pytest tests/beams/test_fea_model.py::test_frame_topology -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/fea_model.py tests/beams/test_fea_model.py
git commit -m "feat(beams): assemble beam-frame from form-beam grid"
```

---

## Task 4: Project panel loads onto beam nodes

**Files:**
- Modify: `src/wing_design/beams/fea_model.py`
- Test: `tests/beams/test_fea_model.py`

- [ ] **Step 1: Write the failing load-conservation test** (append to the test file)

```python
from wing_design.aero.loads import PanelLoads
from wing_design.beams.fea_model import project_panels_to_beam_nodes
from wing_design.structural.projection import R_GEOM_FROM_AERO


def test_projection_conserves_force():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=16, n_levels=10)
    # one fake panel in the aero frame (span along +Y), out on the wing
    panels = PanelLoads(
        centers_xyz=np.array([[0.0, 2.0, 0.1]]),
        areas=np.array([0.5]),
        forces_xyz=np.array([[10.0, 0.0, 500.0]]),
        normals_xyz=np.array([[0.0, 0.0, 1.0]]),
        chords=np.array([0.8]),
        spanwise_widths=np.array([0.3]),
    )
    loads = project_panels_to_beam_nodes(frame, panels, safety_factor=2.0)
    expected = R_GEOM_FROM_AERO @ (panels.forces_xyz[0] * 2.0)
    assert np.allclose(loads[:, :3].sum(axis=0), expected)
    assert np.allclose(loads[:, 3:], 0.0)  # forces only, no applied moments
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_fea_model.py::test_projection_conserves_force -v`
Expected: FAIL — `ImportError: cannot import name 'project_panels_to_beam_nodes'`.

- [ ] **Step 3: Add the projector** (append to `fea_model.py`)

```python
def project_panels_to_beam_nodes(
    frame: BeamFrame,
    panels: PanelLoads,
    *,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """(n_nodes, 6) nodal loads: each panel force (aero→geom) lumped at the nearest node.

    Panel centers and forces are rotated from the aero frame (span +Y) to the
    geom frame (span +Z) with `R_GEOM_FROM_AERO`, then each panel's full force
    vector is assigned to the single nearest beam node. Moments are left zero.
    """
    loads = np.zeros((frame.nodes.shape[0], 6))
    centers_geom = panels.centers_xyz @ R_GEOM_FROM_AERO.T
    forces_geom = (panels.forces_xyz * safety_factor) @ R_GEOM_FROM_AERO.T
    for k in range(panels.n_panels):
        d = np.linalg.norm(frame.nodes - centers_geom[k], axis=1)
        loads[int(d.argmin()), :3] += forces_geom[k]
    return loads
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/beams/test_fea_model.py::test_projection_conserves_force -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/fea_model.py tests/beams/test_fea_model.py
git commit -m "feat(beams): project aero panel loads onto beam nodes"
```

---

## Task 5: Solve + summarize

**Files:**
- Modify: `src/wing_design/beams/fea_model.py`
- Test: `tests/beams/test_fea_model.py`

- [ ] **Step 1: Write the failing solve test** (append)

```python
from wing_design.beams.fea_model import solve_beam_frame


def test_solve_runs_and_deflects():
    spec = WingSpec()
    frame = build_beam_frame(spec, n_beams=16, n_levels=12)
    loads = np.zeros((frame.nodes.shape[0], 6))
    tip = [b * frame.n_levels + 0 for b in range(frame.n_beams)]  # level 0 = tip
    loads[tip, 0] = 50.0  # push the whole tip ring chordwise
    res = solve_beam_frame(frame, loads)
    assert np.isfinite(res.displacements).all()
    assert res.max_translation_m > 0.0
    # clamped base ring stays put
    assert np.allclose(res.displacements[frame.fixed_nodes], 0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_fea_model.py::test_solve_runs_and_deflects -v`
Expected: FAIL — `ImportError: cannot import name 'solve_beam_frame'`.

- [ ] **Step 3: Add `solve_beam_frame`, `FrameMetrics`, `summarize_frame`** (append)

```python
def solve_beam_frame(frame: BeamFrame, loads: np.ndarray) -> FrameResult:
    sections = [frame.section] * frame.elements.shape[0]
    return solve_frame(
        frame.nodes,
        frame.elements,
        sections,
        E=frame.E,
        G=frame.G,
        fixed_nodes=frame.fixed_nodes,
        loads=loads,
    )


@dataclass(frozen=True)
class FrameMetrics:
    case_name: str
    applied_force_N: float
    tip_translation_m: float
    tip_twist_deg: float
    max_axial_stress_Pa: float
    max_bending_stress_Pa: float
    max_vm_stress_Pa: float


def summarize_frame(
    frame: BeamFrame,
    result: FrameResult,
    case_name: str,
    applied_force_N: float,
) -> FrameMetrics:
    """Reduce a FrameResult to the headline tip/stress metrics for a load case."""
    sec = frame.section
    sigma_axial = np.abs(result.axial_force) / sec.A
    sigma_bend = result.bending_moment * sec.r / sec.Iz
    tau = np.abs(result.torsion) * sec.r / sec.J
    vm = np.sqrt((sigma_axial + sigma_bend) ** 2 + 3.0 * tau**2)

    tip_nodes = np.array([b * frame.n_levels + 0 for b in range(frame.n_beams)])
    tip_tr = float(np.linalg.norm(result.displacements[tip_nodes, :3], axis=1).max())
    tip_twist = float(np.abs(result.displacements[tip_nodes, 5]).max())  # rz about span axis

    return FrameMetrics(
        case_name=case_name,
        applied_force_N=applied_force_N,
        tip_translation_m=tip_tr,
        tip_twist_deg=float(np.degrees(tip_twist)),
        max_axial_stress_Pa=float(sigma_axial.max()),
        max_bending_stress_Pa=float(sigma_bend.max()),
        max_vm_stress_Pa=float(vm.max()),
    )
```

- [ ] **Step 4: Run the whole module test file**

Run: `uv run pytest tests/beams/test_fea_model.py -v`
Expected: all PASS (topology, projection, solve).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/fea_model.py tests/beams/test_fea_model.py
git commit -m "feat(beams): solve beam frame + summarize tip/stress metrics"
```

---

## Task 6: Export + runnable example

**Files:**
- Modify: `src/wing_design/beams/__init__.py`
- Create: `examples/21_beam_fea.py`

- [ ] **Step 1: Add `fea_model` exports to `beams/__init__.py`**

Append imports + `__all__` entries (match the file's existing style):

```python
from .fea_model import (
    BeamFrame,
    FrameMetrics,
    build_beam_frame,
    project_panels_to_beam_nodes,
    solve_beam_frame,
    summarize_frame,
)
```

Add to `__all__`:

```python
    "BeamFrame",
    "FrameMetrics",
    "build_beam_frame",
    "project_panels_to_beam_nodes",
    "solve_beam_frame",
    "summarize_frame",
```

- [ ] **Step 2: Create the example**

```python
# examples/21_beam_fea.py
"""Phase-B spike: beam-frame FEA of the form-beam structure under the LL envelope.

Builds the cantilevered space-frame (16 longitudinal form beams + transverse
rings), runs the LiftingLine aero envelope, projects each case's panel loads onto
the nearest beam node, solves, and prints per-case tip deflection, twist, and
member stress. Metrics for the *unsized* structure — the baseline Phase C sizes.
"""
from __future__ import annotations

import numpy as np

from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams.fea_model import (
    build_beam_frame,
    project_panels_to_beam_nodes,
    solve_beam_frame,
    summarize_frame,
)


def main() -> None:
    P = default_scenario()
    spec = P.geometry

    frame = build_beam_frame(
        spec,
        n_beams=16,
        n_levels=20,
        beam_radius=0.02,
        material=P.material,
        knockdown=P.material_iso.skin_E_knockdown,
        nu=P.nu_iso,
    )
    print(f"Frame: {frame.nodes.shape[0]} nodes, {frame.elements.shape[0]} elements")
    print(f"  E = {frame.E / 1e9:.1f} GPa, G = {frame.G / 1e9:.1f} GPa, beam r = 20 mm")
    print(f"  clamped keel-step ring: {frame.fixed_nodes.shape[0]} nodes")

    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane,
        P.load_cases,
        method="lifting_line",
        spanwise_resolution=P.aero.spanwise_resolution,
    )

    print(f"\n  {'case':<14} {'|F|':>9} {'u_tip':>9} {'twist':>8} {'σ_ax':>8} {'σ_bend':>8} {'σ_vm':>8}")
    print(f"  {'':14} {'N':>9} {'mm':>9} {'deg':>8} {'MPa':>8} {'MPa':>8} {'MPa':>8}")
    for ar in envelope:
        if ar.panels is None or abs(ar.factored_normal_force_N) < 1.0:
            continue
        loads = project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
        applied = float(np.linalg.norm(loads[:, :3].sum(axis=0)))
        result = solve_beam_frame(frame, loads)
        m = summarize_frame(frame, result, ar.case.name, applied)
        print(
            f"  {m.case_name:<14} {m.applied_force_N:>9.1f} {m.tip_translation_m * 1e3:>9.2f} "
            f"{m.tip_twist_deg:>8.3f} {m.max_axial_stress_Pa / 1e6:>8.2f} "
            f"{m.max_bending_stress_Pa / 1e6:>8.2f} {m.max_vm_stress_Pa / 1e6:>8.2f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the example end-to-end**

Run: `uv run python examples/21_beam_fea.py`
Expected: a frame summary line, then one metrics row per non-feathered load case (finite, positive tip deflection; stresses in a sane MPa range). If `spsolve` warns about singularity or NaNs appear, STOP — the frame is under-constrained; revisit ring connectivity before continuing.

- [ ] **Step 4: Commit**

```bash
git add src/wing_design/beams/__init__.py examples/21_beam_fea.py
git commit -m "feat(beams): Phase-B beam-frame FEA example (21_beam_fea)"
```

---

## Task 7: Update the plan doc

**Files:**
- Modify: `docs/plan.md`

- [ ] **Step 1: Mark Phase B complete**

In the **Phase B — Structural-eval spike** section, update the deliverable line to note completion and the new solver, e.g. append:

```markdown
**Status: done (2026-06-04).** Beam elements implemented as a fresh in-house 3D
Euler–Bernoulli frame solver (`structural.frame`); skin coupling deferred — the
spike models the skin's transverse stiffening as ring connectors between beams,
not yet a shell. Loads via full panel-force projection onto nearest beam nodes.
Deliverable: `examples/21_beam_fea.py`.
```

Also add a row to the **Decisions log** table:

```markdown
| Phase-B FEA element | Fresh 3D Euler–Bernoulli frame (`structural.frame`); rings stand in for skin shear transfer in the spike (coupled shell = later deepen). |
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest`
Expected: all green (Phase A + new frame + fea_model tests).

- [ ] **Step 3: Commit**

```bash
git add docs/plan.md
git commit -m "docs: mark Phase B done, log frame-FEA decision"
```

---

## Self-Review

- **Spec coverage:** Plan deliverable ("per-load-case stress/deflection metrics for the unsized structure") → Task 6 example output. Beam elements → Task 1. Cantilevered at keel-step → Task 3 `fixed_nodes`. Load-bearing-skin-between-beams: represented in this spike by ring connectors (transverse stiffness) per the chosen "frame w/ ring connectors" fidelity; full shell coupling is explicitly deferred and logged in Task 7. Full panel projection → Task 4.
- **Placeholder scan:** none — all code is concrete.
- **Type consistency:** `BeamSection`, `FrameResult` defined in Task 1 and consumed unchanged in Tasks 3/5. `summarize_frame(frame, result, case_name, applied_force_N)` signature matches its call in the Task 6 example. `R_GEOM_FROM_AERO` imported from `structural.projection` (confirmed to exist). `form_beam_grid(spec, z, n_beams)` and `default_z_levels(spec, n)` match Phase-A signatures.
- **Known spike limitation:** circular sections give `Iy == Iz`, so element roll is irrelevant — the simplified `_element_rotation` is valid here and would need a real principal-axis input only if non-circular sections are introduced (Phase D).
