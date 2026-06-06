# Phase E.1 — Load-Bearing Skin Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Put the load-bearing skin into the structural model — assemble shell panels between the form-beam nodes into the same global stiffness matrix as the beams, with the **skin replacing the ring connectors** as the transverse/torsional load path — and quantify the stiffness gain versus the Phase-B ring frame at the same beam sizes. This directly tests the Phase-C/D finding that the skin, not thicker beams, is the mass-efficient stiffness lever.

**Scope note:** This is **Phase E increment 1** of the four-thread Phase E (skin coupling, MILP stock catalog, buckling+twist constraints, CLT anisotropic skin). It delivers the coupled beam+shell model and a stiffness comparison. **Deferred to later increments:** re-sizing with the skin (E.2), skin membrane-stress recovery + sizing constraints, MILP catalog, buckling/twist constraints, and CLT anisotropy. The skin here is isotropic-equivalent.

**Architecture:** A new `structural/beam_shell.py` assembles a combined sparse stiffness from beam elements (reusing `frame.local_beam_stiffness` + `_element_rotation`) and shell triangles (reusing `shell._element_K`, which is already 18×18 in the global frame with matching 6-DOF node ordering), solves the clamped system, and recovers beam internal forces — returning a `FrameResult` (so the existing `summarize_frame` works unchanged). `beams/shell_model.py` triangulates the beam grid into a skin (2 triangles per quad, wrapping in beam index), builds a `BeamShellModel` (longitudinal beams only — no rings — plus skin), solves it, and the example compares tip deflection beams+rings vs beams+skin.

**Tech Stack:** numpy, scipy.sparse, existing `structural.frame` + `structural.shell`, scenario isotropic-equivalent skin properties.

---

## Background facts (verified)

- `structural/shell.py::_element_K(p1, p2, p3, *, E, nu, t, drilling_factor=1e-4) -> (18,18)` global-frame stiffness, per-node DOF order `[u, v, w, θx, θy, θz]`. CST membrane + DKT bending + θz drilling penalty.
- `structural/frame.py`: `local_beam_stiffness(E, G, sec, L) -> (12,12)`; `_element_rotation(pi, pj) -> (R, L)` (R rows = local axes); `FrameResult(displacements (N,6), axial_force, bending_moment, torsion)`; `BeamSection`. Beam DOF order `[ux,uy,uz,rx,ry,rz]` (matches shell).
- `beams/fea_model.py`: `BeamFrame` (nodes, elements [longitudinal then rings], n_beams, n_levels, fixed_nodes, tip_nodes, section, E, G); `build_beam_frame(...)`; `summarize_frame(frame, result, case_name, applied_force_N) -> FrameMetrics` (uses `frame.tip_nodes`, `frame.section`; reads `result.displacements/axial_force/bending_moment/torsion`); `project_panels_to_beam_nodes`.
- Beam grid node id = `b * n_levels + k` (beam-major/level-minor). Longitudinal elements are the first `n_beams*(n_levels-1)` of `BeamFrame.elements`; rings are the rest.
- Scenario: `P.E_iso_Pa`, `P.nu_iso`, `P.G_iso_Pa`, `P.skin_sizing.t_baseline_m` (0.003), `P.rho_kgm3`, `P.material`, `P.material_iso.skin_E_knockdown`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/wing_design/structural/shell.py` (modify) | Add a public alias `tri_element_stiffness = _element_K` (so the combined solver doesn't import a private name). |
| `src/wing_design/structural/beam_shell.py` (create) | `solve_beam_shell(...)` — combined beam+shell assembly/solve, returns `FrameResult`. |
| `src/wing_design/structural/__init__.py` (modify) | Export `solve_beam_shell`, `tri_element_stiffness`. |
| `src/wing_design/beams/shell_model.py` (create) | `skin_triangles`, `BeamShellModel`, `build_beam_shell_model`, `solve_beam_shell_model`. |
| `src/wing_design/beams/__init__.py` (modify) | Export the new shell-model API. |
| `examples/24_skin_coupling.py` (create) | Compare tip deflection: beams+rings (Phase B) vs beams+skin, per load case. |
| `tests/structural/test_beam_shell.py` (create) | No-skin reduces to frame; skin stiffens a cantilever. |
| `tests/beams/test_shell_model.py` (create) | Skin triangulation counts/validity; model solves; skin stiffer than ring frame. |
| `docs/plan.md` (modify) | Record Phase E.1 done + the deferred E.2+ threads. |

---

## Task 1: Public shell-element alias

**Files:**
- Modify: `src/wing_design/structural/shell.py`
- Modify: `src/wing_design/structural/__init__.py`

- [ ] **Step 1:** In `shell.py`, immediately after the `_element_K` function definition, add a public alias:

```python
# Public alias: the combined beam+shell solver (structural/beam_shell.py) reuses this.
tri_element_stiffness = _element_K
```

- [ ] **Step 2:** In `structural/__init__.py`, add `tri_element_stiffness` to the `from .shell import (...)` block and to `__all__`.

- [ ] **Step 3: Verify**

Run: `uv run python -c "from wing_design.structural import tri_element_stiffness; import numpy as np; K=tri_element_stiffness(np.array([0,0,0.]),np.array([1,0,0.]),np.array([0,1,0.]),E=1e9,nu=0.3,t=0.003); print(K.shape)"`
Expected: `(18, 18)`

- [ ] **Step 4: Commit**

```bash
git add src/wing_design/structural/shell.py src/wing_design/structural/__init__.py
git commit -m "feat(structural): public tri_element_stiffness alias for shell element K"
# end with the Co-Authored-By trailer
```

---

## Task 2: Combined beam+shell solver

**Files:**
- Create: `src/wing_design/structural/beam_shell.py`
- Test: `tests/structural/test_beam_shell.py`

- [ ] **Step 1: Write the failing tests** (`tests/structural/test_beam_shell.py`)

```python
import numpy as np

from wing_design.structural.frame import BeamSection, solve_frame
from wing_design.structural.beam_shell import solve_beam_shell

E_B, NU = 70e9, 0.3
G_B = E_B / (2 * (1 + NU))


def test_no_skin_reduces_to_frame():
    # A 2-element beam along X; with no shell triangles the combined solver must
    # reproduce the pure-frame result exactly.
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    beam_elems = np.array([[0, 1], [1, 2]], dtype=int)
    secs = [BeamSection.circular(0.02)] * 2
    loads = np.zeros((3, 6))
    loads[2, 2] = 200.0
    frame_res = solve_frame(nodes, beam_elems, secs, E=E_B, G=G_B,
                            fixed_nodes=np.array([0]), loads=loads)
    bs_res = solve_beam_shell(
        nodes, beam_elems, secs, np.empty((0, 3), dtype=int),
        E_beam=E_B, G_beam=G_B, E_skin=1e9, nu_skin=0.3, t_skin=0.003,
        fixed_nodes=np.array([0]), loads=loads,
    )
    assert np.allclose(bs_res.displacements, frame_res.displacements, atol=1e-12)


def test_skin_stiffens_cantilever():
    # Two parallel beams + a connecting skin panel (2 triangles) vs the beams alone.
    # Under a shared transverse tip load the skinned structure deflects less.
    nodes = np.array([
        [0, 0, 0], [1, 0, 0],     # beam A: nodes 0,1
        [0, 0.2, 0], [1, 0.2, 0],  # beam B: nodes 2,3
    ], dtype=float)
    beam_elems = np.array([[0, 1], [2, 3]], dtype=int)
    secs = [BeamSection.circular(0.01)] * 2
    loads = np.zeros((4, 6))
    loads[1, 2] = 100.0
    loads[3, 2] = 100.0
    fixed = np.array([0, 2])

    no_skin = solve_beam_shell(
        nodes, beam_elems, secs, np.empty((0, 3), dtype=int),
        E_beam=E_B, G_beam=G_B, E_skin=70e9, nu_skin=0.3, t_skin=0.003,
        fixed_nodes=fixed, loads=loads,
    )
    tris = np.array([[0, 1, 3], [0, 3, 2]], dtype=int)  # skin between the two beams
    with_skin = solve_beam_shell(
        nodes, beam_elems, secs, tris,
        E_beam=E_B, G_beam=G_B, E_skin=70e9, nu_skin=0.3, t_skin=0.003,
        fixed_nodes=fixed, loads=loads,
    )
    d_no = np.abs(no_skin.displacements[[1, 3], 2]).max()
    d_yes = np.abs(with_skin.displacements[[1, 3], 2]).max()
    assert d_yes < d_no            # skin adds stiffness
    assert np.isfinite(d_yes) and d_yes > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/structural/test_beam_shell.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/wing_design/structural/beam_shell.py`

```python
"""Combined beam + shell FEA: form-beam members plus a load-bearing skin.

Assembles the Phase-B frame's beam elements and the DKT+CST shell skin panels into
one global stiffness matrix (both use 6-DOF nodes with the same DOF ordering, so the
shell element's 18×18 global stiffness drops straight in at the shared beam nodes),
solves the clamped system, and recovers beam internal forces — returning a
`FrameResult` so the existing `summarize_frame` works unchanged. The skin replaces
the ring connectors as the transverse/torsional load path.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .frame import BeamSection, FrameResult, _element_rotation, local_beam_stiffness
from .shell import tri_element_stiffness


def solve_beam_shell(
    nodes: np.ndarray,
    beam_elements: np.ndarray,
    beam_sections: list[BeamSection],
    shell_tris: np.ndarray,
    *,
    E_beam: float,
    G_beam: float,
    E_skin: float,
    nu_skin: float,
    t_skin: float,
    fixed_nodes: np.ndarray,
    loads: np.ndarray,
    drilling_factor: float = 1.0e-4,
) -> FrameResult:
    """Solve a clamped beam+shell structure; recover per-beam-element internal forces.

    `shell_tris` is (M, 3) int node indices (may be empty (0,3)). `loads` is (N, 6).
    """
    n_nodes = nodes.shape[0]
    ndof = 6 * n_nodes
    n_beam = beam_elements.shape[0]

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    transforms: list[np.ndarray] = []
    klocals: list[np.ndarray] = []

    # Beam elements
    for e in range(n_beam):
        i, j = int(beam_elements[e, 0]), int(beam_elements[e, 1])
        R, L = _element_rotation(nodes[i], nodes[j])
        kloc = local_beam_stiffness(E_beam, G_beam, beam_sections[e], L)
        T = np.zeros((12, 12))
        for blk in range(4):
            T[3 * blk:3 * blk + 3, 3 * blk:3 * blk + 3] = R
        kg = T.T @ kloc @ T
        transforms.append(T)
        klocals.append(kloc)
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        for a_ in range(12):
            for b_ in range(12):
                rows.append(int(dofs[a_]))
                cols.append(int(dofs[b_]))
                vals.append(kg[a_, b_])

    # Shell (skin) triangles
    for t in range(shell_tris.shape[0]):
        n0, n1, n2 = int(shell_tris[t, 0]), int(shell_tris[t, 1]), int(shell_tris[t, 2])
        ke = tri_element_stiffness(
            nodes[n0], nodes[n1], nodes[n2],
            E=E_skin, nu=nu_skin, t=t_skin, drilling_factor=drilling_factor,
        )
        dofs = np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6]
        for a_ in range(18):
            for b_ in range(18):
                rows.append(int(dofs[a_]))
                cols.append(int(dofs[b_]))
                vals.append(ke[a_, b_])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    f = loads.reshape(-1).astype(float)
    if len(fixed_nodes):
        fixed_dofs = np.concatenate([6 * int(fn) + np.arange(6) for fn in fixed_nodes])
    else:
        fixed_dofs = np.array([], dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)

    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    disp = u.reshape(n_nodes, 6)

    axial = np.zeros(n_beam)
    bending = np.zeros(n_beam)
    torsion = np.zeros(n_beam)
    for e in range(n_beam):
        i, j = int(beam_elements[e, 0]), int(beam_elements[e, 1])
        ue = np.r_[u[6 * i:6 * i + 6], u[6 * j:6 * j + 6]]
        floc = klocals[e] @ (transforms[e] @ ue)
        axial[e] = floc[6]
        bending[e] = max(np.hypot(floc[4], floc[5]), np.hypot(floc[10], floc[11]))
        torsion[e] = floc[9]

    return FrameResult(displacements=disp, axial_force=axial,
                       bending_moment=bending, torsion=torsion)
```

- [ ] **Step 4: Run the tests** — `uv run pytest tests/structural/test_beam_shell.py -v` → 2 PASS. Then `uv run pytest` (full suite green).

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/structural/beam_shell.py tests/structural/test_beam_shell.py
git commit -m "feat(structural): combined beam+shell FEA solver"
# trailer
```

---

## Task 3: Export the combined solver

**Files:**
- Modify: `src/wing_design/structural/__init__.py`

- [ ] **Step 1:** Add `solve_beam_shell` to the imports (`from .beam_shell import solve_beam_shell`) and `__all__`.

- [ ] **Step 2: Verify** — `uv run python -c "from wing_design.structural import solve_beam_shell; print('ok')"` → `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/wing_design/structural/__init__.py
git commit -m "feat(structural): export solve_beam_shell"
# trailer
```

---

## Task 4: Skin triangulation + beam-shell model

**Files:**
- Create: `src/wing_design/beams/shell_model.py`
- Test: `tests/beams/test_shell_model.py`

- [ ] **Step 1: Write the failing tests** (`tests/beams/test_shell_model.py`)

```python
import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.fea_model import build_beam_frame, project_panels_to_beam_nodes
from wing_design.aero.loads import PanelLoads
from wing_design.beams.shell_model import (
    skin_triangles,
    build_beam_shell_model,
    solve_beam_shell_model,
)


def test_skin_triangulation_counts_and_validity():
    nb, nl = 8, 5
    tris = skin_triangles(nb, nl)
    assert tris.shape == (2 * nb * (nl - 1), 3)          # 2 tris per quad, wrap in beam index
    assert tris.min() >= 0 and tris.max() < nb * nl       # all node indices valid
    # every grid node participates in at least one triangle
    assert len(np.unique(tris)) == nb * nl


def test_model_builds_and_solves():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    # beams only (no rings) + skin
    assert model.beam_elements.shape[0] == 8 * (5 - 1)
    assert model.shell_tris.shape[0] == 2 * 8 * (5 - 1)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 20.0
    res = solve_beam_shell_model(model, loads)
    assert np.isfinite(res.displacements).all()
    assert np.allclose(res.displacements[model.fixed_nodes], 0.0)


def test_skin_stiffer_than_ring_frame():
    # Same beams + loads: the skinned model must deflect less than the Phase-B ring frame.
    spec = WingSpec()
    nb, nl, r = 16, 6, 0.02
    model = build_beam_shell_model(spec, n_beams=nb, n_levels=nl, beam_radius=r)
    frame = build_beam_frame(spec, n_beams=nb, n_levels=nl, beam_radius=r)

    panels = PanelLoads(
        centers_xyz=np.array([[0.0, 2.5, 0.1]]),
        areas=np.array([1.0]),
        forces_xyz=np.array([[0.0, 0.0, 1000.0]]),
        normals_xyz=np.array([[0.0, 0.0, 1.0]]),
        chords=np.array([0.8]),
        spanwise_widths=np.array([0.5]),
    )
    loads = project_panels_to_beam_nodes(frame, panels)

    from wing_design.beams.fea_model import solve_beam_frame
    d_ring = np.linalg.norm(solve_beam_frame(frame, loads).displacements[frame.tip_nodes, :3], axis=1).max()
    d_skin = np.linalg.norm(solve_beam_shell_model(model, loads).displacements[model.tip_nodes, :3], axis=1).max()
    assert d_skin < d_ring
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/beams/test_shell_model.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/wing_design/beams/shell_model.py`

```python
"""Beam-shell model of the form-beam structure: longitudinal beams + load-bearing skin.

Builds the Phase-A/B beam grid, keeps the longitudinal beam members, and replaces the
ring connectors with a triangulated skin (the load-bearing shell). Solves with the
combined `structural.solve_beam_shell`. The skin is isotropic-equivalent here; CLT
anisotropy and skin-stress-driven re-sizing are later Phase-E increments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.wing import WingSpec
from ..materials.unidir import T700_EPOXY, UDPly
from ..structural.beam_shell import solve_beam_shell
from ..structural.frame import BeamSection, FrameResult
from .splines import default_z_levels, form_beam_grid


def skin_triangles(n_beams: int, n_levels: int) -> np.ndarray:
    """(2*n_beams*(n_levels-1), 3) skin triangles tiling the beam grid.

    Each quad between beams b,(b+1)%n_beams and levels k,k+1 is split into two
    triangles. Node id = b*n_levels + k (beam-major/level-minor).
    """
    def nid(b: int, k: int) -> int:
        return b * n_levels + k

    tris: list[tuple[int, int, int]] = []
    for k in range(n_levels - 1):
        for b in range(n_beams):
            bn = (b + 1) % n_beams
            tris.append((nid(b, k), nid(bn, k), nid(bn, k + 1)))
            tris.append((nid(b, k), nid(bn, k + 1), nid(b, k + 1)))
    return np.asarray(tris, dtype=int)


@dataclass(frozen=True)
class BeamShellModel:
    nodes: np.ndarray           # (n_beams*n_levels, 3)
    beam_elements: np.ndarray   # (n_beams*(n_levels-1), 2) longitudinal only
    shell_tris: np.ndarray      # (2*n_beams*(n_levels-1), 3)
    n_beams: int
    n_levels: int
    fixed_nodes: np.ndarray
    tip_nodes: np.ndarray
    section: BeamSection
    E_beam: float
    G_beam: float
    E_skin: float
    nu_skin: float
    t_skin: float


def build_beam_shell_model(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
    material: UDPly = T700_EPOXY,
    knockdown: float = 0.5,
    nu: float = 0.32,
    skin_thickness: float = 0.003,
) -> BeamShellModel:
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")
    z = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z, n_beams)
    nodes = grid.reshape(-1, 3)

    def nid(b: int, k: int) -> int:
        return b * n_levels + k

    beam_elems = [(nid(b, k), nid(b, k + 1)) for b in range(n_beams) for k in range(n_levels - 1)]
    beam_elements = np.asarray(beam_elems, dtype=int)

    keel_k = int(np.argmin(z))
    tip_k = int(np.argmax(z))
    fixed_nodes = np.array([nid(b, keel_k) for b in range(n_beams)], dtype=int)
    tip_nodes = np.array([nid(b, tip_k) for b in range(n_beams)], dtype=int)

    E = material.isotropic_equivalent_modulus(knockdown=knockdown)
    G = E / (2.0 * (1.0 + nu))
    return BeamShellModel(
        nodes=nodes,
        beam_elements=beam_elements,
        shell_tris=skin_triangles(n_beams, n_levels),
        n_beams=n_beams,
        n_levels=n_levels,
        fixed_nodes=fixed_nodes,
        tip_nodes=tip_nodes,
        section=BeamSection.circular(beam_radius),
        E_beam=E,
        G_beam=G,
        E_skin=E,        # isotropic-equivalent skin (same UD-knockdown modulus)
        nu_skin=nu,
        t_skin=skin_thickness,
    )


def solve_beam_shell_model(model: BeamShellModel, loads: np.ndarray) -> FrameResult:
    sections = [model.section] * model.beam_elements.shape[0]
    return solve_beam_shell(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam,
        E_skin=model.E_skin, nu_skin=model.nu_skin, t_skin=model.t_skin,
        fixed_nodes=model.fixed_nodes, loads=loads,
    )
```

- [ ] **Step 4: Run the tests** — `uv run pytest tests/beams/test_shell_model.py -v` → 3 PASS. Then `uv run pytest` (full suite). The `test_skin_stiffer_than_ring_frame` is the key check: the skin must reduce tip deflection vs the ring frame at the same beam radius. **If it fails (skin NOT stiffer), STOP and report** — that would contradict the finding and needs investigation (drilling factor, skin thickness, or a model bug), not a weakened test.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/beams/shell_model.py tests/beams/test_shell_model.py
git commit -m "feat(beams): beam-shell model (skin replaces rings)"
# trailer
```

---

## Task 5: Export + comparison example + plan update

**Files:**
- Modify: `src/wing_design/beams/__init__.py`
- Create: `examples/24_skin_coupling.py`
- Modify: `docs/plan.md`

- [ ] **Step 1: Export** — in `src/wing_design/beams/__init__.py` add `from .shell_model import BeamShellModel, build_beam_shell_model, skin_triangles, solve_beam_shell_model` and add those four names to `__all__`.

- [ ] **Step 2: Create `examples/24_skin_coupling.py`**

```python
"""Phase-E.1 spike: quantify the load-bearing skin's stiffness contribution.

Builds the Phase-B ring frame and the Phase-E beam-shell model (skin replaces rings)
at the SAME beam radius, runs the LiftingLine load envelope, and compares tip
deflection per case. Tests the Phase-C/D finding that the skin — not thicker beams —
is the mass-efficient stiffness lever.
"""
from __future__ import annotations

import numpy as np

from wing_design import default_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    solve_beam_frame,
    solve_beam_shell_model,
)


def main() -> None:
    P = default_scenario()
    spec = P.geometry
    n_beams, n_levels, r = 16, 12, 0.02

    frame = build_beam_frame(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=r,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
    )
    model = build_beam_shell_model(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=r,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m,
    )
    print(f"beam radius {r*1e3:.0f} mm; skin thickness {P.skin_sizing.t_baseline_m*1e3:.0f} mm")
    print(f"ring frame: {frame.elements.shape[0]} elements | "
          f"beam-shell: {model.beam_elements.shape[0]} beams + {model.shell_tris.shape[0]} skin tris")

    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane, P.load_cases, method="lifting_line",
        spanwise_resolution=P.aero.spanwise_resolution,
    )

    print(f"\n  {'case':<14} {'u_tip(rings)':>13} {'u_tip(skin)':>12} {'stiffer x':>10}")
    print(f"  {'':14} {'mm':>13} {'mm':>12} {'':>10}")
    for ar in envelope:
        if ar.panels is None or abs(ar.factored_normal_force_N) < 1.0:
            continue
        loads = project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
        d_ring = float(np.linalg.norm(solve_beam_frame(frame, loads).displacements[frame.tip_nodes, :3], axis=1).max())
        d_skin = float(np.linalg.norm(solve_beam_shell_model(model, loads).displacements[model.tip_nodes, :3], axis=1).max())
        ratio = d_ring / d_skin if d_skin > 0 else float("inf")
        print(f"  {ar.case.name:<14} {d_ring*1e3:>13.2f} {d_skin*1e3:>12.2f} {ratio:>9.1f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the example** — `uv run python examples/24_skin_coupling.py`. Expect a table where `u_tip(skin)` < `u_tip(rings)` for every case (stiffer x > 1). **Paste the full output** and report the stiffness factor — this quantifies the finding. (If the skin is *not* stiffer, report it — do not hide it.)

- [ ] **Step 4: Update `docs/plan.md`** — in **Phase E — Deepen structural**, append a status note recording: Phase E.1 done (load-bearing skin coupled, skin replaces rings, isotropic-equivalent); the measured stiffness factor from Step 3; and the deferred E.2+ threads (skin-stress recovery + re-sizing with skin, MILP stock catalog, buckling+twist constraints, CLT anisotropic skin). Add a Decisions-log row:

```markdown
| Phase-E.1 skin coupling | Load-bearing skin assembled as DKT+CST shell panels between beam nodes into the combined `structural.solve_beam_shell`; skin replaces ring connectors. Isotropic-equivalent skin; re-sizing/MILP/buckling/CLT deferred to later E increments. |
```

- [ ] **Step 5: Full suite** — `uv run pytest` → all green.

- [ ] **Step 6: Commit**

```bash
git add src/wing_design/beams/__init__.py examples/24_skin_coupling.py docs/plan.md
git commit -m "feat(beams): Phase-E.1 skin-coupling example + export API; quantify stiffness gain"
# trailer
```

---

## Self-Review

- **Spec coverage:** skin coupling (skin replaces rings) → Tasks 2–4; DOF-compatible shell reuse → Task 1 alias + Task 2 solver; stiffness-gain quantification → Task 5 example. Re-sizing/MILP/buckling/CLT explicitly deferred (scope note).
- **Placeholder scan:** none.
- **Type consistency:** `solve_beam_shell` returns `FrameResult` (consumed by `summarize_frame` and the example's displacement reads); `BeamShellModel.beam_elements` is longitudinal-only; `skin_triangles` node ids match `build_beam_shell_model`'s `nid`; `tri_element_stiffness` is the public alias used by the solver; skin material defaults to the same UD-knockdown modulus as the beams (isotropic-equivalent).
- **Risk:** the shell `_element_K` reuse depends on its global-frame 18×18 output with `[u,v,w,θx,θy,θz]` ordering (verified). `test_no_skin_reduces_to_frame` guards the beam path; `test_skin_stiffens_cantilever` and `test_skin_stiffer_than_ring_frame` guard the coupling. Degenerate skin triangles near the spar (closely-spaced beams) are small but non-zero area; if `_triangle_local_frame` ever divides by zero there, that surfaces in Task 4 — report rather than silently guard.
- **Known limitation (intentional):** isotropic-equivalent skin (CLT later); no skin-stress recovery yet (deflection + beam stress only); the model evaluates at a fixed beam radius (re-sizing-with-skin is E.2).
