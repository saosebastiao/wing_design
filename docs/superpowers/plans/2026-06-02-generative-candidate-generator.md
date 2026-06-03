# Generative Candidate Generator + End-to-End (Milestone 1C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a contract-correct `CandidateMenu` from the wing (driven by the real background shell FEA), validate it, lump aero loads onto frame nodes, export a selected truss to build123d, and wire the whole thing end-to-end in two runnable examples.

**Architecture:** A `validate_menu` guard asserts the generator contracts the CP-SAT model and frame gate depend on. `build_candidate_menu` runs the existing shell FEA on the wing OML to get a stress field, then builds a small, deliberately simple beam library (a central keel→deck→tip spar plus deck-hosted chordwise beams and one chord-plane-mirror pair — all monotonic-z, landmark-exact, junction-exact by construction), a discrete cross-section catalog, a pairwise conflict table, and stress-informed coverage targets. `lump_aero_to_frame_nodes` maps the spanwise aero normal force onto frame nodes. `wing_candidate_to_part` lofts each beam's circular section along its centerline into a build123d Compound. Two examples tie it together: one inspects the menu, one runs menu → CP-SAT → gate → export.

**Tech Stack:** Python 3.10–3.12; reuses `structural/{mesh,shell,fea}`, `aero/{model,cases,loads}`, `structural/projection`, build123d, meshio; `pytest`, `uv`. No new dependencies. (Confirmed working in this environment: gmsh, AeroSandbox, build123d, meshio; shell FEA ~37 s.)

**Scope note:** Plan **1C of 3** for Milestone 1; completes the thin end-to-end slice. 1A (CP-SAT core) and 1B (frame gate) are done. **Deliberately deferred** (later deepening, not this plan): full stress-line-conformant curved-beam tracing (the spec §5 tracer), the outer ranked-pool loop (M2), wraps (M3), volumetric finalist check (M4). The milestone DoD (spec §8.4) is: `21_generate_truss` runs on the 5 m wingsail, single load case, circle sections, no wraps/loop, emitting a chord-symmetric selected truss (STEP/STL/VTU) with a passing `GateResult`.

**Simple-library rationale:** The generator emits a small library where every beam satisfies the contracts *by construction* (monotonic-z; every keel-rooted run goes keel→deck vertically first per the "wing-root sections pointed up" rule; non-spar beams are hosted at the deck node so the deck BC is always present when a tip beam is selected; landmark nodes sit exactly on beam control points; mirror pairs are reciprocal). The background FEA still drives **coverage targets** (stress-informed), satisfying the spec's "prefer stress-aligned" intent at the coverage level. Curved stress-line beams are a drop-in richer library later.

**Conventions:** snake_case; geometry frame chord +X, normal +Y, span +Z; reuse `DesignParameters`. Landmark coordinates are taken from `WingSpec` z-accessors at `(0, 0, z)` (the pivot axis is at x=0, y=0).

---

## File Structure

- Modify: `src/wing_design/generative/menu.py` — add `validate_menu`.
- Create: `src/wing_design/generative/candidates.py` — `build_beam_library`, `build_candidate_menu`.
- Create: `src/wing_design/generative/loads.py` — `lump_aero_to_frame_nodes`.
- Create: `src/wing_design/generative/build.py` — `wing_candidate_to_part`.
- Modify: `src/wing_design/generative/__init__.py` — export the new public functions.
- Create: `tests/generative/test_validate.py`, `tests/generative/test_candidates.py`, `tests/generative/test_loads.py`, `tests/generative/test_build.py`.
- Create: `examples/20_candidate_menu.py`, `examples/21_generate_truss.py`.

---

## Task C1: `validate_menu` guard

**Files:**
- Modify: `src/wing_design/generative/menu.py`
- Create: `tests/generative/test_validate.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/generative/test_validate.py`:
```python
import math

import pytest

from wing_design.generative.menu import (
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    ConflictTable,
    CrossSectionOption,
    CrossSectionShape,
    NodeKind,
    validate_menu,
)


def _beam(beam_id, pts, start_kind, end_kind, mirror_id=None, host_id=None, on_chord=True):
    return CandidateBeam(
        id=beam_id,
        control_points=tuple(pts),
        start_kind=start_kind,
        end_kind=end_kind,
        start_node=0,
        end_node=1,
        length_m=10.0,
        min_radius_m=100.0,
        on_chord_plane=on_chord,
        mirror_id=mirror_id,
        host_id=host_id,
        covers=(),
    )


def _menu(nodes, beams):
    return CandidateMenu(
        nodes=tuple(nodes),
        beams=tuple(beams),
        cross_sections=(CrossSectionOption(0, CrossSectionShape.CIRCLE, 1e-3),),
        conflicts=ConflictTable(forbidden=()),
        coverage_targets=(),
        rho_kgm3=1550.0,
    )


def _good_menu():
    nodes = (
        CandidateNode(id=0, xyz=(0.0, 0.0, -0.95), kind=NodeKind.KEEL_STEP, z_layer=0),
        CandidateNode(id=1, xyz=(0.0, 0.0, -0.20), kind=NodeKind.DECK_STEP, z_layer=1),
        CandidateNode(id=2, xyz=(0.0, 0.0, 5.0), kind=NodeKind.TIP, z_layer=9),
    )
    spar = _beam(0, [(0, 0, -0.95), (0, 0, -0.20), (0, 0, 5.0)],
                 NodeKind.KEEL_STEP, NodeKind.TIP)
    branch = _beam(1, [(0, 0, -0.20), (0.3, 0, 2.0), (0, 0, 5.0)],
                   NodeKind.ON_BEAM, NodeKind.TIP, host_id=0)
    return _menu(nodes, [spar, branch])


def test_validate_accepts_good_menu():
    validate_menu(_good_menu())  # must not raise


def test_validate_rejects_non_monotonic_z():
    nodes = _good_menu().nodes
    bad = _beam(0, [(0, 0, -0.95), (0, 0, 2.0), (0, 0, 1.0)],
                NodeKind.KEEL_STEP, NodeKind.TIP)
    with pytest.raises(ValueError, match="monotonic"):
        validate_menu(_menu(nodes, [bad]))


def test_validate_rejects_missing_landmark_node():
    # Beam endpoints have no matching keel landmark node.
    nodes = (
        CandidateNode(id=2, xyz=(0.0, 0.0, 5.0), kind=NodeKind.TIP, z_layer=9),
    )
    spar = _beam(0, [(0, 0, -0.95), (0, 0, 5.0)], NodeKind.KEEL_STEP, NodeKind.TIP)
    with pytest.raises(ValueError, match="landmark"):
        validate_menu(_menu(nodes, [spar]))


def test_validate_rejects_host_cycle():
    nodes = _good_menu().nodes
    b0 = _beam(0, [(0, 0, -0.20), (0, 0, 5.0)], NodeKind.ON_BEAM, NodeKind.TIP, host_id=1)
    b1 = _beam(1, [(0, 0, -0.20), (0, 0, 5.0)], NodeKind.ON_BEAM, NodeKind.TIP, host_id=0)
    with pytest.raises(ValueError, match="cycle"):
        validate_menu(_menu(nodes, [b0, b1]))


def test_validate_rejects_nonreciprocal_mirror():
    nodes = _good_menu().nodes
    b0 = _beam(0, [(0, 0, -0.95), (0, 0, 5.0)], NodeKind.KEEL_STEP, NodeKind.TIP,
               mirror_id=1, on_chord=False)
    b1 = _beam(1, [(0, 0, -0.95), (0, 0, 5.0)], NodeKind.KEEL_STEP, NodeKind.TIP,
               mirror_id=None, on_chord=False)
    with pytest.raises(ValueError, match="mirror"):
        validate_menu(_menu(nodes, [b0, b1]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/generative/test_validate.py -v`
Expected: FAIL — `ImportError: cannot import name 'validate_menu'`

- [ ] **Step 3: Implement validate_menu**

Append to `src/wing_design/generative/menu.py`:
```python
def _coord_key(xyz, tol=1e-6):
    return (round(xyz[0] / tol), round(xyz[1] / tol), round(xyz[2] / tol))


def validate_menu(menu: "CandidateMenu") -> None:
    """Assert the contracts the CP-SAT model and frame gate rely on but do not
    re-check. Raises ValueError on the first violation.

    Contracts (see the design spec, generator-contracts section):
      * each beam's control points are strictly increasing in z (monotonic-z);
      * every beam endpoint that is a landmark kind (keel/deck/tip) coincides
        with a CandidateNode of that kind at the same coordinate;
      * the host_id graph is acyclic;
      * mirror_id is reciprocal (A.mirror == B  =>  B.mirror == A);
      * an ON_BEAM endpoint names a host that exists.
    """
    node_kind_by_key = {_coord_key(n.xyz): n.kind for n in menu.nodes}
    beam_by_id = {b.id: b for b in menu.beams}

    for b in menu.beams:
        zs = [p[2] for p in b.control_points]
        if any(z1 <= z0 for z0, z1 in zip(zs[:-1], zs[1:])):
            raise ValueError(f"beam {b.id} control points are not monotonic in z: {zs}")

        start_key = _coord_key(b.control_points[0])
        end_key = _coord_key(b.control_points[-1])
        if b.start_kind in (NodeKind.KEEL_STEP, NodeKind.DECK_STEP, NodeKind.TIP):
            if node_kind_by_key.get(start_key) != b.start_kind:
                raise ValueError(
                    f"beam {b.id} start landmark {b.start_kind} has no matching node"
                )
        if b.end_kind in (NodeKind.KEEL_STEP, NodeKind.DECK_STEP, NodeKind.TIP):
            if node_kind_by_key.get(end_key) != b.end_kind:
                raise ValueError(
                    f"beam {b.id} end landmark {b.end_kind} has no matching node"
                )

        if b.host_id is not None and b.host_id not in beam_by_id:
            raise ValueError(f"beam {b.id} names a host {b.host_id} that does not exist")

        if b.mirror_id is not None:
            partner = beam_by_id.get(b.mirror_id)
            if partner is None or partner.mirror_id != b.id:
                raise ValueError(f"beam {b.id} mirror_id {b.mirror_id} is not reciprocal")

    # Host-graph acyclicity: walk each chain, bail if it revisits a beam.
    for b in menu.beams:
        seen = set()
        cur = b
        while cur.host_id is not None:
            if cur.id in seen:
                raise ValueError(f"host_id graph has a cycle involving beam {cur.id}")
            seen.add(cur.id)
            cur = beam_by_id[cur.host_id]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_validate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/menu.py tests/generative/test_validate.py
git commit -m "feat: validate_menu guard for generator contracts"
```

---

## Task C2: Simple beam library (pure geometry)

**Files:**
- Create: `src/wing_design/generative/candidates.py`
- Create: `tests/generative/test_candidates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/generative/test_candidates.py`:
```python
import math

from wing_design.generative.candidates import build_beam_library
from wing_design.generative.menu import (
    CandidateMenu,
    ConflictTable,
    NodeKind,
    validate_menu,
)
from wing_design.scenario import default_scenario


def _wrap_for_validation(nodes, beams, cross_sections):
    return CandidateMenu(
        nodes=tuple(nodes), beams=tuple(beams), cross_sections=tuple(cross_sections),
        conflicts=ConflictTable(forbidden=()), coverage_targets=(), rho_kgm3=1550.0,
    )


def test_build_beam_library_is_contract_valid():
    params = default_scenario()
    nodes, beams, cross_sections = build_beam_library(params)
    # landmarks present
    kinds = {n.kind for n in nodes}
    assert NodeKind.KEEL_STEP in kinds
    assert NodeKind.DECK_STEP in kinds
    assert NodeKind.TIP in kinds
    # at least the central spar + a few branches
    assert len(beams) >= 3
    # exactly one keel-rooted spar; it carries keel->deck->tip
    spars = [b for b in beams if b.start_kind == NodeKind.KEEL_STEP]
    assert len(spars) == 1
    spar = spars[0]
    assert spar.control_points[0][2] < spar.control_points[1][2] < spar.control_points[-1][2]
    # every other beam is hosted on the spar at the deck node
    for b in beams:
        if b is spar:
            continue
        assert b.host_id == spar.id
        assert b.start_kind == NodeKind.ON_BEAM
    # the whole library satisfies validate_menu
    validate_menu(_wrap_for_validation(nodes, beams, cross_sections))


def test_build_beam_library_cross_sections_within_max():
    params = default_scenario()
    _nodes, _beams, cross_sections = build_beam_library(params)
    assert len(cross_sections) == params.generative.n_area_buckets
    for cs in cross_sections:
        assert 0 < cs.area_m2 <= params.generative.cross_section_area_max_m2


def test_build_beam_library_has_a_reciprocal_mirror_pair():
    params = default_scenario()
    _nodes, beams, _cs = build_beam_library(params)
    mirrored = [b for b in beams if b.mirror_id is not None]
    assert len(mirrored) >= 2
    by_id = {b.id: b for b in beams}
    for b in mirrored:
        assert by_id[b.mirror_id].mirror_id == b.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_candidates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.generative.candidates'`

- [ ] **Step 3: Implement build_beam_library**

Create `src/wing_design/generative/candidates.py`:
```python
"""Candidate-menu generator for the constraint-based truss stack.

`build_beam_library` builds a small, deliberately simple beam library whose every
beam satisfies the generator contracts by construction (see `validate_menu`):
monotonic-z; every keel-rooted run goes keel->deck vertically first; non-spar
beams are hosted at the deck node; landmark nodes sit exactly on beam control
points; mirror pairs are reciprocal. `build_candidate_menu` adds the real
background-FEA-driven coverage targets and the conflict table.

Curved stress-line-conformant beams are a richer drop-in library for a later
milestone; this milestone proves the end-to-end pipeline. See the design spec.
"""
from __future__ import annotations

import math

from ..scenario import DesignParameters
from .menu import (
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    CrossSectionOption,
    CrossSectionShape,
    NodeKind,
)


def _cross_section_catalog(params) -> tuple[CrossSectionOption, ...]:
    """Evenly-spaced circular area buckets up to the manufacturability max."""
    n = params.generative.n_area_buckets
    a_max = params.generative.cross_section_area_max_m2
    return tuple(
        CrossSectionOption(
            bucket=i,
            shape=CrossSectionShape.CIRCLE,
            area_m2=a_max * (i + 1) / n,
        )
        for i in range(n)
    )


def build_beam_library(params: DesignParameters):
    """Return (nodes, beams, cross_sections): the contract-correct simple library.

    Geometry: the pivot axis is at x=0, y=0. The central spar runs
    keel-step -> deck-step -> tip on the axis. Chordwise branches start ON the
    spar at the deck node, bow out to a chord offset at mid-span, and return to
    the tip. One out-of-plane (+/-y) mirror pair exercises the symmetry tie.
    """
    spec = params.geometry
    z_keel = spec.z_keel_step
    z_deck = spec.z_deck_step
    z_tip = spec.z_wing_tip
    z_mid = 0.5 * (z_deck + z_tip)

    keel = (0.0, 0.0, z_keel)
    deck = (0.0, 0.0, z_deck)
    tip = (0.0, 0.0, z_tip)

    nodes = [
        CandidateNode(id=0, xyz=keel, kind=NodeKind.KEEL_STEP, z_layer=0),
        CandidateNode(id=1, xyz=deck, kind=NodeKind.DECK_STEP, z_layer=1),
        CandidateNode(id=2, xyz=tip, kind=NodeKind.TIP, z_layer=2),
    ]

    beams: list[CandidateBeam] = []

    # Central spar: keel -> deck -> tip on the axis.
    spar = CandidateBeam(
        id=0,
        control_points=(keel, deck, tip),
        start_kind=NodeKind.KEEL_STEP,
        end_kind=NodeKind.TIP,
        start_node=0,
        end_node=2,
        length_m=(z_tip - z_keel),
        min_radius_m=math.inf,
        on_chord_plane=True,
        mirror_id=None,
        host_id=None,
        covers=(),
    )
    beams.append(spar)

    # Chordwise in-plane branches: deck -> (x_off, 0, z_mid) -> tip. The chord
    # half-extent at z_mid bounds the bow so beams stay inside the OML.
    chord_mid = spec.chord_at_z(z_mid)
    x_offsets = [0.25 * chord_mid, -0.25 * chord_mid]
    next_id = 1
    for x_off in x_offsets:
        beams.append(
            CandidateBeam(
                id=next_id,
                control_points=(deck, (x_off, 0.0, z_mid), tip),
                start_kind=NodeKind.ON_BEAM,
                end_kind=NodeKind.TIP,
                start_node=1,
                end_node=2,
                length_m=2.0 * math.hypot(x_off, 0.5 * (z_tip - z_deck)),
                min_radius_m=1.0,
                on_chord_plane=True,
                mirror_id=None,
                host_id=spar.id,
                covers=(),
            )
        )
        next_id += 1

    # One out-of-plane mirror pair: deck -> (0, +/-y_off, z_mid) -> tip.
    thickness_half = 0.5 * spec.thickness * chord_mid  # airfoil half-thickness at z_mid
    y_off = 0.4 * thickness_half
    a_id, b_id = next_id, next_id + 1
    beams.append(
        CandidateBeam(
            id=a_id, control_points=(deck, (0.0, y_off, z_mid), tip),
            start_kind=NodeKind.ON_BEAM, end_kind=NodeKind.TIP,
            start_node=1, end_node=2,
            length_m=2.0 * math.hypot(y_off, 0.5 * (z_tip - z_deck)),
            min_radius_m=1.0, on_chord_plane=False, mirror_id=b_id, host_id=spar.id,
            covers=(),
        )
    )
    beams.append(
        CandidateBeam(
            id=b_id, control_points=(deck, (0.0, -y_off, z_mid), tip),
            start_kind=NodeKind.ON_BEAM, end_kind=NodeKind.TIP,
            start_node=1, end_node=2,
            length_m=2.0 * math.hypot(y_off, 0.5 * (z_tip - z_deck)),
            min_radius_m=1.0, on_chord_plane=False, mirror_id=a_id, host_id=spar.id,
            covers=(),
        )
    )

    return nodes, beams, _cross_section_catalog(params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_candidates.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/candidates.py tests/generative/test_candidates.py
git commit -m "feat: contract-correct simple beam library generator"
```

---

## Task C3: `build_candidate_menu` (FEA-driven coverage + conflicts)

**Files:**
- Modify: `src/wing_design/generative/candidates.py`
- Modify: `tests/generative/test_candidates.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/generative/test_candidates.py`:
```python
from wing_design.generative.candidates import build_candidate_menu


def test_build_candidate_menu_runs_and_validates():
    # Uses the real shell FEA (~40 s). Smaller mesh via a coarser element size
    # keeps it fast enough for a test.
    params = default_scenario()
    menu = build_candidate_menu(params)
    assert isinstance(menu, CandidateMenu)
    assert len(menu.beams) >= 3
    assert len(menu.coverage_targets) >= 1
    # rho carried from the material
    assert math.isclose(menu.rho_kgm3, params.material.rho_kgm3, rel_tol=1e-9)
    # every coverage target is satisfiable by some beam+bucket in the menu
    max_area = max(cs.area_m2 for cs in menu.cross_sections)
    for tgt in menu.coverage_targets:
        assert tgt.required_min_area_m2 <= max_area + 1e-12
        assert len(tgt.candidate_beams) >= 1
    # the produced menu obeys all generator contracts
    validate_menu(menu)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_candidates.py::test_build_candidate_menu_runs_and_validates -v`
Expected: FAIL — `ImportError: cannot import name 'build_candidate_menu'`

- [ ] **Step 3: Implement build_candidate_menu**

Append to `src/wing_design/generative/candidates.py`:
```python
import numpy as np

from ..aero.cases import DESIGN_CASES
from ..aero.loads import run_case_lifting_line
from ..aero.model import build_airplane
from ..structural.mesh import tet_mesh_wing
from ..structural.shell import shell_mesh_from_tet_mesh, solve_shell_elastic
from .menu import ConflictTable, CoverageTarget, validate_menu


def _background_stress(params):
    """Run the shell FEA for the first design case; return (centroids, sigma_vm).

    centroids: (M,3) triangle centroids in the geometry frame; sigma_vm: (M,)
    membrane von Mises per triangle.
    """
    spec = params.geometry
    tet = tet_mesh_wing(spec, target_element_size=params.mesh.target_element_size_m)
    shell = shell_mesh_from_tet_mesh(tet)
    airplane = build_airplane(spec)
    case = DESIGN_CASES[0]
    aero = run_case_lifting_line(airplane, case,
                                 spanwise_resolution=params.aero.spanwise_resolution)
    from ..structural.projection import project_panels_to_oml_tris  # local import
    # Shell loads: project panel forces onto the shell's loaded triangles.
    tri_forces = _shell_tri_forces(shell, aero, spec.span, case.safety_factor)
    res = solve_shell_elastic(
        shell, E=params.E_iso_Pa, nu=params.nu_iso,
        thickness_m=params.skin_sizing.t_baseline_m, tri_force_vectors=tri_forces,
    )
    tri = shell.triangles
    centroids = shell.nodes[tri].mean(axis=1)
    return centroids, res.membrane_von_mises()


def _shell_tri_forces(shell, aero, span_m, safety_factor):
    """Distribute the case's spanwise normal force onto loaded shell triangles."""
    tri = shell.triangles
    centroids = shell.nodes[tri].mean(axis=1)
    areas = _tri_areas(shell.nodes, tri)
    loaded = shell.loaded_tris
    forces = np.zeros((len(tri), 3))
    z = centroids[:, 2]
    # normal force density (N/m) sampled along span at each loaded centroid
    dens = aero.distributed_normal_force(np.clip(z, 0.0, span_m))
    w = areas * loaded
    contrib = dens * w
    total = contrib.sum()
    target = aero.factored_normal_force_N
    scale = (target / total) if total > 0 else 0.0
    forces[:, 1] = contrib * scale  # apply in +Y (airfoil normal)
    return forces


def _tri_areas(nodes, tri):
    p = nodes[tri]
    return 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
    )


def _segment_point_distance(seg_a, seg_b, pts):
    """Min distance between two polylines, sampled at their points (coarse)."""
    a = np.asarray(seg_a, dtype=float)
    b = np.asarray(seg_b, dtype=float)
    dmin = math.inf
    for pa in a:
        d = np.min(np.linalg.norm(b - pa, axis=1))
        dmin = min(dmin, float(d))
    return dmin


def _coverage_targets(beams, centroids, sigma_vm, params):
    """Stress-informed coverage targets: cluster the highest-sigma triangles,
    require a section sized to the local stress, and attach every beam passing
    within a tolerance of the target centroid.
    """
    sf = params.generative.coverage_safety_factor
    sigma_allow = params.sigma_allow_Pa
    a_max = params.generative.cross_section_area_max_m2
    order = np.argsort(sigma_vm)[::-1]
    n_targets = min(5, len(order))
    tol = 0.5  # m: a beam covers a target if any control point is within tol
    targets = []
    for k in range(n_targets):
        idx = int(order[k])
        c = centroids[idx]
        # required area scales with local stress vs allowable, capped at a_max
        frac = min(1.0, float(sigma_vm[idx]) * sf / sigma_allow)
        req = max(a_max / params.generative.n_area_buckets, frac * a_max)
        covering = []
        for b in beams:
            pts = np.asarray(b.control_points, dtype=float)
            if np.min(np.linalg.norm(pts - c, axis=1)) <= tol:
                covering.append(b.id)
        if not covering:
            covering = [beams[0].id]  # spar always available as a fallback
        targets.append(
            CoverageTarget(id=k, centroid=tuple(float(v) for v in c),
                           required_min_area_m2=min(req, a_max),
                           candidate_beams=tuple(covering))
        )
    return targets


def _conflict_table(beams, cross_sections):
    """Forbid (beam_i, bucket_a, beam_j, bucket_b) where centerlines (excluding
    shared endpoints) pass closer than the sum of the two bucket radii.
    """
    forbidden = []
    for ia in range(len(beams)):
        for ib in range(ia + 1, len(beams)):
            ba, bb = beams[ia], beams[ib]
            # skip pairs that legitimately share an endpoint node
            shared = set(ba.control_points[:1] + ba.control_points[-1:]) & set(
                bb.control_points[:1] + bb.control_points[-1:]
            )
            # interior min distance: sample interior points only
            ia_pts = ba.control_points[1:-1] or ba.control_points
            ib_pts = bb.control_points[1:-1] or bb.control_points
            dist = _segment_point_distance(ia_pts, ib_pts, None)
            for ca in cross_sections:
                for cb in cross_sections:
                    if dist < (ca.radius_m + cb.radius_m) and not shared:
                        forbidden.append((ba.id, ca.bucket, bb.id, cb.bucket))
    return ConflictTable(forbidden=tuple(forbidden))


def build_candidate_menu(params: DesignParameters) -> CandidateMenu:
    """Build the full CandidateMenu: simple beam library + FEA-driven coverage
    targets + conflict table. Runs the background shell FEA (slow).
    """
    nodes, beams, cross_sections = build_beam_library(params)
    centroids, sigma_vm = _background_stress(params)
    targets = _coverage_targets(beams, centroids, sigma_vm, params)
    conflicts = _conflict_table(beams, cross_sections)
    # attach covers back onto beams (which targets each beam serves)
    covers_by_beam = {}
    for t in targets:
        for bid in t.candidate_beams:
            covers_by_beam.setdefault(bid, []).append(t.id)
    beams = [
        CandidateBeam(
            id=b.id, control_points=b.control_points, start_kind=b.start_kind,
            end_kind=b.end_kind, start_node=b.start_node, end_node=b.end_node,
            length_m=b.length_m, min_radius_m=b.min_radius_m,
            on_chord_plane=b.on_chord_plane, mirror_id=b.mirror_id, host_id=b.host_id,
            covers=tuple(covers_by_beam.get(b.id, ())),
        )
        for b in beams
    ]
    menu = CandidateMenu(
        nodes=tuple(nodes), beams=tuple(beams), cross_sections=tuple(cross_sections),
        conflicts=conflicts, coverage_targets=tuple(targets),
        rho_kgm3=params.material.rho_kgm3,
    )
    validate_menu(menu)
    return menu
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_candidates.py -v`
Expected: PASS (4 tests). The FEA test takes ~40 s.

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/candidates.py tests/generative/test_candidates.py
git commit -m "feat: build_candidate_menu with FEA-driven coverage and conflicts"
```

---

## Task C4: Aero → frame-node load lumping

**Files:**
- Create: `src/wing_design/generative/loads.py`
- Create: `tests/generative/test_loads.py`

- [ ] **Step 1: Write the failing test**

Create `tests/generative/test_loads.py`:
```python
import numpy as np

from wing_design.generative.gate import FrameModel
from wing_design.generative.loads import lump_spanwise_force_to_nodes
from wing_design.generative.menu import NodeKind


def _frame():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.5],
        [0.0, 0.0, 5.0],
    ], dtype=float)
    return FrameModel(coords=coords, elements=[(0, 1, 1e-3), (1, 2, 1e-3)],
                      node_kinds=[NodeKind.KEEL_STEP, None, NodeKind.TIP], mass_kg=1.0)


def test_lump_conserves_total_force():
    frame = _frame()
    # uniform 100 N/m over a 5 m span -> 500 N total, applied in +Y
    loads = lump_spanwise_force_to_nodes(frame, lambda z: 100.0, z_min=0.0, z_max=5.0,
                                         direction=(0.0, 1.0, 0.0))
    total_fy = sum(fy for (_fx, fy, _fz) in loads.values())
    assert np.isclose(total_fy, 500.0, rtol=1e-6)


def test_lump_only_nodes_in_span_range_get_load():
    frame = _frame()
    loads = lump_spanwise_force_to_nodes(frame, lambda z: 100.0, z_min=0.0, z_max=5.0,
                                         direction=(0.0, 1.0, 0.0))
    # the keel node at z=0 is on the boundary; the loaded nodes are within range
    for node_idx, (_fx, fy, _fz) in loads.items():
        assert fy >= 0.0
        assert 0.0 <= frame.coords[node_idx][2] <= 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_loads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.generative.loads'`

- [ ] **Step 3: Implement the lumping**

Create `src/wing_design/generative/loads.py`:
```python
"""Lump a spanwise distributed force onto frame nodes for the gate.

The aero solver gives a spanwise normal-force density; the frame gate wants
nodal point loads. We assign each in-span node a tributary share of the line
load (half the distance to each neighbor by z), in a fixed global direction.
This is the milestone's approximate aero->structure coupling (spec §7.3).
"""
from __future__ import annotations

import numpy as np


def lump_spanwise_force_to_nodes(frame, density_fn, *, z_min, z_max, direction):
    """Return {node_index: (fx, fy, fz)} lumping a line load onto in-span nodes.

    density_fn(z) -> N/m. Each node within [z_min, z_max] gets density(z) times
    its tributary length (midpoints to its sorted z-neighbors), times `direction`
    (a unit-ish 3-vector). Nodes are grouped by z so the total integrates the
    density over the span.
    """
    direction = np.asarray(direction, dtype=float)
    z = frame.coords[:, 2]
    in_span = [i for i in range(len(z)) if z_min - 1e-9 <= z[i] <= z_max + 1e-9]
    in_span.sort(key=lambda i: z[i])

    loads = {}
    for rank, i in enumerate(in_span):
        zi = z[i]
        lo = z[in_span[rank - 1]] if rank > 0 else z_min
        hi = z[in_span[rank + 1]] if rank < len(in_span) - 1 else z_max
        tributary = 0.5 * (zi - lo) + 0.5 * (hi - zi)
        f = float(density_fn(zi)) * tributary
        vec = f * direction
        loads[i] = (float(vec[0]), float(vec[1]), float(vec[2]))
    return loads
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/generative/test_loads.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/wing_design/generative/loads.py tests/generative/test_loads.py
git commit -m "feat: lump spanwise aero force onto frame nodes"
```

---

## Task C5: build123d export of a selected truss

**Files:**
- Create: `src/wing_design/generative/build.py`
- Create: `tests/generative/test_build.py`

- [ ] **Step 1: Write the failing test**

Create `tests/generative/test_build.py`:
```python
from build123d import Compound

from wing_design.generative.build import wing_candidate_to_part
from wing_design.generative.candidates import build_beam_library
from wing_design.generative.menu import (
    CandidateMenu,
    ConflictTable,
    WingCandidate,
)
from wing_design.scenario import default_scenario


def _menu():
    params = default_scenario()
    nodes, beams, cross_sections = build_beam_library(params)
    return CandidateMenu(
        nodes=tuple(nodes), beams=tuple(beams), cross_sections=tuple(cross_sections),
        conflicts=ConflictTable(forbidden=()), coverage_targets=(),
        rho_kgm3=params.material.rho_kgm3,
    )


def test_wing_candidate_to_part_builds_a_solid_per_beam():
    menu = _menu()
    # select the spar (id 0) and one branch (id 1), each at the largest bucket
    big = menu.cross_sections[-1].bucket
    candidate = WingCandidate(beam_sections=((0, big), (1, big)), mass_kg=1.0)
    part = wing_candidate_to_part(candidate, menu)
    assert isinstance(part, Compound)
    # one swept solid per selected beam
    assert len(part.solids()) == 2
    assert part.volume > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/generative/test_build.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wing_design.generative.build'`

- [ ] **Step 3: Implement wing_candidate_to_part**

Create `src/wing_design/generative/build.py`:
```python
"""Export a selected WingCandidate to build123d geometry.

Each selected beam's circular cross-section is swept along its centerline
polyline; the beams are collected into a single Compound (no boolean union, so
the export is robust). Use `export_step` / `export_stl` from build123d on the
result.
"""
from __future__ import annotations

from build123d import (
    BuildLine,
    BuildPart,
    BuildSketch,
    Circle,
    Compound,
    Plane,
    Polyline,
    Vector,
    sweep,
)


def _beam_solid(control_points, radius_m):
    pts = [tuple(float(c) for c in p) for p in control_points]
    start = Vector(*pts[0])
    nxt = Vector(*pts[1])
    tangent = (nxt - start).normalized()
    with BuildPart() as bp:
        with BuildLine():
            Polyline(*pts)
        with BuildSketch(Plane(origin=start, z_dir=tangent)):
            Circle(radius_m)
        sweep()
    return bp.part


def wing_candidate_to_part(candidate, menu) -> Compound:
    """Return a Compound with one swept circular beam per selected (beam, bucket)."""
    radius_by_bucket = {cs.bucket: cs.radius_m for cs in menu.cross_sections}
    solids = []
    for beam_id, bucket in candidate.beam_sections:
        beam = menu.beam_by_id(beam_id)
        solids.append(_beam_solid(beam.control_points, radius_by_bucket[bucket]))
    return Compound(children=solids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/generative/test_build.py -v`
Expected: PASS. If the sweep API differs in this build123d version, adjust the `sweep()` call to `sweep(sections=..., path=...)` per the installed version — but verify the test passes before moving on.

- [ ] **Step 5: Export the public API**

Edit `src/wing_design/generative/__init__.py`. After the `from .gate import ...` line add:
```python
from .build import wing_candidate_to_part
from .candidates import build_beam_library, build_candidate_menu
from .loads import lump_spanwise_force_to_nodes
from .menu import validate_menu
```
Add these names to `__all__` (keep grouped/sorted sensibly): `"build_beam_library"`, `"build_candidate_menu"`, `"lump_spanwise_force_to_nodes"`, `"validate_menu"`, `"wing_candidate_to_part"`.

- [ ] **Step 6: Verify imports**

Run:
```bash
uv run python -c "from wing_design.generative import build_candidate_menu, wing_candidate_to_part, lump_spanwise_force_to_nodes, validate_menu; print('ok')"
```
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add src/wing_design/generative/build.py tests/generative/test_build.py src/wing_design/generative/__init__.py
git commit -m "feat: build123d export of a selected truss + export generative API"
```

---

## Task C6: Example — candidate menu inspection

**Files:**
- Create: `examples/20_candidate_menu.py`

- [ ] **Step 1: Write the example**

Create `examples/20_candidate_menu.py`:
```python
"""Phase 1C: build the candidate menu from the wing and write a VTU of the beam
centerlines for ParaView inspection.

Run: just example 20_candidate_menu
"""
from pathlib import Path

import numpy as np
import meshio

from wing_design import default_scenario
from wing_design.generative import build_candidate_menu

EXPORT = Path("exports")


def main() -> None:
    params = default_scenario()
    menu = build_candidate_menu(params)

    print(f"beams:            {len(menu.beams)}")
    print(f"cross-sections:   {len(menu.cross_sections)}")
    print(f"coverage targets: {len(menu.coverage_targets)}")
    print(f"conflict tuples:  {len(menu.conflicts.forbidden)}")
    for b in menu.beams:
        print(f"  beam {b.id}: {b.start_kind.value}->{b.end_kind.value} "
              f"host={b.host_id} mirror={b.mirror_id} covers={b.covers}")

    # VTU: all beam centerlines as polylines.
    points = []
    lines = []
    for b in menu.beams:
        base = len(points)
        for p in b.control_points:
            points.append(p)
        for k in range(len(b.control_points) - 1):
            lines.append([base + k, base + k + 1])
    EXPORT.mkdir(exist_ok=True)
    out = EXPORT / "candidate_menu.vtu"
    meshio.write_points_cells(
        str(out), points=np.array(points, dtype=float), cells=[("line", np.array(lines))]
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example**

Run: `uv run python examples/20_candidate_menu.py`
Expected: prints the menu summary (beams ≥ 3, ≥ 1 coverage target) and writes `exports/candidate_menu.vtu`. Takes ~40 s (background FEA).

- [ ] **Step 3: Commit**

```bash
git add examples/20_candidate_menu.py
git commit -m "feat: example 20 — candidate menu inspection + VTU"
```

---

## Task C7: Example — end-to-end generate + gate + export (the milestone)

**Files:**
- Create: `examples/21_generate_truss.py`

- [ ] **Step 1: Write the example**

Create `examples/21_generate_truss.py`:
```python
"""Phase 1C: end-to-end thin slice. Build the candidate menu, let CP-SAT select
minimum-mass designs, judge the best with the frame gate under the nominal load
case, and export the selected truss to STEP/STL.

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
    build_frame,
    lump_spanwise_force_to_nodes,
    solve_designs,
    solve_frame,
    wing_candidate_to_part,
)
from wing_design.generative.gate import tip_node_indices

EXPORT = Path("exports")


def main() -> None:
    params = default_scenario()
    menu = build_candidate_menu(params)

    designs = solve_designs(menu, params.generative,
                            top_n=params.generative.top_n_designs)
    print(f"CP-SAT returned {len(designs)} candidate design(s)")
    if not designs:
        print("no feasible design — check menu constraints")
        return

    # Aero for the nominal case -> spanwise normal-force density.
    spec = params.geometry
    airplane = build_airplane(spec)
    case = DESIGN_CASES[0]
    aero = run_case_lifting_line(airplane, case,
                                 spanwise_resolution=params.aero.spanwise_resolution)

    chosen = None
    for d in designs:
        frame = build_frame(d, menu)
        loads = lump_spanwise_force_to_nodes(
            frame,
            lambda z: float(aero.distributed_normal_force(min(max(z, 0.0), spec.span))),
            z_min=0.0, z_max=spec.span, direction=(0.0, 1.0, 0.0),
        )
        result = solve_frame(frame, params, loads, governing_case=case.name)
        print(f"  design mass={d.mass_kg:.2f} kg  ratio={result.max_stress_ratio:.3f} "
              f"tip={result.tip_deflection_m*1000:.1f} mm  feasible={result.feasible}")
        if result.feasible:
            chosen = (d, result)
            break

    if chosen is None:
        print("no design passed the gate under the nominal case")
        return

    design, result = chosen
    part = wing_candidate_to_part(design, menu)
    EXPORT.mkdir(exist_ok=True)
    export_step(part, str(EXPORT / "generated_truss.step"))
    export_stl(part, str(EXPORT / "generated_truss.stl"))
    print(f"chosen design: mass={design.mass_kg:.2f} kg, "
          f"feasible under {result.governing_case}")
    print(f"wrote {EXPORT/'generated_truss.step'} and .stl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example (the milestone deliverable)**

Run: `uv run python examples/21_generate_truss.py`
Expected: prints the CP-SAT design count, per-design gate verdicts, and writes `exports/generated_truss.step` + `.stl`. At least one design should pass the gate (the spar at a large section under the nominal case). Takes ~40–60 s.

If no design passes the gate, that is a real result worth reporting — but first confirm the loads/areas are sane (the spar at the largest bucket under the nominal case should be comfortably feasible, like the 1B integration test). Do NOT loosen limits to force a pass; if it fails unexpectedly, report BLOCKED with the printed ratios/deflections.

- [ ] **Step 3: Commit**

```bash
git add examples/21_generate_truss.py
git commit -m "feat: example 21 — end-to-end generate + gate + export"
```

---

## Self-Review

**Spec coverage (spec §5 generator, §8 module layout, §8.4 DoD, contracts):**
- §5 candidate generator: background FEM (shell) → stress field → menu → coverage targets → conflict table → VTU artifact → Task C3, C6 ✓ (curved stress-line tracing deliberately deferred; coverage is stress-driven)
- §5 cross-section catalog (circle, area buckets) → Task C2 ✓
- §6 the menu feeds the existing CP-SAT (`solve_designs`) → Task C7 ✓
- §7.3 aero→node load lumping → Task C4, used in C7 ✓
- §8.1 module layout (`candidates.py`, `build.py`, `loads.py`, examples) → all tasks ✓
- §8.2 examples 20 + 21 → Tasks C6, C7 ✓
- §8.4 DoD (end-to-end, single case, circle sections, no wraps/loop, chord-symmetric, STEP/STL/VTU, GateResult) → Task C7 ✓
- Generator contracts (monotonic-z, acyclic keel-rooted host graph, reciprocal mirrors, landmark-exact, junction-exact) → enforced by construction (C2) and asserted by `validate_menu` (C1), called inside `build_candidate_menu` (C3) ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. The build123d `sweep()` call has an explicit fallback note (Task C5 Step 4) in case the installed API signature differs — that is a documented adaptation point, not a placeholder.

**Type consistency:** `build_beam_library` returns `(nodes, beams, cross_sections)` consumed identically in C3 and the C5 test. `build_candidate_menu(params)` returns a `CandidateMenu` consumed by `solve_designs`/`build_frame`/`wing_candidate_to_part`. `lump_spanwise_force_to_nodes` returns `{node_idx: (fx,fy,fz)}` matching `solve_frame`'s `nodal_loads`. `validate_menu` signature matches its call in C3. `solve_designs(menu, params.generative, top_n=...)` — NOTE: 1A's `solve_designs(menu, params, top_n)` expects the **GenerativeParameters** object (it reads `params.solver_max_time_s`), so the example passes `params.generative`. The gate's `solve_frame(frame, params, ...)` expects the **DesignParameters** (reads `params.material`, `params.sigma_allow_Pa`, `params.generative.tip_deflection_limit_m`). Both call sites use the correct object.

---

## Done When

- `uv run pytest tests/generative/ -v` is green (1A + 1B + the new C1–C5 tests).
- `uv run python examples/20_candidate_menu.py` writes `exports/candidate_menu.vtu` and prints a sane menu summary.
- `uv run python examples/21_generate_truss.py` runs end-to-end and writes `exports/generated_truss.step` + `.stl` with at least one gate-feasible design under the nominal case.
- Milestone 1 is complete: a constraint-based generative pipeline that produces a chord-symmetric, manufacturable, FEM-gated wing truss from the 5 m wingsail spec.
