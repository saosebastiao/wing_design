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

import math
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
    if L < 1e-14:
        raise ValueError(f"degenerate (zero-length) beam element: p0={p0}, p1={p1}")
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


def element_global_stiffness(p0, p1, E, G, A, I, J):
    """Global-frame 12x12 stiffness for the beam from p0 to p1: T^T k_local T."""
    T, L = beam_transform(p0, p1)
    k_local = local_beam_stiffness(E, G, A, I, J, L)
    return T.T @ k_local @ T


def solve_displacements(K, load_vec, fixed_dofs):
    """Solve K u = f with `fixed_dofs` held at zero; return the full vector u.

    Reduces to the free DOFs, solves the dense system, and scatters back.
    """
    n = K.shape[0]
    fixed = set(fixed_dofs)
    free = [d for d in range(n) if d not in fixed]
    Kff = K[np.ix_(free, free)]
    Ff = np.asarray(load_vec, dtype=float)[free]
    try:
        uf = np.linalg.solve(Kff, Ff)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "singular reduced stiffness matrix — the frame is under-constrained "
            "or has disconnected members (e.g. an ON_BEAM junction whose host has "
            "no control point at the junction coordinate)"
        ) from exc
    u = np.zeros(n)
    u[free] = uf
    return u


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
        if max_element_length_m is None or max_element_length_m <= 0:
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


def solve_frame(frame, params, nodal_loads, governing_case="nominal"):
    """Judge a frame under the given nodal loads against stress + deflection limits.

    `params` is a DesignParameters: E from the UD ply's E1, G from G12, the
    tensile allowable from `sigma_allow_Pa`, and the deflection cap from
    `generative.tip_deflection_limit_m`. `nodal_loads` maps a frame node index
    to a global (fx, fy, fz). Returns a GateResult.

    G12 (in-plane shear) is used as the torsional shear modulus; G23 (transverse
    shear, more accurate for a solid UD spar in torsion) is not in the material
    model. Adequate for this milestone-spike gate.
    """
    # The frame can only be judged if its landmark nodes were located (kinds are
    # inherited from the menu by exact coordinate match in build_frame). A missing
    # keel/deck landmark would leave the structure under-constrained and produce a
    # silent rigid-body-drift verdict, so reject it loudly instead.
    kinds = set(frame.node_kinds)
    if NodeKind.KEEL_STEP not in kinds:
        raise ValueError("frame has no KEEL_STEP node; bearing BCs cannot be applied")
    if NodeKind.DECK_STEP not in kinds:
        raise ValueError("frame has no DECK_STEP node; bearing BCs cannot be applied")
    if not tip_node_indices(frame):
        raise ValueError("frame has no TIP node; cannot evaluate tip deflection")

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

    max_ratio = float(recover_max_stress_ratio(frame, disp, E, G, sigma_allow))
    tip_def = float(tip_deflection(frame, disp))
    feasible = bool((max_ratio <= 1.0) and (tip_def <= limit))

    return GateResult(
        feasible=feasible,
        max_stress_ratio=max_ratio,
        tip_deflection_m=tip_def,
        governing_case=governing_case,
        mass_kg=frame.mass_kg,
    )
