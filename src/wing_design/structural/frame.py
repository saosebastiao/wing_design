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

_NEAR_AXIS_COS = 0.99  # |cosθ| above which +Z is too near the element axis to use as the up-vector reference


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

    @classmethod
    def annular(cls, r_outer: float, t_wall: float) -> "BeamSection":
        """Hollow circular tube (P.1 core tube). Exact annulus; t = r → solid."""
        if not (0.0 < t_wall <= r_outer):
            raise ValueError(f"need 0 < t_wall <= r_outer, got t={t_wall}, r={r_outer}")
        ri = r_outer - t_wall
        area = np.pi * (r_outer**2 - ri**2)
        second = 0.25 * np.pi * (r_outer**4 - ri**4)
        polar = 2.0 * second
        return cls(A=area, Iy=second, Iz=second, J=polar, r=r_outer)


@dataclass(frozen=True)
class FrameResult:
    displacements: np.ndarray       # (n_nodes, 6)
    axial_force: np.ndarray         # (n_elem,) tension positive [N]
    bending_moment: np.ndarray      # (n_elem,) max sqrt(My²+Mz²) over the two element ends [N·m];
                                    # a valid bending-stress proxy only when Iy == Iz (circular sections)
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
    up = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < _NEAR_AXIS_COS else np.array([0.0, 1.0, 0.0])
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
    if len(sections) != n_elem:
        raise ValueError(f"expected {n_elem} sections, got {len(sections)}")
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


def von_mises_per_element(result: FrameResult, sections: list[BeamSection]) -> np.ndarray:
    """Per-element combined-stress proxy (Pa) from a solved frame.

    sigma_n = |N|/A + M_resultant * r / Iz   (axial + bending at the outer fibre)
    tau     = |T| * r / J                     (St-Venant torsional shear)
    sigma_vm = sqrt(sigma_n**2 + 3*tau**2)

    Assumes circular sections (Iy == Iz); `M_resultant` is the resultant bending
    moment already stored on `FrameResult.bending_moment`.
    """
    A = np.array([s.A for s in sections])
    r = np.array([s.r for s in sections])
    Iz = np.array([s.Iz for s in sections])
    J = np.array([s.J for s in sections])
    sigma_n = np.abs(result.axial_force) / A + result.bending_moment * r / Iz
    tau = np.abs(result.torsion) * r / J
    return np.sqrt(sigma_n**2 + 3.0 * tau**2)
