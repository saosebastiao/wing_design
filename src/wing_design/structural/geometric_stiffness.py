"""Geometric (stress) stiffness Kσ for linear buckling of the beam+shell structure.

Sign convention: built from the actual signed stress state (tension positive →
stabilizing, positive-semidefinite contribution; compression → destabilizing). The
linear buckling problem is then K φ = −λ Kσ φ with λ the load multiplier
(see `eigen_buckling.linear_buckling`).

Approximations (documented, validated in tests/structural/test_eigen_buckling.py):
- Beam: consistent Euler–Bernoulli geometric stiffness from the element axial force
  only (Przemieniecki); Wagner/torsional-flexural coupling neglected (circular
  sections). Column references converge to <0.3% at 8 elements.
- Shell triangle: membrane-stress geometric stiffness on the transverse-deflection
  field only, with linear (CST-style) w interpolation: Kg = t·A·Gᵀ[σ]G on the three
  local w DOFs. In-plane geometric terms and DKT-consistent w interpolation omitted
  (standard faceted-shell practice). SS-plate kc=4 reference: ~3% at a 12×12 grid.
"""
from __future__ import annotations

import numpy as np

from .frame import _element_rotation
from .shell import _triangle_local_frame


def beam_geometric_K(N: float, L: float) -> np.ndarray:
    """12×12 local geometric stiffness for axial force ``N`` (tension positive).

    DOF order matches `frame.local_beam_stiffness`: [u v w θx θy θz] × 2 nodes;
    bending xy-plane couples (v, θz) with + signs, xz-plane (w, θy) with the
    flipped coupling signs (same convention as the elastic matrix).
    """
    kg = np.zeros((12, 12))
    s = N / (30.0 * L)
    # xy-plane: v(1), θz(5), v2(7), θz2(11) — standard signs
    i1, i2, i3, i4 = 1, 5, 7, 11
    kg[i1, i1] = 36.0 * s;        kg[i1, i2] = 3.0 * L * s
    kg[i1, i3] = -36.0 * s;       kg[i1, i4] = 3.0 * L * s
    kg[i2, i2] = 4.0 * L**2 * s;  kg[i2, i3] = -3.0 * L * s
    kg[i2, i4] = -(L**2) * s
    kg[i3, i3] = 36.0 * s;        kg[i3, i4] = -3.0 * L * s
    kg[i4, i4] = 4.0 * L**2 * s
    # xz-plane: w(2), θy(4), w2(8), θy2(10) — coupling signs flipped
    i1, i2, i3, i4 = 2, 4, 8, 10
    kg[i1, i1] = 36.0 * s;        kg[i1, i2] = -3.0 * L * s
    kg[i1, i3] = -36.0 * s;       kg[i1, i4] = -3.0 * L * s
    kg[i2, i2] = 4.0 * L**2 * s;  kg[i2, i3] = 3.0 * L * s
    kg[i2, i4] = -(L**2) * s
    kg[i3, i3] = 36.0 * s;        kg[i3, i4] = 3.0 * L * s
    kg[i4, i4] = 4.0 * L**2 * s
    return kg + kg.T - np.diag(np.diag(kg))


def tri_geometric_K(p1, p2, p3, *, t: float, sigma_local: np.ndarray) -> np.ndarray:
    """18×18 global geometric stiffness of a shell triangle.

    ``sigma_local`` = [σxx, σyy, σxy] in the triangle's local frame
    (`_triangle_local_frame` convention — the frame `recover_membrane_stress_C`
    reports in). Linear-w field: Kg_w = t·A·Gᵀ[σ]G on the local w DOFs.
    """
    R, local, area = _triangle_local_frame(p1, p2, p3)
    x1, y1 = local[0]; x2, y2 = local[1]; x3, y3 = local[2]
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    Gm = np.vstack([b, c]) / (2.0 * area)              # (2,3): nodal w -> ∇w
    S = np.array([[sigma_local[0], sigma_local[2]],
                  [sigma_local[2], sigma_local[1]]])
    kg_w = t * area * (Gm.T @ S @ Gm)                  # (3,3)

    kg_local = np.zeros((18, 18))
    wdofs = [2, 8, 14]
    for a_i, da in enumerate(wdofs):
        for b_i, db in enumerate(wdofs):
            kg_local[da, db] = kg_w[a_i, b_i]

    T = np.zeros((18, 18))
    for n in range(3):
        for k in range(2):
            base = 6 * n + 3 * k
            T[base:base + 3, base:base + 3] = R
    return T @ kg_local @ T.T


def assemble_geometric_stiffness(
    nodes: np.ndarray,
    *,
    beam_elements: np.ndarray | None,
    axial_forces: np.ndarray | None,
    shell_tris: np.ndarray | None,
    t_tri: np.ndarray | None,
    tri_stress_local: np.ndarray | None,
) -> np.ndarray:
    """Global (ndof, ndof) Kσ from a solved stress state.

    ``axial_forces`` per beam element (tension positive, the `FrameResult.axial_force`
    convention); ``tri_stress_local`` (M, 3) per-triangle local membrane stress
    (the `recover_membrane_stress_C` output). Either part may be None.
    """
    ndof = 6 * nodes.shape[0]
    Kg = np.zeros((ndof, ndof))
    if beam_elements is not None:
        for e in range(beam_elements.shape[0]):
            i, j = int(beam_elements[e, 0]), int(beam_elements[e, 1])
            R, L = _element_rotation(nodes[i], nodes[j])
            T = np.kron(np.eye(4), R)
            kg = T.T @ beam_geometric_K(float(axial_forces[e]), L) @ T
            dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            Kg[np.ix_(dofs, dofs)] += kg
    if shell_tris is not None:
        for m in range(shell_tris.shape[0]):
            n0, n1, n2 = (int(shell_tris[m, 0]), int(shell_tris[m, 1]),
                          int(shell_tris[m, 2]))
            kg = tri_geometric_K(nodes[n0], nodes[n1], nodes[n2],
                                 t=float(t_tri[m]), sigma_local=tri_stress_local[m])
            dofs = np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6]
            Kg[np.ix_(dofs, dofs)] += kg
    return Kg
