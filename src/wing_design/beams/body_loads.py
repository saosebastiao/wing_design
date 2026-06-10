"""Self-weight / inertial body loads from the design's own mass (V.4, backlog V#5).

The structure's mass distribution depends on the design vector (beam radii, band
thicknesses), so these loads are rebuilt every evaluate and the analytic adjoint
gains a +λᵀ·∂f/∂x term (see `body_load_jacobian` and sensitivity.py). Acceleration
vectors are expressed in the wing frame (z = span); the caller composes gravity,
heel, and slam into them (e.g. 30° heel: g·(sinθ·ŷ − cosθ·ẑ)).
"""
from __future__ import annotations

import numpy as np


def body_load_vector(model, radii, t_tri, *, rho: float, accel) -> np.ndarray:
    """(N, 6) lumped nodal forces from structural mass under ``accel`` (m/s²).

    Beam element mass ρπr²L lumps half to each end node; triangle mass ρ·t·A a
    third to each corner. Forces only (no moment lumping — consistent with the
    aero projection). The tip gusset is massless by convention and contributes
    nothing.
    """
    a = np.asarray(accel, dtype=float)
    nodes = model.nodes
    out = np.zeros((nodes.shape[0], 6))
    be = model.beam_elements
    r = np.asarray(radii, dtype=float)
    for e in range(be.shape[0]):
        i, j = int(be[e, 0]), int(be[e, 1])
        L = float(np.linalg.norm(nodes[j] - nodes[i]))
        m_half = 0.5 * rho * np.pi * r[e] ** 2 * L
        out[i, :3] += m_half * a
        out[j, :3] += m_half * a
    tris = model.shell_tris
    t = np.asarray(t_tri, dtype=float)
    for m in range(tris.shape[0]):
        n0, n1, n2 = int(tris[m, 0]), int(tris[m, 1]), int(tris[m, 2])
        v1 = nodes[n1] - nodes[n0]
        v2 = nodes[n2] - nodes[n0]
        area = 0.5 * float(np.linalg.norm(np.cross(v1, v2)))
        m_third = rho * t[m] * area / 3.0
        for n in (n0, n1, n2):
            out[n, :3] += m_third * a
    return out


def body_load_jacobian(
    model,
    radii,
    *,
    group_of_element,
    band_of_tri,
    rho: float,
    accel,
    G: int,
    B: int,
    L: int,
) -> np.ndarray:
    """∂f/∂x as a dense (ndof, nx) matrix, x = [r_group(G), t_band(B), f0(L), f45(L)].

    Radius groups: ∂(ρπr²L)/∂r = 2ρπrL per element, half to each end node's
    translational DOFs. Thickness bands: ∂(ρtA)/∂t = ρA per triangle, a third per
    corner. Layup fractions: zero (mass is fraction-independent).
    """
    a = np.asarray(accel, dtype=float)
    nodes = model.nodes
    ndof = 6 * nodes.shape[0]
    nx = G + B + 2 * L
    dF = np.zeros((ndof, nx))
    be = model.beam_elements
    r = np.asarray(radii, dtype=float)
    for e in range(be.shape[0]):
        i, j = int(be[e, 0]), int(be[e, 1])
        Le = float(np.linalg.norm(nodes[j] - nodes[i]))
        dm_half = 0.5 * rho * 2.0 * np.pi * r[e] * Le
        g = int(group_of_element[e])
        for n in (i, j):
            dF[6 * n:6 * n + 3, g] += dm_half * a
    tris = model.shell_tris
    for m in range(tris.shape[0]):
        n0, n1, n2 = int(tris[m, 0]), int(tris[m, 1]), int(tris[m, 2])
        v1 = nodes[n1] - nodes[n0]
        v2 = nodes[n2] - nodes[n0]
        area = 0.5 * float(np.linalg.norm(np.cross(v1, v2)))
        dm_third = rho * area / 3.0
        col = G + int(band_of_tri[m])
        for n in (n0, n1, n2):
            dF[6 * n:6 * n + 3, col] += dm_third * a
    return dF
