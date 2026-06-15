"""Self-weight / inertial body loads from the design's own mass (V.4, backlog V#5).

The structure's mass distribution depends on the design vector (beam radii, band
thicknesses), so these loads are rebuilt every evaluate and the analytic adjoint
gains a +λᵀ·∂f/∂x term (see `body_load_jacobian` and sensitivity.py). Acceleration
vectors are expressed in the wing frame (z = span); the caller composes gravity,
heel, and slam into them (e.g. 30° heel: g·(sinθ·ŷ − cosθ·ẑ)).
"""
from __future__ import annotations

import numpy as np


def body_load_vector(model, radii, t_tri, *, rho: float, accel,
                     areas: np.ndarray | None = None,
                     extra_skin_density: np.ndarray | None = None) -> np.ndarray:
    """(N, 6) lumped nodal forces from structural mass under ``accel`` (m/s²).

    Beam element mass ρ·A·L lumps half to each end node (A = πr² from ``radii``,
    or per-element ``areas`` when given — the P.1 tube path with annular
    sections); triangle mass ρ·t·A a third to each corner. Forces only (no moment
    lumping — consistent with the aero projection). The tip gusset and tube bonds
    are massless by convention and contribute nothing.
    """
    a = np.asarray(accel, dtype=float)
    nodes = model.nodes
    out = np.zeros((nodes.shape[0], 6))
    be = model.beam_elements
    r = np.asarray(radii, dtype=float)
    for e in range(be.shape[0]):
        i, j = int(be[e, 0]), int(be[e, 1])
        L = float(np.linalg.norm(nodes[j] - nodes[i]))
        if areas is None:
            # exact original product association — regrouping via an areas array
            # is a 1-ulp change that shifts cold-start basins (P.0 ulp lesson)
            m_half = 0.5 * rho * np.pi * r[e] ** 2 * L
        else:
            m_half = 0.5 * rho * float(areas[e]) * L
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
        if extra_skin_density is not None:
            # P.3 core surface density rho_c * c [kg/m^2]
            m_third += float(extra_skin_density[m]) * area / 3.0
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
    S: int = 0,
    tube_elements: np.ndarray | None = None,
    tube_r: np.ndarray | None = None,
    tube_t: np.ndarray | None = None,
    tube_r_cols: np.ndarray | None = None,
    tube_t_cols: np.ndarray | None = None,
    nx_extra: int | None = None,
    core_rho: float | None = None,
    core_col_of_tri: np.ndarray | None = None,
    brace_elements: np.ndarray | None = None,
) -> np.ndarray:
    """∂f/∂x as a dense (ndof, nx) matrix,
    x = [r_group(G), t_band(B), f0(L), f45(L) | r_tube(S), t_wall(S)].

    Radius groups: ∂(ρπr²L)/∂r = 2ρπrL per form element, half per end node.
    Tube segments (P.1, annulus A = π(2rt − t²)): ∂A/∂r = 2πt, ∂A/∂t = 2π(r−t).
    Thickness bands: ∂(ρtA)/∂t = ρA per triangle, a third per corner. Layup
    fractions: zero (mass is fraction-independent).
    Brace elements (P.2 lateral rings): skipped — their radius is fixed in-loop
    at config.brace_r_min, so ∂f/∂brace_radius is zero here; the mass gradient
    w.r.t. brace_radius is handled separately in the mass objective.
    """
    a = np.asarray(accel, dtype=float)
    nodes = model.nodes
    ndof = 6 * nodes.shape[0]
    nx = G + B + 2 * L + (nx_extra if nx_extra is not None else 2 * S)
    dF = np.zeros((ndof, nx))
    be = model.beam_elements
    r = np.asarray(radii, dtype=float)
    tube_seg = {int(e): s for s, e in enumerate(tube_elements)} if tube_elements is not None else {}
    brace_set = (set(int(e) for e in brace_elements)
                 if brace_elements is not None else set())
    for e in range(be.shape[0]):
        i, j = int(be[e, 0]), int(be[e, 1])
        Le = float(np.linalg.norm(nodes[j] - nodes[i]))
        if e in brace_set:
            continue  # brace radius fixed in-loop; gradient handled in mass objective
        if e in tube_seg:
            s = tube_seg[e]
            rt, tt = float(tube_r[s]), float(tube_t[s])
            col_r = int(tube_r_cols[s]) if tube_r_cols is not None else G + B + 2 * L + s
            col_t = int(tube_t_cols[s]) if tube_t_cols is not None else G + B + 2 * L + S + s
            dm_r = 0.5 * rho * 2.0 * np.pi * tt * Le
            dm_t = 0.5 * rho * 2.0 * np.pi * (rt - tt) * Le
            for n in (i, j):
                dF[6 * n:6 * n + 3, col_r] += dm_r * a
                dF[6 * n:6 * n + 3, col_t] += dm_t * a
            continue
        dm_half = 0.5 * rho * 2.0 * np.pi * r[e] * Le
        g = int(group_of_element[e])
        for n in (i, j):
            dF[6 * n:6 * n + 3, g] += dm_half * a
    tris = model.shell_tris
    for m in range(tris.shape[0]):
        n0, n1, n2 = int(tris[m, 0]), int(tris[m, 1]), int(tris[m, 2])
        if core_rho is not None:
            v1c = nodes[n1] - nodes[n0]
            v2c = nodes[n2] - nodes[n0]
            area_c = 0.5 * float(np.linalg.norm(np.cross(v1c, v2c)))
            dmc = core_rho * area_c / 3.0
            for nn in (n0, n1, n2):
                dF[6 * nn:6 * nn + 3, int(core_col_of_tri[m])] += dmc * a
        v1 = nodes[n1] - nodes[n0]
        v2 = nodes[n2] - nodes[n0]
        area = 0.5 * float(np.linalg.norm(np.cross(v1, v2)))
        dm_third = rho * area / 3.0
        col = G + int(band_of_tri[m])
        for n in (n0, n1, n2):
            dF[6 * n:6 * n + 3, col] += dm_third * a
    return dF
