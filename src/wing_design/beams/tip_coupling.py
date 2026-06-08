"""Hard tip-coupling experiment: a stiff 'gusset' tying all wing-tip beam nodes.

Models a rigid tip joint (clamp/gusset) as a clique of very stiff connector beams
among the tip nodes, appended to the model's longitudinal beams and solved by the
existing combined beam+shell solver. Lets load transfer directly beam-to-beam at the
tip (instead of only through skin shear). `gusset_radius` is the stiffness knob --
larger = closer to rigid. Returns the real-beam count so callers can slice the
original beams' internal forces (the coupling elements are not design members).

Implementation note — Z-coupling springs
-----------------------------------------
All tip nodes share the same spanwise (Z) coordinate, so horizontal gusset beam
elements couple the tip nodes in the XY-plane but their out-of-plane (Z) bending
stiffness is mediated by rotational DOFs in series, leaving the effective Z
spring constant (12EI/L³ × rotational-flexibility factor ≈ 3EI/L³) too low to
overcome the closed-tube shear already provided by the skin.  Instead, each
clique edge is modelled as a pair of structural contributions:

1. A full beam element (assembled via ``solve_beam_shell``) for in-plane and
   torsional coupling — these are the "n_beam … + clique" elements visible in the
   returned FrameResult.
2. A **scalar Z-direction penalty spring** added on top of the global stiffness
   matrix before solving: K[6i+2, 6i+2] += k_z, K[6j+2, 6j+2] += k_z,
   K[6i+2, 6j+2] -= k_z, with k_z = E_beam · π r² / L (axial stiffness of a rod
   of the same cross-section and length).  This directly couples the spanwise
   displacement of every tip-node pair without rotational-DOF mediation.

When gusset_radius is None no springs are added and no extra elements are appended,
so the result is identical to the plain solve.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..structural.frame import BeamSection, FrameResult, _element_rotation, local_beam_stiffness
from ..structural.shell import tri_element_stiffness
from .shell_model import BeamShellModel


def tip_clique_elements(tip_nodes: np.ndarray) -> np.ndarray:
    """(n*(n-1)/2, 2) all unordered node-index pairs among the tip nodes."""
    t = np.asarray(tip_nodes).astype(int)
    pairs = [(int(t[i]), int(t[j])) for i in range(t.shape[0]) for j in range(i + 1, t.shape[0])]
    return np.asarray(pairs, dtype=int)


def solve_beam_shell_tip_coupled(
    model: BeamShellModel,
    loads: np.ndarray,
    *,
    gusset_radius: float | None,
    beam_sections: list[BeamSection] | None = None,
) -> tuple[FrameResult, int]:
    """Solve the beam-shell model with a stiff tip gusset; return (result, n_beam).

    ``gusset_radius`` None => no coupling (identical to the plain solve).  Otherwise
    each pair among the tip nodes is connected by:

    * A full beam element (``BeamSection.circular(gusset_radius)``) for in-plane and
      torsional stiffness.
    * A scalar Z-direction penalty spring (stiffness E_beam · π r² / L) that directly
      enforces equal spanwise displacements across the tip ring regardless of the skin's
      existing torsional coupling.

    The first ``n_beam`` elements in the returned ``FrameResult`` are the original
    longitudinal beams; clique elements follow and are **not** design members.
    """
    n_beam = model.beam_elements.shape[0]
    if beam_sections is None:
        beam_sections = [model.section] * n_beam

    clique: np.ndarray | None = None
    if gusset_radius is not None:
        clique = tip_clique_elements(model.tip_nodes)
        gusset_sec = BeamSection.circular(float(gusset_radius))
        all_elements = np.vstack([model.beam_elements, clique])
        all_sections = list(beam_sections) + [gusset_sec] * clique.shape[0]
    else:
        all_elements = model.beam_elements
        all_sections = list(beam_sections)

    # --- Assemble global stiffness matrix (beams + skin) -----------------
    n_nodes = model.nodes.shape[0]
    ndof = 6 * n_nodes
    n_elem = all_elements.shape[0]

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    transforms: list[np.ndarray] = []
    klocals: list[np.ndarray] = []

    for e in range(n_elem):
        i, j = int(all_elements[e, 0]), int(all_elements[e, 1])
        R, L = _element_rotation(model.nodes[i], model.nodes[j])
        kloc = local_beam_stiffness(model.E_beam, model.G_beam, all_sections[e], L)
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

    for t in range(model.shell_tris.shape[0]):
        n0, n1, n2 = (int(model.shell_tris[t, 0]),
                      int(model.shell_tris[t, 1]),
                      int(model.shell_tris[t, 2]))
        ke = tri_element_stiffness(
            model.nodes[n0], model.nodes[n1], model.nodes[n2],
            E=model.E_skin, nu=model.nu_skin, t=model.t_skin,
        )
        dofs = np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6]
        for a_ in range(18):
            for b_ in range(18):
                rows.append(int(dofs[a_]))
                cols.append(int(dofs[b_]))
                vals.append(ke[a_, b_])

    # --- Z-direction penalty springs for gusset clique -------------------
    if clique is not None:
        A_gusset = math.pi * float(gusset_radius) ** 2
        for pair in clique:
            pi_idx, pj_idx = int(pair[0]), int(pair[1])
            L = float(np.linalg.norm(model.nodes[pi_idx] - model.nodes[pj_idx]))
            k_z = model.E_beam * A_gusset / L  # axial stiffness of a rod of this section
            doi = 6 * pi_idx + 2   # uz DOF of node i
            doj = 6 * pj_idx + 2   # uz DOF of node j
            rows.extend([doi, doj, doi, doj])
            cols.extend([doi, doj, doj, doi])
            vals.extend([k_z, k_z, -k_z, -k_z])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # --- Boundary conditions and solve -----------------------------------
    f = loads.reshape(-1).astype(float)
    if len(model.fixed_nodes):
        fixed_dofs = np.concatenate([6 * int(fn) + np.arange(6) for fn in model.fixed_nodes])
    else:
        fixed_dofs = np.array([], dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)

    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    disp = u.reshape(n_nodes, 6)

    # --- Recover beam internal forces (original beams only) --------------
    axial = np.zeros(n_elem)
    bending = np.zeros(n_elem)
    torsion = np.zeros(n_elem)
    for e in range(n_elem):
        i, j = int(all_elements[e, 0]), int(all_elements[e, 1])
        ue = np.r_[u[6 * i:6 * i + 6], u[6 * j:6 * j + 6]]
        floc = klocals[e] @ (transforms[e] @ ue)
        axial[e] = floc[6]
        bending[e] = max(np.hypot(floc[4], floc[5]), np.hypot(floc[10], floc[11]))
        torsion[e] = floc[9]

    return FrameResult(displacements=disp, axial_force=axial,
                       bending_moment=bending, torsion=torsion), n_beam
