"""Combined beam + shell FEA: form-beam members plus a load-bearing skin.

Assembles the Phase-B frame's beam elements and the DKT+CST shell skin panels into
one global stiffness matrix (both use 6-DOF nodes with the same DOF ordering, so the
shell element's 18×18 global stiffness drops straight in at the shared beam nodes),
solves the clamped system, and recovers beam internal forces — returning a
`FrameResult` so the existing `summarize_frame` works unchanged. The skin replaces
the ring connectors as the transverse/torsional load path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .frame import BeamSection, FrameResult, _element_rotation, local_beam_stiffness
from .shell import tri_element_stiffness, tri_element_stiffness_laminate


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
    Degenerate (collinear) shell triangles raise from `tri_element_stiffness`.
    """
    n_nodes = nodes.shape[0]
    ndof = 6 * n_nodes
    n_beam = beam_elements.shape[0]
    if len(beam_sections) != n_beam:
        raise ValueError(f"expected {n_beam} beam_sections, got {len(beam_sections)}")

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


def _assemble_beam_shell_laminate(
    nodes: np.ndarray,
    beam_elements: np.ndarray,
    beam_sections: list[BeamSection],
    shell_tris: np.ndarray,
    *,
    E_beam: float,
    G_beam: float,
    A_skin: np.ndarray,
    D_skin: np.ndarray,
    fixed_nodes: np.ndarray,
    drilling_factor: float = 1.0e-4,
    gusset_elements: np.ndarray | None = None,
    gusset_section: "BeamSection | None" = None,
):
    """Assemble the global stiffness matrix for a beam+CLT-shell structure.

    Returns ``(K, free, fixed_dofs, ndof, n_beam, transforms, klocals)`` where
    ``K`` is a CSR sparse matrix, ``free``/``fixed_dofs`` are DOF index arrays,
    ``transforms`` and ``klocals`` are per-beam-element lists for force recovery.
    """
    A_skin = np.asarray(A_skin, dtype=float)
    D_skin = np.asarray(D_skin, dtype=float)
    a_per = A_skin.ndim == 3
    d_per = D_skin.ndim == 3
    n_nodes = nodes.shape[0]
    ndof = 6 * n_nodes
    n_beam = beam_elements.shape[0]
    if len(beam_sections) != n_beam:
        raise ValueError(f"expected {n_beam} beam_sections, got {len(beam_sections)}")

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

    # Gusset elements — assembled into K but excluded from transforms/klocals
    if gusset_elements is not None:
        if gusset_section is None:
            raise ValueError("gusset_section required when gusset_elements given")
        for e in range(gusset_elements.shape[0]):
            i, j = int(gusset_elements[e, 0]), int(gusset_elements[e, 1])
            R, L = _element_rotation(nodes[i], nodes[j])
            kloc = local_beam_stiffness(E_beam, G_beam, gusset_section, L)
            T = np.zeros((12, 12))
            for blk in range(4):
                T[3 * blk:3 * blk + 3, 3 * blk:3 * blk + 3] = R
            kg = T.T @ kloc @ T
            dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            for a_ in range(12):
                for b_ in range(12):
                    rows.append(int(dofs[a_]))
                    cols.append(int(dofs[b_]))
                    vals.append(kg[a_, b_])

    # Shell (skin) triangles — CLT laminate stiffness
    for tri_idx in range(shell_tris.shape[0]):
        n0, n1, n2 = (int(shell_tris[tri_idx, 0]),
                      int(shell_tris[tri_idx, 1]),
                      int(shell_tris[tri_idx, 2]))
        ke = tri_element_stiffness_laminate(
            nodes[n0], nodes[n1], nodes[n2],
            A=(A_skin[tri_idx] if a_per else A_skin),
            D=(D_skin[tri_idx] if d_per else D_skin),
            drilling_factor=drilling_factor,
        )
        dofs = np.r_[6 * n0:6 * n0 + 6, 6 * n1:6 * n1 + 6, 6 * n2:6 * n2 + 6]
        for a_ in range(18):
            for b_ in range(18):
                rows.append(int(dofs[a_]))
                cols.append(int(dofs[b_]))
                vals.append(ke[a_, b_])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    if len(fixed_nodes):
        fixed_dofs = np.concatenate([6 * int(fn) + np.arange(6) for fn in fixed_nodes])
    else:
        fixed_dofs = np.array([], dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed_dofs)

    return K, free, fixed_dofs, ndof, n_beam, transforms, klocals


def _recover_beam_forces(
    beam_elements: np.ndarray,
    n_beam: int,
    u: np.ndarray,
    transforms: list,
    klocals: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover per-beam-element internal forces from the displacement vector."""
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
    return axial, bending, torsion


@dataclass
class FactoredBeamShell:
    """Result of `solve_beam_shell_laminate_factored`, carrying the LU factorization."""
    result: FrameResult
    lu: object                  # splu of K_ff
    K_ff: object                # csc K_ff (for adjoint matvecs / tests)
    free: np.ndarray
    ndof: int
    u: np.ndarray               # full ndof displacement vector
    # per-element data for sensitivity assembly:
    beam_elements: np.ndarray
    transforms: list
    klocals: list


def solve_beam_shell_laminate(
    nodes: np.ndarray,
    beam_elements: np.ndarray,
    beam_sections: list[BeamSection],
    shell_tris: np.ndarray,
    *,
    E_beam: float,
    G_beam: float,
    A_skin: np.ndarray,
    D_skin: np.ndarray,
    fixed_nodes: np.ndarray,
    loads: np.ndarray,
    drilling_factor: float = 1.0e-4,
    gusset_elements: np.ndarray | None = None,
    gusset_section: "BeamSection | None" = None,
) -> FrameResult:
    """Solve a clamped beam+shell structure with a CLT laminate skin; recover per-beam internal forces.

    Mirrors `solve_beam_shell` exactly, except the skin triangles are assembled
    using `tri_element_stiffness_laminate` with the CLT membrane stiffness matrix
    ``A_skin`` and bending stiffness matrix ``D_skin`` instead of the isotropic
    (E, nu, t) parameterisation.

    ``A_skin`` may be (3,3) [N/m] — one matrix applied to all triangles — or
    (M,3,3) — one per-triangle membrane stiffness matrix.  Likewise ``D_skin``
    may be (3,3) [N·m] or (M,3,3).  Mixed shapes (one uniform, one per-tri) are
    supported.  The beam assembly, boundary-condition elimination, linear solve,
    and beam-force recovery are identical to `solve_beam_shell`, so when
    ``A_skin = t·Dm`` and ``D_skin = (t³/12)·Dm`` the two solvers produce
    numerically identical results.

    `shell_tris` is (M, 3) int node indices (may be empty (0,3)).  `loads` is (N, 6).
    Degenerate (collinear) shell triangles raise from `tri_element_stiffness_laminate`.
    """
    K, free, _fixed_dofs, ndof, n_beam, transforms, klocals = _assemble_beam_shell_laminate(
        nodes, beam_elements, beam_sections, shell_tris,
        E_beam=E_beam, G_beam=G_beam, A_skin=A_skin, D_skin=D_skin,
        fixed_nodes=fixed_nodes, drilling_factor=drilling_factor,
        gusset_elements=gusset_elements, gusset_section=gusset_section,
    )

    f = loads.reshape(-1).astype(float)

    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    disp = u.reshape(nodes.shape[0], 6)

    axial, bending, torsion = _recover_beam_forces(
        beam_elements, n_beam, u, transforms, klocals
    )

    return FrameResult(displacements=disp, axial_force=axial,
                       bending_moment=bending, torsion=torsion)


def solve_beam_shell_laminate_factored(
    nodes: np.ndarray,
    beam_elements: np.ndarray,
    beam_sections: list[BeamSection],
    shell_tris: np.ndarray,
    *,
    E_beam: float,
    G_beam: float,
    A_skin: np.ndarray,
    D_skin: np.ndarray,
    fixed_nodes: np.ndarray,
    loads: np.ndarray,
    drilling_factor: float = 1.0e-4,
    gusset_elements: np.ndarray | None = None,
    gusset_section: "BeamSection | None" = None,
) -> FactoredBeamShell:
    """Like `solve_beam_shell_laminate`, but returns a `FactoredBeamShell` with the LU factorization.

    Uses `spla.splu(K_ff)` instead of `spla.spsolve`, so the factorization can be
    reused for adjoint sensitivity solves without refactorizing per design variable.
    The returned `FactoredBeamShell.result` is numerically identical to the output
    of `solve_beam_shell_laminate` with the same arguments.
    """
    K, free, _fixed_dofs, ndof, n_beam, transforms, klocals = _assemble_beam_shell_laminate(
        nodes, beam_elements, beam_sections, shell_tris,
        E_beam=E_beam, G_beam=G_beam, A_skin=A_skin, D_skin=D_skin,
        fixed_nodes=fixed_nodes, drilling_factor=drilling_factor,
        gusset_elements=gusset_elements, gusset_section=gusset_section,
    )

    f = loads.reshape(-1).astype(float)

    K_ff = K[free][:, free].tocsc()
    lu = spla.splu(K_ff)

    u = np.zeros(ndof)
    u[free] = lu.solve(f[free])
    disp = u.reshape(nodes.shape[0], 6)

    axial, bending, torsion = _recover_beam_forces(
        beam_elements, n_beam, u, transforms, klocals
    )

    result = FrameResult(displacements=disp, axial_force=axial,
                         bending_moment=bending, torsion=torsion)

    return FactoredBeamShell(
        result=result,
        lu=lu,
        K_ff=K_ff,
        free=free,
        ndof=ndof,
        u=u,
        beam_elements=beam_elements,
        transforms=transforms,
        klocals=klocals,
    )
