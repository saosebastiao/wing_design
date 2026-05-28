"""Linear elastic 3D FEA on tetrahedral meshes (Phase 4 spike).

Isotropic-equivalent CFRP per Arora et al.: a quasi-isotropic E with the same
mass density as the chosen UD ply, applied uniformly through the volume. Real
anisotropic homogenization (skin + beams) lives in Phase 8.

Implementation notes:

  * Linear (constant-strain) tetrahedra; one Gauss point per element. B is
    constant per element, K_e = V_e · Bᵀ D B.
  * Assembly via SciPy COO triplets → CSR for the solve. Solve uses
    `scipy.sparse.linalg.spsolve` (sparse LU). For our 5 m wing at ~3 cm
    elements that's ~10⁵ DOF — perfectly fine on a single core.
  * Stress recovery: σ_e = D B u_e per element. Nodal stresses are
    volume-weighted averages over the incident tets (Loubignac smoothing).
  * BCs: Dirichlet `u = 0` on the spar-base disk (the mast bearing); surface
    tractions on the OML facets supplied as one (3,) force vector per facet,
    distributed evenly to its three nodes (consistent for constant traction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh import TetMesh


@dataclass(frozen=True)
class FEAResult:
    displacements: np.ndarray         # (N, 3)
    stress_per_tet: np.ndarray        # (M, 6) Voigt: σxx σyy σzz σyz σxz σxy
    nodal_stress: np.ndarray          # (N, 6) Loubignac-smoothed
    element_volumes: np.ndarray       # (M,)

    @property
    def max_displacement_m(self) -> float:
        return float(np.linalg.norm(self.displacements, axis=1).max())

    @property
    def von_mises_per_tet(self) -> np.ndarray:
        s = self.stress_per_tet
        return _von_mises_voigt(s)

    @property
    def nodal_von_mises(self) -> np.ndarray:
        return _von_mises_voigt(self.nodal_stress)


def isotropic_D_matrix(E: float, nu: float) -> np.ndarray:
    """6x6 elasticity matrix (Voigt, engineering shear). Ordering: xx, yy, zz, yz, xz, xy."""
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lam = c * nu
    mu_eff = c * (1.0 - 2.0 * nu) / 2.0     # = E / (2(1+ν))
    diag = c * (1.0 - nu)
    D = np.array(
        [
            [diag, lam, lam, 0, 0, 0],
            [lam, diag, lam, 0, 0, 0],
            [lam, lam, diag, 0, 0, 0],
            [0, 0, 0, mu_eff, 0, 0],
            [0, 0, 0, 0, mu_eff, 0],
            [0, 0, 0, 0, 0, mu_eff],
        ],
        dtype=float,
    )
    return D


def _tet_B_and_volume(coords4x3: np.ndarray) -> tuple[np.ndarray, float]:
    """Strain-displacement matrix B (6×12) and volume V for a linear tet."""
    v1 = coords4x3[1] - coords4x3[0]
    v2 = coords4x3[2] - coords4x3[0]
    v3 = coords4x3[3] - coords4x3[0]
    J = np.column_stack([v1, v2, v3])     # 3×3
    V = np.linalg.det(J) / 6.0
    if V <= 0:
        # Flip orientation so V > 0
        coords4x3 = coords4x3[[0, 2, 1, 3]]
        v1 = coords4x3[1] - coords4x3[0]
        v2 = coords4x3[2] - coords4x3[0]
        v3 = coords4x3[3] - coords4x3[0]
        J = np.column_stack([v1, v2, v3])
        V = np.linalg.det(J) / 6.0
    # Shape function gradients in global coords: ∇_x N_i = J⁻ᵀ ∇_ξ N_i.
    # For i = 1, 2, 3, ∇_ξ N_i is a unit vector, so ∇_x N_i is the i-th column of J⁻ᵀ
    # (equivalently the i-th row of J⁻¹). ∇_x N_0 = -Σ ∇_x N_i.
    JinvT = np.linalg.inv(J).T
    grads = np.zeros((3, 4))
    grads[:, 1:] = JinvT
    grads[:, 0] = -grads[:, 1:].sum(axis=1)
    B = np.zeros((6, 12))
    for i in range(4):
        bx, by, bz = grads[:, i]
        B[0, 3 * i + 0] = bx
        B[1, 3 * i + 1] = by
        B[2, 3 * i + 2] = bz
        B[3, 3 * i + 1] = bz
        B[3, 3 * i + 2] = by
        B[4, 3 * i + 0] = bz
        B[4, 3 * i + 2] = bx
        B[5, 3 * i + 0] = by
        B[5, 3 * i + 1] = bx
    return B, V


def _von_mises_voigt(s: np.ndarray) -> np.ndarray:
    """Von Mises stress for stress tensors in Voigt form (..., 6)."""
    sxx, syy, szz, syz, sxz, sxy = (s[..., i] for i in range(6))
    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (syz**2 + sxz**2 + sxy**2)
    return np.sqrt(np.maximum(vm2, 0.0))


def _assemble_stiffness(
    mesh: TetMesh,
    D: np.ndarray,
) -> tuple[sp.csr_matrix, list[np.ndarray], np.ndarray, np.ndarray]:
    """Assemble global K plus per-element B matrices, volumes, and the D used.

    `D` is either a single 6×6 matrix (used for every tet) or a (M, 6, 6) stack
    of per-tet D matrices — the latter lets `solve_linear_elastic` model
    spatially-varying material (e.g. the thin-shell / foam-core sandwich in
    `structural.sandwich`).
    """
    N = mesh.n_nodes
    M = mesh.n_tets
    per_tet = (D.ndim == 3)
    if per_tet and D.shape != (M, 6, 6):
        raise ValueError(f"per-tet D must have shape ({M}, 6, 6), got {D.shape}")

    rows = np.empty(M * 144, dtype=np.int64)
    cols = np.empty(M * 144, dtype=np.int64)
    vals = np.empty(M * 144, dtype=np.float64)
    Bs: list[np.ndarray] = []
    Vs = np.empty(M, dtype=np.float64)
    Ds: np.ndarray = D if per_tet else np.broadcast_to(D, (M, 6, 6))

    for i, tet in enumerate(mesh.tets):
        x = mesh.nodes[tet]
        B, V = _tet_B_and_volume(x)
        D_e = Ds[i]
        K_e = V * (B.T @ D_e @ B)
        Bs.append(B)
        Vs[i] = V
        dofs = np.empty(12, dtype=np.int64)
        for k in range(4):
            base = 3 * int(tet[k])
            dofs[3 * k] = base
            dofs[3 * k + 1] = base + 1
            dofs[3 * k + 2] = base + 2
        rr = np.repeat(dofs, 12)
        cc = np.tile(dofs, 12)
        s = 144 * i
        rows[s : s + 144] = rr
        cols[s : s + 144] = cc
        vals[s : s + 144] = K_e.ravel()

    K = sp.coo_matrix((vals, (rows, cols)), shape=(3 * N, 3 * N)).tocsr()
    return K, Bs, Vs, Ds


def solve_linear_elastic(
    mesh: TetMesh,
    *,
    E: float | np.ndarray,
    nu: float,
    tri_force_vectors: np.ndarray,
) -> FEAResult:
    """Solve K u = f for a tet mesh with surface tractions and a clamped Dirichlet region.

    Parameters
    ----------
    mesh : TetMesh
        Output of `mesh.tet_mesh_wing`. `mesh.dirichlet_nodes` are clamped.
    E : float or (M,) array
        Young's modulus, either a single scalar (used for every tet) or a
        per-tet array — the latter lets you model spatially-varying material
        such as the thin-shell / foam-core sandwich in `structural.sandwich`.
    nu : float
        Isotropic-equivalent Poisson ratio (constant across the volume).
    tri_force_vectors : (K_oml, 3) float
        Total force vector applied to each `mesh.oml_tris` facet [N]. Distributed
        evenly to the three vertices of each facet (consistent for constant
        traction over the facet).
    """
    if tri_force_vectors.shape != (mesh.oml_tris.shape[0], 3):
        raise ValueError(
            f"tri_force_vectors must be ({mesh.oml_tris.shape[0]}, 3), got {tri_force_vectors.shape}"
        )

    E_arr = np.asarray(E, dtype=float)
    if E_arr.ndim == 0:
        D = isotropic_D_matrix(float(E_arr), nu)
    else:
        if E_arr.shape != (mesh.n_tets,):
            raise ValueError(f"per-tet E must have shape ({mesh.n_tets},), got {E_arr.shape}")
        D = np.stack([isotropic_D_matrix(float(e), nu) for e in E_arr])
    K, Bs, Vs, Ds = _assemble_stiffness(mesh, D)
    N = mesh.n_nodes

    # Distribute facet forces equally to facet vertices
    f = np.zeros(3 * N, dtype=np.float64)
    third = tri_force_vectors / 3.0
    for tri, F3 in zip(mesh.oml_tris, third):
        for n in tri:
            base = 3 * int(n)
            f[base] += F3[0]
            f[base + 1] += F3[1]
            f[base + 2] += F3[2]

    # Dirichlet: u = 0 on the spar-base disk
    free = np.ones(3 * N, dtype=bool)
    for n in mesh.dirichlet_nodes:
        base = 3 * int(n)
        free[base : base + 3] = False

    K_ff = K[free][:, free]
    u = np.zeros(3 * N, dtype=np.float64)
    u[free] = spla.spsolve(K_ff.tocsc(), f[free])

    displacements = u.reshape(N, 3)

    # Stress recovery per element (uses each tet's own D matrix)
    stress = np.zeros((mesh.n_tets, 6), dtype=np.float64)
    for i, (B, tet) in enumerate(zip(Bs, mesh.tets)):
        u_e = displacements[tet].ravel()
        stress[i] = Ds[i] @ (B @ u_e)

    # Loubignac (volume-weighted) nodal smoothing
    nodal_stress = np.zeros((N, 6), dtype=np.float64)
    weight = np.zeros(N, dtype=np.float64)
    for i, tet in enumerate(mesh.tets):
        V = Vs[i]
        nodal_stress[tet] += V * stress[i]
        weight[tet] += V
    nodal_stress /= np.maximum(weight[:, None], 1.0e-30)

    return FEAResult(
        displacements=displacements,
        stress_per_tet=stress,
        nodal_stress=nodal_stress,
        element_volumes=Vs,
    )
