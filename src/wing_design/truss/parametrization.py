"""Poisson parametrization φ: Ω → ℝ³ aligned with a frame field R(x).

Arora et al. (Volumetric Michell Trusses) place beams along integer isocurves
of a scalar field φ whose gradient matches a per-node SO(3) frame R(x). We
implement the gradient-fitting Poisson:

    minimize_φ  Σ_T V_T ‖∇φ|_T − v(T)‖²

over the tet mesh, where v(T) is the target direction at tet T (the average
of the four corner eigenvectors after sign alignment, normalized). The first-
order optimality condition is L φ = b with

    L_ij  =  Σ_T V_T (∇N_i · ∇N_j)
    b_i,k =  Σ_T V_T (∇N_i · v_k(T))

L is the standard FEM cotangent Laplacian on linear tets and is singular by a
constant shift; we pin one node to fix the gauge.

This is one scalar Poisson per family (k = 0..2). The three φ_k fields are
the global parametrization Phase 5b promises: their integer level sets
(Phase 5c) carve the wing volume into a regular truss lattice.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..structural.mesh import TetMesh
from .frame_field import PrincipalFrame


def _tet_gradients_and_volumes(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray]:
    """Per-tet shape function gradients (M, 3, 4) and signed volumes (M,)."""
    M = mesh.n_tets
    grads = np.empty((M, 3, 4), dtype=np.float64)
    vols = np.empty(M, dtype=np.float64)

    for ti, tet in enumerate(mesh.tets):
        x = mesh.nodes[tet]
        v1 = x[1] - x[0]
        v2 = x[2] - x[0]
        v3 = x[3] - x[0]
        J = np.column_stack([v1, v2, v3])
        V = float(np.linalg.det(J)) / 6.0
        # Orientation flip for inverted tets — preserves the gradient identity
        if V <= 0:
            x = x[[0, 2, 1, 3]]
            v1 = x[1] - x[0]
            v2 = x[2] - x[0]
            v3 = x[3] - x[0]
            J = np.column_stack([v1, v2, v3])
            V = float(np.linalg.det(J)) / 6.0
        JinvT = np.linalg.inv(J).T
        g = np.empty((3, 4), dtype=np.float64)
        g[:, 1:] = JinvT
        g[:, 0] = -g[:, 1:].sum(axis=1)
        grads[ti] = g
        vols[ti] = abs(V)
    return grads, vols


def _tet_targets(mesh: TetMesh, frame: PrincipalFrame) -> np.ndarray:
    """Per-tet target direction per family (M, 3, 3): columns are v_1, v_2, v_3."""
    M = mesh.n_tets
    targets = np.empty((M, 3, 3), dtype=np.float64)
    eigvecs = frame.eigenvectors

    for ti, tet in enumerate(mesh.tets):
        ref = eigvecs[int(tet[0])]            # (3, 3)
        avg = ref.copy()
        for j in (1, 2, 3):
            v = eigvecs[int(tet[j])]
            for k in range(3):
                if float(v[:, k] @ ref[:, k]) < 0.0:
                    avg[:, k] += -v[:, k]
                else:
                    avg[:, k] += v[:, k]
        avg *= 0.25  # didn't subtract first contribution, so divide by 4
        # Renormalize
        n = np.linalg.norm(avg, axis=0)
        n = np.where(n > 1e-12, n, 1.0)
        targets[ti] = avg / n
    return targets


def fit_parametrization(
    mesh: TetMesh,
    frame: PrincipalFrame,
    *,
    pin_node: int | None = None,
) -> np.ndarray:
    """Solve the gradient-fitting Poisson and return φ of shape (n_nodes, 3).

    The k-th column of the return value is φ_k, with ∇φ_k ≈ R[:, k] in the
    least-squares sense. One node (default: index 0) is pinned to φ = 0 to
    fix the additive-constant nullspace.
    """
    N = mesh.n_nodes
    M = mesh.n_tets

    grads, vols = _tet_gradients_and_volumes(mesh)            # (M, 3, 4), (M,)
    targets = _tet_targets(mesh, frame)                       # (M, 3, 3)

    # Element 4×4 stiffness K_e = V (∇N_i · ∇N_j); stack to (M, 4, 4) for sparse assembly.
    Ke = np.einsum("mki,mkj->mij", grads, grads) * vols[:, None, None]
    rows = np.broadcast_to(mesh.tets[:, :, None], (M, 4, 4))
    cols = np.broadcast_to(mesh.tets[:, None, :], (M, 4, 4))
    L = sp.coo_matrix(
        (Ke.ravel(), (rows.ravel(), cols.ravel())),
        shape=(N, N),
    ).tocsr()

    # Right-hand side b_k[i] = Σ_T V_T (∇N_i · v_k(T)); accumulate per family.
    rhs = np.zeros((N, 3), dtype=np.float64)
    # be[T, i, k] = V_T (∇N_i · v_k(T))  — note grads is (M, 3, 4), targets is (M, 3, 3)
    be = np.einsum("mki,mkr->mir", grads, targets) * vols[:, None, None]   # (M, 4, 3)
    flat_nodes = mesh.tets.ravel()
    flat_be = be.reshape(-1, 3)
    for k in range(3):
        np.add.at(rhs[:, k], flat_nodes, flat_be[:, k])

    # Pin one node to break the constant-shift nullspace
    pin = 0 if pin_node is None else int(pin_node)
    L_lil = L.tolil()
    L_lil.rows[pin] = [pin]
    L_lil.data[pin] = [1.0]
    L_pinned = L_lil.tocsr()
    L_pinned = L_pinned + 0  # densify-then-resparsify edge case guard
    rhs_pinned = rhs.copy()
    rhs_pinned[pin] = 0.0

    phi = np.empty((N, 3), dtype=np.float64)
    for k in range(3):
        phi[:, k] = spla.spsolve(L_pinned, rhs_pinned[:, k])
    return phi


def gradient_fit_residual(
    mesh: TetMesh,
    frame: PrincipalFrame,
    phi: np.ndarray,
) -> np.ndarray:
    """Per-tet ‖∇φ|_T − v(T)‖ in target units (consistency diagnostic, shape (M, 3))."""
    grads, vols = _tet_gradients_and_volumes(mesh)
    targets = _tet_targets(mesh, frame)
    # ∇φ|_T = Σ_i grads[T, :, i] phi[tet[i], k]   → (M, 3, 3)
    phi_per_tet = phi[mesh.tets]                                # (M, 4, 3)
    grad_phi = np.einsum("mki,mij->mkj", grads, phi_per_tet)    # (M, 3, 3)
    residual = np.linalg.norm(grad_phi - targets, axis=1)       # (M, 3)
    return residual
