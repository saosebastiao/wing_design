"""Linear (eigenvalue) buckling solve: K φ = −λ Kσ φ on the free DOFs.

λ is the multiplier on the *stress state* that produced Kσ: the structure buckles
at λ × (the applied load). Solved dense via Cholesky reduction to a standard
symmetric eigenproblem — the sizing meshes are small (medium 16×8 ≈ 800 free DOFs),
where dense is both trivial and far more robust than sparse shift-invert near
clustered modes. Revisit sparse only at a measured need (V.0 discipline).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

LAMBDA_CAP = 1.0e6   # multipliers above this are reported as "no buckling"


def linear_buckling(K_ff, Kg_ff, *, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Smallest positive buckling multipliers of (K + λ Kσ) φ = 0.

    Parameters: free-DOF partitions of the elastic stiffness ``K_ff`` (must be
    positive definite) and geometric stiffness ``Kg_ff`` (sign convention:
    tension-stabilizing positive — `geometric_stiffness` builds it that way).
    Dense or sparse accepted.

    Returns ``(lams, modes)``: up to ``k`` multipliers ascending (may be empty if
    nothing buckles below LAMBDA_CAP) and the corresponding mode vectors as columns
    (free-DOF basis, unit 2-norm).
    """
    Kd = K_ff.toarray() if sp.issparse(K_ff) else np.asarray(K_ff, dtype=float)
    Gd = Kg_ff.toarray() if sp.issparse(Kg_ff) else np.asarray(Kg_ff, dtype=float)
    Kd = 0.5 * (Kd + Kd.T)
    Gd = 0.5 * (Gd + Gd.T)

    # K = L Lᵀ; K φ = −λ Kσ φ  →  (L⁻¹ (−Kσ) L⁻ᵀ) ψ = (1/λ) ψ,  ψ = Lᵀ φ.
    L = sla.cholesky(Kd, lower=True)
    B = sla.solve_triangular(L, Gd, lower=True)
    Bt = sla.solve_triangular(L, B.T, lower=True)      # L⁻¹ Kσ L⁻ᵀ
    Bt = -0.5 * (Bt + Bt.T)                            # −Kσ reduced, symmetrized
    mu, psi = sla.eigh(Bt)                             # ascending

    pos = mu > 1.0 / LAMBDA_CAP                        # μ = 1/λ > 0 → finite buckling
    mu_pos = mu[pos][::-1]                             # largest μ first = smallest λ
    psi_pos = psi[:, pos][:, ::-1]
    lams = 1.0 / mu_pos[:k]
    modes = sla.solve_triangular(L, psi_pos[:, :k], lower=True, trans="T")
    norms = np.linalg.norm(modes, axis=0)
    modes = modes / np.where(norms > 0.0, norms, 1.0)
    return lams, modes
