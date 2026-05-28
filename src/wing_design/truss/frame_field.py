"""Per-node principal stress frame field.

Computes eigendecomposition of the Cauchy stress tensor at every mesh node and
returns it sorted as (σ₁ ≥ σ₂ ≥ σ₃) — i.e. e₁ is the max-tension direction
(positive σ₁) and e₃ is the max-compression direction (most negative σ₃).

The spike pass uses the raw Loubignac-smoothed nodal stress directly. The
faithful Arora et al. approach replaces this with a Laplacian-smoothed
SO(3) frame field; that's Phase 5b.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrincipalFrame:
    eigenvalues: np.ndarray   # (N, 3) sorted descending: σ_1 ≥ σ_2 ≥ σ_3
    eigenvectors: np.ndarray  # (N, 3, 3) unit; column k is the eigenvector for σ_{k+1}

    @property
    def n_nodes(self) -> int:
        return int(self.eigenvalues.shape[0])

    def eigenvector(self, family: int) -> np.ndarray:
        """(N, 3) array of unit eigenvectors for the given family (0, 1, or 2)."""
        return self.eigenvectors[..., family]

    def eigenvalue(self, family: int) -> np.ndarray:
        return self.eigenvalues[..., family]


def voigt_to_tensor(s: np.ndarray) -> np.ndarray:
    """(..., 6) Voigt → (..., 3, 3) symmetric tensor.

    Voigt ordering: σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy.
    """
    sxx = s[..., 0]
    syy = s[..., 1]
    szz = s[..., 2]
    syz = s[..., 3]
    sxz = s[..., 4]
    sxy = s[..., 5]
    T = np.empty(s.shape[:-1] + (3, 3), dtype=s.dtype)
    T[..., 0, 0] = sxx
    T[..., 1, 1] = syy
    T[..., 2, 2] = szz
    T[..., 1, 2] = T[..., 2, 1] = syz
    T[..., 0, 2] = T[..., 2, 0] = sxz
    T[..., 0, 1] = T[..., 1, 0] = sxy
    return T


def principal_frame_from_voigt(nodal_stress_voigt: np.ndarray) -> PrincipalFrame:
    """Eigendecompose nodal stress (Voigt form) and return descending-sorted frames."""
    T = voigt_to_tensor(nodal_stress_voigt)
    # np.linalg.eigh returns ascending eigenvalues; reverse to descending.
    eigvals, eigvecs = np.linalg.eigh(T)
    eigvals = eigvals[..., ::-1]
    eigvecs = eigvecs[..., ::-1]
    return PrincipalFrame(eigenvalues=eigvals.copy(), eigenvectors=eigvecs.copy())


def _node_adjacency(tets: np.ndarray, n_nodes: int) -> list[list[int]]:
    """Adjacency list: which nodes share a tet with each node."""
    neighbors: list[set[int]] = [set() for _ in range(n_nodes)]
    for tet in tets:
        a, b, c, d = (int(t) for t in tet)
        for u, v in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            neighbors[u].add(v)
            neighbors[v].add(u)
    return [sorted(s) for s in neighbors]


def align_signs_bfs(frame: PrincipalFrame, tets: np.ndarray) -> PrincipalFrame:
    """BFS sign-alignment of the per-node eigenvector field.

    Each eigenvector has an ambiguous ±sign at every node. We pick the sign at
    a high-σ root node, then propagate aligned signs to neighbors via BFS so
    adjacent nodes agree on direction. The result is the same eigenvalue field
    (untouched) but with eigenvectors continuous across the mesh — streamlines
    traced through this field don't suffer from per-step sign-flip noise.

    This is the first ingredient of the Arora et al. SO(3) frame field. A
    proper SO(3) fit (Phase 5c) additionally couples the three families and
    smooths under a Laplacian energy; BFS sign-alignment is the lazy version
    that handles the dominant failure mode (random sign flips) without an
    optimization solve.
    """
    n_nodes = frame.n_nodes
    eigvecs = frame.eigenvectors.copy()

    neighbors = _node_adjacency(tets, n_nodes)

    # Seed at the node where σ_1 is strongest — most reliable sign anchor.
    root = int(np.argmax(np.abs(frame.eigenvalues[:, 0])))
    visited = np.zeros(n_nodes, dtype=bool)
    visited[root] = True
    frontier = [root]

    while frontier:
        next_frontier: list[int] = []
        for u in frontier:
            parent_vecs = eigvecs[u]  # (3, 3): columns are the three families
            for v in neighbors[u]:
                if visited[v]:
                    continue
                vv = eigvecs[v]
                for k in range(3):
                    if float(vv[:, k] @ parent_vecs[:, k]) < 0.0:
                        vv[:, k] = -vv[:, k]
                eigvecs[v] = vv
                visited[v] = True
                next_frontier.append(v)
        frontier = next_frontier

    return PrincipalFrame(eigenvalues=frame.eigenvalues, eigenvectors=eigvecs)
