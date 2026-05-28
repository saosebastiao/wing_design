"""RK2 streamline integration of a per-node principal direction field.

The fields we trace are eigenvector fields of a symmetric tensor — they have a
±sign ambiguity at every node. We resolve it greedily by flipping the local
vector if it disagrees with the previous step's direction (`dot < 0`).

The integrator stops when:
  * the streamline leaves the mesh (nearest-node distance exceeds a tolerance),
  * the local principal stress drops below `min_abs_eigenvalue` (no information),
  * we've taken `max_steps`,
  * the streamline turns back on itself (consecutive directions nearly anti-parallel).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..structural.mesh import TetMesh
from .frame_field import PrincipalFrame


class StreamlineIntegrator:
    """Look up principal directions / eigenvalues at arbitrary points via nearest node."""

    def __init__(self, mesh: TetMesh, frame: PrincipalFrame) -> None:
        self.nodes = mesh.nodes
        self.tree = cKDTree(mesh.nodes)
        self.eigenvectors = frame.eigenvectors
        self.eigenvalues = frame.eigenvalues
        # Use median spacing of nearest-neighbor pairs as the "in-domain" tolerance
        d_nn, _ = self.tree.query(mesh.nodes, k=2)
        self._domain_tol = float(np.median(d_nn[:, 1]) * 2.0)

    def nearest(self, point: np.ndarray) -> tuple[float, int]:
        d, i = self.tree.query(point)
        return float(d), int(i)

    def in_domain(self, point: np.ndarray) -> bool:
        d, _ = self.tree.query(point)
        return d <= self._domain_tol

    def eigenvector_at(self, point: np.ndarray, family: int) -> tuple[np.ndarray, float]:
        _, idx = self.tree.query(point)
        return self.eigenvectors[idx, :, family], float(self.eigenvalues[idx, family])


def _aligned(v: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
    if ref is None:
        return v
    return v if np.dot(v, ref) >= 0 else -v


def trace_streamline(
    start: np.ndarray,
    family: int,
    integrator: StreamlineIntegrator,
    *,
    step_size: float,
    max_steps: int = 2000,
    min_abs_eigenvalue: float = 0.0,
    initial_direction: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate one half of a streamline from `start` (call twice for both halves).

    Direction convention: the streamline initially moves along `initial_direction`
    (or the raw eigenvector at the start if None). To trace the full curve from a
    seed, call this twice with opposite `initial_direction` and concatenate.
    """
    if not integrator.in_domain(start):
        return np.array([start])

    path: list[np.ndarray] = [start.copy()]
    prev_dir = initial_direction.copy() if initial_direction is not None else None

    for _ in range(max_steps):
        p = path[-1]
        v0, lam0 = integrator.eigenvector_at(p, family)
        if abs(lam0) < min_abs_eigenvalue:
            break
        v0 = _aligned(v0, prev_dir)

        # RK2: probe at half step, re-align to the leading direction, then full step
        p_half = p + 0.5 * step_size * v0
        if not integrator.in_domain(p_half):
            break
        v_half, lam_half = integrator.eigenvector_at(p_half, family)
        if abs(lam_half) < min_abs_eigenvalue:
            break
        v_half = _aligned(v_half, v0)

        new_p = p + step_size * v_half
        if not integrator.in_domain(new_p):
            break

        step = new_p - p
        if prev_dir is not None and np.dot(step, prev_dir) < 0.0:
            # Streamline reversed — likely hit a degenerate point; stop.
            break

        path.append(new_p)
        prev_dir = step

    return np.asarray(path)
