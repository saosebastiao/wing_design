"""Boundary-seeded extraction of stress-aligned curves.

For each principal-direction family k ∈ {0, 1, 2} we:

  1. Seed points on the OML surface (loaded skin).
  2. From each seed, trace a streamline both directions along the family's
     eigenvector field through the volume.
  3. Concatenate the two halves into one polyline.

The spike returns a list of polylines per family. Phase 6 (ALP) treats these
as the centerlines of unidirectional beams and assigns each a stock
cross-section.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..structural.mesh import TetMesh
from .frame_field import PrincipalFrame
from .streamline import StreamlineIntegrator, trace_streamline


@dataclass(frozen=True)
class StreamlineFamily:
    family: int                       # 0 (max tension), 1 (mid), 2 (max compression)
    polylines: list[np.ndarray]       # each (n_pts, 3) — variable length

    @property
    def n_lines(self) -> int:
        return len(self.polylines)

    @property
    def total_length_m(self) -> float:
        return float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in self.polylines))


def mirror_family_across_chord_plane(fam: StreamlineFamily) -> StreamlineFamily:
    """Reflect every polyline across the y=0 chord plane.

    The wingsail is a symmetric airfoil functioning as a sail: it flips with
    the tack, so every (AWS, +α) load case has a mirror twin (AWS, −α) the
    structure must also handle. Because the airfoil is symmetric across y=0
    and the LL panel loads under −α are the y-reflection of those under +α,
    the σ(x) field for the mirrored case is exactly the y-reflection of the
    primary case, and so are its principal eigenvector fields and the
    streamlines that trace them. Mirroring curves we already traced is
    therefore equivalent to (and 4× cheaper than) re-running FEA on −α.
    """
    mirrored: list[np.ndarray] = []
    for poly in fam.polylines:
        m = poly.copy()
        m[:, 1] = -m[:, 1]
        mirrored.append(m)
    return StreamlineFamily(family=fam.family, polylines=mirrored)


def seed_oml_surface(mesh: TetMesh, max_seeds: int = 200) -> np.ndarray:
    """Pick up to `max_seeds` triangle centroids from the OML, area-weighted (sort of).

    For a wingsail we want seeds spread across the skin. Simple stratified sample:
    take the OML triangle centroids in order and pick every k-th one to land near
    `max_seeds` total.
    """
    n_oml = mesh.oml_tris.shape[0]
    if n_oml == 0:
        return np.empty((0, 3))
    stride = max(1, n_oml // max_seeds)
    chosen = mesh.oml_tris[::stride]
    centroids = mesh.nodes[chosen].mean(axis=1)
    return centroids


def seed_volumetric_by_stress(
    mesh: TetMesh,
    eigenvalue_magnitude: np.ndarray,
    *,
    max_seeds: int = 200,
    min_spacing_m: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Stress-weighted Poisson-disk seeding through the mesh volume.

    Picks up to `max_seeds` mesh nodes, biased toward high `eigenvalue_magnitude`,
    keeping every accepted seed at least `min_spacing_m` away from any other.
    The result lives anywhere in the volume — including the interior load
    paths near the spar attachment that OML-surface seeding entirely misses.

    `eigenvalue_magnitude` is typically `|σ_k|` for the family you're about to
    trace, so the seeds land in regions where that family actually carries load.
    `min_spacing_m` defaults to ~1/12 of the mesh bbox diagonal so the seeds
    are sparse enough to span the structure without clumping.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    weights = np.maximum(eigenvalue_magnitude.astype(float), 0.0)
    if weights.sum() <= 0:
        return np.empty((0, 3))
    if min_spacing_m is None:
        diag = float(np.linalg.norm(np.ptp(mesh.nodes, axis=0)))
        min_spacing_m = diag / 12.0

    # Importance-sample candidates without replacement (way more than max_seeds),
    # then accept greedily under the minimum-spacing constraint.
    probs = weights / weights.sum()
    n_candidates = min(mesh.n_nodes, 50 * max_seeds)
    order = rng.choice(mesh.n_nodes, size=n_candidates, replace=False, p=probs)

    accepted: list[int] = []
    accepted_coords: list[np.ndarray] = []
    min_sq = min_spacing_m * min_spacing_m
    for idx in order:
        p = mesh.nodes[idx]
        ok = True
        for q in accepted_coords:
            if float(np.dot(p - q, p - q)) < min_sq:
                ok = False
                break
        if ok:
            accepted.append(int(idx))
            accepted_coords.append(p)
            if len(accepted) >= max_seeds:
                break
    return np.asarray(accepted_coords) if accepted_coords else np.empty((0, 3))


def extract_stress_lines(
    mesh: TetMesh,
    frame: PrincipalFrame,
    *,
    family: int,
    seeds: np.ndarray,
    step_size: float | None = None,
    max_steps: int = 2000,
    min_abs_eigenvalue: float = 0.0,
) -> StreamlineFamily:
    """Trace one streamline per seed along the given eigenvector family.

    `step_size` defaults to the median nearest-neighbor spacing in the mesh.
    """
    integrator = StreamlineIntegrator(mesh, frame)
    if step_size is None:
        step_size = 0.5 * integrator._domain_tol  # half a "domain unit"

    polylines: list[np.ndarray] = []
    for s in seeds:
        if not integrator.in_domain(s):
            continue
        v_seed, lam_seed = integrator.eigenvector_at(s, family)
        if abs(lam_seed) < min_abs_eigenvalue:
            continue
        forward = trace_streamline(
            s,
            family,
            integrator,
            step_size=step_size,
            max_steps=max_steps,
            min_abs_eigenvalue=min_abs_eigenvalue,
            initial_direction=v_seed,
        )
        backward = trace_streamline(
            s,
            family,
            integrator,
            step_size=step_size,
            max_steps=max_steps,
            min_abs_eigenvalue=min_abs_eigenvalue,
            initial_direction=-v_seed,
        )
        # Concatenate: reversed backward (minus the duplicate seed) + forward
        if backward.shape[0] > 1:
            polyline = np.vstack([backward[::-1], forward[1:]])
        else:
            polyline = forward
        if polyline.shape[0] >= 2:
            polylines.append(polyline)

    return StreamlineFamily(family=family, polylines=polylines)
