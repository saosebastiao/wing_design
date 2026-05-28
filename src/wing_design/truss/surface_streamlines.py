"""Surface streamline tracing on a triangulated shell (Phase 5e).

Given a `ShellMesh` and a `ShellFEAResult`, walk along the per-triangle
principal-stress directions to trace **stress lines on the OML surface**.
These polylines are the centerlines of the **spar caps** of the internal
frame — the spanwise members that lie against the skin and carry the
wing-bending moment as axial tension (bottom skin) or compression (top
skin).

Algorithm
---------

1. Each triangle has two 3-D principal-direction unit vectors `e_1, e_2`
   (already computed by `ShellFEAResult.membrane_principal_dirs_3d`,
   lying in the triangle's tangent plane). Family 0 traces along `e_1`
   (max σ_1); family 1 along `e_2` (min σ_2 — i.e. max compression).
2. Step within the current triangle using simple Euler integration:
   `p_next = p + h · e_k`. If `p_next` stays inside the triangle
   (all barycentric coords ≥ 0), commit the step.
3. If `p_next` crosses an edge into a neighbor triangle, find the
   exact crossing point by interpolating barycentric coords along the
   step, commit *that* point, then switch to the neighbor.
4. Sign-align the new triangle's principal direction with the previous
   step direction (`dot ≥ 0`) to keep the streamline continuous across
   the tangent-plane change.
5. Terminate when: a step exits the mesh boundary (no neighbor); the
   principal-stress magnitude falls below `min_abs_sigma`; or the
   step count exceeds `max_steps`.

Seeding is biased toward high σ_VM triangles with a 3-D minimum spacing
so curves cover the load path without clumping.

Each seed produces two streamlines per family (forward + backward from
the seed centroid), kept as separate polylines so the caller can decide
how to merge or chain them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..structural.shell import ShellFEAResult, ShellMesh


@dataclass(frozen=True)
class SurfaceStreamline:
    """One traced polyline on the OML surface.

    `points` are 3-D coordinates, all lying on the mesh; `triangles[i]`
    is the triangle that segment `points[i] → points[i+1]` traversed;
    `sigma_along[i]` is the principal eigenvalue at the segment midpoint
    (positive in family 0 = tension; negative in family 1 = compression).
    """

    points: np.ndarray          # (n, 3)
    triangles: np.ndarray       # (n-1,) int
    sigma_along: np.ndarray     # (n-1,)
    family: int                 # 0 (σ_1, max tension) or 1 (σ_2, max compression)

    @property
    def n_segments(self) -> int:
        return int(self.triangles.shape[0])

    def length_m(self) -> float:
        if self.points.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


def _edge_adjacency(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map each undirected edge (a, b), a < b, to the triangles sharing it."""
    adj: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ti, tri in enumerate(triangles):
        for k in range(3):
            u, v = int(tri[k]), int(tri[(k + 1) % 3])
            edge = (min(u, v), max(u, v))
            adj[edge].append(ti)
    return adj


def _barycentric(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray | None:
    """Barycentric coords of p projected onto triangle (v0, v1, v2). Returns None on degenerate."""
    e0 = v1 - v0
    e1 = v2 - v0
    e2 = p - v0
    d00 = float(e0 @ e0)
    d01 = float(e0 @ e1)
    d11 = float(e1 @ e1)
    d20 = float(e2 @ e0)
    d21 = float(e2 @ e1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1.0e-30:
        return None
    inv = 1.0 / denom
    b1 = (d11 * d20 - d01 * d21) * inv
    b2 = (d00 * d21 - d01 * d20) * inv
    b0 = 1.0 - b1 - b2
    return np.array([b0, b1, b2])


def seed_high_sigma_triangles(
    shell: ShellMesh,
    fea: ShellFEAResult,
    *,
    max_seeds: int = 60,
    min_spacing_m: float = 0.20,
    sigma_floor_fraction: float = 0.05,
) -> list[int]:
    """Greedy σ-weighted Poisson-disk seeding on the OML triangles.

    Triangles are visited in descending σ_VM order; a candidate is
    accepted if no previously-accepted seed lies within `min_spacing_m`
    (3-D distance between triangle centroids). Triangles with σ_VM below
    `sigma_floor_fraction` of the p99 are skipped entirely.
    """
    sigma_vm = fea.membrane_von_mises()
    centroids = shell.nodes[shell.triangles].mean(axis=1)
    sigma_floor = sigma_floor_fraction * float(np.percentile(sigma_vm, 99))
    order = np.argsort(-sigma_vm)
    accepted: list[int] = []
    accepted_pts: list[np.ndarray] = []
    spacing_sq = min_spacing_m * min_spacing_m
    for ti in order:
        if sigma_vm[ti] < sigma_floor:
            break
        c = centroids[ti]
        ok = True
        for q in accepted_pts:
            if float((c - q) @ (c - q)) < spacing_sq:
                ok = False
                break
        if ok:
            accepted.append(int(ti))
            accepted_pts.append(c)
            if len(accepted) >= max_seeds:
                break
    return accepted


def trace_surface_streamline(
    shell: ShellMesh,
    fea: ShellFEAResult,
    *,
    start_tri: int,
    family: int,
    step_size_m: float,
    max_steps: int = 2000,
    min_abs_sigma_Pa: float = 0.0,
    direction: int = +1,
    edge_adj: dict[tuple[int, int], list[int]] | None = None,
) -> SurfaceStreamline:
    """One streamline starting at `start_tri`'s centroid, walking forward (+1) or backward (-1)."""
    if edge_adj is None:
        edge_adj = _edge_adjacency(shell.triangles)

    principal_dirs = fea.membrane_principal_dirs_3d()      # (M, 2, 3)
    principal_vals, _ = fea.membrane_principal_2d()        # (M, 2)
    tri_verts = shell.nodes[shell.triangles]                # (M, 3, 3)

    current = int(start_tri)
    p = tri_verts[current].mean(axis=0)
    initial_v = principal_dirs[current, family, :] * direction
    prev_v = initial_v.copy()

    points = [p.copy()]
    tris_visited: list[int] = []
    sigmas_along: list[float] = []

    for _ in range(max_steps):
        v_local = principal_dirs[current, family, :]
        if float(v_local @ prev_v) < 0.0:
            v_local = -v_local
        sigma = float(principal_vals[current, family])
        if abs(sigma) < min_abs_sigma_Pa:
            break

        p_next = p + step_size_m * v_local
        v0, v1, v2 = tri_verts[current]
        bary_next = _barycentric(p_next, v0, v1, v2)
        if bary_next is None:
            break

        if (bary_next >= -1.0e-9).all():
            # Stayed in this triangle — commit the step.
            tris_visited.append(current)
            sigmas_along.append(sigma)
            points.append(p_next.copy())
            prev_v = v_local
            p = p_next
            continue

        # We crossed an edge. Find which by interpolating bary_p → bary_next.
        bary_p = _barycentric(p, v0, v1, v2)
        if bary_p is None:
            break
        ts = np.full(3, np.inf)
        for ie in range(3):
            if bary_p[ie] > 0 and bary_next[ie] < 0:
                ts[ie] = bary_p[ie] / (bary_p[ie] - bary_next[ie])
        if not np.isfinite(ts.min()):
            # Degenerate — give up at this step.
            break
        edge_idx = int(np.argmin(ts))
        t = float(ts[edge_idx])
        p_cross = p + t * (p_next - p)

        # Commit the in-triangle segment to the edge-crossing point.
        tris_visited.append(current)
        sigmas_along.append(sigma)
        points.append(p_cross.copy())

        # Look up the neighbor across this edge.
        tri_node_ids = shell.triangles[current]
        n_a = int(tri_node_ids[(edge_idx + 1) % 3])
        n_b = int(tri_node_ids[(edge_idx + 2) % 3])
        edge_key = (min(n_a, n_b), max(n_a, n_b))
        candidates = [t_idx for t_idx in edge_adj[edge_key] if t_idx != current]
        if not candidates:
            # Boundary of the shell — streamline terminates.
            break
        current = int(candidates[0])
        prev_v = v_local
        p = p_cross

    return SurfaceStreamline(
        points=np.asarray(points),
        triangles=np.asarray(tris_visited, dtype=np.int64),
        sigma_along=np.asarray(sigmas_along),
        family=int(family),
    )


def trace_surface_streamlines(
    shell: ShellMesh,
    fea: ShellFEAResult,
    *,
    families: tuple[int, ...] = (0, 1),
    max_seeds: int = 60,
    min_spacing_m: float = 0.20,
    step_size_m: float | None = None,
    max_steps: int = 4000,
    min_abs_sigma_Pa: float | None = None,
    sigma_floor_fraction: float = 0.05,
) -> list[SurfaceStreamline]:
    """Seed by σ + trace both directions for each family.

    `step_size_m` defaults to about half the median triangle "diameter"
    (sqrt of triangle area), so each step stays within or near the current
    triangle. `min_abs_sigma_Pa` defaults to `sigma_floor_fraction` × p99
    of the membrane σ_VM.
    """
    edge_adj = _edge_adjacency(shell.triangles)
    sigma_vm = fea.membrane_von_mises()
    if min_abs_sigma_Pa is None:
        min_abs_sigma_Pa = sigma_floor_fraction * float(np.percentile(sigma_vm, 99))
    if step_size_m is None:
        # Roughly half a triangle "diameter"
        med_area = float(np.median(fea.element_areas))
        step_size_m = 0.5 * np.sqrt(2.0 * med_area)

    seeds = seed_high_sigma_triangles(
        shell, fea, max_seeds=max_seeds, min_spacing_m=min_spacing_m,
        sigma_floor_fraction=sigma_floor_fraction,
    )

    out: list[SurfaceStreamline] = []
    for seed_tri in seeds:
        for family in families:
            for direction in (+1, -1):
                sl = trace_surface_streamline(
                    shell, fea,
                    start_tri=seed_tri, family=family,
                    step_size_m=step_size_m, max_steps=max_steps,
                    min_abs_sigma_Pa=min_abs_sigma_Pa,
                    direction=direction,
                    edge_adj=edge_adj,
                )
                if sl.points.shape[0] >= 2:
                    out.append(sl)
    return out
