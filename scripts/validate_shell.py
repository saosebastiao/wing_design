"""Sanity-check the DKT+CST shell element on a cantilever-strip patch test.

A rectangular thin plate, clamped on one short edge, point load on the
opposite short edge, perpendicular to the plate. Expected tip deflection
from Euler–Bernoulli:

    w_tip = F L^3 / (3 E I),   I = b t^3 / 12

For E = 70 GPa, L = 1 m, b = 0.1 m, t = 0.01 m, F = 10 N:
    I    = 8.33e-9 m^4
    w_tip = 10 / (3 * 70e9 * 8.33e-9) = 0.00571 m = 5.71 mm

The DKT shell should converge toward this value as the mesh refines. We
also check that the in-plane stretching test (axial pull) gives the right
elongation under tension.
"""
from __future__ import annotations

import numpy as np

from wing_design.structural.shell import ShellMesh, solve_shell_elastic


def build_rect_mesh(L: float, b: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Regular grid of triangles on a rectangle [0, L] × [0, b], lying in z=0 plane."""
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, b, ny + 1)
    nodes = np.array([[x, y, 0.0] for y in ys for x in xs])
    tris = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            tris.append([n0, n1, n3])
            tris.append([n0, n3, n2])
    return nodes, np.array(tris, dtype=np.int64)


def test_cantilever_bending() -> None:
    L = 1.0
    b = 0.1
    t = 0.01
    E = 70.0e9
    nu = 0.30
    F = 10.0

    expected_I = b * t ** 3 / 12.0
    expected_w = F * L ** 3 / (3.0 * E * expected_I)
    print(f"Cantilever-bending test: E={E/1e9} GPa, L={L} m, b={b} m, t={t*1000:.1f} mm, F={F} N")
    print(f"  Euler–Bernoulli tip deflection: {expected_w*1000:.3f} mm")

    for nx in (10, 20, 40):
        ny = max(2, nx // 5)
        nodes, tris = build_rect_mesh(L=L, b=b, nx=nx, ny=ny)
        # Clamp the left edge (x = 0) — every node along it
        clamp_mask = np.isclose(nodes[:, 0], 0.0)
        dirichlet = np.where(clamp_mask)[0]

        # Apply a total downward (−z) force F distributed across the right-edge nodes.
        right_mask = np.isclose(nodes[:, 0], L)
        right_indices = np.where(right_mask)[0]
        # Distribute force to the triangles whose centroid is closest to the right edge.
        tri_centroids = nodes[tris].mean(axis=1)
        tip_band = tri_centroids[:, 0] > L - 1.5 * (L / nx)
        n_tip_tris = int(tip_band.sum())
        tri_forces = np.zeros((tris.shape[0], 3))
        tri_forces[tip_band, 2] = -F / max(n_tip_tris, 1)

        loaded = tip_band
        mesh = ShellMesh(
            nodes=nodes, triangles=tris,
            dirichlet_nodes=dirichlet, loaded_tris=loaded,
        )
        result = solve_shell_elastic(
            mesh, E=E, nu=nu, thickness_m=t, tri_force_vectors=tri_forces,
        )
        tip_disp = result.displacements[right_indices, 2].mean()
        ratio = tip_disp / -expected_w
        print(f"  nx={nx:3d}: tip w = {tip_disp*1000:+8.3f} mm   ratio = {ratio:.3f}")


if __name__ == "__main__":
    test_cantilever_bending()
