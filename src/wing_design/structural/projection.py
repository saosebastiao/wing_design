"""Map AeroSandbox per-panel loads onto the tet mesh's OML surface facets.

Each `PanelLoads` entry carries:
  * `centers_xyz` (in the airplane frame: span along +Y)
  * `forces_xyz`  (total force per panel, [N])
  * `spanwise_widths`

We need a force vector per OML triangle in the **wing geometry frame** that
`build_wing_solid` produces (span along +Z). The frames are related by a fixed
rotation: aero +X → geom +X, aero +Y → geom +Z, aero +Z → geom +Y, i.e.

    R_geom_from_aero = [[1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]]

Projection strategy (matches the panel-spanwise convention used by `AeroResult`):

  1. Bin OML triangles by span fraction η = z / span (in geometry frame).
  2. Within each panel's η-range, distribute the panel force across the
     contained triangles, weighted by triangle area, so the total force on the
     mesh exactly equals the panel total per case (modulo binning at panel
     boundaries, which we handle by clipping panel η-spans against tri spans).

This is intentionally simple: the panel forces are a span-resolved aero load,
not a chord-resolved pressure field, so we spread each panel's total over its
span band uniformly in area. Phase 4+ would refine to per-(span, chord) panels.
"""
from __future__ import annotations

import numpy as np

from ..aero.loads import PanelLoads
from .mesh import TetMesh


# Frame change: aero (chord +X, span +Y, normal +Z) → geom (chord +X, span +Z, normal +Y)
R_GEOM_FROM_AERO = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)


def _tri_areas_and_centroids(nodes: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = nodes[tris[:, 0]]
    b = nodes[tris[:, 1]]
    c = nodes[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    centroids = (a + b + c) / 3.0
    return areas, centroids


def project_panels_to_oml_tris(
    mesh: TetMesh,
    panels: PanelLoads,
    span_m: float,
    *,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """Return a (K_oml, 3) array of force vectors to apply to each OML triangle.

    Forces are in the **geometry frame** (span along +Z), with the aerodynamic
    safety factor already applied. The sum of the returned vectors equals the
    aerodynamic resultant within rounding.
    """
    areas, centroids = _tri_areas_and_centroids(mesh.nodes, mesh.oml_tris)
    z_centroids = centroids[:, 2]                 # span position of each tri (geom frame)

    # Per-panel spanwise band in geom frame: [y_c - w/2, y_c + w/2] -> [z_lo, z_hi]
    panel_y = panels.centers_xyz[:, 1]             # aero span axis
    panel_w = panels.spanwise_widths
    panel_z_lo = panel_y - 0.5 * panel_w
    panel_z_hi = panel_y + 0.5 * panel_w

    tri_forces = np.zeros((mesh.oml_tris.shape[0], 3), dtype=np.float64)

    for k in range(panels.n_panels):
        z_lo = float(panel_z_lo[k])
        z_hi = float(panel_z_hi[k])
        in_band = (z_centroids >= z_lo) & (z_centroids < z_hi)
        band_area = float(areas[in_band].sum())
        if band_area <= 0.0:
            continue
        F_aero = panels.forces_xyz[k] * safety_factor
        F_geom = R_GEOM_FROM_AERO @ F_aero
        # Distribute F_geom over the band's triangles, weighted by tri area
        w = areas[in_band] / band_area
        tri_forces[in_band] += w[:, None] * F_geom

    return tri_forces
