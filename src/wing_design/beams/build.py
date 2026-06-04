"""build123d geometry for the form beams, the skin wrap, and their assembly.

Spike fidelity: beams are fixed-radius circular cross-sections inset inward from
the shell along the radial direction from the pivot, lofted along each beam
spline. The skin is the full OML solid minus its inward offset (a wall-thickness
hollow shell). Both are refined in Phase D (area-sized inward-arc sections,
beam-endpoint-driven wrap, true surface normals).
"""
from __future__ import annotations

import numpy as np
from build123d import (
    BuildPart,
    BuildSketch,
    Circle,
    Compound,
    Locations,
    Plane,
    loft,
)

from ..geometry.wing import WingSpec, build_wing_solid
from .splines import default_z_levels, form_beam_grid


def build_form_beams(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
) -> list:
    """One lofted solid per form beam (crude fixed-radius circular section)."""
    z_levels = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z_levels, n_beams)  # (n_beams, n_levels, 3)
    beams = []
    for b in range(n_beams):
        with BuildPart() as bp:
            for k in range(n_levels):
                px, py, pz = grid[b, k]
                radial = np.array([px, py])
                norm = float(np.linalg.norm(radial))
                n_out = radial / norm if norm > 1e-9 else np.array([1.0, 0.0])
                cx, cy = radial - beam_radius * n_out  # inset inward from shell
                with BuildSketch(Plane.XY.offset(float(pz))):
                    with Locations((float(cx), float(cy))):
                        Circle(beam_radius)
            loft(ruled=True)
        beams.append(bp.part)
    return beams


def build_skin_wrap(spec: WingSpec, *, wall: float = 0.003):
    """Phase-A skin: the solid OML.

    ``wall`` is accepted for forward-compatibility but not applied here. A true
    wall-thickness, load-bearing skin is built in Phase D from the beam-arc
    endpoints (the doc's intended winding-formed wrap), not by hollow-offsetting
    this lofted solid — 3D solid ``offset`` is unreliable on the morphing
    spar/transition geometry, and the offset approach is superseded anyway.
    """
    del wall  # not applied at spike fidelity; see docstring
    return build_wing_solid(spec)


def build_assembly(
    spec: WingSpec,
    *,
    n_beams: int = 16,
    n_levels: int = 20,
    beam_radius: float = 0.02,
    wall: float = 0.003,
) -> Compound:
    """Compound of the skin wrap + all form beams."""
    beams = build_form_beams(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=beam_radius
    )
    skin = build_skin_wrap(spec, wall=wall)
    for i, b in enumerate(beams):
        b.label = f"beam_{i:02d}"
    skin.label = "skin"
    return Compound(label="wingsail", children=[skin, *beams])
