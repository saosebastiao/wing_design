"""Build the wingsail outer solid (lofted airfoil + rotation spar) with build123d."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from build123d import (
    Align,
    Axis,
    BuildLine,
    BuildPart,
    BuildSketch,
    Cylinder,
    Locations,
    Mode,
    Plane,
    Polyline,
    add,
    loft,
    make_face,
)

from .airfoil import naca_00xx_coords


@dataclass(frozen=True)
class WingSpec:
    span: float = 5.0           # m, root to tip
    root_chord: float = 1.0     # m
    tip_chord: float = 0.6      # m
    thickness: float = 0.18     # NACA00xx t/c
    pivot_frac: float = 0.25    # x/c of the rotation axis at every station
    spar_length: float = 0.75   # m, below the root section
    spar_radius: float = 0.05   # m
    n_sections: int = 5
    n_airfoil_points: int = 160


def _airfoil_face(chord: float, thickness: float, pivot_frac: float, n_pts: int):
    coords = naca_00xx_coords(thickness, n=n_pts) - np.array([pivot_frac, 0.0])
    coords = coords * chord
    pts = [(float(x), float(y)) for x, y in coords]
    with BuildSketch() as sk:
        with BuildLine():
            Polyline(*pts, close=True)
        make_face()
    return sk.sketch


def build_wing_solid(spec: WingSpec = WingSpec()):
    """Loft a wingsail solid between `n_sections` airfoil stations and attach a spar stub.

    Axes: chord along +X, semi-span along +Z, airfoil normal along +Y.
    The pivot axis (`pivot_frac`*chord) is the line X=0, Y=0.
    """
    sections = []
    for i in range(spec.n_sections):
        frac = i / (spec.n_sections - 1)
        chord = spec.root_chord + frac * (spec.tip_chord - spec.root_chord)
        face_xy = _airfoil_face(chord, spec.thickness, spec.pivot_frac, spec.n_airfoil_points)
        plane = Plane(origin=(0, 0, frac * spec.span))
        sections.append(plane * face_xy)

    with BuildPart() as part:
        for face in sections:
            add(face)
        loft()
        with Locations((0.0, 0.0, -spec.spar_length)):
            Cylinder(
                radius=spec.spar_radius,
                height=spec.spar_length,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.ADD,
            )
    return part.part
