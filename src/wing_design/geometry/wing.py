"""Build the wingsail outer solid (lofted airfoil + fairing + rotation spar) with build123d.

Axes: chord along +X, span along +Z, airfoil normal along +Y. The pivot axis
(`pivot_frac` * chord) lies on X=0, Y=0.

Z layout (top to bottom):
    z = span                          tip airfoil (chord = tip_chord)
    ...
    z = 0                             root airfoil (chord = root_chord, t = 0)
    z = -transition_length            full circle (diameter = spar_diameter, t = 1)
    z = -(transition_length+spar_len) bottom of the cylindrical spar (same circle)

Sections between z=0 and z=-transition_length morph linearly between the root
airfoil and the spar circle, producing a smooth fairing instead of an abrupt
step where the spar meets the airfoil.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from build123d import (
    BuildLine,
    BuildPart,
    BuildSketch,
    Plane,
    Spline,
    add,
    loft,
    make_face,
)

from .airfoil import naca_00xx_coords


@dataclass(frozen=True)
class WingSpec:
    span: float = 5.0                  # m, root (z=0) to tip
    root_chord: float = 1.0            # m
    tip_chord: float = 0.6             # m
    thickness: float = 0.18            # NACA00xx t/c
    pivot_frac: float = 0.25           # x/c of the rotation axis at every station
    spar_length: float = 0.75          # m, cylindrical portion below the fairing
    spar_diameter: float = 0.15        # m, diameter of the cylindrical spar / fairing base
    transition_length: float = 0.20    # m, height of airfoil->circle fairing below root
    n_transition_sections: int = 4     # interior morph sections in the fairing
    n_sections: int = 5                # spanwise airfoil sections (root..tip inclusive)
    n_airfoil_points: int = 160        # polyline resolution on the airfoil


def _airfoil_to_circle_polyline(
    chord: float,
    thickness: float,
    pivot_frac: float,
    diameter: float,
    blend: float,
    n_pts: int,
) -> np.ndarray:
    """Polyline that blends a NACA00xx airfoil (blend=0) to a circle (blend=1).

    The airfoil keeps its native cosine-spaced point distribution. For each
    airfoil point at angle θ_i from the pivot, the corresponding circle point
    is placed at the same θ_i with constant radius R. The blend is a per-point
    linear interpolation between the two — which is a *radial* morph because
    the two polylines share angles point-for-point. This keeps point density
    high where the airfoil curvature is high (TE and LE), so the loft sees
    consistent, well-sampled sections at every blend value.

    Relies on the airfoil being star-shaped w.r.t. the pivot, which holds for
    NACA00xx with pivot_frac in roughly [0.05, 0.95].
    """
    airfoil = naca_00xx_coords(thickness, n=n_pts)
    airfoil = (airfoil - np.array([pivot_frac, 0.0])) * chord
    theta = np.arctan2(airfoil[:, 1], airfoil[:, 0])
    circle = 0.5 * diameter * np.column_stack([np.cos(theta), np.sin(theta)])
    return (1.0 - blend) * airfoil + blend * circle


def _section_face(
    chord: float,
    thickness: float,
    pivot_frac: float,
    diameter: float,
    blend: float,
    n_pts: int,
):
    """A planar Face whose boundary is a closed periodic spline through the morphed points.

    The cosine-spaced airfoil polyline ends with a duplicate at the trailing edge
    (closed_te=True). We drop that duplicate so the periodic spline can close
    cleanly without a zero-length segment.
    """
    coords = _airfoil_to_circle_polyline(chord, thickness, pivot_frac, diameter, blend, n_pts)
    pts = [(float(x), float(y)) for x, y in coords[:-1]]
    with BuildSketch() as sk:
        with BuildLine():
            Spline(*pts, periodic=True)
        make_face()
    return sk.sketch


def build_wing_solid(spec: WingSpec = WingSpec()):
    """Loft the wing + fairing + cylindrical spar as a single solid.

    Sections are added top (tip) to bottom (spar base) so the loft proceeds in
    one direction. The cylindrical spar is represented by two identical
    circular sections at z = -transition_length and z = -(transition_length +
    spar_length).

    Uses `loft(ruled=True)` so build123d connects each adjacent pair of
    sections with a developable ruled surface. BSpline-smooth lofting
    (`ruled=False`, the default) overshoots wildly when adjacent sections
    change shape topology — the airfoil→circle morph would produce a bbox
    several times the size of any individual section.
    """
    sections: list = []

    # Wing: tip (high z) down to root (z = 0), pure airfoil at each station
    for i in range(spec.n_sections - 1, -1, -1):
        frac = i / (spec.n_sections - 1)
        chord = spec.root_chord + frac * (spec.tip_chord - spec.root_chord)
        face = _section_face(
            chord, spec.thickness, spec.pivot_frac, spec.spar_diameter, blend=0.0,
            n_pts=spec.n_airfoil_points,
        )
        sections.append(Plane(origin=(0, 0, frac * spec.span)) * face)

    # Fairing: airfoil (blend=0 at z=0, already added) → circle (blend=1 at z=-T)
    for j in range(1, spec.n_transition_sections + 2):
        blend = j / (spec.n_transition_sections + 1)
        z = -spec.transition_length * blend
        face = _section_face(
            spec.root_chord, spec.thickness, spec.pivot_frac, spec.spar_diameter,
            blend=blend, n_pts=spec.n_airfoil_points,
        )
        sections.append(Plane(origin=(0, 0, z)) * face)

    # Cylindrical spar: same circle, extended down by spar_length
    bottom_face = _section_face(
        spec.root_chord, spec.thickness, spec.pivot_frac, spec.spar_diameter,
        blend=1.0, n_pts=spec.n_airfoil_points,
    )
    z_bottom = -(spec.transition_length + spec.spar_length)
    sections.append(Plane(origin=(0, 0, z_bottom)) * bottom_face)

    with BuildPart() as part:
        for face in sections:
            add(face)
        loft(ruled=True)
    return part.part
