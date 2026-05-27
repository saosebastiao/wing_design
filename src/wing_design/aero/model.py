"""Map a `WingSpec` onto an AeroSandbox `Airplane` for aerodynamic analysis.

ASB convention used here: chord along +X, span along +Y, airfoil normal along +Z
(standard fixed-wing aircraft). The wingsail is modeled as a horizontal cantilever
wing rooted at y=0; the "lift" force ASB reports becomes the wingsail's
heel-direction force (perpendicular to the apparent wind, horizontal in the boat
frame).
"""
from __future__ import annotations

import aerosandbox as asb

from ..geometry.wing import WingSpec


def _naca_00xx_name(thickness: float) -> str:
    """ASB airfoil name for a NACA 00xx with `thickness` = t/c (e.g. 0.18 -> 'naca0018')."""
    tt = int(round(thickness * 100))
    if not 1 <= tt <= 99:
        raise ValueError(f"NACA 00xx thickness must round to 1..99% chord; got t/c={thickness}")
    return f"naca00{tt:02d}"


def build_asb_wing(spec: WingSpec) -> asb.Wing:
    """Two-station tapered ASB wing matching `spec` (linear taper)."""
    airfoil = asb.Airfoil(_naca_00xx_name(spec.thickness))
    root = asb.WingXSec(
        xyz_le=[-spec.pivot_frac * spec.root_chord, 0.0, 0.0],
        chord=spec.root_chord,
        twist=0.0,
        airfoil=airfoil,
    )
    tip = asb.WingXSec(
        xyz_le=[-spec.pivot_frac * spec.tip_chord, spec.span, 0.0],
        chord=spec.tip_chord,
        twist=0.0,
        airfoil=airfoil,
    )
    return asb.Wing(name="wingsail", xsecs=[root, tip], symmetric=False)


def build_airplane(spec: WingSpec) -> asb.Airplane:
    """ASB Airplane wrapping the wingsail. `xyz_ref` is the pivot at the root."""
    wing = build_asb_wing(spec)
    mean_chord = 0.5 * (spec.root_chord + spec.tip_chord)
    return asb.Airplane(
        name="wingsail",
        xyz_ref=[0.0, 0.0, 0.0],
        wings=[wing],
        s_ref=spec.span * mean_chord,
        b_ref=spec.span,
        c_ref=mean_chord,
    )
