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
    """ASB wing matching `spec` — one WingXSec per taper-profile knot.

    Note the axis swap: structural geometry frame has span in +Z, but ASB's
    convention puts span in +Y. So a structural z-fraction maps to an ASB
    y-coordinate of `frac * spec.span`.
    """
    airfoil = asb.Airfoil(_naca_00xx_name(spec.thickness))
    xsecs = []
    for frac in spec.section_z_fractions:
        chord = spec.chord_at_z(frac * spec.span)
        xsecs.append(asb.WingXSec(
            xyz_le=[-spec.pivot_frac * chord, frac * spec.span, 0.0],
            chord=chord,
            twist=0.0,
            airfoil=airfoil,
        ))
    return asb.Wing(name="wingsail", xsecs=xsecs, symmetric=False)


def build_airplane(spec: WingSpec) -> asb.Airplane:
    """ASB Airplane wrapping the wingsail. `xyz_ref` is the pivot at the root."""
    wing = build_asb_wing(spec)
    # Mean aerodynamic chord ≈ area-weighted average over the span.
    z_fracs = spec.section_z_fractions
    chord_samples = [spec.chord_at_z(f * spec.span) for f in z_fracs]
    if len(z_fracs) >= 2:
        area = 0.0
        for i in range(len(z_fracs) - 1):
            dz = (z_fracs[i + 1] - z_fracs[i]) * spec.span
            area += 0.5 * (chord_samples[i] + chord_samples[i + 1]) * dz
        mean_chord = area / spec.span
    else:
        mean_chord = chord_samples[0]
    return asb.Airplane(
        name="wingsail",
        xyz_ref=[0.0, 0.0, 0.0],
        wings=[wing],
        s_ref=spec.span * mean_chord,
        b_ref=spec.span,
        c_ref=mean_chord,
    )
