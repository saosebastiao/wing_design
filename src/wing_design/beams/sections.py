"""Pure-numpy beam cross-section primitives for Phase-D sized geometry.

`oml_outward_normals` gives the true in-plane outward surface normal at each beam
position (replacing Phase-A's radial-from-pivot approximation). `lens_section_polyline`
builds an area-exact inward-arc ("lens") section: a semicircular segment whose flat
(outer) edge sits on the OML surface and whose arc bulges inward along -normal. The
flat edge approximates the locally-flat OML; the area equals the Phase-C sized area.
"""
from __future__ import annotations

import numpy as np

from ..geometry.wing import WingSpec
from .cross_section import beam_section_points


def oml_outward_normals(spec: WingSpec, z: float, n_beams: int) -> np.ndarray:
    """(n_beams, 2) unit outward normals at the arc-spaced OML beam positions at ``z``."""
    if n_beams < 3:
        raise ValueError(f"n_beams must be >= 3, got {n_beams}")
    P = beam_section_points(spec, z, n_beams)
    tang = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)   # central difference around the loop
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.column_stack([tang[:, 1], -tang[:, 0]])         # rotate tangent -90°
    radial = P / np.linalg.norm(P, axis=1, keepdims=True)    # outward reference (from pivot)
    sign = np.sign(np.sum(nrm * radial, axis=1))
    sign[sign == 0] = 1.0
    return nrm * sign[:, None]


def lens_section_polyline(
    center: np.ndarray,
    normal: np.ndarray,
    area: float,
    n_arc: int = 24,
) -> np.ndarray:
    """Closed (n_arc+1, 2) inward-arc lens of exactly ``area`` at ``center``.

    Semicircular segment: flat diameter along the surface tangent through ``center``
    (the outer edge, on the OML), arc bulging inward along ``-normal``. The polygon's
    closing edge (last→first vertex) is the flat outer edge.

    ``rho`` is chosen so that the DISCRETIZED polygon (n_arc chords) has area exactly
    ``area`` for any n_arc: the polygon area is ``(n_arc/2) * rho**2 * sin(pi/n_arc)``,
    so ``rho = sqrt(2*area / (n_arc * sin(pi/n_arc)))``. This converges to the analytic
    half-disk formula ``rho = sqrt(2*area/pi)`` as n_arc → infinity.
    """
    if area <= 0.0:
        raise ValueError(f"area must be positive, got {area}")
    if n_arc < 2:
        raise ValueError(f"n_arc must be >= 2, got {n_arc}")
    rho = float(np.sqrt(2.0 * area / (n_arc * np.sin(np.pi / n_arc))))
    n = np.asarray(normal, dtype=float)
    nrm = np.linalg.norm(n)
    if nrm == 0.0:
        raise ValueError("normal must be a non-zero vector")
    n = n / nrm
    t = np.array([-n[1], n[0]])                              # tangent ⟂ normal
    c = np.asarray(center, dtype=float)
    theta = np.linspace(0.0, np.pi, n_arc + 1)
    # θ=0 → c+ρt ; θ=π → c−ρt ; bulge along −n
    return c + rho * (np.cos(theta)[:, None] * t - np.sin(theta)[:, None] * n)
