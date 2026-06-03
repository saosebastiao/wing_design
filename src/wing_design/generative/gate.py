"""1D linear-elastic 3D frame (beam-element) solver — the truss gate.

Judges a WingCandidate against stress and tip-deflection limits. Each beam
centerline is discretized into 2-node 3D Euler-Bernoulli beam elements
(6 DOF/node: ux, uy, uz, theta_x, theta_y, theta_z). UD carbon with fiber along
the spline is modeled with E1 axially and G12 in torsion. The structure is
reacted by a bearing couple (keel-step + deck-step), not a clamped base.

Validated against closed-form cantilever / axial / torsion solutions. See
docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md §7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .menu import GateResult, NodeKind


def section_properties(area_m2: float) -> tuple[float, float, float]:
    """Return (A, I, J) for a solid circular section of the given area.

    I is the second moment of area (equal about both transverse axes for a
    circle); J = 2 I is the polar moment (torsion constant for a circle).
    """
    A = area_m2
    radius = math.sqrt(A / math.pi)
    I = math.pi * radius**4 / 4.0
    J = 2.0 * I
    return A, I, J
