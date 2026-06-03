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


def local_beam_stiffness(E, G, A, I, J, L):
    """12x12 local stiffness for a 2-node 3D Euler-Bernoulli beam element.

    DOF order per node: [u (axial, local x), v (local y), w (local z),
    theta_x (torsion), theta_y, theta_z]. Circular section, so the two
    transverse second moments are equal (passed as a single I).
    """
    k = np.zeros((12, 12))
    ea = E * A / L
    gj = G * J / L
    a = 12.0 * E * I / L**3
    b = 6.0 * E * I / L**2
    c = 4.0 * E * I / L
    d = 2.0 * E * I / L

    # Axial (u_i, u_j)
    k[0, 0] = ea
    k[0, 6] = -ea
    k[6, 0] = -ea
    k[6, 6] = ea

    # Torsion (theta_x_i, theta_x_j)
    k[3, 3] = gj
    k[3, 9] = -gj
    k[9, 3] = -gj
    k[9, 9] = gj

    # Bending in local x-y plane: v (1,7) and theta_z (5,11)
    k[1, 1] = a
    k[1, 5] = b
    k[1, 7] = -a
    k[1, 11] = b
    k[5, 1] = b
    k[5, 5] = c
    k[5, 7] = -b
    k[5, 11] = d
    k[7, 1] = -a
    k[7, 5] = -b
    k[7, 7] = a
    k[7, 11] = -b
    k[11, 1] = b
    k[11, 5] = d
    k[11, 7] = -b
    k[11, 11] = c

    # Bending in local x-z plane: w (2,8) and theta_y (4,10)
    k[2, 2] = a
    k[2, 4] = -b
    k[2, 8] = -a
    k[2, 10] = -b
    k[4, 2] = -b
    k[4, 4] = c
    k[4, 8] = b
    k[4, 10] = d
    k[8, 2] = -a
    k[8, 4] = b
    k[8, 8] = a
    k[8, 10] = b
    k[10, 2] = -b
    k[10, 4] = d
    k[10, 8] = b
    k[10, 10] = c

    return k
