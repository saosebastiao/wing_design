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


def beam_transform(p0, p1):
    """Return (T, L): the 12x12 local->global transform and element length.

    The local x-axis runs from p0 to p1. A reference vector (global +Z, or
    global +Y when the element is itself nearly parallel to +Z) fixes the local
    y/z axes. Because the wing spans along +Z, most beams are near-vertical, so
    the +Z-parallel branch is the common path, not an edge case.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    L = float(np.linalg.norm(delta))
    x_local = delta / L

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(x_local, ref))) > 0.9999:
        ref = np.array([0.0, 1.0, 0.0])

    y_local = np.cross(ref, x_local)
    y_local /= np.linalg.norm(y_local)
    z_local = np.cross(x_local, y_local)

    R = np.vstack([x_local, y_local, z_local])  # rows = local axes in global coords
    T = np.zeros((12, 12))
    for blk in range(4):
        T[3 * blk:3 * blk + 3, 3 * blk:3 * blk + 3] = R
    return T, L


def element_global_stiffness(p0, p1, E, G, A, I, J):
    """Global-frame 12x12 stiffness for the beam from p0 to p1: T^T k_local T."""
    T, L = beam_transform(p0, p1)
    k_local = local_beam_stiffness(E, G, A, I, J, L)
    return T.T @ k_local @ T


def solve_displacements(K, load_vec, fixed_dofs):
    """Solve K u = f with `fixed_dofs` held at zero; return the full vector u.

    Reduces to the free DOFs, solves the dense system, and scatters back.
    """
    n = K.shape[0]
    fixed = set(fixed_dofs)
    free = [d for d in range(n) if d not in fixed]
    Kff = K[np.ix_(free, free)]
    Ff = np.asarray(load_vec, dtype=float)[free]
    uf = np.linalg.solve(Kff, Ff)
    u = np.zeros(n)
    u[free] = uf
    return u
