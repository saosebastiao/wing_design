"""Closed-form buckling utilization checks for the beam-shell structure.

Each function returns a per-element *utilization* = demand·safety_factor / capacity,
so a feasibility constraint is simply `utilization <= 1`. Approximations (documented):
beam buckling uses the element length as the effective buckling length (the skin
laterally restrains intermediate nodes); panel buckling treats each skin triangle as
a simply-supported flat plate of characteristic width `b = sqrt(area)` with a fixed
coefficient `kc`. Tension never buckles (utilization 0).
"""
from __future__ import annotations

import numpy as np


def beam_euler_utilization(
    axial_force: np.ndarray,
    radii: np.ndarray,
    lengths: np.ndarray,
    *,
    E: float,
    K: float = 1.0,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """Per-beam Euler buckling utilization = (compression·SF) / Pcr.

    `axial_force` tension-positive (compression negative). Pcr = π²·E·I / (K·L)²,
    I = π r⁴/4. Tension members return 0.
    """
    comp = np.maximum(0.0, -np.asarray(axial_force, dtype=float))
    r = np.asarray(radii, dtype=float)
    L = np.asarray(lengths, dtype=float)
    second_moment = np.pi * r**4 / 4.0
    pcr = np.pi**2 * E * second_moment / (K * L) ** 2
    return comp * safety_factor / np.maximum(pcr, 1e-30)


def panel_buckling_utilization(
    membrane_stress: np.ndarray,
    areas: np.ndarray,
    *,
    D11: float,
    t: float,
    kc: float = 4.0,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """Per-panel plate-buckling utilization = (compressive principal stress·SF) / σcr.

    σcr = kc·π²·D11 / (b²·t), b = sqrt(area). `membrane_stress` is (M,3) [σxx,σyy,σxy];
    the most-compressive principal stress drives buckling. Tension panels return 0.
    Approximate (triangle-as-plate, fixed kc) — see module docstring.
    """
    s = np.asarray(membrane_stress, dtype=float)
    mean = 0.5 * (s[:, 0] + s[:, 1])
    radius = np.sqrt(0.25 * (s[:, 0] - s[:, 1]) ** 2 + s[:, 2] ** 2)
    s_min = mean - radius
    comp = np.maximum(0.0, -s_min)
    b = np.sqrt(np.maximum(np.asarray(areas, dtype=float), 1e-30))
    sigma_cr = kc * np.pi**2 * D11 / (b**2 * t)
    return comp * safety_factor / np.maximum(sigma_cr, 1e-30)
