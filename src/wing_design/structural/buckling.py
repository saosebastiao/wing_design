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


def beam_euler_utilization_I(
    axial_force: np.ndarray,
    I: np.ndarray,
    lengths: np.ndarray,
    *,
    E: float,
    K: float = 1.0,
    safety_factor: float = 1.0,
) -> np.ndarray:
    """Per-element Euler utilization from explicit second moments (P.1).

    Generalizes `beam_euler_utilization` to arbitrary sections (annular tube
    segments): Pcr = π²·E·I/(K·L)². Tension members return 0.
    """
    comp = np.maximum(0.0, -np.asarray(axial_force, dtype=float))
    pcr = np.pi**2 * E * np.asarray(I, dtype=float) / (K * np.asarray(lengths, dtype=float)) ** 2
    return comp * safety_factor / np.maximum(pcr, 1e-30)


TUBE_WALL_KNOCKDOWN = 0.65   # NASA SP-8007-class imperfection knockdown (recorded)


def tube_wall_utilization(
    axial_force: np.ndarray,
    r_outer: np.ndarray,
    t_wall: np.ndarray,
    A: np.ndarray,
    *,
    E: float,
    safety_factor: float = 1.0,
    knockdown: float = TUBE_WALL_KNOCKDOWN,
) -> np.ndarray:
    """Per-tube-segment wall crimping (axial-compression cylinder buckling, P.1).

    σ_cr = knockdown · 0.605 · E · t / r (classical cylinder bifurcation with an
    imperfection knockdown); utilization = SF · max(0, −N)/A / σ_cr. Shear/torsion
    buckling and bending-gradient relief are deferred (recorded caveats).
    """
    comp_stress = np.maximum(0.0, -np.asarray(axial_force, dtype=float)) / np.asarray(A, dtype=float)
    sigma_cr = knockdown * 0.605 * E * np.asarray(t_wall, dtype=float) / np.asarray(r_outer, dtype=float)
    return comp_stress * safety_factor / np.maximum(sigma_cr, 1e-30)


def panel_buckling_utilization(
    membrane_stress: np.ndarray,
    areas: np.ndarray,
    *,
    D11: float,
    t: float,
    kc: float = 4.0,
    safety_factor: float = 1.0,
    b_widths: np.ndarray | None = None,
) -> np.ndarray:
    """Per-panel plate-buckling utilization = (compressive principal stress·SF) / σcr.

    σcr = kc·π²·D11 / (b²·t). Default b = sqrt(area) (historical; mis-scales with
    beam count and over-sizes b — backlog V#1). Pass ``b_widths`` (per-panel physical
    strip widths, `beams.shell_model.skin_panel_widths`) for the V.3b width-based
    check — the long-strip SS plate formula with the chordwise beam spacing as b,
    eigen-calibrated 2026-06-10. `membrane_stress` is (M,3) [σxx,σyy,σxy]; the
    most-compressive principal stress drives buckling. Tension panels return 0.
    Caveats: fixed kc=4 assumes simply-supported edges (flexible beam edges may be
    softer); D11 is the triangle-local-frame value (direction mismatch V#2 open).
    """
    s = np.asarray(membrane_stress, dtype=float)
    mean = 0.5 * (s[:, 0] + s[:, 1])
    radius = np.sqrt(0.25 * (s[:, 0] - s[:, 1]) ** 2 + s[:, 2] ** 2)
    s_min = mean - radius
    comp = np.maximum(0.0, -s_min)
    if b_widths is not None:
        b2 = np.maximum(np.asarray(b_widths, dtype=float), 1e-30) ** 2
    else:
        b2 = np.maximum(np.asarray(areas, dtype=float), 1e-30)
    sigma_cr = kc * np.pi**2 * D11 / (b2 * t)
    return comp * safety_factor / np.maximum(sigma_cr, 1e-30)


def beam_foundation_utilization(
    axial_force,
    EI,
    k_found,
    *,
    safety_factor: float = 1.0,
) -> "np.ndarray":
    """Per-element buckling utilization for a stringer on an elastic foundation
    (V.3c): Pcr = 2*sqrt(k*EI) — the continuous-mode minimum for a long beam
    continuously restrained by the bonded skin (conservative vs the integer-mode
    optimum; valid when the foundation half-wave fits the beam, which holds by
    orders of magnitude for the in-wing spans). Tension returns 0.
    """
    comp = np.maximum(0.0, -np.asarray(axial_force, dtype=float))
    pcr = 2.0 * np.sqrt(np.asarray(k_found, dtype=float) * np.asarray(EI, dtype=float))
    return comp * safety_factor / np.maximum(pcr, 1e-30)


def k_ring_stiffness(r_brace, E, L_ring, s, alpha=3.0):
    """Distributed elastic-foundation stiffness (force / length / length) that a
    transverse ring member of circular radius ``r_brace`` contributes to a braced
    form-beam segment, smeared over the spanwise tributary length ``s``.

    Model: the ring acts as a transverse spring of stiffness alpha*E*I_ring/L_ring^3
    (I_ring = pi r^4 / 4, alpha = end-restraint constant, 3 = cantilever, conservative),
    smeared over the segment's spanwise length s -> k = spring / s. ``L_ring`` and ``s``
    may be arrays (per element); ``r_brace`` is the single shared DV (scalar)."""
    I = np.pi * r_brace**4 / 4.0
    denom = np.maximum(np.asarray(L_ring, float) ** 3 * np.asarray(s, float), 1e-30)
    return alpha * E * I / denom


def dk_ring_dr(r_brace, E, L_ring, s, alpha=3.0):
    """d k_ring / d r_brace = 4 * k_ring / r_brace (quartic in r)."""
    return 4.0 * k_ring_stiffness(r_brace, E, L_ring, s, alpha) / np.maximum(r_brace, 1e-30)
