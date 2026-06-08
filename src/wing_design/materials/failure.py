"""Per-ply Tsai-Wu failure criterion for a unidirectional CFRP ply.

Tsai-Wu in the ply material axes:
    FI = F1 s1 + F2 s2 + F11 s1^2 + F22 s2^2 + F66 t12^2 + 2 F12 s1 s2,  failure when FI >= 1.
The strength ratio R is the factor by which the current stress can scale before FI = 1
(failure when R < 1); it is the interpretable margin used by the sizer's constraint.

`ply_strength_ratio` / `laminate_min_strength_ratio` take the element-local membrane
strain [exx, eyy, gxy] (engineering shear), transform it into each present ply's
material axes, recover the ply stress with the material-axis reduced stiffness Q, and
return the (min) strength ratio.
"""
from __future__ import annotations

import numpy as np

from .unidir import UDPly, reduced_stiffness_Q

_SAFE_R = 1.0e9  # returned when there is effectively no stress (no failure)


def tsai_wu_coefficients(ply: UDPly) -> tuple[float, float, float, float, float, float]:
    """(F1, F2, F11, F22, F66, F12) with F12 = -0.5*sqrt(F11*F22)."""
    F1 = 1.0 / ply.Xt_Pa - 1.0 / ply.Xc_Pa
    F2 = 1.0 / ply.Yt_Pa - 1.0 / ply.Yc_Pa
    F11 = 1.0 / (ply.Xt_Pa * ply.Xc_Pa)
    F22 = 1.0 / (ply.Yt_Pa * ply.Yc_Pa)
    F66 = 1.0 / (ply.S12_Pa ** 2)
    F12 = -0.5 * np.sqrt(F11 * F22)
    return F1, F2, F11, F22, F66, F12


def tsai_wu_index(sigma123, ply: UDPly) -> float:
    """Tsai-Wu failure index FI for material-axis stress [s1, s2, t12]; failure at FI>=1."""
    s1, s2, t12 = (float(v) for v in sigma123)
    F1, F2, F11, F22, F66, F12 = tsai_wu_coefficients(ply)
    return float(F1 * s1 + F2 * s2 + F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2.0 * F12 * s1 * s2)


def tsai_wu_strength_ratio(sigma123, ply: UDPly) -> float:
    """Strength ratio R (R*sigma reaches the envelope); failure when R<1.

    Solves a*R^2 + b*R - 1 = 0 with a = quadratic terms, b = linear terms. Returns a
    large finite value when there is essentially no stress (a,b ~ 0).
    """
    s1, s2, t12 = (float(v) for v in sigma123)
    F1, F2, F11, F22, F66, F12 = tsai_wu_coefficients(ply)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2.0 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    if a <= 1.0e-30:
        if b <= 1.0e-30:
            return _SAFE_R
        return min(_SAFE_R, 1.0 / b)
    R = (-b + np.sqrt(b * b + 4.0 * a)) / (2.0 * a)
    return float(min(_SAFE_R, R))


def _strain_to_ply_axes(eps_local, angle_deg: float) -> np.ndarray:
    """Transform element-local engineering strain [exx,eyy,gxy] to ply axes at angle_deg."""
    ex, ey, gxy = (float(v) for v in eps_local)
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    e1 = ex * c * c + ey * s * s + gxy * s * c
    e2 = ex * s * s + ey * c * c - gxy * s * c
    g12 = -2.0 * ex * s * c + 2.0 * ey * s * c + gxy * (c * c - s * s)
    return np.array([e1, e2, g12])


def ply_strength_ratio(ply: UDPly, eps_local, angle_deg: float) -> float:
    """Tsai-Wu strength ratio of a single ply at `angle_deg` under element-local strain."""
    eps123 = _strain_to_ply_axes(eps_local, angle_deg)
    sigma123 = reduced_stiffness_Q(ply) @ eps123
    return tsai_wu_strength_ratio(sigma123, ply)


def laminate_min_strength_ratio(
    ply: UDPly, eps_local, *, f0: float, f45: float, f90: float, offset_deg: float,
) -> float:
    """Min Tsai-Wu strength ratio over the PRESENT ply orientations.

    Orientations are 0/+45/-45/90 (relative to the datum) shifted into element-local
    axes by ``offset_deg``; an orientation is checked only if its area fraction > 0.
    """
    eps_tol = 1.0e-9
    angles: list[float] = []
    if f0 > eps_tol:
        angles.append(0.0 + offset_deg)
    if f45 > eps_tol:
        angles.append(45.0 + offset_deg)
        angles.append(-45.0 + offset_deg)
    if f90 > eps_tol:
        angles.append(90.0 + offset_deg)
    if not angles:
        return _SAFE_R
    return min(ply_strength_ratio(ply, eps_local, a) for a in angles)
