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


def tsai_wu_strength_ratio_grad(sigma123, ply: UDPly) -> np.ndarray:
    """Analytic ∂R/∂[s1,s2,t12] for the Tsai-Wu strength ratio R.

    Implicit differentiation of a R² + b R − 1 = 0 gives
    dR = −(R² da + R db)/(2 a R + b). With
    ∂a/∂s1 = 2F11 s1 + 2F12 s2, ∂a/∂s2 = 2F22 s2 + 2F12 s1, ∂a/∂t12 = 2F66 t12,
    ∂b/∂s1 = F1, ∂b/∂s2 = F2, ∂b/∂t12 = 0.
    Returns zeros where there is essentially no stress (denominator ~0 or R capped).
    """
    s1, s2, t12 = (float(v) for v in sigma123)
    F1, F2, F11, F22, F66, F12 = tsai_wu_coefficients(ply)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2.0 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    if a <= 1.0e-30:
        # Linear regime R = 1/b (or _SAFE_R); gradient is tiny / ill-defined -> zeros.
        return np.zeros(3)
    R = (-b + np.sqrt(b * b + 4.0 * a)) / (2.0 * a)
    if R >= _SAFE_R:
        return np.zeros(3)
    denom = 2.0 * a * R + b
    if abs(denom) <= 1.0e-30:
        return np.zeros(3)
    da = np.array([
        2.0 * F11 * s1 + 2.0 * F12 * s2,
        2.0 * F22 * s2 + 2.0 * F12 * s1,
        2.0 * F66 * t12,
    ])
    db = np.array([F1, F2, 0.0])
    return -(R * R * da + R * db) / denom


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


def _strength_ratio_array(s1, s2, t12, coeffs) -> np.ndarray:
    """Vectorized Tsai-Wu strength ratio over material-axis stress arrays (each (M,))."""
    F1, F2, F11, F22, F66, F12 = coeffs
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    t12 = np.asarray(t12, dtype=float)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2.0 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    a_safe = np.where(a > 1.0e-30, a, 1.0)
    R_quad = (-b + np.sqrt(b * b + 4.0 * a_safe)) / (2.0 * a_safe)
    b_safe = np.where(b > 1.0e-30, b, 1.0)
    R_lin = np.where(b > 1.0e-30, 1.0 / b_safe, _SAFE_R)
    R = np.where(a > 1.0e-30, R_quad, R_lin)
    return np.minimum(R, _SAFE_R)


def laminate_min_strength_ratio_batch(
    ply: UDPly, eps_local, *, f0, f45, f90, offset_deg,
) -> np.ndarray:
    """(M,) min Tsai-Wu strength ratio per triangle over the present ply orientations.

    ``eps_local`` is (M,3) element-local engineering strain. ``offset_deg`` and the
    fractions ``f0/f45/f90`` may each be a scalar (broadcast) or a length-M array (the
    per-band-layup case). Each of the 0/+45/-45/90 orientations is gated element-wise by
    its fraction (>1e-9 to count as present); the min is over present orientations.
    Vectorized equivalent of ``laminate_min_strength_ratio`` per row.
    """
    eps = np.asarray(eps_local, dtype=float)
    if eps.ndim != 2 or eps.shape[1] != 3:
        raise ValueError(f"eps_local must be (M,3), got {eps.shape}")
    M = eps.shape[0]
    off = np.broadcast_to(np.asarray(offset_deg, dtype=float), (M,))
    f0a = np.broadcast_to(np.asarray(f0, dtype=float), (M,))
    f45a = np.broadcast_to(np.asarray(f45, dtype=float), (M,))
    f90a = np.broadcast_to(np.asarray(f90, dtype=float), (M,))
    ex, ey, gxy = eps[:, 0], eps[:, 1], eps[:, 2]
    Q = reduced_stiffness_Q(ply)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    coeffs = tsai_wu_coefficients(ply)
    eps_tol = 1.0e-9

    orientations = ((0.0, f0a), (45.0, f45a), (-45.0, f45a), (90.0, f90a))
    ratios = np.full((len(orientations), M), _SAFE_R)
    for k, (base, frac) in enumerate(orientations):
        th = np.radians(base + off)
        c, s = np.cos(th), np.sin(th)
        e1 = ex * c * c + ey * s * s + gxy * s * c
        e2 = ex * s * s + ey * c * c - gxy * s * c
        g12 = -2.0 * ex * s * c + 2.0 * ey * s * c + gxy * (c * c - s * s)
        s1 = Q11 * e1 + Q12 * e2
        s2 = Q12 * e1 + Q22 * e2
        t12 = Q66 * g12
        r = _strength_ratio_array(s1, s2, t12, coeffs)
        ratios[k] = np.where(frac > eps_tol, r, _SAFE_R)
    return ratios.min(axis=0)


def laminate_min_strength_ratio(
    ply: UDPly, eps_local, *, f0: float, f45: float, f90: float, offset_deg: float,
) -> float:
    """Min Tsai-Wu strength ratio over the present ply orientations (scalar; see batch)."""
    out = laminate_min_strength_ratio_batch(
        ply, np.asarray(eps_local, dtype=float).reshape(1, 3),
        f0=f0, f45=f45, f90=f90, offset_deg=np.asarray([offset_deg], dtype=float),
    )
    return float(out[0])
