"""Symmetric NACA 4-digit airfoil coordinate generation."""
from __future__ import annotations

import numpy as np


def naca_00xx_thickness(x: np.ndarray, t: float, closed_te: bool = True) -> np.ndarray:
    a4 = -0.1036 if closed_te else -0.1015
    return (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        + a4 * x**4
    )


def naca_00xx_coords(
    t: float,
    n: int = 200,
    closed_te: bool = True,
) -> np.ndarray:
    """(N, 2) airfoil contour, traversed upper TE -> LE -> lower TE.

    Uses cosine spacing along the chord for higher resolution near LE/TE.
    The loop is open (caller closes it).
    """
    half = max(2, n // 2)
    beta = np.linspace(0.0, np.pi, half + 1)
    x = 0.5 * (1.0 - np.cos(beta))
    y = naca_00xx_thickness(x, t, closed_te=closed_te)
    upper = np.column_stack([x[::-1], y[::-1]])
    lower = np.column_stack([x[1:], -y[1:]])
    return np.vstack([upper, lower])
