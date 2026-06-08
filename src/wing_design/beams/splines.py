"""Form-beam spanwise point grids and per-beam B-spline fits."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep

from ..geometry.wing import WingSpec
from .cross_section import beam_section_points


def default_z_levels(spec: WingSpec, n: int = 20) -> np.ndarray:
    """``n`` spanwise sampling heights from the tip (z = span) down to the keel-step."""
    z_top = spec.span
    z_bottom = -(spec.transition_length + spec.spar_length)
    return np.linspace(z_top, z_bottom, n)


def form_beam_grid(
    spec: WingSpec,
    z_levels: np.ndarray,
    n_beams: int,
    arc_fractions: np.ndarray | None = None,
) -> np.ndarray:
    """(n_beams, n_levels, 3) array of on-shell points.

    Entry ``[b, k]`` is the (x, y, z) location of form beam ``b`` at
    ``z_levels[k]``. Beam ordering matches ``beam_section_points`` (0 = TE,
    ``n_beams // 2`` = LE).

    If ``arc_fractions`` is given (shape (n_beams,), values in [0, 1)) it is
    forwarded to ``beam_section_points`` at every level to place beams at
    non-uniform arc positions. Omit for the default even spacing.
    """
    n_levels = len(z_levels)
    grid = np.empty((n_beams, n_levels, 3))
    for k, z in enumerate(z_levels):
        xy = beam_section_points(spec, float(z), n_beams, arc_fractions=arc_fractions)
        grid[:, k, 0] = xy[:, 0]
        grid[:, k, 1] = xy[:, 1]
        grid[:, k, 2] = float(z)
    return grid


def fit_beam_splines(grid: np.ndarray, smoothing: float = 0.0) -> list:
    """Fit one cubic B-spline per beam to its (x, y, z) point sequence.

    Returns a list of scipy ``splprep`` ``tck`` tuples, one per beam. With
    ``smoothing = 0`` the spline interpolates the on-shell points, so it lies on
    the shell to sampling tolerance.
    """
    splines = []
    for b in range(grid.shape[0]):
        pts = grid[b]  # (n_levels, 3)
        k = min(3, pts.shape[0] - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=smoothing, k=k)
        splines.append(tck)
    return splines


def sample_spline(tck, n: int = 50) -> np.ndarray:
    """Sample a fitted spline at ``n`` evenly spaced parameter values -> (n, 3)."""
    u = np.linspace(0.0, 1.0, n)
    x, y, z = splev(u, tck)
    return np.column_stack([x, y, z])
