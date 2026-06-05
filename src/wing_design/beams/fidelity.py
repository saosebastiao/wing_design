"""On-surface fidelity metric for the form-beam splines.

Each beam spline interpolates the OML at the chosen z-levels; between levels it
may drift off the true surface. We measure that drift: for many z between tip and
keel, compare each beam spline's (x, y) to the true OML beam position at that z
(`beam_section_points`), and report the worst distance over all beams. Lower is
better; it shrinks as the z-level count rises.
"""
from __future__ import annotations

import numpy as np

from ..geometry.wing import WingSpec
from .cross_section import beam_section_points
from .splines import default_z_levels, fit_beam_splines, form_beam_grid, sample_spline


def beam_path_at_z(spec: WingSpec, z: float, n_beams: int) -> np.ndarray:
    """True on-OML (x, y, z) of every beam at height ``z`` (the arc-spaced position)."""
    xy = beam_section_points(spec, z, n_beams)
    return np.column_stack([xy, np.full(n_beams, z)])


def spline_surface_error(
    spec: WingSpec,
    *,
    n_beams: int,
    n_levels: int,
    smoothing: float = 0.0,
    n_samples: int = 400,
) -> float:
    """Worst-beam max distance [m] between the fitted splines and the true OML path.

    Fits the beam splines through ``n_levels`` z-levels, then samples each spline
    densely and, at each sample's own z, compares its (x, y) to the true OML beam
    position. Returns the maximum over all samples and beams.
    """
    z_levels = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z_levels, n_beams)
    splines = fit_beam_splines(grid, smoothing=smoothing)
    worst = 0.0
    for b in range(n_beams):
        pts = sample_spline(splines[b], n=n_samples)  # (n_samples, 3)
        for px, py, pz in pts:
            true_xy = beam_section_points(spec, float(pz), n_beams)[b]
            worst = max(worst, float(np.hypot(px - true_xy[0], py - true_xy[1])))
    return worst
