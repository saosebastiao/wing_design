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


def spline_surface_error(
    spec: WingSpec,
    *,
    n_beams: int,
    n_levels: int,
    smoothing: float = 0.0,
    n_samples: int = 400,
    z_min: float = -np.inf,
) -> float:
    """Worst-beam max distance [m] between the fitted splines and the true OML path.

    Fits the beam splines through ``n_levels`` z-levels, then samples each spline
    densely and, at each sample's own z, compares its (x, y) to the true OML beam
    position. Returns the maximum over all samples and beams.

    Parameters
    ----------
    z_min:
        Only include samples whose spanwise height ``pz >= z_min``.  Use ``0.0``
        to restrict the metric to the aerodynamic wing surface only (above the
        keel/root transition).  Default ``-np.inf`` includes all samples and
        preserves the original behaviour.

    Note on transition-region overshoot
    ------------------------------------
    Single-spline interpolation overshoots through the airfoil→circle collapse
    that occurs in roughly z ∈ [-0.2, 0] (the keel-to-wing-root transition).
    Because uniform z-levels barely sample this 0.2 m band, the global worst-case
    error plateaus around ~0.1 m regardless of how many levels are used.  On the
    aero surface (z >= 0) the error is smaller and does decrease with more levels.
    The long-term fix is to use denser z-levels through the transition band; in
    the meantime, pass ``z_min=0.0`` to measure fidelity on the aerodynamic surface
    only.
    """
    z_levels = default_z_levels(spec, n_levels)
    grid = form_beam_grid(spec, z_levels, n_beams)
    splines = fit_beam_splines(grid, smoothing=smoothing)
    worst = 0.0
    for b in range(n_beams):
        pts = sample_spline(splines[b], n=n_samples)  # (n_samples, 3)
        for px, py, pz in pts:
            if pz < z_min:
                continue
            true_xy = beam_section_points(spec, float(pz), n_beams)[b]
            worst = max(worst, float(np.hypot(px - true_xy[0], py - true_xy[1])))
    return worst
