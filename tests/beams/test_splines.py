import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.beams.splines import (
    default_z_levels,
    form_beam_grid,
    fit_beam_splines,
    sample_spline,
)


def test_z_levels_span_tip_to_keelstep():
    spec = small_wingsail
    z = default_z_levels(spec, 10)
    assert z.shape == (10,)
    assert abs(z[0] - spec.span) < 1e-9
    assert abs(z[-1] - (-(spec.transition_length + spec.spar_length))) < 1e-9
    assert np.all(np.diff(z) < 0)  # monotonically descending


def test_grid_shape_and_z_consistency():
    spec = small_wingsail
    z = default_z_levels(spec, 12)
    grid = form_beam_grid(spec, z, n_beams=16)
    assert grid.shape == (16, 12, 3)
    for k in range(12):
        assert np.allclose(grid[:, k, 2], z[k])


def test_splines_interpolate_their_points():
    spec = small_wingsail
    z = default_z_levels(spec, 12)
    grid = form_beam_grid(spec, z, n_beams=16)
    splines = fit_beam_splines(grid, smoothing=0.0)
    assert len(splines) == 16
    # A densely sampled interpolating spline must pass near every input point.
    # (Dense sampling so the geometric nearest-sample check resolves the knots
    # even through the high-curvature transition region.)
    for b in range(16):
        curve = sample_spline(splines[b], n=5000)
        for p in grid[b]:
            d = np.linalg.norm(curve - p, axis=1).min()
            assert d < 1e-3
