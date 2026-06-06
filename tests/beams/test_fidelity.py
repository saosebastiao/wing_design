from wing_design.geometry import WingSpec
from wing_design.beams.fidelity import spline_surface_error


def test_surface_error_decreases_with_more_levels():
    spec = WingSpec()
    coarse = spline_surface_error(spec, n_beams=16, n_levels=6)
    fine = spline_surface_error(spec, n_beams=16, n_levels=24)
    assert fine < coarse           # more levels → closer to the true OML
    assert fine >= 0.0


def test_transition_dominates_surface_error():
    spec = WingSpec()
    # The airfoil->circle transition (z in [-0.2, 0]) carries the worst error;
    # restricting to the aero surface (z >= 0) gives a meaningfully smaller value.
    full = spline_surface_error(spec, n_beams=16, n_levels=40)
    wing = spline_surface_error(spec, n_beams=16, n_levels=40, z_min=0.0)
    assert wing < full


def test_wing_surface_error_decreases_with_more_levels():
    spec = WingSpec()
    # On the aero surface, more z-levels still reduce the (overshoot-driven) error.
    coarse = spline_surface_error(spec, n_beams=16, n_levels=12, z_min=0.0)
    fine = spline_surface_error(spec, n_beams=16, n_levels=40, z_min=0.0)
    assert fine < coarse
    assert fine >= 0.0
