from wing_design.geometry import WingSpec
from wing_design.beams.fidelity import spline_surface_error


def test_surface_error_decreases_with_more_levels():
    spec = WingSpec()
    coarse = spline_surface_error(spec, n_beams=16, n_levels=6)
    fine = spline_surface_error(spec, n_beams=16, n_levels=24)
    assert fine < coarse           # more levels → closer to the true OML
    assert fine >= 0.0


def test_surface_error_is_small_when_fine():
    spec = WingSpec()
    err = spline_surface_error(spec, n_beams=16, n_levels=40)
    assert err < 0.01              # < 1 cm worst-case on a 5 m wing
