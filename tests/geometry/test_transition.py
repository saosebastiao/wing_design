import numpy as np

from wing_design.geometry import small_wingsail, oml_section_polyline


def _xspan(spec, z):
    p = oml_section_polyline(spec, z)
    return float(p[:, 0].max() - p[:, 0].min())


def test_transition_morph_endpoints():
    spec = small_wingsail
    assert abs(_xspan(spec, 0.0) - spec.root_chord) < 1e-6
    assert abs(_xspan(spec, -spec.transition_length) - spec.spar_diameter) < 1e-6


def test_transition_is_eased_at_junctions():
    spec = small_wingsail
    T = spec.transition_length
    eps = T * 0.02

    def slope(z0):
        return abs(_xspan(spec, z0 + eps) - _xspan(spec, z0 - eps)) / (2 * eps)

    s_top = slope(-1.5 * eps)        # just below the z=0 junction
    s_mid = slope(-T / 2)            # middle of the transition
    s_bot = slope(-T + 1.5 * eps)    # just above the z=-T junction
    # Smoothstep eases the morph at both ends, so junction slopes are far smaller
    # than the mid-transition slope (a linear morph would make them roughly equal).
    assert s_top < 0.5 * s_mid
    assert s_bot < 0.5 * s_mid
