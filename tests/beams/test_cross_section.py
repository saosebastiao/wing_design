import numpy as np

from wing_design.geometry import WingSpec, oml_section_polyline


def test_oml_section_wing_region_is_airfoil():
    spec = WingSpec()
    poly = oml_section_polyline(spec, z=0.0)
    # Root chord = 1.0, pivot at 0.25c: TE at +0.75, LE at -0.25.
    assert abs(poly[:, 0].max() - (1.0 - spec.pivot_frac) * spec.root_chord) < 1e-6
    assert abs(poly[:, 0].min() - (-spec.pivot_frac * spec.root_chord)) < 1e-6


def test_oml_section_spar_region_is_circle():
    spec = WingSpec()
    z = -(spec.transition_length + spec.spar_length * 0.5)  # deep in the spar
    poly = oml_section_polyline(spec, z=z)
    r = np.hypot(poly[:, 0], poly[:, 1])
    assert np.allclose(r, spec.spar_diameter / 2.0, atol=1e-6)
