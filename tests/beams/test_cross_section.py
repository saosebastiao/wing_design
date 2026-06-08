import numpy as np

from wing_design.geometry import small_wingsail, oml_section_polyline
from wing_design.beams.cross_section import (
    resample_closed_polyline,
    beam_section_points,
)


def test_oml_section_wing_region_is_airfoil():
    spec = small_wingsail
    poly = oml_section_polyline(spec, z=0.0)
    # Root chord = 1.0, pivot at 0.25c: TE at +0.75, LE at -0.25.
    assert abs(poly[:, 0].max() - (1.0 - spec.pivot_frac) * spec.root_chord) < 1e-6
    assert abs(poly[:, 0].min() - (-spec.pivot_frac * spec.root_chord)) < 1e-6


def test_oml_section_spar_region_is_circle():
    spec = small_wingsail
    z = -(spec.transition_length + spec.spar_length * 0.5)  # deep in the spar
    poly = oml_section_polyline(spec, z=z)
    r = np.hypot(poly[:, 0], poly[:, 1])
    assert np.allclose(r, spec.spar_diameter / 2.0, atol=1e-6)


def test_resample_square_corners():
    # Unit square, closed (last == first). Perimeter = 4.
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    out = resample_closed_polyline(sq, 4)
    assert np.allclose(out[0], [0, 0])
    assert np.allclose(out[1], [1, 0])
    assert np.allclose(out[2], [1, 1])
    assert np.allclose(out[3], [0, 1])


def test_resample_square_midpoints():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    out = resample_closed_polyline(sq, 8)
    assert np.allclose(out[1], [0.5, 0.0])  # midpoint of the first edge


def test_beam_points_le_and_te_are_members():
    spec = small_wingsail
    pts = beam_section_points(spec, z=0.0, n_beams=16)
    assert pts.shape == (16, 2)
    # Beam 0 sits at the TE, beam 8 at the LE (symmetric section, even count).
    assert abs(pts[0, 0] - (1.0 - spec.pivot_frac) * spec.root_chord) < 1e-3
    assert abs(pts[0, 1]) < 1e-6
    assert abs(pts[8, 0] - (-spec.pivot_frac * spec.root_chord)) < 1e-3
    assert abs(pts[8, 1]) < 1e-6
    # Upper/lower symmetry: beam i mirrors beam (16 - i) in y.
    assert np.allclose(pts[1, 1], -pts[15, 1], atol=1e-6)


def test_resample_with_arc_fractions_matches_even_default():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    from wing_design.beams.cross_section import resample_closed_polyline
    a = resample_closed_polyline(sq, 8)
    b = resample_closed_polyline(sq, 8, arc_fractions=np.arange(8) / 8.0)
    assert np.allclose(a, b)
