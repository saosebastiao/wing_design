import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.beams.cross_section import beam_section_points
from wing_design.beams.sections import oml_outward_normals, lens_section_polyline


def test_normals_unit_and_outward():
    spec = small_wingsail
    P = beam_section_points(spec, 1.0, 16)
    n = oml_outward_normals(spec, 1.0, 16)
    assert n.shape == (16, 2)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)
    # outward: each normal points away from the pivot (positive dot with the radial dir)
    radial = P / np.linalg.norm(P, axis=1, keepdims=True)
    assert np.all(np.sum(n * radial, axis=1) > 0.0)


def test_lens_area_is_exact_default_and_fine():
    center = np.array([0.3, 0.05])
    normal = np.array([0.0, 1.0])
    area = 1.2e-3
    for n_arc in (8, 24, 64):
        poly = lens_section_polyline(center, normal, area, n_arc=n_arc)
        x, y = poly[:, 0], poly[:, 1]
        shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert abs(shoelace - area) / area < 1e-9, f"n_arc={n_arc}: {shoelace} vs {area}"


def test_lens_bulges_inward():
    center = np.array([0.3, 0.05])
    normal = np.array([0.0, 1.0])   # outward = +y, so the lens must bulge toward -y
    area = 1.2e-3
    poly = lens_section_polyline(center, normal, area, n_arc=64)
    assert poly[:, 1].min() < center[1]                 # bulges below the chord
    assert abs(poly[:, 1].max() - center[1]) < 1e-9     # flat (outer) edge on the surface
