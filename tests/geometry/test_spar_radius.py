import numpy as np

from wing_design.geometry import small_wingsail


def test_spar_radius_is_floored_inscribed_circle():
    spec = small_wingsail  # NACA0018, root_chord=1, pivot 0.25c
    # Max inscribed circle centered at the pivot ≈ local half-thickness ≈ 0.089 m,
    # floored to the nearest cm → 0.08 m.
    assert abs(spec.spar_radius - 0.08) < 1e-9
    # It is floored, so it never exceeds the true min distance from pivot to surface.
    from wing_design.geometry.airfoil import naca_00xx_coords
    coords = (naca_00xx_coords(spec.thickness, n=spec.n_airfoil_points)
              - np.array([spec.pivot_frac, 0.0])) * spec.root_chord
    true_min = float(np.min(np.hypot(coords[:, 0], coords[:, 1])))
    assert spec.spar_radius <= true_min + 1e-12


def test_spar_diameter_is_twice_radius():
    spec = small_wingsail
    assert abs(spec.spar_diameter - 2.0 * spec.spar_radius) < 1e-12
