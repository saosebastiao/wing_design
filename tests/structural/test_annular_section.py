"""P.1: annular (tube) beam section."""
import numpy as np
import pytest

from wing_design.structural.frame import BeamSection


def test_full_wall_equals_solid():
    # t = r degenerates to the solid circular section exactly.
    a = BeamSection.annular(0.03, 0.03)
    c = BeamSection.circular(0.03)
    assert a.A == pytest.approx(c.A)
    assert a.Iy == pytest.approx(c.Iy)
    assert a.Iz == pytest.approx(c.Iz)
    assert a.J == pytest.approx(c.J)
    assert a.r == c.r


def test_thin_wall_limits():
    r, t = 0.05, 0.001
    a = BeamSection.annular(r, t)
    assert a.A == pytest.approx(2 * np.pi * r * t, rel=2e-2)       # 2*pi*r*t
    assert a.Iy == pytest.approx(np.pi * r**3 * t, rel=5e-2)       # pi*r^3*t
    assert a.J == pytest.approx(2 * a.Iy)


def test_tube_beats_solid_rod_at_equal_mass():
    # The P.1 premise: at equal area (mass), the tube has ~(r_tube/t)x the I.
    solid = BeamSection.circular(0.01)
    r, t = 0.05, None
    # solve t for equal area: pi(2rt - t^2) = pi r_s^2
    t = r - np.sqrt(r**2 - solid.A / np.pi)
    tube = BeamSection.annular(r, t)
    assert tube.A == pytest.approx(solid.A, rel=1e-12)
    assert tube.Iy / solid.Iy > 20.0


def test_validity():
    with pytest.raises(ValueError):
        BeamSection.annular(0.02, 0.03)    # t > r
    with pytest.raises(ValueError):
        BeamSection.annular(0.02, 0.0)     # no wall
