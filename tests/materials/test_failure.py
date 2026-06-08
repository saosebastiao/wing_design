import numpy as np

from wing_design.materials.unidir import T700_EPOXY
from wing_design.materials.failure import (
    tsai_wu_coefficients, tsai_wu_index, tsai_wu_strength_ratio,
    ply_strength_ratio, laminate_min_strength_ratio,
)

PLY = T700_EPOXY


def test_index_unit_at_uniaxial_tension():
    assert np.isclose(tsai_wu_index([PLY.Xt_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)


def test_strength_ratio_one_at_each_strength_point():
    assert np.isclose(tsai_wu_strength_ratio([PLY.Xt_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([-PLY.Xc_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, PLY.Yt_Pa, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, -PLY.Yc_Pa, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, 0.0, PLY.S12_Pa], PLY), 1.0, rtol=1e-9)


def test_strength_ratio_scales_inverse_with_shear():
    r1 = tsai_wu_strength_ratio([0.0, 0.0, 1.0e7], PLY)
    r2 = tsai_wu_strength_ratio([0.0, 0.0, 5.0e6], PLY)
    assert np.isclose(r2, 2.0 * r1, rtol=1e-9)


def test_zero_stress_is_safe():
    assert tsai_wu_strength_ratio([0.0, 0.0, 0.0], PLY) >= 1.0e8


def test_f12_convention():
    F1, F2, F11, F22, F66, F12 = tsai_wu_coefficients(PLY)
    assert np.isclose(F12, -0.5 * np.sqrt(F11 * F22), rtol=1e-12)


def test_ply_strength_ratio_fibre_vs_transverse():
    eps = [1.0e-3, 0.0, 0.0]
    r0 = ply_strength_ratio(PLY, eps, 0.0)
    r90 = ply_strength_ratio(PLY, eps, 90.0)
    assert r0 > r90


def test_laminate_min_skips_absent_orientations():
    eps = [1.0e-3, 0.0, 0.0]
    r_all = laminate_min_strength_ratio(PLY, eps, f0=0.34, f45=0.33, f90=0.33, offset_deg=0.0)
    r_0only = laminate_min_strength_ratio(PLY, eps, f0=1.0, f45=0.0, f90=0.0, offset_deg=0.0)
    assert r_0only >= r_all
    assert np.isclose(r_0only, ply_strength_ratio(PLY, eps, 0.0), rtol=1e-9)
