import numpy as np
import pytest

from wing_design.geometry import WingSpec
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import skin_areas, skin_band_map, skin_band_areas


def _model(n_beams=8, n_levels=6):
    return build_beam_shell_model(WingSpec(), n_beams=n_beams, n_levels=n_levels, beam_radius=0.02)


def test_band_map_single_band_all_zero():
    m = _model()
    bm = skin_band_map(m, 1)
    assert bm.shape == (m.shell_tris.shape[0],)
    assert np.all(bm == 0)


def test_band_map_contiguous_and_complete():
    m = _model(n_beams=8, n_levels=6)  # 5 levels -> bands
    n_bands = 3
    bm = skin_band_map(m, n_bands)
    assert bm.min() == 0 and bm.max() == n_bands - 1
    assert set(np.unique(bm)) == set(range(n_bands))
    n_beams = m.n_beams
    e = np.arange(m.shell_tris.shape[0])
    level = (e // 2) // n_beams
    for lv in range(m.n_levels - 1):
        bands_at_level = np.unique(bm[level == lv])
        assert bands_at_level.shape == (1,)
    level_band = np.array([bm[level == lv][0] for lv in range(m.n_levels - 1)])
    assert np.all(np.diff(level_band) >= 0)


def test_band_map_invalid_raises():
    m = _model(n_beams=8, n_levels=6)  # 5 segment-levels
    with pytest.raises(ValueError):
        skin_band_map(m, 0)
    with pytest.raises(ValueError):
        skin_band_map(m, 6)  # > n_levels-1


def test_band_areas_sum_to_total():
    m = _model(n_beams=8, n_levels=6)
    bm = skin_band_map(m, 3)
    ba = skin_band_areas(m, bm, 3)
    assert ba.shape == (3,)
    assert np.isclose(ba.sum(), skin_areas(m).sum())


from wing_design.structural.buckling import panel_buckling_utilization


def test_panel_buckling_accepts_array_t():
    # two compressive panels
    ms = np.array([[-1.0e7, 0.0, 0.0], [-1.0e7, 0.0, 0.0]])
    areas = np.array([0.01, 0.01])
    D11 = 50.0
    u_scalar = panel_buckling_utilization(ms, areas, D11=D11, t=0.002, kc=4.0, safety_factor=1.0)
    u_array = panel_buckling_utilization(ms, areas, D11=D11, t=np.array([0.002, 0.002]),
                                         kc=4.0, safety_factor=1.0)
    assert np.allclose(u_scalar, u_array)
    # With D11 held fixed, sigma_cr = kc*pi^2*D11/(b^2*t) so utilization scales as t
    # element-wise: halving t halves utilization. (Confirms array-t broadcasting.)
    u_thin = panel_buckling_utilization(ms, areas, D11=D11, t=np.array([0.002, 0.001]),
                                        kc=4.0, safety_factor=1.0)
    assert np.isclose(u_thin[1], 0.5 * u_thin[0])
