import numpy as np

from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups


def _model(n_beams=8, n_levels=5, arc_fractions=None):
    P = small_scenario()
    return build_beam_shell_model(P.geometry, n_beams=n_beams, n_levels=n_levels,
                                  beam_radius=0.02, arc_fractions=arc_fractions)


def test_symmetric_default_groups_reduced():
    m = _model(n_beams=8, n_levels=5)
    g, ng = beam_radius_groups(m)
    n = m.beam_elements.shape[0]
    assert g.shape == (n,)
    # even n_beams: U = n_beams//2 + 1 unique beams
    assert ng == (8 // 2 + 1) * (5 - 1)
    # mirror beams b and 8-b share group ids per level
    seg = 5 - 1
    for b in range(1, 4):
        mb = (8 - b) % 8
        assert np.array_equal(g[b * seg:(b + 1) * seg], g[mb * seg:(mb + 1) * seg])
    # chord-line beams 0 and 4 are NOT shared with any other beam
    g0 = set(g[0:seg]); g4 = set(g[4 * seg:5 * seg])
    others = set(g) - g0 - g4
    assert g0.isdisjoint(others) and g4.isdisjoint(others)


def test_asymmetric_placement_falls_back():
    nb, nl = 8, 5
    # deliberately asymmetric arc placement (not mirror-symmetric)
    af = np.linspace(0.0, 1.0, nb, endpoint=False) ** 1.5
    m = _model(n_beams=nb, n_levels=nl, arc_fractions=af)
    g, ng = beam_radius_groups(m)
    n = m.beam_elements.shape[0]
    assert ng == n
    assert np.array_equal(g, np.arange(n))
