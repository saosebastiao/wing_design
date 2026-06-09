import numpy as np
from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model, model_with_tip_gusset
from wing_design.beams.tip_coupling import tip_clique_elements


def _m(**kw):
    return build_beam_shell_model(small_scenario().geometry, n_beams=8, n_levels=5, beam_radius=0.02, **kw)


def test_default_no_gusset():
    m = _m()
    assert m.tip_gusset_elements is None and m.tip_gusset_radius is None


def test_build_with_gusset():
    m = _m(tip_gusset_radius=0.05)
    assert m.tip_gusset_radius == 0.05
    exp = tip_clique_elements(m.tip_nodes)
    assert m.tip_gusset_elements.shape == exp.shape
    assert np.array_equal(m.tip_gusset_elements, exp)


def test_model_with_tip_gusset_roundtrip():
    m = _m()
    g = model_with_tip_gusset(m, 0.06)
    assert g.tip_gusset_radius == 0.06 and g.tip_gusset_elements is not None
    assert m.tip_gusset_elements is None  # original unchanged
