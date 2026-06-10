"""V.3b: physical panel-strip widths + strip-mode panel buckling in the sizer."""
import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import T700_EPOXY
from wing_design.beams.shell_model import build_beam_shell_model, skin_panel_widths
from wing_design.beams.shell_sizing import skin_areas
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig,
    size_beam_shell_laminate,
)


def test_widths_match_chordwise_beam_spacing():
    model = build_beam_shell_model(small_wingsail, n_beams=8, n_levels=4)
    w = skin_panel_widths(model)
    assert w.shape == (model.shell_tris.shape[0],)
    assert np.all(w > 0.0)
    # every triangle classified into a 2-beam strip (no sqrt-area fallback):
    # widths must equal the node spacing between its two bounding beams.
    nl = model.n_levels
    for e in range(model.shell_tris.shape[0]):
        beams = np.unique(model.shell_tris[e] // nl)
        assert beams.size == 2
        ks = np.unique(model.shell_tris[e] % nl)
        ref = np.mean([np.linalg.norm(model.nodes[beams[0] * nl + k]
                                      - model.nodes[beams[1] * nl + k]) for k in ks])
        assert w[e] == pytest.approx(ref)


def test_widths_scale_inversely_with_beam_count():
    # The whole point of V.3b: b must scale ~1/n_beams (sqrt(area) only ~1/sqrt(n)).
    w8 = skin_panel_widths(build_beam_shell_model(small_wingsail, n_beams=8, n_levels=4))
    w16 = skin_panel_widths(build_beam_shell_model(small_wingsail, n_beams=16, n_levels=4))
    ratio = np.median(w8) / np.median(w16)
    assert 1.7 < ratio < 2.3


def test_widths_smaller_than_sqrt_area_on_long_panels():
    # On a span-coarse mesh the strips are long: physical width < sqrt(area),
    # so strip mode credits capacity (the V#1 conservatism being removed).
    model = build_beam_shell_model(small_wingsail, n_beams=8, n_levels=3)
    w = skin_panel_widths(model)
    b_old = np.sqrt(skin_areas(model))
    assert np.median(w / b_old) < 1.0


@pytest.mark.sizing  # full SLSQP sizing run (small model)
def test_strip_mode_sizing_feasible():
    model = build_beam_shell_model(small_wingsail, n_beams=8, n_levels=4)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 800.0
    loads[model.tip_nodes, 0] = 400.0
    def cfg(mode):
        return LaminateSizingConfig(
            sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
            r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
            buckling_safety_factor=1.5, use_analytic_jacobian=True,
            panel_width_mode=mode)
    strip = size_beam_shell_laminate(model, [loads], cfg("strip"), ply=T700_EPOXY,
                                     rho=1550.0, maxiter=120)
    assert strip.max_panel_buckling_util <= 1.05
    assert strip.max_beam_buckling_util <= 1.05
    legacy = size_beam_shell_laminate(model, [loads], cfg("sqrt_area"), ply=T700_EPOXY,
                                      rho=1550.0, maxiter=120)
    # Strip widths < sqrt(area) here -> higher sigma_cr -> never heavier.
    assert strip.mass_kg <= legacy.mass_kg * 1.01


def test_unknown_width_mode_raises():
    model = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 100.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        panel_width_mode="bogus")
    with pytest.raises(ValueError, match="panel_width_mode"):
        size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=5)
