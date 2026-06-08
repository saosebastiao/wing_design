import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.shell_model import build_beam_shell_model, solve_beam_shell_model
from wing_design.beams.tip_coupling import tip_clique_elements, solve_beam_shell_tip_coupled


def test_clique_element_count():
    nb = 8
    elems = tip_clique_elements(np.arange(nb))
    assert elems.shape == (nb * (nb - 1) // 2, 2)
    assert elems.min() >= 0 and elems.max() < nb


def test_tip_coupling_reduces_relative_tip_motion():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes[0], 2] = 500.0     # push ONE tip node only
    base = solve_beam_shell_model(model, loads)
    coupled, n_beam = solve_beam_shell_tip_coupled(model, loads, gusset_radius=0.08)
    tip = model.tip_nodes
    spread_base = np.std(base.displacements[tip, 2])
    spread_coupled = np.std(coupled.displacements[tip, 2])
    assert spread_coupled < 0.5 * spread_base
    assert n_beam == model.beam_elements.shape[0]


def test_no_gusset_matches_plain_solve():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=6, n_levels=4, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 50.0
    plain = solve_beam_shell_model(model, loads)
    none_res, _ = solve_beam_shell_tip_coupled(model, loads, gusset_radius=None)
    assert np.allclose(plain.displacements, none_res.displacements, atol=1e-12)
