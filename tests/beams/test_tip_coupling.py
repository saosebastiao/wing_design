import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.beams.shell_model import build_beam_shell_model, solve_beam_shell_model
from wing_design.beams.tip_coupling import tip_clique_elements, solve_beam_shell_tip_coupled


def test_clique_element_count():
    nb = 8
    elems = tip_clique_elements(np.arange(nb))
    assert elems.shape == (nb * (nb - 1) // 2, 2)
    assert elems.min() >= 0 and elems.max() < nb


def test_no_gusset_matches_plain_solve():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=6, n_levels=4, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 50.0
    plain = solve_beam_shell_model(model, loads)
    none_res, _ = solve_beam_shell_tip_coupled(model, loads, gusset_radius=None)
    assert np.allclose(plain.displacements, none_res.displacements, atol=1e-12)


def test_gusset_solve_valid_and_sliced():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 100.0
    res, n_beam = solve_beam_shell_tip_coupled(model, loads, gusset_radius=0.08)
    assert n_beam == model.beam_elements.shape[0]
    # internal forces sliced to the DESIGN beams only (clique excluded)
    assert res.axial_force.shape == (n_beam,)
    assert np.isfinite(res.displacements).all()
    assert np.allclose(res.displacements[model.fixed_nodes], 0.0)


def test_gusset_couples_in_plane_tip_motion():
    # In-plane asymmetric tip load: the stiff clique ties tip nodes axially, so the
    # in-plane (chordwise, x) spread of tip-node displacement shrinks vs skin-only.
    # (Spanwise/z coupling is already provided by the skin, so the gusset's added
    # effect shows up in-plane.)
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes[0], 0] = 500.0     # push ONE tip node chordwise
    base = solve_beam_shell_model(model, loads)
    coupled, _ = solve_beam_shell_tip_coupled(model, loads, gusset_radius=0.08)
    spread_base = np.std(base.displacements[model.tip_nodes, 0])
    spread_coupled = np.std(coupled.displacements[model.tip_nodes, 0])
    assert spread_coupled < spread_base   # the gusset stiffens in-plane tip coupling
