import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.shell_model import build_beam_shell_model, solve_beam_shell_model
from wing_design.structural.shell import recover_membrane_strain, recover_membrane_stress_C


def test_recover_membrane_strain_matches_identity_C():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 0] = 200.0
    res = solve_beam_shell_model(model, loads)
    eps = recover_membrane_strain(model.nodes, model.shell_tris, res.displacements)
    eps_via_C = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=np.eye(3))
    assert eps.shape == (model.shell_tris.shape[0], 3)
    assert np.allclose(eps, eps_via_C, atol=1e-12)
