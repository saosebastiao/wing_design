import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_lengths, skin_areas, beam_mass, skin_mass


def test_beam_lengths_match_geometry():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    L = beam_lengths(model)
    assert L.shape[0] == model.beam_elements.shape[0]
    i, j = model.beam_elements[0]
    assert abs(L[0] - np.linalg.norm(model.nodes[j] - model.nodes[i])) < 1e-12


def test_beam_mass_uniform():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    L = beam_lengths(model)
    r, rho = 0.02, 1550.0
    m = beam_mass(model, np.full(model.beam_elements.shape[0], r), rho=rho)
    assert abs(m - rho * np.pi * r**2 * L.sum()) / m < 1e-12


def test_skin_mass_scales_with_thickness():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    A = skin_areas(model).sum()
    rho = 1550.0
    assert abs(skin_mass(model, 0.003, rho=rho) - rho * 0.003 * A) / (rho * 0.003 * A) < 1e-12
    assert abs(skin_mass(model, 0.006, rho=rho) - 2 * skin_mass(model, 0.003, rho=rho)) < 1e-9
