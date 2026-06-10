import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_lengths, skin_areas, beam_mass, skin_mass, BeamShellSizingConfig, size_beam_shell


def test_beam_lengths_match_geometry():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    L = beam_lengths(model)
    assert L.shape[0] == model.beam_elements.shape[0]
    i, j = model.beam_elements[0]
    assert abs(L[0] - np.linalg.norm(model.nodes[j] - model.nodes[i])) < 1e-12


def test_beam_mass_uniform():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    L = beam_lengths(model)
    r, rho = 0.02, 1550.0
    m = beam_mass(model, np.full(model.beam_elements.shape[0], r), rho=rho)
    assert abs(m - rho * np.pi * r**2 * L.sum()) / m < 1e-12


def test_skin_mass_scales_with_thickness():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    A = skin_areas(model).sum()
    rho = 1550.0
    assert abs(skin_mass(model, 0.003, rho=rho) - rho * 0.003 * A) / (rho * 0.003 * A) < 1e-12
    assert abs(skin_mass(model, 0.006, rho=rho) - 2 * skin_mass(model, 0.003, rho=rho)) < 1e-9


def test_size_beam_shell_reduces_mass_and_feasible():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    n = model.beam_elements.shape[0]
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 10.0
    cfg = BeamShellSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.01,
    )
    rho = 1550.0
    start = beam_mass(model, np.full(n, cfg.r_max), rho=rho) + skin_mass(model, cfg.t_max, rho=rho)
    res = size_beam_shell(model, [loads], cfg, rho=rho, maxiter=40)
    assert res.radii.shape == (n,)
    assert cfg.t_min - 1e-9 <= res.t_skin <= cfg.t_max + 1e-9
    assert res.mass_kg < start
    assert res.max_beam_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert res.max_skin_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert np.all(res.radii >= cfg.r_min - 1e-9) and np.all(res.radii <= cfg.r_max + 1e-9)


def test_size_beam_shell_binding_constraint():
    # Heavy load + tight allowable: a stress constraint must bind, so the optimizer
    # is pushed off minimum gauge (not the trivial all-min solution).
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 5000.0
    cfg = BeamShellSizingConfig(
        sigma_allow_Pa=1.0e7, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.05, t_min=0.0005, t_max=0.02,
    )
    res = size_beam_shell(model, [loads], cfg, rho=1550.0, maxiter=80)
    worst = max(res.max_beam_vm_Pa, res.max_skin_vm_Pa)
    assert worst <= cfg.sigma_allow_Pa * 1.05      # feasible
    assert worst > 0.5 * cfg.sigma_allow_Pa        # genuinely constraint-limited, not min-gauge


@pytest.mark.sizing  # measured 8 s (2026-06-10)
def test_buckling_constraint_respected():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=4)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 800.0
    loads[model.tip_nodes, 0] = 400.0
    cfg = BeamShellSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=1.5,
    )
    res = size_beam_shell(model, [loads], cfg, rho=1550.0, maxiter=80)
    assert res.max_beam_buckling_util <= 1.05
    assert res.max_panel_buckling_util <= 1.05
