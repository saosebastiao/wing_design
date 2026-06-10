import pytest

pytestmark = pytest.mark.sizing  # full SLSQP sizing runs (measured 15-99 s each)

import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import T700_EPOXY
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig, size_beam_shell_laminate, laminate_result_is_feasible,
)


def _problem():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 800.0
    loads[model.tip_nodes, 0] = 400.0
    return model, [loads]


def test_default_is_fd():
    # use_analytic_jacobian defaults False; a small sizing runs & is feasible.
    model, load_arrays = _problem()
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=1.5, n_skin_bands=2,
    )
    assert cfg.use_analytic_jacobian is False
    res = size_beam_shell_laminate(model, load_arrays, cfg, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    assert laminate_result_is_feasible(res, cfg, tol=5e-3)


def test_analytic_matches_fd_optimum():
    model, load_arrays = _problem()
    base = dict(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=1.5, n_skin_bands=2,
    )
    cfg_fd = LaminateSizingConfig(**base, use_analytic_jacobian=False)
    cfg_an = LaminateSizingConfig(**base, use_analytic_jacobian=True)
    r_fd = size_beam_shell_laminate(model, load_arrays, cfg_fd, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    r_an = size_beam_shell_laminate(model, load_arrays, cfg_an, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    assert laminate_result_is_feasible(r_fd, cfg_fd, tol=5e-3)
    assert laminate_result_is_feasible(r_an, cfg_an, tol=5e-3)
    assert np.isclose(r_fd.mass_kg, r_an.mass_kg, rtol=2e-2)


def test_analytic_with_tsai_wu_matches_fd():
    model, load_arrays = _problem()
    base = dict(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=1.5, n_skin_bands=2,
        skin_failure="tsai_wu", tsai_wu_safety_factor=1.5,
    )
    cfg_fd = LaminateSizingConfig(**base, use_analytic_jacobian=False)
    cfg_an = LaminateSizingConfig(**base, use_analytic_jacobian=True)
    r_fd = size_beam_shell_laminate(model, load_arrays, cfg_fd, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    r_an = size_beam_shell_laminate(model, load_arrays, cfg_an, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    assert laminate_result_is_feasible(r_fd, cfg_fd, tol=5e-3)
    assert laminate_result_is_feasible(r_an, cfg_an, tol=5e-3)
    assert np.isclose(r_fd.mass_kg, r_an.mass_kg, rtol=2e-2)
