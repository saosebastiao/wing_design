import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import T700_EPOXY
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.laminate_sizing import LaminateSizingConfig, size_beam_shell_laminate


def test_clt_cosizing_feasible_and_valid_fractions():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    n = model.beam_elements.shape[0]
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 200.0
    loads[model.tip_nodes, 0] = 200.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=2.0,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.01,
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    assert res.radii.shape == (n,)
    assert res.f0 >= -1e-6 and res.f45 >= -1e-6 and res.f90 >= -1e-6
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6
    assert cfg.t_min - 1e-9 <= res.t_skin <= cfg.t_max + 1e-9
    assert res.max_beam_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert res.max_skin_vm_Pa <= cfg.sigma_allow_Pa * 1.05


def test_clt_tailors_layup_when_twist_binds():
    # A torsion-dominated load with a tight twist limit should drive the optimizer
    # to a ±45-rich layup (maximizing skin shear stiffness) — above the quasi-iso
    # 1/3 start it begins from.
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 0] = 500.0   # chordwise
    loads[model.tip_nodes, 2] = 500.0   # normal — offset pair drives twist
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=0.02,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.02,
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=120)
    assert res.f45 > 0.45                                  # layup tailored toward ±45
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6    # valid partition


def test_laminate_buckling_constraint_respected():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=4)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 800.0
    loads[model.tip_nodes, 0] = 400.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=1.5,
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=80)
    assert res.max_beam_buckling_util <= 1.05
    assert res.max_panel_buckling_util <= 1.05
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6


def test_clt_datum_sizing_runs_and_valid():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 200.0
    loads[model.tip_nodes, 0] = 200.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=2.0,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.01,
        ply_angle_datum=(0.0, 0.0, 1.0),
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=60)
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6
    assert res.max_beam_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert res.max_skin_vm_Pa <= cfg.sigma_allow_Pa * 1.05


def test_clt_datum_with_buckling_feasible():
    # The datum + buckling combination is the "final design" path; verify the
    # per-triangle D11 (array) feeds panel buckling correctly and stays feasible.
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 400.0
    loads[model.tip_nodes, 0] = 200.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5,
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY, rho=1550.0, maxiter=80)
    assert res.max_beam_buckling_util <= 1.05
    assert res.max_panel_buckling_util <= 1.05
    assert abs(res.f0 + res.f45 + res.f90 - 1.0) < 1e-6
