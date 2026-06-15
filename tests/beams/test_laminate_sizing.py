import numpy as np
import pytest

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


@pytest.mark.sizing  # measured 7 s (2026-06-10)
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


def test_design_vector_from_result_roundtrip():
    # Warm-start helper: rebuild x from a result and reseed — including across a
    # feature-chain step that ADDS blocks (tube/hollow/sandwich) the prior
    # result lacks (those fall back to cold defaults).
    from wing_design.materials.unidir import PVC_H80
    from wing_design.beams.laminate_sizing import design_vector_from_result

    spec = small_wingsail
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=2.0e8, tip_defl_max_m=0.02, tip_twist_max_deg=2.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5,
        use_analytic_jacobian=True,
    )
    model0 = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model0.nodes.shape[0], 6))
    loads[model0.tip_nodes, 2] = 400.0
    loads[model0.tip_nodes, 0] = 200.0
    r0 = size_beam_shell_laminate(model0, [loads], cfg, ply=T700_EPOXY,
                                  rho=1550.0, maxiter=30)

    # same-config roundtrip: x has the right length and reseeds cleanly
    x_same = design_vector_from_result(model0, cfg, r0)
    r_same = size_beam_shell_laminate(model0, [loads], cfg, ply=T700_EPOXY,
                                      rho=1550.0, maxiter=2, x0=x_same)
    assert np.allclose(r_same.t_bands, r0.t_bands, rtol=0.5)

    # chain step: add tube + hollow + sandwich; missing blocks get defaults
    import dataclasses
    model1 = build_beam_shell_model(spec, n_beams=4, n_levels=3,
                                    core_tube=True, hollow_beams=True)
    loads1 = np.zeros((model1.nodes.shape[0], 6))
    loads1[model1.tip_nodes, 2] = 400.0
    loads1[model1.tip_nodes, 0] = 200.0
    cfg1 = dataclasses.replace(cfg, core=PVC_H80, ks_rho=50.0)
    x_up = design_vector_from_result(model1, cfg1, r0)
    r1 = size_beam_shell_laminate(model1, [loads1], cfg1, ply=T700_EPOXY,
                                  rho=1550.0, maxiter=2, x0=x_up)
    assert r1.t_core is not None and r1.r_tube is not None


def test_braced_cold_start_runs_without_length_mismatch():
    # Smoke test: braced cold-start (x0=None) must not crash with a length mismatch.
    # Uses small_wingsail + small_scenario() so it stays fast (~few seconds).
    from wing_design.geometry import small_wingsail
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.fea_model import project_panels_to_skin
    from wing_design.aero import build_airplane, sweep_envelope
    from wing_design import small_scenario
    from wing_design.beams.laminate_sizing import LaminateSizingConfig, size_beam_shell_laminate
    P = small_scenario()
    model = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3,
                                   lateral_bracing=True)
    ap = build_airplane(small_wingsail)
    env = sweep_envelope(ap, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
                               tip_defl_max_m=0.05 * small_wingsail.span,
                               tip_twist_max_deg=5.0)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=2)
    assert r is not None and np.isfinite(r.mass_kg)   # cold-start braced solve completes


def test_braced_solve_includes_brace_mass_in_total():
    # Production-path check: size_beam_shell_laminate (braced) must expose a positive
    # brace_mass_kg that is genuinely included in the reported mass_kg.
    # Uses the same tiny fixture as test_braced_cold_start_runs_without_length_mismatch
    # so it stays fast (~seconds).
    import numpy as np
    from wing_design.geometry import small_wingsail
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.fea_model import project_panels_to_skin
    from wing_design.aero import build_airplane, sweep_envelope
    from wing_design import small_scenario
    from wing_design.beams.laminate_sizing import LaminateSizingConfig, size_beam_shell_laminate
    P = small_scenario()
    model = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3,
                                   lateral_bracing=True)
    ap = build_airplane(small_wingsail)
    env = sweep_envelope(ap, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
                               tip_defl_max_m=0.05 * small_wingsail.span,
                               tip_twist_max_deg=5.0)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=2)
    assert r.brace_mass_kg > 0.0, (
        f"brace_mass_kg should be positive for a braced model, got {r.brace_mass_kg}"
    )
    # brace mass must be genuinely included in the reported total (components sum to total)
    parts = (r.beam_mass_kg + r.skin_mass_kg + r.tube_mass_kg
             + r.core_mass_kg + r.brace_mass_kg)
    assert abs(r.mass_kg - parts) <= 1e-6 * max(1.0, r.mass_kg), (
        f"mass_kg={r.mass_kg} != sum of parts={parts}; brace not included in total?"
    )


def test_brace_radius_block_present_and_bounded():
    import numpy as np
    from wing_design.geometry import medium_wingsail
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.laminate_sizing import LaminateSizingConfig, laminate_design_bounds
    m = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=6, lateral_bracing=True)
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=2.0e8, tip_defl_max_m=0.5, tip_twist_max_deg=5.0,
        brace_r_min=0.002, brace_r_max=0.05,
    )
    lo, hi = laminate_design_bounds(m, cfg)
    assert lo[-1] == 0.002 and hi[-1] == 0.05          # brace_radius is the LAST var
    m0 = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=6)   # unbraced
    lo0, _ = laminate_design_bounds(m0, cfg)
    assert lo.size == lo0.size + 1                      # exactly one extra var when braced


def test_brace_mass_and_objective_gradient_fd():
    import numpy as np
    from wing_design.geometry import medium_wingsail
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.laminate_sizing import (
        LaminateSizingConfig, laminate_design_bounds, _objective_and_grad_for_test)
    P_rho = 1600.0
    m = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=5, lateral_bracing=True)
    cfg = LaminateSizingConfig(sigma_allow_Pa=6e8, tip_defl_max_m=0.4, tip_twist_max_deg=5.0,
                               brace_r_min=0.002, brace_r_max=0.05)
    lo, hi = laminate_design_bounds(m, cfg)
    x = 0.5 * (lo + hi)
    f0, g = _objective_and_grad_for_test(m, cfg, x, rho=P_rho)
    e = np.zeros_like(x); e[-1] = 1.0
    fp, _ = _objective_and_grad_for_test(m, cfg, x + 1e-7 * e, rho=P_rho)
    fm, _ = _objective_and_grad_for_test(m, cfg, x - 1e-7 * e, rho=P_rho)
    fd = (fp - fm) / 2e-7
    assert abs(g[-1] - fd) <= 1e-5 * max(1.0, abs(fd))
    assert g[-1] > 0.0     # heavier brace => heavier objective


def test_no_separate_brace_vm_constraint():
    # Brace stress is constrained by the EXISTING beam_vm constraint (whose
    # element vector already spans the brace ring elements). There must be NO
    # separate brace_vm / brace_vm_ks constraint (the redundant one from
    # commit 0b20976 was removed).
    from wing_design.geometry import small_wingsail
    from wing_design import small_scenario
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.laminate_sizing import (
        LaminateSizingConfig, _constraint_names_for_test)
    P = small_scenario()
    model = build_beam_shell_model(small_wingsail, n_beams=6, n_levels=4,
                                   lateral_bracing=True)
    assert getattr(model, "brace_elements", None) is not None
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
                               tip_defl_max_m=0.05 * small_wingsail.span, tip_twist_max_deg=5.0,
                               ply_angle_datum=(0, 0, 1), ks_rho=50.0, buckling_safety_factor=1.5,
                               use_analytic_jacobian=True, beam_buckling_model="foundation",
                               panel_width_mode="strip")
    names = _constraint_names_for_test(model, cfg)
    assert not any("brace_vm" in nm for nm in names), \
        f"redundant brace_vm constraint still present: {names}"
    # the von-Mises constraint that covers braces is still there (vector form,
    # not KS-aggregated)
    assert "beam_vm" in names, f"beam_vm missing: {names}"


def test_design_vector_from_result_brace_roundtrip():
    import numpy as np
    from wing_design.geometry import small_wingsail
    from wing_design import small_scenario
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.fea_model import project_panels_to_skin
    from wing_design.aero import build_airplane, sweep_envelope
    from wing_design.beams.laminate_sizing import (
        LaminateSizingConfig, size_beam_shell_laminate, design_vector_from_result,
        laminate_design_bounds)
    from wing_design.materials.unidir import T700_EPOXY
    P = small_scenario()
    model = build_beam_shell_model(small_wingsail, n_beams=6, n_levels=4, lateral_bracing=True)
    env = sweep_envelope(build_airplane(small_wingsail), P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
                               tip_defl_max_m=0.05 * small_wingsail.span, tip_twist_max_deg=5.0)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=2)
    assert r.brace_radius is not None and r.brace_radius > 0.0
    x = design_vector_from_result(model, cfg, r)
    lo, hi = laminate_design_bounds(model, cfg)
    assert x.shape == lo.shape                       # full-length vector (incl brace block)
    assert np.isclose(x[-1], r.brace_radius)         # brace_radius round-trips as the last var


def test_braced_production_config_smoke():
    """Production-shaped braced config smoke test on a tiny mesh.

    Exercises lateral_bracing=True + core_tube=True + hollow_beams=True +
    sandwich core + accel_vectors + full production flags on a tiny mesh
    (n_beams=6, n_levels=4, maxiter=2). This covers the bug class where code
    iterating all beam_elements indexes form-only arrays (e.g. radii, areas).
    """
    import dataclasses
    import numpy as np
    from wing_design.geometry import small_wingsail
    from wing_design import small_scenario
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.fea_model import project_panels_to_skin, panel_pressure_per_tri
    from wing_design.aero import build_airplane, sweep_envelope
    from wing_design.beams.laminate_sizing import LaminateSizingConfig, size_beam_shell_laminate
    from wing_design.materials.unidir import T700_EPOXY, PVC_H80

    G0 = 9.81
    # gravity + one lateral case (mimics slam_braced production accels)
    accel_vectors = (
        (0.0, 0.0, -G0),
        (0.0, G0, -G0),
    )

    P = small_scenario()
    model = build_beam_shell_model(
        small_wingsail, n_beams=6, n_levels=4,
        core_tube=True, hollow_beams=True, lateral_bracing=True,
    )
    env = sweep_envelope(
        build_airplane(small_wingsail), P.load_cases,
        method="lifting_line", spanwise_resolution=P.aero.spanwise_resolution,
    )
    active = [ar for ar in env
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.05 * small_wingsail.span,
        tip_twist_max_deg=5.0,
        core=PVC_H80,
        ks_rho=50.0,
        accel_vectors=accel_vectors,
        beam_buckling_model="foundation",
        panel_d_mode="datum_ortho",
        panel_width_mode="strip",
        use_analytic_jacobian=True,
        buckling_safety_factor=1.5,
        ply_angle_datum=(0.0, 0.0, 1.0),
    )
    r = size_beam_shell_laminate(
        model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
        maxiter=2, panel_pressures=press,
    )
    assert np.isfinite(r.mass_kg), f"mass_kg not finite: {r.mass_kg}"
    assert r.brace_radius is not None and r.brace_radius > 0, (
        f"brace_radius should be positive, got {r.brace_radius}"
    )
