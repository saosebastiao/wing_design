import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.beams.shell_model import build_beam_shell_model, solve_beam_shell_model
from wing_design.structural.shell import recover_membrane_strain, recover_membrane_stress_C


def test_recover_membrane_strain_matches_identity_C():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 0] = 200.0
    res = solve_beam_shell_model(model, loads)
    eps = recover_membrane_strain(model.nodes, model.shell_tris, res.displacements)
    eps_via_C = recover_membrane_stress_C(model.nodes, model.shell_tris, res.displacements, C=np.eye(3))
    assert eps.shape == (model.shell_tris.shape[0], 3)
    assert np.allclose(eps, eps_via_C, atol=1e-12)


from wing_design.scenario import small_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig, build_beam_frame, build_beam_shell_model,
    project_panels_to_beam_nodes, size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY


def _scenario_small(n_beams=8, n_levels=5):
    P = small_scenario()
    spec = P.geometry
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    return P, spec, model, loads


@pytest.mark.sizing  # measured 16 s (2026-06-10)
def test_von_mises_default_has_none_ratio():
    P, spec, model, loads = _scenario_small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15)
    assert r.min_skin_strength_ratio is None


@pytest.mark.sizing  # measured 22 s (2026-06-10)
def test_tsai_wu_feasible_and_active():
    P, spec, model, loads = _scenario_small()
    sf = 2.0
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), skin_failure="tsai_wu", tsai_wu_safety_factor=sf)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    assert r.min_skin_strength_ratio is not None
    assert r.min_skin_strength_ratio >= sf - 0.05
