import pytest

pytestmark = pytest.mark.sizing  # full SLSQP sizing runs (measured 15-99 s each)

import numpy as np

from wing_design.scenario import small_scenario
from wing_design.beams import (
    LaminateSizingConfig, build_beam_frame, build_beam_shell_model,
    project_panels_to_beam_nodes, size_beam_shell_laminate,
)
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.materials.unidir import T700_EPOXY


def _small(n_beams=8, n_levels=5):
    P = small_scenario()
    spec = P.geometry
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    return P, spec, model, loads


def test_global_layup_default_bands_broadcast():
    P, spec, model, loads = _small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0, n_skin_bands=3)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15)
    assert r.f0_bands.shape == (3,)
    assert np.allclose(r.f0_bands, r.f0_bands[0])  # global layup -> all bands equal
    assert np.isclose(r.f0, r.f0_bands[0])


def test_per_band_layup_feasible():
    P, spec, model, loads = _small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        buckling_safety_factor=1.5, n_skin_bands=3, per_band_layup=True)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    assert r.f0_bands.shape == (3,) and r.f45_bands.shape == (3,) and r.f90_bands.shape == (3,)
    assert np.all(r.f0_bands >= -1e-9) and np.all(r.f45_bands >= -1e-9) and np.all(r.f90_bands >= -1e-9)
    assert np.allclose(r.f0_bands + r.f45_bands + r.f90_bands, 1.0, atol=1e-6)


def test_per_band_layup_with_tsai_wu():
    P, spec, model, loads = _small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), skin_failure="tsai_wu", tsai_wu_safety_factor=2.0,
        n_skin_bands=3, per_band_layup=True)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    assert r.min_skin_strength_ratio is not None
    assert r.min_skin_strength_ratio >= 2.0 - 0.05
