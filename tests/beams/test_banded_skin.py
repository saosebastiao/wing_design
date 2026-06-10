import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import skin_areas, skin_band_map, skin_band_areas


def _model(n_beams=8, n_levels=6):
    return build_beam_shell_model(small_wingsail, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02)


def test_band_map_single_band_all_zero():
    m = _model()
    bm = skin_band_map(m, 1)
    assert bm.shape == (m.shell_tris.shape[0],)
    assert np.all(bm == 0)


def test_band_map_contiguous_and_complete():
    m = _model(n_beams=8, n_levels=6)  # 5 levels -> bands
    n_bands = 3
    bm = skin_band_map(m, n_bands)
    assert bm.min() == 0 and bm.max() == n_bands - 1
    assert set(np.unique(bm)) == set(range(n_bands))
    n_beams = m.n_beams
    e = np.arange(m.shell_tris.shape[0])
    level = (e // 2) // n_beams
    for lv in range(m.n_levels - 1):
        bands_at_level = np.unique(bm[level == lv])
        assert bands_at_level.shape == (1,)
    level_band = np.array([bm[level == lv][0] for lv in range(m.n_levels - 1)])
    assert np.all(np.diff(level_band) >= 0)


def test_band_map_invalid_raises():
    m = _model(n_beams=8, n_levels=6)  # 5 segment-levels
    with pytest.raises(ValueError):
        skin_band_map(m, 0)
    with pytest.raises(ValueError):
        skin_band_map(m, 6)  # > n_levels-1


def test_band_areas_sum_to_total():
    m = _model(n_beams=8, n_levels=6)
    bm = skin_band_map(m, 3)
    ba = skin_band_areas(m, bm, 3)
    assert ba.shape == (3,)
    assert np.isclose(ba.sum(), skin_areas(m).sum())


from wing_design.structural.buckling import panel_buckling_utilization


def test_panel_buckling_accepts_array_t():
    # two compressive panels
    ms = np.array([[-1.0e7, 0.0, 0.0], [-1.0e7, 0.0, 0.0]])
    areas = np.array([0.01, 0.01])
    D11 = 50.0
    u_scalar = panel_buckling_utilization(ms, areas, D11=D11, t=0.002, kc=4.0, safety_factor=1.0)
    u_array = panel_buckling_utilization(ms, areas, D11=D11, t=np.array([0.002, 0.002]),
                                         kc=4.0, safety_factor=1.0)
    assert np.allclose(u_scalar, u_array)
    # With D11 held fixed, sigma_cr = kc*pi^2*D11/(b^2*t) so utilization scales as t
    # element-wise: halving t halves utilization. (Confirms array-t broadcasting.)
    u_thin = panel_buckling_utilization(ms, areas, D11=D11, t=np.array([0.002, 0.001]),
                                        kc=4.0, safety_factor=1.0)
    assert np.isclose(u_thin[1], 0.5 * u_thin[0])


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
def test_uniform_band_one_is_consistent():
    P, spec, model, loads = _scenario_small()
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0)  # n_skin_bands defaults to 1
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15)
    assert r.t_bands.shape == (1,)
    assert np.isclose(r.t_skin, r.t_bands[0])
    from wing_design.beams.shell_sizing import skin_areas
    assert np.isclose(r.skin_mass_kg, P.rho_kgm3 * r.t_skin * skin_areas(model).sum(), rtol=1e-6)


@pytest.mark.sizing  # measured 166 s (2026-06-10)
def test_banded_feasible_and_not_heavier():
    P, spec, model, loads = _scenario_small()
    base_cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0, buckling_safety_factor=1.5)
    band_cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0, buckling_safety_factor=1.5,
        n_skin_bands=4)
    uni = size_beam_shell_laminate(model, loads, base_cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    ban = size_beam_shell_laminate(model, loads, band_cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=40)
    assert ban.t_bands.shape == (4,)
    assert np.all(ban.t_bands >= band_cfg.t_min - 1e-9)
    assert np.all(ban.t_bands <= band_cfg.t_max + 1e-9)
    assert ban.mass_kg <= uni.mass_kg + 0.25
    from wing_design.beams.shell_sizing import skin_areas, skin_band_map
    bm = skin_band_map(model, 4)
    t_tri = ban.t_bands[bm]
    assert np.isclose(ban.skin_mass_kg, P.rho_kgm3 * np.sum(t_tri * skin_areas(model)), rtol=1e-6)


def test_band_helpers_exported():
    import wing_design.beams as b
    assert hasattr(b, "skin_band_map")
    assert hasattr(b, "skin_band_areas")
