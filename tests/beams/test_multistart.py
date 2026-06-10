import numpy as np
import pytest

from wing_design.scenario import small_scenario
from wing_design.beams import (
    LaminateSizingConfig, LaminateSizingResult, build_beam_frame, build_beam_shell_model,
    project_panels_to_beam_nodes, size_beam_shell_laminate,
    laminate_design_bounds, laminate_result_is_feasible,
)
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.materials.unidir import T700_EPOXY


def _cfg(**kw):
    base = dict(sigma_allow_Pa=1.0e9, tip_defl_max_m=0.1, tip_twist_max_deg=5.0)
    base.update(kw)
    return LaminateSizingConfig(**base)


def _result(**kw):
    base = dict(
        radii=np.array([0.01]), t_skin=0.002, t_bands=np.array([0.002]),
        f0=0.33, f45=0.34, f90=0.33,
        f0_bands=np.array([0.33]), f45_bands=np.array([0.34]), f90_bands=np.array([0.33]),
        mass_kg=10.0, beam_mass_kg=4.0, skin_mass_kg=6.0, converged=True, n_iter=10,
        max_beam_vm_Pa=1.0e8, max_skin_vm_Pa=1.0e8, tip_defl_m=0.05, tip_twist_deg=2.0,
        max_beam_buckling_util=0.9, max_panel_buckling_util=0.9, min_skin_strength_ratio=None,
    )
    base.update(kw)
    return LaminateSizingResult(**base)


def _small():
    P = small_scenario()
    spec = P.geometry
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    return P, spec, model, loads


def test_design_bounds_shape():
    P, spec, model, loads = _small()
    cfg = _cfg(n_skin_bands=3, per_band_layup=True)
    lo, hi = laminate_design_bounds(model, cfg)
    _, G = beam_radius_groups(model)
    assert lo.shape == hi.shape == (G + 3 + 2 * 3,)
    assert np.all(lo < hi)
    # fraction entries (the last 2*L) bounded [0,1]
    assert np.all(lo[G + 3:] == 0.0) and np.all(hi[G + 3:] == 1.0)


def test_feasible_check_true_and_false():
    cfg = _cfg(buckling_safety_factor=1.5)
    assert laminate_result_is_feasible(_result(), cfg) is True
    assert laminate_result_is_feasible(_result(max_beam_buckling_util=1.7), cfg) is False
    assert laminate_result_is_feasible(_result(tip_twist_deg=9.9), cfg) is False
    # tsai_wu skin: needs min_skin_strength_ratio >= SF
    cfg_tw = _cfg(skin_failure="tsai_wu", tsai_wu_safety_factor=2.0)
    assert laminate_result_is_feasible(_result(min_skin_strength_ratio=2.5), cfg_tw) is True
    assert laminate_result_is_feasible(_result(min_skin_strength_ratio=1.2), cfg_tw) is False


@pytest.mark.sizing  # measured 31 s (2026-06-10)
def test_x0_roundtrip_backward_compat():
    P, spec, model, loads = _small()
    cfg = _cfg(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
               tip_twist_max_deg=5.0)
    a = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15)
    lo, hi = laminate_design_bounds(model, cfg)
    _, G = beam_radius_groups(model)
    default = np.concatenate([np.full(G, cfg.r_max), np.full(1, cfg.t_max), [1/3], [1/3]])
    b = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=15, x0=default)
    assert np.allclose(a.radii, b.radii) and np.isclose(a.mass_kg, b.mass_kg)


import wing_design.beams.laminate_sizing as ls_mod
from wing_design.beams import size_beam_shell_laminate_multistart, MultiStartResult


def test_multistart_selects_best_feasible(monkeypatch):
    # stub returns canned results by call order; start 2 is lightest-but-infeasible.
    masses = [10.0, 8.0, 6.0, 9.0]
    feas =   [True, True, False, True]
    calls = {"i": 0}

    def fake(model, loads, config, *, ply, rho, maxiter, ftol, x0=None, panel_pressures=None):
        i = calls["i"]; calls["i"] += 1
        buck = 1.7 if not feas[i] else 0.9
        return _result(mass_kg=masses[i], max_beam_buckling_util=buck)

    monkeypatch.setattr(ls_mod, "size_beam_shell_laminate", fake)
    cfg = _cfg(buckling_safety_factor=1.5)
    res = size_beam_shell_laminate_multistart(None, None, cfg, ply=None, rho=1600.0,
                                              n_starts=4, seed=0, maxiter=5)
    assert isinstance(res, MultiStartResult)
    assert res.n_starts == 4 and res.n_feasible == 3
    assert res.best.mass_kg == 8.0          # lightest FEASIBLE (6.0 was infeasible)
    assert res.best_start_index == 1
    assert len(res.start_masses) == 4 and len(res.start_feasible) == 4


def test_multistart_deterministic(monkeypatch):
    seen = []

    def fake(model, loads, config, *, ply, rho, maxiter, ftol, x0=None, panel_pressures=None):
        # mass depends on x0 so different starts differ; record x0 to check determinism
        seen.append(None if x0 is None else float(np.sum(x0)))
        m = 10.0 if x0 is None else float(5.0 + np.sum(x0))
        return _result(mass_kg=m)

    monkeypatch.setattr(ls_mod, "size_beam_shell_laminate", fake)
    cfg = _cfg(n_skin_bands=2)
    P = small_scenario()
    model = build_beam_shell_model(P.geometry, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    r1 = size_beam_shell_laminate_multistart(model, [], cfg, ply=None, rho=1600.0, n_starts=3, seed=42)
    r2 = size_beam_shell_laminate_multistart(model, [], cfg, ply=None, rho=1600.0, n_starts=3, seed=42)
    assert r1.best_start_index == r2.best_start_index
    assert r1.start_masses == r2.start_masses


def test_multistart_start0_is_default(monkeypatch):
    x0s = []

    def fake(model, loads, config, *, ply, rho, maxiter, ftol, x0=None, panel_pressures=None):
        x0s.append(x0)
        return _result(mass_kg=10.0)

    monkeypatch.setattr(ls_mod, "size_beam_shell_laminate", fake)
    cfg = _cfg(n_skin_bands=2)
    P = small_scenario()
    model = build_beam_shell_model(P.geometry, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    size_beam_shell_laminate_multistart(model, [], cfg, ply=None, rho=1600.0, n_starts=3, seed=1)
    assert x0s[0] is None                    # start 0 uses the sizer's default guess
    assert all(x is not None for x in x0s[1:])


def test_multistart_random_starts_on_simplex(monkeypatch):
    # Regression: make_start must project random starts onto the f0+f45<=1 simplex using
    # the symmetric group count G (not n). Pre-fix, the slice indexed past the vector and
    # the projection silently no-op'd, leaving off-simplex random starts.
    import wing_design.beams.laminate_sizing as ls_mod2
    from wing_design.beams.shell_sizing import beam_radius_groups
    captured = []

    def fake(model, loads, config, *, ply, rho, maxiter, ftol, x0=None, panel_pressures=None):
        captured.append(x0)
        return _result(mass_kg=10.0)

    monkeypatch.setattr(ls_mod2, "size_beam_shell_laminate", fake)
    P = small_scenario()
    model = build_beam_shell_model(P.geometry, n_beams=8, n_levels=5, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    cfg = _cfg(n_skin_bands=2)
    size_beam_shell_laminate_multistart(model, [], cfg, ply=None, rho=1600.0, n_starts=4, seed=3)
    _, G = beam_radius_groups(model)
    B, L = 2, 1
    f0_lo, f45_lo = G + B, G + B + L
    for x in captured[1:]:  # start 0 is None (default)
        assert x is not None and x.shape[0] == G + B + 2 * L
        assert np.all(x[f0_lo:f0_lo + L] + x[f45_lo:f45_lo + L] <= 1.0 + 1e-9)


@pytest.mark.sizing  # real (tiny) SLSQP runs in subprocesses
def test_parallel_multistart_matches_serial():
    # V.0.4: the process-pool map must reproduce the serial map exactly — starts
    # are generated up front from the seed and are independent.
    import numpy as np
    from wing_design.geometry import small_wingsail
    from wing_design.materials.unidir import T700_EPOXY
    from wing_design.beams.shell_model import build_beam_shell_model
    from wing_design.beams.laminate_sizing import (
        LaminateSizingConfig, size_beam_shell_laminate_multistart)

    model = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 400.0
    loads[model.tip_nodes, 0] = 200.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        buckling_safety_factor=1.5, use_analytic_jacobian=True)
    kw = dict(ply=T700_EPOXY, rho=1550.0, n_starts=3, seed=7, maxiter=40)
    ser = size_beam_shell_laminate_multistart(model, [loads], cfg, **kw)
    par = size_beam_shell_laminate_multistart(model, [loads], cfg, n_workers=3, **kw)
    assert par.start_masses == ser.start_masses
    assert par.best_start_index == ser.best_start_index
    assert par.best.mass_kg == ser.best.mass_kg
