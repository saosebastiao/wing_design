"""V.1 shadow prices: SLSQP KKT multipliers converted to physical kg-per-unit.

The conversion (normalized constraint g = 1 - v/L, normalized objective m/m_ref ->
dm*/dL = -m_ref * lambda / L) is validated by central-differencing one limit on a
small problem: re-optimize at L*(1 +/- h) warm-started from the optimum and compare
the mass slope against the reported shadow price (nlp-sizing skill requirement).
Optimization-level FD is noisy (ftol + basin scatter), so the tolerance is loose
compared to gradient FD tests — sign and magnitude are the contract.
"""
import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import T700_EPOXY
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig,
    size_beam_shell_laminate,
)


def _x0_from_result(model, r):
    group_of_element, G = beam_radius_groups(model)
    r_group = np.zeros(G)
    seen = set()
    for e, g in enumerate(group_of_element):
        if g not in seen:
            r_group[g] = r.radii[e]
            seen.add(g)
    return np.concatenate([r_group, np.asarray(r.t_bands, float),
                           [float(r.f0_bands[0])], [float(r.f45_bands[0])]])


def _twist_bound_setup():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 0] = 500.0
    loads[model.tip_nodes, 2] = 500.0
    return model, loads


def _cfg(twist_max):
    return LaminateSizingConfig(
        sigma_allow_Pa=1.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=twist_max,
        r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.02,
        use_analytic_jacobian=True,
    )


def test_shadow_prices_reported_with_expected_signs():
    model, loads = _twist_bound_setup()
    res = size_beam_shell_laminate(model, [loads], _cfg(0.02), ply=T700_EPOXY,
                                   rho=1550.0, maxiter=150, ftol=1.0e-8)
    sp = res.shadow_prices
    assert sp is not None
    # All limit-type prices are dm*/dlimit <= 0 (relaxing a limit never adds mass).
    assert sp["tip_twist_max_deg"] < 0.0          # binding -> strictly negative
    assert sp["tip_defl_max_m"] <= 0.0
    assert sp["sigma_allow_Pa"] <= 0.0
    # No buckling constraint configured -> no buckling shadow prices.
    assert "buckling_sf_beam" not in sp and "buckling_sf_panel" not in sp
    # Slack deflection limit (1 m on a ~5 m wing) must carry ~zero price vs twist's.
    assert abs(sp["tip_defl_max_m"]) < 0.01 * abs(sp["tip_twist_max_deg"])


def test_defl_shadow_price_matches_finite_difference():
    # Deflection (unlike twist, which the optimizer buys with mass-free layup
    # fractions) can only be bought with stiffness = mass, so its shadow price is
    # strictly nonzero and FD-checkable.
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 800.0

    def cfg(defl_max):
        return LaminateSizingConfig(
            sigma_allow_Pa=1.0e9, tip_defl_max_m=defl_max, tip_twist_max_deg=90.0,
            r_min=0.004, r_max=0.03, t_min=0.0005, t_max=0.02,
            use_analytic_jacobian=True,
        )

    # 0.6 mm binds hard on this 4x3 model at 800 N (min-gauge defl is ~1.25 mm at
    # 10 mm limit -> slack; measured 2026-06-10: shadow -9841 kg/m, FD relerr 0.5%
    # at h=5%, 0.02% at h=2%).
    defl_max = 0.0006
    res = size_beam_shell_laminate(model, [loads], cfg(defl_max), ply=T700_EPOXY,
                                   rho=1550.0, maxiter=200, ftol=1.0e-8)
    assert res.converged
    assert res.tip_defl_m > 0.99 * defl_max          # the limit actually binds
    sp = res.shadow_prices["tip_defl_max_m"]
    assert sp < 0.0

    h = 0.05 * defl_max
    x0 = _x0_from_result(model, res)
    masses = []
    for dm in (defl_max - h, defl_max + h):
        r = size_beam_shell_laminate(model, [loads], cfg(dm), ply=T700_EPOXY,
                                     rho=1550.0, maxiter=200, ftol=1.0e-8, x0=x0)
        assert r.converged
        masses.append(r.mass_kg)
    fd = (masses[1] - masses[0]) / (2.0 * h)

    assert fd < 0.0
    assert abs(sp - fd) <= 0.10 * abs(fd), f"shadow={sp:.4f} vs FD={fd:.4f} kg/m"


def test_buckling_shadow_prices_present_and_nonnegative():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=4, n_levels=3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 800.0
    loads[model.tip_nodes, 0] = 400.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=1.5, use_analytic_jacobian=True,
    )
    res = size_beam_shell_laminate(model, [loads], cfg, ply=T700_EPOXY,
                                   rho=1550.0, maxiter=120)
    sp = res.shadow_prices
    assert sp is not None
    # Raising a safety factor can only cost mass: dm*/dSF >= 0.
    assert sp["buckling_sf_beam"] >= 0.0
    assert sp["buckling_sf_panel"] >= 0.0
    # Something must bind: total price over all constraints is nonzero.
    total = sum(abs(v) for v in sp.values())
    assert total > 0.0
