"""V.0.3/P#7 KS smoothing: property, FD validation, and KS-vs-hard optima."""
import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import PVC_H80, T700_EPOXY
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig,
    laminate_result_is_feasible,
    size_beam_shell_laminate,
)
from wing_design.beams.sensitivity import ks_aggregate

RHO = 1550.0
SF = 1.5


def test_ks_overestimates_max_and_converges():
    rng = np.random.default_rng(0)
    v = rng.uniform(-1.0, 1.0, size=200)
    for rho in (10.0, 50.0, 200.0):
        ks, w = ks_aggregate(v, rho)
        assert ks >= v.max() - 1e-12
        assert ks - v.max() <= np.log(len(v)) / rho + 1e-12
        assert w.sum() == pytest.approx(1.0)
    assert ks_aggregate(np.array([0.7]), 50.0)[0] == pytest.approx(0.7)


def _setup(core=None, tube=False):
    m = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3,
                               core_tube=tube, hollow_beams=tube)
    l1 = np.zeros((m.nodes.shape[0], 6))
    l1[m.tip_nodes, 2] = 800.0
    l1[m.tip_nodes, 0] = 400.0
    l2 = np.zeros((m.nodes.shape[0], 6))
    l2[m.tip_nodes, 2] = -600.0       # second case: different binding pattern
    l2[m.tip_nodes, 0] = -300.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=2.0e8, tip_defl_max_m=0.01, tip_twist_max_deg=2.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=SF, use_analytic_jacobian=True,
        core=core, ks_rho=50.0)
    return m, [l1, l2], cfg


def test_ks_requires_analytic():
    m, loads, cfg = _setup()
    import dataclasses
    bad = dataclasses.replace(cfg, use_analytic_jacobian=False)
    with pytest.raises(ValueError, match="analytic"):
        size_beam_shell_laminate(m, loads, bad, ply=T700_EPOXY, rho=RHO, maxiter=2)


@pytest.mark.sizing
def test_ks_optimum_feasible_and_near_hard_max():
    # KS is conservative: the KS optimum must be hard-feasible and within a few
    # percent of the hard-max optimum (rho = 50 on normalized constraints).
    import dataclasses
    m, loads, cfg = _setup(core=PVC_H80, tube=True)
    hard = dataclasses.replace(cfg, ks_rho=None)
    r_hard = size_beam_shell_laminate(m, loads, hard, ply=T700_EPOXY, rho=RHO,
                                      maxiter=300)
    r_ks = size_beam_shell_laminate(m, loads, cfg, ply=T700_EPOXY, rho=RHO,
                                    maxiter=300)
    assert laminate_result_is_feasible(r_hard, hard)
    # KS result judged with the HARD feasibility check (the real requirement)
    assert laminate_result_is_feasible(r_ks, hard)
    assert r_ks.mass_kg >= r_hard.mass_kg * 0.98       # can't beat the true optimum
    assert r_ks.mass_kg <= r_hard.mass_kg * 1.08       # conservatism stays small


def test_ks_gradients_match_fd_through_scipy_closures():
    # End-to-end FD of the KS constraint closures (fun vs jac) at a fixed x:
    # exercises every wired family (panel, beam-Euler, tube-wall, wrinkle,
    # crimp, skin, defl, twist) including multi-combo softmax weighting.
    from scipy.optimize import approx_fprime
    import wing_design.beams.laminate_sizing as ls
    m, loads, cfg = _setup(core=PVC_H80, tube=True)

    captured = {}
    orig = ls.minimize

    def spy(fun, x0, jac=None, method=None, bounds=None, constraints=None, options=None):
        captured["constraints"] = constraints
        captured["x0"] = np.asarray(x0, dtype=float)

        class R:
            x = np.asarray(x0, dtype=float)
            success = False
            nit = 0
        return R()

    ls.minimize = spy
    try:
        size_beam_shell_laminate(m, loads, cfg, ply=T700_EPOXY, rho=RHO, maxiter=1)
    finally:
        ls.minimize = orig

    x0 = captured["x0"] * 0.9 + 0.001     # interior-ish point, off the bounds
    checked = 0
    for con in captured["constraints"]:
        if "jac" not in con:
            continue
        f = con["fun"]; J = con["jac"]
        val = np.atleast_1d(f(x0))
        if val.shape[0] != 1:
            continue                       # vector rows (beam_vm) not KS — skip
        g = np.atleast_2d(J(x0))[0]
        fd = approx_fprime(x0, lambda x: float(np.atleast_1d(f(x))[0]), 1.5e-7)
        scale = max(np.abs(fd).max(), 1e-12)
        assert np.allclose(g, fd, rtol=5e-3, atol=5e-4 * scale), \
            f"KS jac mismatch (max err {np.abs(g - fd).max():.3e}, scale {scale:.3e})"
        checked += 1
    assert checked >= 7                    # all 8 KS families minus any inactive
