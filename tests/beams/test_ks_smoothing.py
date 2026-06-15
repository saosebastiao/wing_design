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


def test_ks_foundation_gradients_match_fd():
    # V.3c: beam-on-elastic-foundation model — FD-audit the beam_buck KS closure
    # (k chains through t_band/f0/f45/t_core + EI chains through r/t_hollow).
    from scipy.optimize import approx_fprime
    import dataclasses
    import wing_design.beams.laminate_sizing as ls
    m, loads, cfg = _setup(core=PVC_H80, tube=True)
    cfg = dataclasses.replace(cfg, ply_angle_datum=(0.0, 0.0, 1.0),
                              beam_buckling_model="foundation")

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

    x0 = captured["x0"] * 0.9 + 0.001
    checked = 0
    for con in captured["constraints"]:
        if "jac" not in con:
            continue
        f = con["fun"]; J = con["jac"]
        if np.atleast_1d(f(x0)).shape[0] != 1:
            continue
        g = np.atleast_2d(J(x0))[0]
        fd = approx_fprime(x0, lambda x: float(np.atleast_1d(f(x))[0]), 1.5e-7)
        scale = max(np.abs(fd).max(), 1e-12)
        assert np.allclose(g, fd, rtol=5e-3, atol=5e-4 * scale), \
            f"foundation KS jac mismatch (max err {np.abs(g - fd).max():.3e})"
        checked += 1
    assert checked >= 7


def test_ks_beam_buck_brace_radius_gradient_matches_fd():
    # Task 5 crux: lateral-bracing rings credit the in-loop sizing ONLY through
    # the elastic-FOUNDATION stiffness (k_ring) augmentation. The analytic
    # beam_buck KS Jacobian's brace_radius column (the LAST column) must be
    # NONZERO (proves the optimizer now "sees" the rings) and match a central
    # difference of the constraint value at that column.
    import wing_design.beams.laminate_sizing as ls
    from wing_design.geometry import medium_wingsail

    m = build_beam_shell_model(medium_wingsail, n_beams=8, n_levels=5,
                               core_tube=False, hollow_beams=False,
                               lateral_bracing=True)
    assert getattr(m, "brace_elements", None) is not None
    l1 = np.zeros((m.nodes.shape[0], 6))
    l1[m.tip_nodes, 2] = 800.0
    l1[m.tip_nodes, 0] = 400.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=6e8, tip_defl_max_m=0.4, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), ks_rho=50.0,
        buckling_safety_factor=1.5, use_analytic_jacobian=True,
        beam_buckling_model="foundation", panel_width_mode="strip",
        brace_r_min=0.004, brace_r_max=0.05)

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
        size_beam_shell_laminate(m, [l1], cfg, ply=T700_EPOXY, rho=RHO, maxiter=1)
    finally:
        ls.minimize = orig

    x0 = captured["x0"] * 0.9 + 0.001
    n_dv = x0.shape[0]

    # Locate the beam_buck KS constraint (scalar-valued, has analytic jac).
    beam_buck = None
    for con in captured["constraints"]:
        if "jac" not in con:
            continue
        f = con["fun"]; J = con["jac"]
        if np.atleast_1d(f(x0)).shape[0] != 1:
            continue
        g = np.atleast_2d(J(x0))[0]
        # The brace_radius DV is the LAST column; only the beam_buck KS closure
        # depends on it (mass is a separate objective, not in `constraints`).
        if abs(g[-1]) > 0.0:
            beam_buck = (f, J)
            break
    assert beam_buck is not None, "no KS constraint had a nonzero brace_radius column"

    f, J = beam_buck
    g = np.atleast_2d(J(x0))[0]
    brace_col = n_dv - 1
    # NONZERO is essential: it proves the optimizer now sees the rings.
    assert abs(g[brace_col]) > 1e-9, \
        f"analytic brace_radius column is ~0 ({g[brace_col]:.3e}) — rings not credited"

    # Central-difference the constraint value at the brace_radius column.
    h = 1e-7
    xp = x0.copy(); xp[brace_col] += h
    xm = x0.copy(); xm[brace_col] -= h
    fp = float(np.atleast_1d(f(xp))[0])
    fm = float(np.atleast_1d(f(xm))[0])
    fd = (fp - fm) / (2.0 * h)
    rel = abs(g[brace_col] - fd) / max(abs(fd), 1e-12)
    assert rel <= 1e-4, (
        f"brace_radius FD-through-KS mismatch: analytic={g[brace_col]:.6e} "
        f"fd={fd:.6e} rel={rel:.3e}")


def _braced_ks_model_and_cfg():
    """Small braced foundation+KS+strip model used by the brace-vm gradient tests."""
    m = build_beam_shell_model(small_wingsail, n_beams=8, n_levels=5,
                               core_tube=False, hollow_beams=False,
                               lateral_bracing=True)
    assert getattr(m, "brace_elements", None) is not None
    l1 = np.zeros((m.nodes.shape[0], 6))
    l1[m.tip_nodes, 2] = 800.0
    l1[m.tip_nodes, 0] = 400.0
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=6e8, tip_defl_max_m=0.4, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), ks_rho=50.0,
        buckling_safety_factor=1.5, use_analytic_jacobian=True,
        beam_buckling_model="foundation", panel_width_mode="strip",
        brace_r_min=0.004, brace_r_max=0.05)
    return m, [l1], cfg


def test_beam_vm_gradient_on_brace_row_matches_fd():
    # Regression for the brace IndexError bug: beam_con_jac loops over ALL
    # elements (including the transverse brace rings) and must NOT crash, and
    # the brace row's gradient w.r.t a structural DV (a t_band column) must
    # match a central difference of beam_con's brace component. The brace
    # element has a FIXED nominal radius (no radius-DV column), so its gradient
    # flows only through the displacement adjoint w.r.t. the other DVs.
    import wing_design.beams.laminate_sizing as ls
    from wing_design.beams.shell_sizing import beam_radius_groups

    m, loads, cfg = _braced_ks_model_and_cfg()

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

    x0 = captured["x0"] * 0.9 + 0.001
    n = int(m.beam_elements.shape[0])

    # beam_vm is the constraint whose fun(x) returns a length-n vector (one row
    # per beam element); it is NOT KS-aggregated, so it keeps the full vector.
    beam_vm = None
    for con in captured["constraints"]:
        if "jac" not in con:
            continue
        f = con["fun"]
        if np.atleast_1d(f(x0)).shape[0] == n:
            beam_vm = (f, con["jac"])
            break
    assert beam_vm is not None, "no length-n (beam_vm) vector constraint found"
    f, J = beam_vm

    # This call exercises _beam_vm_grad_one on the brace rows — it must NOT raise.
    Jrows = np.atleast_2d(J(x0))
    assert Jrows.shape[0] == n

    brace_row = int(m.brace_elements[0])
    G = beam_radius_groups(m)[1]   # first t_band column = r_group block width G
    t_band_col = G                  # FD a structural DV that the brace row sees

    g = Jrows[brace_row, t_band_col]
    h = 1e-7
    xp = x0.copy(); xp[t_band_col] += h
    xm = x0.copy(); xm[t_band_col] -= h
    fp = float(np.atleast_1d(f(xp))[brace_row])
    fm = float(np.atleast_1d(f(xm))[brace_row])
    fd = (fp - fm) / (2.0 * h)
    rel = abs(g - fd) / max(abs(fd), 1e-12)
    assert rel <= 1e-4, (
        f"brace-row beam_vm gradient mismatch: analytic={g:.6e} "
        f"fd={fd:.6e} rel={rel:.3e}")


def test_braced_solve_analytic_jac_runs():
    # The whole point: a REAL braced solve with the analytic Jacobian must run
    # SLSQP iterations over brace elements without crashing (the IndexError was
    # on iteration 1). Asserts a finite mass comes back.
    m, loads, cfg = _braced_ks_model_and_cfg()
    res = size_beam_shell_laminate(m, loads, cfg, ply=T700_EPOXY, rho=RHO, maxiter=3)
    assert np.isfinite(res.mass_kg)
    assert res.mass_kg > 0.0


def test_foundation_requires_ks_and_datum():
    import dataclasses
    m, loads, cfg = _setup()
    bad = dataclasses.replace(cfg, beam_buckling_model="foundation",
                              ply_angle_datum=None)
    with pytest.raises(ValueError, match="foundation"):
        size_beam_shell_laminate(m, loads, bad, ply=T700_EPOXY, rho=RHO, maxiter=2)


def test_panel_datum_ortho_requires_ks_and_datum():
    import dataclasses
    m, loads, cfg = _setup()
    bad = dataclasses.replace(cfg, panel_d_mode="datum_ortho",
                              ply_angle_datum=None)
    with pytest.raises(ValueError, match="datum_ortho"):
        size_beam_shell_laminate(m, loads, bad, ply=T700_EPOXY, rho=RHO, maxiter=2)


def test_ortho_plate_qstar_isotropic_and_fd():
    # V#2 unit checks: (1) quasi-isotropic fractions make the smeared Qeff
    # isotropic, so Qstar = 2·Qeff00 exactly (the kc=4 long-plate identity:
    # ½·Qstar stands in for Qeff00 with no model change for isotropic skins);
    # (2) dQstar_df matches central-difference FD of Qstar over (f0, f45).
    from wing_design.materials.unidir import laminate_stiffness
    from wing_design.beams.sensitivity import (dQeff_df, dQstar_df,
                                               ortho_plate_Qstar)
    ply = T700_EPOXY
    Qiso = laminate_stiffness(ply, f0=0.25, f45=0.5, f90=0.25, thickness=1.0)[2]
    assert ortho_plate_Qstar(Qiso) == pytest.approx(2.0 * Qiso[0, 0], rel=1e-9)

    def qstar(f0, f45):
        Qd = laminate_stiffness(ply, f0=f0, f45=f45, f90=1.0 - f0 - f45,
                                thickness=1.0)[2]
        return ortho_plate_Qstar(Qd)

    f0, f45, h = 0.32, 0.53, 1e-6
    Qd = laminate_stiffness(ply, f0=f0, f45=f45, f90=1.0 - f0 - f45,
                            thickness=1.0)[2]
    for which, fd in (
        ("f0", (qstar(f0 + h, f45) - qstar(f0 - h, f45)) / (2 * h)),
        ("f45", (qstar(f0, f45 + h) - qstar(f0, f45 - h)) / (2 * h)),
    ):
        an = dQstar_df(Qd, dQeff_df(ply, which=which, offset_deg=0.0))
        assert an == pytest.approx(fd, rel=1e-4)


def test_ks_panel_datum_ortho_gradients_match_fd():
    # V#2: datum-frame orthotropic strip σcr — FD-audit every KS closure with
    # the mode on (panel f-chains now run through Qstar; t/c chains shared).
    from scipy.optimize import approx_fprime
    import dataclasses
    import wing_design.beams.laminate_sizing as ls
    m, loads, cfg = _setup(core=PVC_H80, tube=True)
    cfg = dataclasses.replace(cfg, ply_angle_datum=(0.0, 0.0, 1.0),
                              panel_d_mode="datum_ortho",
                              beam_buckling_model="foundation")

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

    x0 = captured["x0"] * 0.9 + 0.001
    checked = 0
    for con in captured["constraints"]:
        if "jac" not in con:
            continue
        f = con["fun"]; J = con["jac"]
        if np.atleast_1d(f(x0)).shape[0] != 1:
            continue
        g = np.atleast_2d(J(x0))[0]
        fd = approx_fprime(x0, lambda x: float(np.atleast_1d(f(x))[0]), 1.5e-7)
        scale = max(np.abs(fd).max(), 1e-12)
        assert np.allclose(g, fd, rtol=5e-3, atol=5e-4 * scale), \
            f"datum_ortho KS jac mismatch (max err {np.abs(g - fd).max():.3e})"
        checked += 1
    assert checked >= 7


@pytest.mark.sizing
def test_incumbent_guard_never_worse_than_feasible_seed():
    # The user-requested property: a run seeded with a feasible design can never
    # return anything worse — the best hard-feasible iterate is kept even when
    # the optimizer's endpoint wanders or fails (tiny maxiter forces that here).
    import dataclasses
    m, loads, cfg = _setup(core=PVC_H80, tube=True)
    good = size_beam_shell_laminate(m, loads, cfg, ply=T700_EPOXY, rho=RHO,
                                    maxiter=300)
    assert laminate_result_is_feasible(good, dataclasses.replace(cfg, ks_rho=None))
    # reconstruct x0 from the optimum and rerun with a starved budget
    from wing_design.beams.shell_sizing import beam_radius_groups
    goe, G = beam_radius_groups(m)
    rg = np.zeros(G); seen = set()
    for e, g in enumerate(goe):
        if g not in seen: rg[g] = good.radii[e]; seen.add(g)
    hgroups = sorted({int(goe[e]) for e in m.hollow_elements})
    x0 = np.concatenate([rg, np.asarray(good.t_bands, float),
                         [float(good.f0_bands[0])], [float(good.f45_bands[0])],
                         good.r_tube, good.t_wall, good.t_hollow, good.t_core])
    starved = size_beam_shell_laminate(m, loads, cfg, ply=T700_EPOXY, rho=RHO,
                                       maxiter=3, x0=x0)
    assert laminate_result_is_feasible(starved, dataclasses.replace(cfg, ks_rho=None))
    assert starved.mass_kg <= good.mass_kg * (1 + 1e-6)
