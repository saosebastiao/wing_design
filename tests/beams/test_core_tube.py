"""P.1 core tube: FD-validated gradients + sizing behavior (small models)."""
import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig,
    laminate_design_bounds,
    laminate_result_is_feasible,
    size_beam_shell_laminate,
)
from wing_design.beams.sensitivity import (
    DesignSens, grad_tube_wall_one, prepare_sensitivity,
)
from wing_design.structural.buckling import TUBE_WALL_KNOCKDOWN, tube_wall_utilization
from wing_design.structural.frame import BeamSection, _element_rotation
from wing_design.structural.beam_shell import solve_beam_shell_laminate_factored

RHO = 1550.0
SF = 1.5


def _tube_case():
    m = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3, core_tube=True)
    goe, G = beam_radius_groups(m)
    n_form = m.n_form_elements
    S = len(m.tube_elements)
    M = m.shell_tris.shape[0]
    n_all = m.beam_elements.shape[0]
    beam_len = np.empty(n_all)
    for e in range(n_all):
        i, j = int(m.beam_elements[e, 0]), int(m.beam_elements[e, 1])
        _R, L = _element_rotation(m.nodes[i], m.nodes[j])
        beam_len[e] = L
    loads = np.zeros((m.nodes.shape[0], 6))
    loads[m.tip_nodes, 2] = -2000.0     # spanwise compression -> tube axial comp
    loads[m.tip_nodes, 0] = 400.0
    return m, goe, G, S, M, beam_len, loads


def _decode_solve(m, goe, G, S, beam_len, loads, x):
    """x = [r_group(G), t, f0, f45, r_tube(S), t_wall(S)] -> factored solve."""
    radii = x[:G][goe]
    t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2])
    r_t = x[G + 3:G + 3 + S]
    t_w = x[G + 3 + S:G + 3 + 2 * S]
    A, D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45,
                                    f90=1.0 - f0 - f45, thickness=t)
    secs = ([BeamSection.circular(float(r)) for r in radii]
            + [BeamSection.annular(float(r_t[s]), float(t_w[s])) for s in range(S)])
    fac = solve_beam_shell_laminate_factored(
        m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=t * Qeff, D_skin=(t**3 / 12) * Qeff,
        fixed_nodes=m.fixed_nodes, loads=loads,
        gusset_elements=m.tube_bond_elements, gusset_section=BeamSection.circular(0.05))
    return fac, Qeff, r_t, t_w


def _build_ds(m, goe, G, S, M, beam_len, x):
    radii = x[:G][goe]
    t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2])
    _A, _D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45,
                                      f90=1.0 - f0 - f45, thickness=t)
    return DesignSens(
        model=m, G=G, B=1, L=1, group_of_element=goe,
        band_of_tri=np.zeros(M, dtype=int), layup_group_of_band=np.zeros(1, dtype=int),
        radii_full=radii, beam_lengths=beam_len, t_tri=np.full(M, t),
        Qeff_tri=np.broadcast_to(Qeff, (M, 3, 3)).copy(), offset_tri=np.zeros(M),
        ply=T700_EPOXY, S=S, tube_elements=m.tube_elements,
        tube_r=x[G + 3:G + 3 + S].copy(), tube_t=x[G + 3 + S:G + 3 + 2 * S].copy())


def test_grad_tube_wall_matches_fd():
    m, goe, G, S, M, beam_len, loads = _tube_case()
    nx = G + 3 + 2 * S
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1/3, 1/3],
                         0.7 * m.tube_r_bounds, np.full(S, 0.004)])
    fac0, _Q, r_t, t_w = _decode_solve(m, goe, G, S, beam_len, loads, x0)
    ds0 = _build_ds(m, goe, G, S, M, beam_len, x0)
    cache = prepare_sensitivity(ds0, fac0)

    s_star = 0   # keel-most segment carries the most compression
    con0, grad = grad_tube_wall_one(fac0, ds0, s_star, safety_factor=SF,
                                    E=m.E_beam, knockdown=TUBE_WALL_KNOCKDOWN,
                                    cache=cache)
    assert con0 < 1.0   # compressive

    def con_value(x):
        fac, _Qx, r_tx, t_wx = _decode_solve(m, goe, G, S, beam_len, loads, x)
        sec_A = np.pi * (r_tx**2 - (r_tx - t_wx)**2)
        u = tube_wall_utilization(
            fac.result.axial_force[m.tube_elements], r_tx, t_wx, sec_A,
            E=m.E_beam, safety_factor=SF, knockdown=TUBE_WALL_KNOCKDOWN)
        return 1.0 - float(u[s_star])

    assert np.isclose(con0, con_value(x0))
    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6],
                            np.full(S, 1e-7), np.full(S, 1e-8)])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd[i] = (con_value(xp) - con_value(xm)) / (2 * h)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max()), \
        f"max err {np.abs(grad - fd).max():.3e}"


def _tube_cfg(analytic):
    return LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=0.01, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=SF, use_analytic_jacobian=analytic)


@pytest.mark.sizing
def test_tube_sizing_feasible_and_bounds_respected():
    m, goe, G, S, M, beam_len, loads = _tube_case()
    r = size_beam_shell_laminate(m, [loads], _tube_cfg(True), ply=T700_EPOXY,
                                 rho=RHO, maxiter=150)
    assert r.r_tube is not None and r.r_tube.shape == (S,)
    assert laminate_result_is_feasible(r, _tube_cfg(True))
    assert np.all(r.r_tube <= m.tube_r_bounds * (1 + 1e-9))
    assert np.all(r.t_wall <= r.r_tube * (1 + 1e-6))
    assert r.max_tube_wall_util <= 1.05
    assert r.tube_mass_kg > 0.0
    lo, hi = laminate_design_bounds(m, _tube_cfg(True))
    assert lo.shape == (G + 1 + 2 + 2 * S,)


@pytest.mark.sizing
def test_tube_fd_and_analytic_agree_cold():
    # Equivalence contract (the historical FD-vs-analytic pattern): both cold
    # starts converge feasible and land within the basin tolerance (measured
    # 2026-06-10: 49.63 vs 49.43 kg = 0.4%). NOTE: warm-starting SLSQP-FD exactly
    # AT an optimum is pathological (degenerate working set + FD noise on the
    # kinked max-constraints walks it infeasible) — do not test that.
    m, goe, G, S, M, beam_len, loads = _tube_case()
    an = size_beam_shell_laminate(m, [loads], _tube_cfg(True), ply=T700_EPOXY,
                                  rho=RHO, maxiter=300)
    fd = size_beam_shell_laminate(m, [loads], _tube_cfg(False), ply=T700_EPOXY,
                                  rho=RHO, maxiter=300)
    assert laminate_result_is_feasible(an, _tube_cfg(True))
    assert laminate_result_is_feasible(fd, _tube_cfg(False))
    assert np.isclose(fd.mass_kg, an.mass_kg, rtol=2e-2)


def test_sizer_assembles_tube_bonds(monkeypatch):
    # Regression for the 2026-06-10 bug: tube_bond_elements existed on the model
    # but never reached the sizing FEA (only tip_gusset_elements were passed), so
    # the tube was optimized while floating. Capture the solver's gusset args.
    import wing_design.beams.laminate_sizing as ls
    m, goe, G, S, M, beam_len, loads = _tube_case()
    captured = {}
    orig = ls.solve_beam_shell_laminate_factored

    def spy(*args, **kw):
        captured["gusset"] = kw.get("gusset_elements")
        return orig(*args, **kw)

    monkeypatch.setattr(ls, "solve_beam_shell_laminate_factored", spy)
    cfg = _tube_cfg(True)
    size_beam_shell_laminate(m, [loads], cfg, ply=T700_EPOXY, rho=RHO, maxiter=2)
    g = captured["gusset"]
    assert g is not None
    bond_set = {tuple(sorted(r)) for r in m.tube_bond_elements.tolist()}
    gusset_set = {tuple(sorted(r)) for r in np.asarray(g).tolist()}
    assert bond_set <= gusset_set, "tube bonds missing from the sizing FEA"


# --- P#1b hollow form-beam segments ----------------------------------------


def _hollow_case():
    m = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=4,
                               hollow_beams=True)
    goe, G = beam_radius_groups(m)
    M = m.shell_tris.shape[0]
    loads = np.zeros((m.nodes.shape[0], 6))
    loads[m.tip_nodes, 2] = 800.0
    loads[m.tip_nodes, 0] = 400.0
    return m, goe, G, M, loads


def test_hollow_model_tags_in_wing_segments():
    m, goe, G, M, loads = _hollow_case()
    z = m.nodes[:m.n_levels, 2]
    seg_in_wing = [(z[k] >= -1e-9 and z[k + 1] >= -1e-9) for k in range(m.n_levels - 1)]
    expected = [b * (m.n_levels - 1) + k
                for b in range(m.n_beams) for k in range(m.n_levels - 1)
                if seg_in_wing[k]]
    assert m.hollow_elements.tolist() == expected


@pytest.mark.sizing
def test_hollow_sizing_feasible_never_heavier():
    m, goe, G, M, loads = _hollow_case()
    solid = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=4)

    def cfg():
        return LaminateSizingConfig(
            sigma_allow_Pa=1.0e9, tip_defl_max_m=0.01, tip_twist_max_deg=90.0,
            r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
            buckling_safety_factor=SF, use_analytic_jacobian=True)

    base = size_beam_shell_laminate(solid, [loads], cfg(), ply=T700_EPOXY,
                                    rho=RHO, maxiter=200)
    hol = size_beam_shell_laminate(m, [loads], cfg(), ply=T700_EPOXY,
                                   rho=RHO, maxiter=300)
    assert laminate_result_is_feasible(base, cfg())
    assert laminate_result_is_feasible(hol, cfg())
    assert hol.t_hollow is not None
    assert np.all(hol.t_hollow > 0)
    assert hol.max_tube_wall_util <= 1.05
    # hollow strictly contains solid designs (t = r) -> never materially heavier
    assert hol.mass_kg <= base.mass_kg * 1.02


@pytest.mark.sizing
def test_hollow_fd_and_analytic_agree_cold():
    # The decisive gradient check for the hollow path: cold FD and analytic land
    # in the same basin (hollow r-cols REUSE the radius-group columns, so any
    # error in the annular dK/dr mapping would split the optima immediately).
    m, goe, G, M, loads = _hollow_case()
    cfg_an = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=0.01, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=SF, use_analytic_jacobian=True)
    cfg_fd = LaminateSizingConfig(
        sigma_allow_Pa=1.0e9, tip_defl_max_m=0.01, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        buckling_safety_factor=SF, use_analytic_jacobian=False)
    an = size_beam_shell_laminate(m, [loads], cfg_an, ply=T700_EPOXY, rho=RHO, maxiter=300)
    fd = size_beam_shell_laminate(m, [loads], cfg_fd, ply=T700_EPOXY, rho=RHO, maxiter=300)
    assert laminate_result_is_feasible(an, cfg_an)
    assert laminate_result_is_feasible(fd, cfg_fd)
    assert np.isclose(fd.mass_kg, an.mass_kg, rtol=2e-2)
