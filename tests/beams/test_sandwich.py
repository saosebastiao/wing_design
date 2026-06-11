"""P.3 sandwich skin: FD-validated gradients + sizing behavior (small models)."""
import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import (
    CoreMaterial, PVC_H80, T700_EPOXY, laminate_stiffness, sandwich_D_factor,
)
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups, skin_areas
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig,
    laminate_result_is_feasible,
    size_beam_shell_laminate,
)
from wing_design.beams.sensitivity import (
    DesignSens, grad_core_check, grad_panel_buckling, prepare_sensitivity,
    sandwich_geom_factor,
)
from wing_design.structural.beam_shell import solve_beam_shell_laminate_factored
from wing_design.structural.buckling import panel_buckling_utilization
from wing_design.structural.frame import BeamSection, _element_rotation
from wing_design.structural.shell import recover_membrane_stress_C

RHO = 1550.0
SF = 1.5
CORE = PVC_H80


def test_sandwich_factor_monolithic_at_zero_core():
    t = np.array([0.001, 0.003, 0.02])
    assert np.allclose(sandwich_D_factor(t, 0.0), 1.0, rtol=0, atol=1e-14)
    # 10 mm core on 2 mm faces: D multiplies ~90x
    assert 80 < sandwich_D_factor(0.002, 0.01) < 100


def _case():
    m = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3)
    goe, G = beam_radius_groups(m)
    M = m.shell_tris.shape[0]
    n = m.beam_elements.shape[0]
    beam_len = np.empty(n)
    for e in range(n):
        i, j = int(m.beam_elements[e, 0]), int(m.beam_elements[e, 1])
        _R, L = _element_rotation(m.nodes[i], m.nodes[j])
        beam_len[e] = L
    loads = np.zeros((m.nodes.shape[0], 6))
    loads[m.tip_nodes, 2] = -1.2e5     # strong compression -> compressive panels
    for a, node in enumerate(m.tip_nodes):
        loads[node, 0] = 60.0 + 11.0 * a
    return m, goe, G, M, beam_len, loads


def _decode_solve(m, goe, G, beam_len, loads, x):
    """x = [r_group(G), t, f0, f45, c] -> factored solve with sandwich D."""
    radii = x[:G][goe]
    t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2]); c = float(x[G + 3])
    A, D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45,
                                    f90=1.0 - f0 - f45, thickness=t)
    D = D * float(sandwich_D_factor(t, c))
    secs = [BeamSection.circular(float(r)) for r in radii]
    fac = solve_beam_shell_laminate_factored(
        m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D,
        fixed_nodes=m.fixed_nodes, loads=loads)
    return fac, Qeff


def _build_ds(m, goe, G, M, beam_len, x):
    radii = x[:G][goe]
    t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2]); c = float(x[G + 3])
    _A, _D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45,
                                      f90=1.0 - f0 - f45, thickness=t)
    return DesignSens(
        model=m, G=G, B=1, L=1, group_of_element=goe,
        band_of_tri=np.zeros(M, dtype=int), layup_group_of_band=np.zeros(1, dtype=int),
        radii_full=radii, beam_lengths=beam_len, t_tri=np.full(M, t),
        Qeff_tri=np.broadcast_to(Qeff, (M, 3, 3)).copy(), offset_tri=np.zeros(M),
        ply=T700_EPOXY, nx_extra=1,
        core=CORE, c_tri=np.full(M, c),
        core_col_of_tri=np.full(M, G + 3, dtype=int))


def _fd(con_value, x0, G):
    nx = G + 4
    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6, 1e-7]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd[i] = (con_value(xp) - con_value(xm)) / (2 * h)
    return fd


def test_grad_panel_buckling_sandwich_matches_fd():
    m, goe, G, M, beam_len, loads = _case()
    areas = skin_areas(m)
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1/3, 1/3, 0.006]])
    fac0, _Q = _decode_solve(m, goe, G, beam_len, loads, x0)
    ds0 = _build_ds(m, goe, G, M, beam_len, x0)
    cache = prepare_sensitivity(ds0, fac0)
    con0, grad = grad_panel_buckling(fac0, ds0, panel_kc=4.0, safety_factor=SF,
                                     areas=areas, cache=cache)
    assert con0 < 1.0

    def con_value(x):
        fac, Qx = _decode_solve(m, goe, G, beam_len, loads, x)
        C = np.broadcast_to(Qx, (M, 3, 3)).copy()
        s = recover_membrane_stress_C(m.nodes, m.shell_tris, fac.result.displacements, C=C)
        t = float(x[G]); c = float(x[G + 3])
        D11 = Qx[0, 0] * sandwich_geom_factor(t, c) / t * t   # D11_sand
        u = panel_buckling_utilization(np.asarray(s), areas,
                                       D11=Qx[0, 0] * sandwich_geom_factor(t, c),
                                       t=np.full(M, t), kc=4.0, safety_factor=SF)
        return 1.0 - float(u.max())

    assert np.isclose(con0, con_value(x0))
    fd = _fd(con_value, x0, G)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max()), \
        f"max err {np.abs(grad - fd).max():.3e}"


@pytest.mark.parametrize("which", ["wrinkle", "crimp"])
def test_grad_core_check_matches_fd(which):
    from wing_design.materials.unidir import CORE_RAMP_M, crimping_stress, wrinkling_stress
    m, goe, G, M, beam_len, loads = _case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1/3, 1/3, 0.006]])
    fac0, _Q = _decode_solve(m, goe, G, beam_len, loads, x0)
    ds0 = _build_ds(m, goe, G, M, beam_len, x0)
    cache = prepare_sensitivity(ds0, fac0)
    con0, grad = grad_core_check(fac0, ds0, which=which, safety_factor=SF, cache=cache)
    assert con0 < 1.0

    def con_value(x):
        fac, Qx = _decode_solve(m, goe, G, beam_len, loads, x)
        C = np.broadcast_to(Qx, (M, 3, 3)).copy()
        s = np.asarray(recover_membrane_stress_C(m.nodes, m.shell_tris,
                                                 fac.result.displacements, C=C))
        mean = 0.5 * (s[:, 0] + s[:, 1])
        rad = np.sqrt(0.25 * (s[:, 0] - s[:, 1])**2 + s[:, 2]**2)
        comp = np.maximum(0.0, -(mean - rad))
        t = float(x[G]); c = float(x[G + 3])
        if which == "wrinkle":
            sig = wrinkling_stress(Qx[0, 0], CORE)
        else:
            sig = crimping_stress(t, c, CORE)
        ramp = c / (c + CORE_RAMP_M)
        return 1.0 - float((comp * SF / sig * ramp).max())

    assert np.isclose(con0, con_value(x0))
    fd = _fd(con_value, x0, G)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max()), \
        f"max err {np.abs(grad - fd).max():.3e}"


@pytest.mark.sizing
def test_sandwich_sizing_feasible_never_heavier():
    m, goe, G, M, beam_len, loads_unused = _case()
    loads = np.zeros((m.nodes.shape[0], 6))
    loads[m.tip_nodes, 2] = 800.0
    loads[m.tip_nodes, 0] = 400.0

    def cfg(core):
        return LaminateSizingConfig(
            sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
            r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
            buckling_safety_factor=SF, use_analytic_jacobian=True, core=core)

    base = size_beam_shell_laminate(m, [loads], cfg(None), ply=T700_EPOXY,
                                    rho=RHO, maxiter=150)
    sand = size_beam_shell_laminate(m, [loads], cfg(CORE), ply=T700_EPOXY,
                                    rho=RHO, maxiter=250)
    assert laminate_result_is_feasible(base, cfg(None))
    assert laminate_result_is_feasible(sand, cfg(CORE))
    assert sand.t_core is not None
    assert sand.max_wrinkle_util <= 1.05 and sand.max_crimp_util <= 1.05
    # c = 0 (monolithic) is in the sandwich design space
    assert sand.mass_kg <= base.mass_kg * 1.02


@pytest.mark.sizing
def test_ipopt_matches_slsqp_on_small_problem():
    # Toolbox #4 fallback: IPOPT must reach the same feasible optimum class as
    # SLSQP on a problem SLSQP handles fine (basin tolerance 2%).
    m, goe, G, M, beam_len, _ = _case()
    loads = np.zeros((m.nodes.shape[0], 6))
    loads[m.tip_nodes, 2] = 800.0
    loads[m.tip_nodes, 0] = 400.0

    def cfg(opt):
        return LaminateSizingConfig(
            sigma_allow_Pa=1.0e9, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
            r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
            buckling_safety_factor=SF, use_analytic_jacobian=True,
            core=CORE, optimizer=opt)

    s = size_beam_shell_laminate(m, [loads], cfg("slsqp"), ply=T700_EPOXY,
                                 rho=RHO, maxiter=250)
    i = size_beam_shell_laminate(m, [loads], cfg("ipopt"), ply=T700_EPOXY,
                                 rho=RHO, maxiter=600)
    assert laminate_result_is_feasible(s, cfg("slsqp"))
    assert i.converged
    assert laminate_result_is_feasible(i, cfg("ipopt"))
    assert i.mass_kg <= s.mass_kg * 1.02
