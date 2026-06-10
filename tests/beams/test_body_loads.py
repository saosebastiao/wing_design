"""V.4 self-weight/inertial body loads: lumping conservation + the lam^T dF/dx
adjoint term, FD-validated per house convention (central difference, tiny mesh).

The FD reference re-solves with the design-dependent load f(x) = aero + body(x)
rebuilt at every perturbed x — exactly what the dF term must account for.
"""
import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import T700_EPOXY
from wing_design.beams.body_loads import body_load_jacobian, body_load_vector
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups, skin_areas, skin_band_map
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig,
    laminate_result_is_feasible,
    size_beam_shell_laminate,
)
from wing_design.beams.sensitivity import (
    DesignSens, grad_tip_defl, grad_panel_buckling, prepare_sensitivity,
)
from wing_design.materials.unidir import laminate_stiffness
from wing_design.structural.beam_shell import solve_beam_shell_laminate_factored
from wing_design.structural.frame import BeamSection, _element_rotation

RHO = 1550.0
ACCEL = (4.0, 2.0, -9.81)   # heeled-ish gravity: lateral + axial components


def _model_case():
    m = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3)
    goe, G = beam_radius_groups(m)
    n = m.beam_elements.shape[0]
    M = m.shell_tris.shape[0]
    beam_len = np.empty(n)
    for e in range(n):
        i, j = int(m.beam_elements[e, 0]), int(m.beam_elements[e, 1])
        _R, L = _element_rotation(m.nodes[i], m.nodes[j])
        beam_len[e] = L
    aero = np.zeros((m.nodes.shape[0], 6))
    aero[m.tip_nodes, 0] = 120.0
    aero[m.tip_nodes, 2] = -250.0
    return m, goe, G, n, M, beam_len, aero


def _decode_solve_with_body(m, goe, G, beam_len, aero, x):
    """f(x) = aero + body(x); returns (fac, Qeff, t_tri)."""
    radii = x[:G][goe]
    t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2])
    f90 = 1.0 - f0 - f45
    A, D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=t)
    t_tri = np.full(m.shell_tris.shape[0], t)
    loads = aero + body_load_vector(m, radii, t_tri, rho=RHO, accel=ACCEL)
    secs = [BeamSection.circular(float(r)) for r in radii]
    fac = solve_beam_shell_laminate_factored(
        m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D,
        fixed_nodes=m.fixed_nodes, loads=loads)
    return fac, Qeff, t_tri


def _build_ds(m, goe, G, M, beam_len, x):
    radii = x[:G][goe]
    t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2])
    f90 = 1.0 - f0 - f45
    _A, _D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=t)
    return DesignSens(
        model=m, G=G, B=1, L=1,
        group_of_element=goe, band_of_tri=np.zeros(M, dtype=int),
        layup_group_of_band=np.zeros(1, dtype=int),
        radii_full=radii, beam_lengths=beam_len,
        t_tri=np.full(M, t), Qeff_tri=np.broadcast_to(Qeff, (M, 3, 3)).copy(),
        offset_tri=np.zeros(M), ply=T700_EPOXY)


def test_body_load_total_force_equals_mass_times_accel():
    m, goe, G, n, M, beam_len, _aero = _model_case()
    radii = np.full(n, 0.012)
    t_tri = np.full(M, 0.002)
    f = body_load_vector(m, radii, t_tri, rho=RHO, accel=ACCEL)
    mass = RHO * np.sum(np.pi * radii**2 * beam_len) + RHO * np.sum(0.002 * skin_areas(m))
    assert np.allclose(f[:, :3].sum(axis=0), mass * np.asarray(ACCEL), rtol=1e-9)
    assert np.allclose(f[:, 3:], 0.0)


def test_body_load_jacobian_matches_fd():
    m, goe, G, n, M, beam_len, _aero = _model_case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1/3, 1/3]])
    band_of_tri = np.zeros(M, dtype=int)
    dF = body_load_jacobian(m, x0[:G][goe], group_of_element=goe,
                            band_of_tri=band_of_tri, rho=RHO, accel=ACCEL,
                            G=G, B=1, L=1)
    nx = G + 3

    def fvec(x):
        radii = x[:G][goe]
        return body_load_vector(m, radii, np.full(M, float(x[G])), rho=RHO,
                                accel=ACCEL).reshape(-1)

    for i in range(nx):
        h = 1e-7
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd = (fvec(xp) - fvec(xm)) / (2 * h)
        assert np.allclose(dF[:, i], fd, rtol=1e-6, atol=1e-6 * max(np.abs(fd).max(), 1.0)), i


def test_grad_tip_defl_with_body_load_matches_fd():
    m, goe, G, n, M, beam_len, aero = _model_case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1/3, 1/3]])
    nx = G + 3
    fac0, _Q, _t = _decode_solve_with_body(m, goe, G, beam_len, aero, x0)
    ds0 = _build_ds(m, goe, G, M, beam_len, x0)
    cache = prepare_sensitivity(ds0, fac0)
    dF = body_load_jacobian(m, x0[:G][goe], group_of_element=goe,
                            band_of_tri=np.zeros(M, dtype=int), rho=RHO,
                            accel=ACCEL, G=G, B=1, L=1)
    g0, grad = grad_tip_defl(fac0, ds0, cache=cache, dF=dF)

    def metric(x):
        fac, _Qx, _tx = _decode_solve_with_body(m, goe, G, beam_len, aero, x)
        u = fac.u.reshape(-1, 6)
        return float(np.linalg.norm(u[m.tip_nodes, :3], axis=1).max())

    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd[i] = (metric(xp) - metric(xm)) / (2 * h)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


def test_grad_panel_buckling_with_body_load_matches_fd():
    m, goe, G, n, M, beam_len, aero = _model_case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1/3, 1/3]])
    nx = G + 3
    areas = skin_areas(m)
    fac0, Q0, t0 = _decode_solve_with_body(m, goe, G, beam_len, aero, x0)
    ds0 = _build_ds(m, goe, G, M, beam_len, x0)
    cache = prepare_sensitivity(ds0, fac0)
    dF = body_load_jacobian(m, x0[:G][goe], group_of_element=goe,
                            band_of_tri=np.zeros(M, dtype=int), rho=RHO,
                            accel=ACCEL, G=G, B=1, L=1)
    con0, grad = grad_panel_buckling(fac0, ds0, panel_kc=4.0, safety_factor=1.5,
                                     areas=areas, cache=cache, dF=dF)
    assert con0 < 1.0   # compressive active panel

    from wing_design.structural.buckling import panel_buckling_utilization
    from wing_design.structural.shell import recover_membrane_stress_C

    def con_value(x):
        fac, Qx, t_tri = _decode_solve_with_body(m, goe, G, beam_len, aero, x)
        C = np.broadcast_to(Qx, (M, 3, 3)).copy()
        s = recover_membrane_stress_C(m.nodes, m.shell_tris, fac.result.displacements, C=C)
        D11 = (float(x[G]) ** 3 / 12.0) * Qx[0, 0]
        u = panel_buckling_utilization(np.asarray(s), areas, D11=D11, t=t_tri,
                                       kc=4.0, safety_factor=1.5)
        return 1.0 - float(u.max())

    assert np.isclose(con0, con_value(x0))
    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd[i] = (con_value(xp) - con_value(xm)) / (2 * h)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


@pytest.mark.sizing  # two small SLSQP runs
def test_sizing_with_gravity_feasible_and_heavier():
    m = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3)
    aero = np.zeros((m.nodes.shape[0], 6))
    aero[m.tip_nodes, 2] = 800.0
    aero[m.tip_nodes, 0] = 400.0

    def cfg(accels):
        return LaminateSizingConfig(
            sigma_allow_Pa=1.0e9, tip_defl_max_m=0.005, tip_twist_max_deg=90.0,
            r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
            buckling_safety_factor=1.5, use_analytic_jacobian=True,
            accel_vectors=accels)

    base = size_beam_shell_laminate(m, [aero], cfg(()), ply=T700_EPOXY,
                                    rho=RHO, maxiter=150)
    grav = size_beam_shell_laminate(m, [aero], cfg(((30.0, 0.0, -9.81),)),
                                    ply=T700_EPOXY, rho=RHO, maxiter=150)
    assert laminate_result_is_feasible(grav, cfg(((30.0, 0.0, -9.81),)))
    # a strong lateral acceleration adds demand: never lighter than aero-only
    assert grav.mass_kg >= base.mass_kg * 0.999
