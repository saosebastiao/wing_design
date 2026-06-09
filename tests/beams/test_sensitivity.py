import numpy as np
from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.structural.frame import BeamSection
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness
from wing_design.structural.beam_shell import solve_beam_shell_laminate, solve_beam_shell_laminate_factored


def _case(n_beams=6, n_levels=4):
    P = small_scenario()
    m = build_beam_shell_model(P.geometry, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    secs = [BeamSection.circular(0.01)] * m.beam_elements.shape[0]
    A, D, _ = laminate_stiffness(T700_EPOXY, f0=0.34, f45=0.33, f90=0.33, thickness=0.0015)
    loads = np.zeros((m.nodes.shape[0], 6)); loads[m.tip_nodes, 0] = 100.0
    return m, secs, A, D, loads


def test_factored_solve_matches_spsolve():
    m, secs, A, D, loads = _case()
    base = solve_beam_shell_laminate(m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D, fixed_nodes=m.fixed_nodes, loads=loads)
    fac = solve_beam_shell_laminate_factored(m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D, fixed_nodes=m.fixed_nodes, loads=loads)
    assert np.allclose(base.displacements, fac.result.displacements, atol=1e-12)
    # the factorization solves K_ff x = rhs
    free = fac.free
    rhs = np.random.default_rng(0).normal(size=free.shape[0])
    x = fac.lu.solve(rhs)
    assert np.allclose(fac.K_ff @ x, rhs, atol=1e-8)


# --- Task 2: element-stiffness derivative primitives (FD-validated) ---
from wing_design.structural.frame import local_beam_stiffness
from wing_design.structural.shell import tri_element_stiffness_laminate
from wing_design.materials.unidir import (
    reduced_stiffness_Q, transformed_Qbar, laminate_stiffness_offset,
)
from wing_design.beams.sensitivity import (
    central_diff, dkloc_dr, dke_dAD, dAD_dt, dQeff_df, dAD_df,
)

_TRI = (np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]), np.array([0.0, 0.0, 0.4]))


def test_dkloc_dr_matches_fd():
    E, G, r, L = 1e10, 4e9, 0.01, 0.5
    ana = dkloc_dr(E, G, r, L)
    fd = central_diff(
        lambda rr: local_beam_stiffness(E, G, BeamSection.circular(rr), L), r, 1e-8
    )
    assert np.allclose(ana, fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())


def test_dke_dAD_linearity():
    p1, p2, p3 = _TRI
    rng = np.random.default_rng(1)
    A0, D0, _ = laminate_stiffness_offset(
        T700_EPOXY, f0=0.4, f45=0.4, f90=0.2, thickness=0.002, offset_deg=0.0
    )
    dA = rng.normal(size=(3, 3)); dA = 0.5 * (dA + dA.T) * 1e6
    dD = rng.normal(size=(3, 3)); dD = 0.5 * (dD + dD.T) * 1e0
    s = 1e-3
    ke0 = tri_element_stiffness_laminate(p1, p2, p3, A=A0, D=D0)
    ke1 = tri_element_stiffness_laminate(p1, p2, p3, A=A0 + s * dA, D=D0 + s * dD)
    pred = ke0 + s * dke_dAD(p1, p2, p3, dA, dD)
    assert np.allclose(ke1, pred, rtol=1e-9, atol=1e-9 * np.abs(ke1).max())


def test_dAD_dt_matches_fd():
    t = 0.0018
    f0, f45, f90 = 0.4, 0.35, 0.25
    _, _, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=t)
    dA_ana, dD_ana = dAD_dt(Qeff, t)
    dA_fd = central_diff(
        lambda tt: laminate_stiffness(T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=tt)[0],
        t, 1e-7,
    )
    dD_fd = central_diff(
        lambda tt: laminate_stiffness(T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=tt)[1],
        t, 1e-7,
    )
    assert np.allclose(dA_ana, dA_fd, rtol=1e-6, atol=1e-6 * np.abs(dA_fd).max())
    assert np.allclose(dD_ana, dD_fd, rtol=1e-6, atol=1e-6 * np.abs(dD_fd).max())


def test_dQeff_df_matches_fd():
    f0, f45, t = 0.4, 0.35, 0.0018
    qeff = lambda ff0, ff45: laminate_stiffness(
        T700_EPOXY, f0=ff0, f45=ff45, f90=1.0 - ff0 - ff45, thickness=t
    )[2]
    # f0
    ana0 = dQeff_df(T700_EPOXY, which="f0")
    fd0 = central_diff(lambda x: qeff(x, f45), f0, 1e-7)
    assert np.allclose(ana0, fd0, rtol=1e-6, atol=1e-6 * np.abs(fd0).max())
    # f45
    ana45 = dQeff_df(T700_EPOXY, which="f45")
    fd45 = central_diff(lambda x: qeff(f0, x), f45, 1e-7)
    assert np.allclose(ana45, fd45, rtol=1e-6, atol=1e-6 * np.abs(fd45).max())


def test_dQeff_df_offset():
    f0, f45, t, o = 0.4, 0.35, 0.0018, 30.0
    qeff = lambda ff0, ff45: laminate_stiffness_offset(
        T700_EPOXY, f0=ff0, f45=ff45, f90=1.0 - ff0 - ff45, thickness=t, offset_deg=o
    )[2]
    ana0 = dQeff_df(T700_EPOXY, which="f0", offset_deg=o)
    fd0 = central_diff(lambda x: qeff(x, f45), f0, 1e-7)
    assert np.allclose(ana0, fd0, rtol=1e-6, atol=1e-6 * np.abs(fd0).max())
    ana45 = dQeff_df(T700_EPOXY, which="f45", offset_deg=o)
    fd45 = central_diff(lambda x: qeff(f0, x), f45, 1e-7)
    assert np.allclose(ana45, fd45, rtol=1e-6, atol=1e-6 * np.abs(fd45).max())


# --- Task 3: adjoint engine + tip deflection/twist gradients (FD-validated) ---
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.beams.sensitivity import (
    DesignSens, grad_tip_defl, grad_tip_twist,
)
from wing_design.structural.frame import _element_rotation


def _adjoint_case():
    """Small uniform case: n_beams=6, n_levels=4, 1 thickness band, no datum, 1 layup group."""
    m, _secs, _A, _D, _loads = _case(n_beams=6, n_levels=4)
    group_of_element, G = beam_radius_groups(m)
    n = m.beam_elements.shape[0]
    M = m.shell_tris.shape[0]
    # beam lengths (depend on geometry only, fixed across radius/thickness DVs)
    beam_lengths = np.empty(n)
    for e in range(n):
        i, j = int(m.beam_elements[e, 0]), int(m.beam_elements[e, 1])
        _R, L = _element_rotation(m.nodes[i], m.nodes[j])
        beam_lengths[e] = L
    # tip chordwise load so both deflection and twist are nonzero
    loads = np.zeros((m.nodes.shape[0], 6))
    loads[m.tip_nodes, 0] = 120.0
    return m, group_of_element, G, n, M, beam_lengths, loads


def _decode_solve(m, group_of_element, G, beam_lengths, loads, x):
    """Decode x = [r_group(G), t, f0, f45] -> FactoredBeamShell + Qeff used."""
    r_groups = x[:G]
    t = float(x[G])
    f0 = float(x[G + 1])
    f45 = float(x[G + 2])
    f90 = 1.0 - f0 - f45
    radii_full = r_groups[group_of_element]
    secs = [BeamSection.circular(float(radii_full[e])) for e in range(len(radii_full))]
    _A, _D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=t)
    A = t * Qeff
    D = (t**3 / 12.0) * Qeff
    fac = solve_beam_shell_laminate_factored(
        m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D,
        fixed_nodes=m.fixed_nodes, loads=loads,
    )
    return fac, Qeff


def _build_ds(m, group_of_element, G, M, beam_lengths, x):
    r_groups = x[:G]
    t = float(x[G])
    f0 = float(x[G + 1])
    f45 = float(x[G + 2])
    f90 = 1.0 - f0 - f45
    radii_full = r_groups[group_of_element]
    _A, _D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=t)
    return DesignSens(
        model=m, G=G, B=1, L=1,
        group_of_element=group_of_element,
        band_of_tri=np.zeros(M, dtype=int),
        layup_group_of_band=np.zeros(1, dtype=int),
        radii_full=radii_full,
        beam_lengths=beam_lengths,
        t_tri=np.full(M, t),
        Qeff_tri=np.broadcast_to(Qeff, (M, 3, 3)).copy(),
        offset_tri=np.zeros(M),
        ply=T700_EPOXY,
    )


def _g_defl(fac, m):
    u = fac.u.reshape(-1, 6)
    return float(np.max(np.linalg.norm(u[m.tip_nodes, :3], axis=1)))


def _g_twist(fac, m):
    u = fac.u.reshape(-1, 6)
    return float(np.max(np.abs(u[m.tip_nodes, 5])))


def test_grad_defl_twist_match_fd():
    m, group_of_element, G, n, M, beam_lengths, loads = _adjoint_case()
    x0 = np.concatenate([np.full(G, 0.01), [0.0015, 1.0 / 3.0, 1.0 / 3.0]])
    nx = G + 1 + 2

    fac0, _ = _decode_solve(m, group_of_element, G, beam_lengths, loads, x0)
    ds0 = _build_ds(m, group_of_element, G, M, beam_lengths, x0)

    g_defl, grad_defl = grad_tip_defl(fac0, ds0)
    g_twist, grad_twist = grad_tip_twist(fac0, ds0)
    assert g_defl > 0 and g_twist > 0
    assert np.isclose(g_defl, _g_defl(fac0, m))
    assert np.isclose(g_twist, _g_twist(fac0, m))

    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd_defl = np.zeros(nx)
    fd_twist = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        facp, _ = _decode_solve(m, group_of_element, G, beam_lengths, loads, xp)
        facm, _ = _decode_solve(m, group_of_element, G, beam_lengths, loads, xm)
        fd_defl[i] = (_g_defl(facp, m) - _g_defl(facm, m)) / (2.0 * h)
        fd_twist[i] = (_g_twist(facp, m) - _g_twist(facm, m)) / (2.0 * h)

    assert np.allclose(grad_defl, fd_defl, rtol=2e-4, atol=1e-4 * np.abs(fd_defl).max())
    assert np.allclose(grad_twist, fd_twist, rtol=2e-4, atol=1e-4 * np.abs(fd_twist).max())
