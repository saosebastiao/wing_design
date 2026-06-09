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
