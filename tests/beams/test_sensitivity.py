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


# --- Task 4: beam von-Mises + Euler buckling gradients (FD-validated) ---
from wing_design.beams.sensitivity import grad_beam_vm, grad_beam_buckling
from wing_design.structural.frame import von_mises_per_element
from wing_design.structural.buckling import beam_euler_utilization

_SIGMA_ALLOW = 6.0e8
_EULER_K = 1.0
_SF = 1.5


def _buckling_case():
    """Like _adjoint_case but a load that compresses some beams (so buckling is active)."""
    m, group_of_element, G, n, M, beam_lengths, _loads = _adjoint_case()
    # Span is +Z (clamped at min z, tip at max z). A tip load with a strong −Z
    # (spanwise, toward root) component compresses the longitudinal beams; the
    # chordwise (+X) component creates bending/torsion so vM is nonzero. The
    # asymmetric per-node X push breaks buckling ties between the beams.
    loads = np.zeros((m.nodes.shape[0], 6))
    loads[m.tip_nodes, 2] = -8.0e4   # spanwise compression
    for a, node in enumerate(m.tip_nodes):
        loads[node, 0] = 40.0 + 7.0 * a  # chordwise, distinct per beam -> unique active
    return m, group_of_element, G, n, M, beam_lengths, loads


def _sections_of(radii_full):
    return [BeamSection.circular(float(r)) for r in radii_full]


def test_grad_beam_vm_matches_fd():
    m, group_of_element, G, n, M, beam_lengths, loads = _buckling_case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1.0 / 3.0, 1.0 / 3.0]])
    nx = G + 1 + 2

    fac0, _ = _decode_solve(m, group_of_element, G, beam_lengths, loads, x0)
    ds0 = _build_ds(m, group_of_element, G, M, beam_lengths, x0)

    con0, grad = grad_beam_vm(fac0, ds0, _SIGMA_ALLOW)

    def beam_con(x):
        fac, _ = _decode_solve(m, group_of_element, G, beam_lengths, loads, x)
        radii = x[:G][group_of_element]
        secs = _sections_of(radii)
        return 1.0 - float(np.max(von_mises_per_element(fac.result, secs))) / _SIGMA_ALLOW

    assert np.isclose(con0, beam_con(x0))

    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd[i] = (beam_con(xp) - beam_con(xm)) / (2.0 * h)

    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


def test_grad_beam_buckling_matches_fd():
    m, group_of_element, G, n, M, beam_lengths, loads = _buckling_case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1.0 / 3.0, 1.0 / 3.0]])
    nx = G + 1 + 2

    fac0, _ = _decode_solve(m, group_of_element, G, beam_lengths, loads, x0)
    ds0 = _build_ds(m, group_of_element, G, M, beam_lengths, x0)

    con0, grad = grad_beam_buckling(fac0, ds0, euler_K=_EULER_K, safety_factor=_SF)

    def beam_buck_con(x):
        fac, _ = _decode_solve(m, group_of_element, G, beam_lengths, loads, x)
        radii = x[:G][group_of_element]
        util = beam_euler_utilization(
            fac.result.axial_force, radii, beam_lengths,
            E=m.E_beam, K=_EULER_K, safety_factor=_SF,
        )
        return 1.0 - float(np.max(util))

    assert np.isclose(con0, beam_buck_con(x0))
    # ensure compression is actually active
    assert np.min(fac0.result.axial_force) < 0.0

    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd[i] = (beam_buck_con(xp) - beam_buck_con(xm)) / (2.0 * h)

    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


# --- Task 1 (caching): SensCache equivalence ---


def _adjoint_small_case():
    """Build a small (factored, ds) pair exercising radius, thickness AND layup DVs.

    Uses a datum offset and >=2 thickness bands (with >=2 layup groups) so the
    band/layup-group dv-index plumbing in the cache is genuinely exercised, not
    just the all-zeros degenerate case.
    """
    from wing_design.materials.unidir import laminate_stiffness_offset

    m, group_of_element, G, n, M, beam_lengths, loads = _adjoint_case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1.0 / 3.0, 1.0 / 3.0]])
    offset_deg = 20.0

    t = float(x0[G]); f0 = float(x0[G + 1]); f45 = float(x0[G + 2])
    f90 = 1.0 - f0 - f45
    radii_full = x0[:G][group_of_element]
    secs = [BeamSection.circular(float(r)) for r in radii_full]
    _A, _D, Qeff = laminate_stiffness_offset(
        T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=t, offset_deg=offset_deg
    )
    A = t * Qeff; D = (t**3 / 12.0) * Qeff
    fac = solve_beam_shell_laminate_factored(
        m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D,
        fixed_nodes=m.fixed_nodes, loads=loads,
    )

    # 2 thickness bands, 2 layup groups (alternate triangles) so band/layup
    # dv-indexing is exercised. All bands share the same physical t/Qeff here
    # (single solve), but the dv-index bookkeeping differs per band/group.
    band_of_tri = (np.arange(M) % 2).astype(int)
    layup_group_of_band = np.array([0, 1], dtype=int)
    ds = DesignSens(
        model=m, G=G, B=2, L=2,
        group_of_element=group_of_element,
        band_of_tri=band_of_tri,
        layup_group_of_band=layup_group_of_band,
        radii_full=radii_full,
        beam_lengths=beam_lengths,
        t_tri=np.full(M, t),
        Qeff_tri=np.broadcast_to(Qeff, (M, 3, 3)).copy(),
        offset_tri=np.full(M, offset_deg),
        ply=T700_EPOXY,
    )
    return fac, ds


def test_cached_lambdaT_dK_x_matches_uncached():
    from wing_design.beams.sensitivity import (
        prepare_sensitivity, lambdaT_dK_x, lambdaT_dK_x_cached,
    )
    factored, ds = _adjoint_small_case()
    rng = np.random.default_rng(0)
    lam = rng.normal(size=factored.ndof)
    cache = prepare_sensitivity(ds, factored)
    a = lambdaT_dK_x(factored, ds, lam)
    b = lambdaT_dK_x_cached(cache, lam, factored.u)
    assert np.allclose(a, b, atol=1e-12, rtol=0.0)


# --- Task 5: skin von-Mises + panel buckling gradients (FD-validated) ---
from wing_design.beams.sensitivity import grad_skin_vm, grad_panel_buckling
from wing_design.structural.shell import (
    recover_membrane_stress_C, membrane_von_mises, _triangle_local_frame,
)
from wing_design.structural.buckling import panel_buckling_utilization

_SKIN_SIGMA_ALLOW = 6.0e8
_PANEL_KC = 4.0
_PANEL_SF = 1.5


def _tri_areas(m):
    areas = np.empty(m.shell_tris.shape[0])
    for t in range(m.shell_tris.shape[0]):
        n0, n1, n2 = (int(v) for v in m.shell_tris[t])
        _R, _loc, a = _triangle_local_frame(m.nodes[n0], m.nodes[n1], m.nodes[n2])
        areas[t] = a
    return areas


def _build_ds_off(m, group_of_element, G, M, beam_lengths, x, offset_deg):
    ds = _build_ds(m, group_of_element, G, M, beam_lengths, x)
    if offset_deg != 0.0:
        from wing_design.materials.unidir import laminate_stiffness_offset
        t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2])
        f90 = 1.0 - f0 - f45
        _A, _D, Qeff = laminate_stiffness_offset(
            T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=t, offset_deg=offset_deg
        )
        ds.Qeff_tri = np.broadcast_to(Qeff, (M, 3, 3)).copy()
        ds.offset_tri = np.full(M, offset_deg)
    return ds


def _decode_solve_off(m, group_of_element, G, beam_lengths, loads, x, offset_deg):
    """Decode + solve, applying a ply-angle datum offset to the laminate."""
    from wing_design.materials.unidir import laminate_stiffness_offset
    r_groups = x[:G]
    t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2])
    f90 = 1.0 - f0 - f45
    radii_full = r_groups[group_of_element]
    secs = [BeamSection.circular(float(r)) for r in radii_full]
    _A, _D, Qeff = laminate_stiffness_offset(
        T700_EPOXY, f0=f0, f45=f45, f90=f90, thickness=t, offset_deg=offset_deg
    )
    A = t * Qeff; D = (t**3 / 12.0) * Qeff
    fac = solve_beam_shell_laminate_factored(
        m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D,
        fixed_nodes=m.fixed_nodes, loads=loads,
    )
    return fac, Qeff


def _skin_con_value(m, fac, Qeff, sigma_allow):
    C = np.broadcast_to(Qeff, (m.shell_tris.shape[0], 3, 3)).copy()
    s = recover_membrane_stress_C(m.nodes, m.shell_tris, fac.result.displacements, C=C)
    return 1.0 - float(np.max(membrane_von_mises(s))) / sigma_allow


def _panel_buck_con_value(m, fac, Qeff, t, areas, kc, SF):
    C = np.broadcast_to(Qeff, (m.shell_tris.shape[0], 3, 3)).copy()
    s = recover_membrane_stress_C(m.nodes, m.shell_tris, fac.result.displacements, C=C)
    util = np.empty(m.shell_tris.shape[0])
    for k in range(m.shell_tris.shape[0]):
        D11 = (t**3 / 12.0) * float(Qeff[0, 0])
        util[k] = panel_buckling_utilization(
            s[k:k + 1], areas[k:k + 1], D11=D11, t=t, kc=kc, safety_factor=SF
        )[0]
    return 1.0 - float(np.max(util))


def _skin_case():
    """Tip load with strong chordwise + spanwise components -> nonzero skin stress
    and clearly compressive panels."""
    m, group_of_element, G, n, M, beam_lengths, _loads = _adjoint_case()
    loads = np.zeros((m.nodes.shape[0], 6))
    loads[m.tip_nodes, 2] = -1.2e5   # spanwise compression -> compressive panels
    for a, node in enumerate(m.tip_nodes):
        loads[node, 0] = 60.0 + 11.0 * a   # chordwise, distinct -> unique active tri
    return m, group_of_element, G, n, M, beam_lengths, loads


def _run_skin_vm(offset_deg):
    m, group_of_element, G, n, M, beam_lengths, loads = _skin_case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1.0 / 3.0, 1.0 / 3.0]])
    nx = G + 1 + 2

    fac0, Qeff0 = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, x0, offset_deg)
    ds0 = _build_ds_off(m, group_of_element, G, M, beam_lengths, x0, offset_deg)

    con0, grad = grad_skin_vm(fac0, ds0, _SKIN_SIGMA_ALLOW)
    assert np.isclose(con0, _skin_con_value(m, fac0, Qeff0, _SKIN_SIGMA_ALLOW))

    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        facp, Qp = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, xp, offset_deg)
        facm, Qm = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, xm, offset_deg)
        fd[i] = (
            _skin_con_value(m, facp, Qp, _SKIN_SIGMA_ALLOW)
            - _skin_con_value(m, facm, Qm, _SKIN_SIGMA_ALLOW)
        ) / (2.0 * h)
    return grad, fd


def test_grad_skin_vm_matches_fd():
    grad, fd = _run_skin_vm(offset_deg=0.0)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


def test_grad_skin_vm_matches_fd_offset():
    grad, fd = _run_skin_vm(offset_deg=30.0)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


def _run_panel_buckling(offset_deg, use_strip_widths=False):
    m, group_of_element, G, n, M, beam_lengths, loads = _skin_case()
    if use_strip_widths:
        # V.3b strip mode: b^2 = physical panel width^2 enters where area did.
        from wing_design.beams.shell_model import skin_panel_widths
        areas = skin_panel_widths(m) ** 2
    else:
        areas = _tri_areas(m)
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1.0 / 3.0, 1.0 / 3.0]])
    nx = G + 1 + 2

    fac0, Qeff0 = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, x0, offset_deg)
    ds0 = _build_ds_off(m, group_of_element, G, M, beam_lengths, x0, offset_deg)

    con0, grad = grad_panel_buckling(
        fac0, ds0, panel_kc=_PANEL_KC, safety_factor=_PANEL_SF, areas=areas
    )
    t0 = float(x0[G])
    assert np.isclose(
        con0, _panel_buck_con_value(m, fac0, Qeff0, t0, areas, _PANEL_KC, _PANEL_SF)
    )
    # ensure a clearly compressive active panel
    assert con0 < 1.0

    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        facp, Qp = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, xp, offset_deg)
        facm, Qm = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, xm, offset_deg)
        tp = float(xp[G]); tm = float(xm[G])
        fd[i] = (
            _panel_buck_con_value(m, facp, Qp, tp, areas, _PANEL_KC, _PANEL_SF)
            - _panel_buck_con_value(m, facm, Qm, tm, areas, _PANEL_KC, _PANEL_SF)
        ) / (2.0 * h)
    return grad, fd


def test_grad_panel_buckling_matches_fd():
    grad, fd = _run_panel_buckling(offset_deg=0.0)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


def test_grad_panel_buckling_matches_fd_offset():
    grad, fd = _run_panel_buckling(offset_deg=30.0)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


def test_grad_panel_buckling_matches_fd_strip_widths():
    # V.3b: the strip width is design-independent, so the analytic gradient must
    # stay exact when widths^2 replaces triangle areas as b^2.
    grad, fd = _run_panel_buckling(offset_deg=30.0, use_strip_widths=True)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


# --- Task 6: per-ply Tsai-Wu skin gradient (FD-validated) ---
from wing_design.beams.sensitivity import grad_skin_tsai_wu
from wing_design.materials.failure import laminate_min_strength_ratio_batch
from wing_design.structural.shell import recover_membrane_strain

_TSAI_SF = 1.5


def _skin_tsai_con_value(m, fac, f0, f45, f90, offsets, SF):
    """min over triangles of laminate_min_strength_ratio_batch / SF − 1."""
    eps = recover_membrane_strain(m.nodes, m.shell_tris, fac.result.displacements)
    R = laminate_min_strength_ratio_batch(
        T700_EPOXY, eps, f0=f0, f45=f45, f90=f90, offset_deg=offsets,
    )
    return float(np.min(R)) / SF - 1.0


def test_grad_skin_tsai_wu_matches_fd():
    # DATUM offset so ply angles are nonzero; layup with all orientations present.
    offset_deg = 25.0
    m, group_of_element, G, n, M, beam_lengths, loads = _skin_case()
    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1.0 / 3.0, 1.0 / 3.0]])
    nx = G + 1 + 2
    offsets = np.full(M, offset_deg)

    fac0, _Qeff0 = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, x0, offset_deg)
    ds0 = _build_ds_off(m, group_of_element, G, M, beam_lengths, x0, offset_deg)
    # attach per-triangle fractions (all four orientations present at f0=f45=1/3)
    f0v = float(x0[G + 1]); f45v = float(x0[G + 2]); f90v = 1.0 - f0v - f45v
    ds0.f0_tri = np.full(M, f0v)
    ds0.f45_tri = np.full(M, f45v)
    ds0.f90_tri = np.full(M, f90v)

    con0, grad = grad_skin_tsai_wu(fac0, ds0, safety_factor=_TSAI_SF)
    ref = _skin_tsai_con_value(m, fac0, f0v, f45v, f90v, offsets, _TSAI_SF)
    assert np.isclose(con0, ref), (con0, ref)

    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        facp, _ = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, xp, offset_deg)
        facm, _ = _decode_solve_off(m, group_of_element, G, beam_lengths, loads, xm, offset_deg)
        f0p = float(xp[G + 1]); f45p = float(xp[G + 2]); f90p = 1.0 - f0p - f45p
        f0m = float(xm[G + 1]); f45m = float(xm[G + 2]); f90m = 1.0 - f0m - f45m
        fd[i] = (
            _skin_tsai_con_value(m, facp, f0p, f45p, f90p, offsets, _TSAI_SF)
            - _skin_tsai_con_value(m, facm, f0m, f45m, f90m, offsets, _TSAI_SF)
        ) / (2.0 * h)

    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())
