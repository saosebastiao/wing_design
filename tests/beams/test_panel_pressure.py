"""V.5 panel pressure: projection conservation + the strip-bending failure term
(FD-validated gradient per house convention)."""
import numpy as np
import pytest

from wing_design.geometry import small_wingsail
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness
from wing_design.beams.shell_model import build_beam_shell_model, skin_panel_widths
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig,
    laminate_result_is_feasible,
    size_beam_shell_laminate,
)
from wing_design.beams.sensitivity import DesignSens, grad_skin_vm, prepare_sensitivity
from wing_design.structural.beam_shell import solve_beam_shell_laminate_factored
from wing_design.structural.frame import BeamSection, _element_rotation
from wing_design.structural.shell import membrane_von_mises, recover_membrane_stress_C

SIGMA_ALLOW = 2.0e8


class _FakePanels:
    """Minimal PanelLoads stand-in (centers/forces in the AERO frame: span +Y)."""
    def __init__(self, centers, forces):
        self.centers_xyz = np.asarray(centers, dtype=float)
        self.forces_xyz = np.asarray(forces, dtype=float)
        self.n_panels = self.centers_xyz.shape[0]


def _panels_near(model, k=5, f=(40.0, 0.0, 10.0)):
    """Fake aero panels near the first k tri centroids (geom->aero rotation inverse)."""
    from wing_design.structural.projection import R_GEOM_FROM_AERO
    centroids = model.nodes[model.shell_tris[:k]].mean(axis=1)
    centers_aero = centroids @ R_GEOM_FROM_AERO          # inverse of geom = R @ aero
    forces_aero = np.tile(np.asarray(f) @ R_GEOM_FROM_AERO, (k, 1))
    return _FakePanels(centers_aero, forces_aero)


def test_skin_projection_conserves_total_force():
    from wing_design.beams.fea_model import project_panels_to_skin
    model = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3)
    panels = _panels_near(model, k=6)
    loads = project_panels_to_skin(model, panels, safety_factor=1.5)
    from wing_design.structural.projection import R_GEOM_FROM_AERO
    expected = 1.5 * (panels.forces_xyz @ R_GEOM_FROM_AERO.T).sum(axis=0)
    assert np.allclose(loads[:, :3].sum(axis=0), expected, rtol=1e-12)
    assert np.allclose(loads[:, 3:], 0.0)


def test_panel_pressure_maps_force_over_area():
    from wing_design.beams.fea_model import panel_pressure_per_tri
    model = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3)
    panels = _panels_near(model, k=3)
    q = panel_pressure_per_tri(model, panels)
    assert q.shape == (model.shell_tris.shape[0],)
    assert (q > 0).sum() >= 1            # at least one loaded triangle
    assert np.all(q >= 0.0)


def test_grad_skin_vm_with_strip_bending_matches_fd():
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
    # a deliberately uneven pressure field so the adder moves the argmax around
    rng = np.random.default_rng(3)
    qw2 = 0.75 * rng.uniform(0.0, 4.0e3, size=M) * skin_panel_widths(m) ** 2

    x0 = np.concatenate([np.linspace(0.011, 0.014, G), [0.0015, 1/3, 1/3]])
    nx = G + 3

    def decode_solve(x):
        radii = x[:G][goe]
        t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2])
        A, D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45,
                                        f90=1.0 - f0 - f45, thickness=t)
        secs = [BeamSection.circular(float(r)) for r in radii]
        fac = solve_beam_shell_laminate_factored(
            m.nodes, m.beam_elements, secs, m.shell_tris,
            E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D,
            fixed_nodes=m.fixed_nodes, loads=aero)
        return fac, Qeff

    def build_ds(x):
        radii = x[:G][goe]
        t = float(x[G]); f0 = float(x[G + 1]); f45 = float(x[G + 2])
        _A, _D, Qeff = laminate_stiffness(T700_EPOXY, f0=f0, f45=f45,
                                          f90=1.0 - f0 - f45, thickness=t)
        return DesignSens(
            model=m, G=G, B=1, L=1, group_of_element=goe,
            band_of_tri=np.zeros(M, dtype=int),
            layup_group_of_band=np.zeros(1, dtype=int),
            radii_full=radii, beam_lengths=beam_len, t_tri=np.full(M, t),
            Qeff_tri=np.broadcast_to(Qeff, (M, 3, 3)).copy(),
            offset_tri=np.zeros(M), ply=T700_EPOXY)

    fac0, _Q0 = decode_solve(x0)
    ds0 = build_ds(x0)
    cache = prepare_sensitivity(ds0, fac0)
    con0, grad = grad_skin_vm(fac0, ds0, SIGMA_ALLOW, cache=cache, qw2_tri=qw2)

    def con_value(x):
        fac, Qx = decode_solve(x)
        C = np.broadcast_to(Qx, (M, 3, 3)).copy()
        s = recover_membrane_stress_C(m.nodes, m.shell_tris, fac.result.displacements, C=C)
        metric = membrane_von_mises(s) + qw2 / float(x[G]) ** 2
        return 1.0 - float(metric.max()) / SIGMA_ALLOW

    assert np.isclose(con0, con_value(x0))
    steps = np.concatenate([np.full(G, 1e-7), [1e-7, 1e-6, 1e-6]])
    fd = np.zeros(nx)
    for i in range(nx):
        h = steps[i]
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd[i] = (con_value(xp) - con_value(xm)) / (2 * h)
    assert np.allclose(grad, fd, rtol=2e-4, atol=1e-4 * np.abs(fd).max())


@pytest.mark.sizing
def test_sizing_with_pressure_bending_feasible_never_lighter():
    m = build_beam_shell_model(small_wingsail, n_beams=4, n_levels=3)
    aero = np.zeros((m.nodes.shape[0], 6))
    aero[m.tip_nodes, 2] = 800.0
    aero[m.tip_nodes, 0] = 400.0
    q = np.full(m.shell_tris.shape[0], 3.0e3)   # 3 kPa everywhere

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=2.0e8, tip_defl_max_m=1.0, tip_twist_max_deg=90.0,
        r_min=0.004, r_max=0.04, t_min=0.0005, t_max=0.02,
        use_analytic_jacobian=True)
    base = size_beam_shell_laminate(m, [aero], cfg, ply=T700_EPOXY, rho=1550.0,
                                    maxiter=150)
    withq = size_beam_shell_laminate(m, [aero], cfg, ply=T700_EPOXY, rho=1550.0,
                                     maxiter=150, panel_pressures=[q])
    assert laminate_result_is_feasible(withq, cfg)
    assert withq.max_skin_vm_Pa <= cfg.sigma_allow_Pa * 1.05
    assert withq.mass_kg >= base.mass_kg * 0.999
