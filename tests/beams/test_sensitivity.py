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
