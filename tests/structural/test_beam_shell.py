import numpy as np

from wing_design.structural.frame import BeamSection, solve_frame
from wing_design.structural.beam_shell import solve_beam_shell, solve_beam_shell_laminate

E_B, NU = 70e9, 0.3
G_B = E_B / (2 * (1 + NU))


def test_no_skin_reduces_to_frame():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    beam_elems = np.array([[0, 1], [1, 2]], dtype=int)
    secs = [BeamSection.circular(0.02)] * 2
    loads = np.zeros((3, 6))
    loads[2, 2] = 200.0
    frame_res = solve_frame(nodes, beam_elems, secs, E=E_B, G=G_B,
                            fixed_nodes=np.array([0]), loads=loads)
    bs_res = solve_beam_shell(
        nodes, beam_elems, secs, np.empty((0, 3), dtype=int),
        E_beam=E_B, G_beam=G_B, E_skin=1e9, nu_skin=0.3, t_skin=0.003,
        fixed_nodes=np.array([0]), loads=loads,
    )
    assert np.allclose(bs_res.displacements, frame_res.displacements, atol=1e-12)


def test_skin_stiffens_cantilever():
    nodes = np.array([
        [0, 0, 0], [1, 0, 0],
        [0, 0.2, 0], [1, 0.2, 0],
    ], dtype=float)
    beam_elems = np.array([[0, 1], [2, 3]], dtype=int)
    secs = [BeamSection.circular(0.01)] * 2
    loads = np.zeros((4, 6))
    loads[1, 2] = 100.0
    loads[3, 2] = 100.0
    fixed = np.array([0, 2])

    no_skin = solve_beam_shell(
        nodes, beam_elems, secs, np.empty((0, 3), dtype=int),
        E_beam=E_B, G_beam=G_B, E_skin=70e9, nu_skin=0.3, t_skin=0.003,
        fixed_nodes=fixed, loads=loads,
    )
    tris = np.array([[0, 1, 3], [0, 3, 2]], dtype=int)
    with_skin = solve_beam_shell(
        nodes, beam_elems, secs, tris,
        E_beam=E_B, G_beam=G_B, E_skin=70e9, nu_skin=0.3, t_skin=0.003,
        fixed_nodes=fixed, loads=loads,
    )
    d_no = np.abs(no_skin.displacements[[1, 3], 2]).max()
    d_yes = np.abs(with_skin.displacements[[1, 3], 2]).max()
    assert d_yes < d_no
    assert np.isfinite(d_yes) and d_yes > 0.0


def test_laminate_solver_reduces_to_isotropic():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 0.2, 0], [1, 0.2, 0]], dtype=float)
    beam_elems = np.array([[0, 1], [2, 3]], dtype=int)
    secs = [BeamSection.circular(0.01)] * 2
    loads = np.zeros((4, 6)); loads[1, 2] = 100.0; loads[3, 2] = 100.0
    fixed = np.array([0, 2])
    tris = np.array([[0, 1, 3], [0, 3, 2]], dtype=int)
    E, nu, t = 70e9, 0.3, 0.003
    Dm = (E / (1 - nu**2)) * np.array([[1, nu, 0.0], [nu, 1, 0.0], [0, 0, (1 - nu) / 2]])
    A, D = t * Dm, (t**3 / 12.0) * Dm
    iso = solve_beam_shell(nodes, beam_elems, secs, tris, E_beam=E, G_beam=E / 2.6,
                           E_skin=E, nu_skin=nu, t_skin=t, fixed_nodes=fixed, loads=loads)
    lam = solve_beam_shell_laminate(nodes, beam_elems, secs, tris, E_beam=E, G_beam=E / 2.6,
                                    A_skin=A, D_skin=D, fixed_nodes=fixed, loads=loads)
    assert np.allclose(iso.displacements, lam.displacements, rtol=1e-9, atol=1e-12)


def test_laminate_solver_accepts_per_triangle_stiffness():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 0.2, 0], [1, 0.2, 0]], dtype=float)
    beam_elems = np.array([[0, 1], [2, 3]], dtype=int)
    secs = [BeamSection.circular(0.01)] * 2
    loads = np.zeros((4, 6)); loads[1, 2] = 100.0; loads[3, 2] = 100.0
    fixed = np.array([0, 2])
    tris = np.array([[0, 1, 3], [0, 3, 2]], dtype=int)
    E, nu, t = 70e9, 0.3, 0.003
    Dm = (E / (1 - nu**2)) * np.array([[1, nu, 0.0], [nu, 1, 0.0], [0, 0, (1 - nu) / 2]])
    A, D = t * Dm, (t**3 / 12.0) * Dm
    single = solve_beam_shell_laminate(nodes, beam_elems, secs, tris, E_beam=E, G_beam=E / 2.6,
                                       A_skin=A, D_skin=D, fixed_nodes=fixed, loads=loads)
    A_tris = np.repeat(A[None], len(tris), axis=0)
    D_tris = np.repeat(D[None], len(tris), axis=0)
    multi = solve_beam_shell_laminate(nodes, beam_elems, secs, tris, E_beam=E, G_beam=E / 2.6,
                                      A_skin=A_tris, D_skin=D_tris, fixed_nodes=fixed, loads=loads)
    assert np.allclose(single.displacements, multi.displacements, rtol=1e-12)
