"""V.3 eigenvalue buckling: Kσ formulations validated against closed-form references.

Beam geometric stiffness -> Euler column loads (pinned-pinned, cantilever);
shell triangle geometric stiffness -> simply-supported plate kc = 4 formula.
References are exact classical solutions, so tolerances reflect discretization
error only (documented per test).
"""
import numpy as np
import pytest

from wing_design.structural.frame import BeamSection, local_beam_stiffness, _element_rotation
from wing_design.structural.geometric_stiffness import (
    beam_geometric_K,
    assemble_geometric_stiffness,
)
from wing_design.structural.eigen_buckling import linear_buckling
from wing_design.structural.shell import _element_K, _triangle_local_frame

E = 70.0e9
G = 27.0e9


def _column(n_el, L, r, axis=np.array([0.0, 0.0, 1.0])):
    """Straight n_el-element column along `axis`; returns nodes, elements, sections."""
    nodes = np.outer(np.linspace(0.0, L, n_el + 1), axis)
    elements = np.column_stack([np.arange(n_el), np.arange(1, n_el + 1)])
    sections = [BeamSection.circular(r)] * n_el
    return nodes, elements, sections


def _assemble_beam_K(nodes, elements, sections):
    ndof = 6 * nodes.shape[0]
    K = np.zeros((ndof, ndof))
    for e in range(elements.shape[0]):
        i, j = int(elements[e, 0]), int(elements[e, 1])
        R, L = _element_rotation(nodes[i], nodes[j])
        T = np.kron(np.eye(4), R)
        k = T.T @ local_beam_stiffness(E, G, sections[e], L) @ T
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        K[np.ix_(dofs, dofs)] += k
    return K


def _free_dofs(n_nodes, fixed: dict[int, list[int]]):
    """fixed: node -> list of local dof indices (0..5) to clamp."""
    mask = np.ones(6 * n_nodes, dtype=bool)
    for node, dofs in fixed.items():
        for d in dofs:
            mask[6 * node + d] = False
    return np.where(mask)[0]


def _column_buckling_lambda(n_el, fixed, *, unit_compression=1.0e4):
    L_tot, r = 2.0, 0.02
    nodes, elements, sections = _column(n_el, L_tot, r)
    n_nodes = nodes.shape[0]
    K = _assemble_beam_K(nodes, elements, sections)
    axial = np.full(n_el, -unit_compression)          # uniform compression
    Kg = assemble_geometric_stiffness(
        nodes, beam_elements=elements, axial_forces=axial,
        shell_tris=None, t_tri=None, tri_stress_local=None)
    free = _free_dofs(n_nodes, fixed)
    lams, _modes = linear_buckling(K[np.ix_(free, free)], Kg[np.ix_(free, free)], k=3)
    I = np.pi * r**4 / 4.0
    return lams[0] * unit_compression, E * I, L_tot


def test_pinned_pinned_column_euler_load():
    # torsion rx must be restrained somewhere; pin translations both ends.
    n_el = 8
    fixed = {0: [0, 1, 2, 3], n_el: [0, 1]}   # base: pin + torsion; tip: lateral pins
    Pcr, EI, L = _column_buckling_lambda(n_el, fixed)
    ref = np.pi**2 * EI / L**2
    assert Pcr == pytest.approx(ref, rel=2.0e-3)   # consistent Kg converges fast


def test_cantilever_column_euler_load():
    n_el = 8
    fixed = {0: [0, 1, 2, 3, 4, 5]}
    Pcr, EI, L = _column_buckling_lambda(n_el, fixed)
    ref = np.pi**2 * EI / (2.0 * L) ** 2
    assert Pcr == pytest.approx(ref, rel=2.0e-3)


def test_tension_state_has_no_buckling_mode():
    n_el = 4
    L_tot, r = 2.0, 0.02
    nodes, elements, sections = _column(n_el, L_tot, r)
    K = _assemble_beam_K(nodes, elements, sections)
    axial = np.full(n_el, +1.0e4)                      # tension: stabilizing
    Kg = assemble_geometric_stiffness(
        nodes, beam_elements=elements, axial_forces=axial,
        shell_tris=None, t_tri=None, tri_stress_local=None)
    free = _free_dofs(nodes.shape[0], {0: [0, 1, 2, 3, 4, 5]})
    lams, _ = linear_buckling(K[np.ix_(free, free)], Kg[np.ix_(free, free)], k=3)
    assert lams.size == 0 or lams[0] > 1.0e3           # no physical positive multiplier


def _plate_mesh(nx, ny, a, b):
    """Structured triangulation of an a x b plate in the global x-y plane."""
    xs = np.linspace(0.0, a, nx + 1)
    ys = np.linspace(0.0, b, ny + 1)
    nodes = np.array([[x, y, 0.0] for y in ys for x in xs])
    tris = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            tris.append([n0, n1, n3])
            tris.append([n0, n3, n2])
    return nodes, np.array(tris, dtype=int)


def test_ss_plate_uniaxial_compression_kc4():
    # Square SS plate, uniaxial sigma_x: sigma_cr = 4 pi^2 D / (b^2 t).
    # w-linear Kg + DKT bending on a 12x12 grid: a few-% discretization error.
    a = b = 1.0
    t = 0.004
    nu = 0.3
    nx = ny = 12
    nodes, tris = _plate_mesh(nx, ny, a, b)
    n_nodes = nodes.shape[0]
    ndof = 6 * n_nodes

    K = np.zeros((ndof, ndof))
    for tri in tris:
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        ke = _element_K(p1, p2, p3, E=E, nu=nu, t=t)
        dofs = np.r_[6 * tri[0]:6 * tri[0] + 6, 6 * tri[1]:6 * tri[1] + 6,
                     6 * tri[2]:6 * tri[2] + 6]
        K[np.ix_(dofs, dofs)] += ke

    sigma0 = 1.0e6
    stress = np.tile([-sigma0, 0.0, 0.0], (tris.shape[0], 1))   # uniform sigma_x comp.
    # NOTE: stress must be expressed in each triangle's LOCAL frame; for this flat
    # z=0 plate every local frame is a rotation within the plate plane, handled by
    # the assembler's local-frame contract via per-tri rotation of the given stress.
    # Here we pass global-plane stress and rotate per triangle below.
    stress_local = np.empty_like(stress)
    for k_t, tri in enumerate(tris):
        R, _loc, _area = _triangle_local_frame(nodes[tri[0]], nodes[tri[1]], nodes[tri[2]])
        ex, ey = R[:, 0], R[:, 1]      # columns are the local axes (global = R @ local)
        s = np.array([[-sigma0, 0.0], [0.0, 0.0]])              # global (x,y) plane
        Q = np.array([[ex[0], ex[1]], [ey[0], ey[1]]])
        sl = Q @ s @ Q.T
        stress_local[k_t] = [sl[0, 0], sl[1, 1], sl[0, 1]]

    Kg = assemble_geometric_stiffness(
        nodes, beam_elements=None, axial_forces=None,
        shell_tris=tris, t_tri=np.full(tris.shape[0], t), tri_stress_local=stress_local)

    # BCs: SS on w along all edges; clamp ALL in-plane + drilling DOFs (membrane
    # state is prescribed, not solved); rotations rx/ry free everywhere.
    fixed: dict[int, list[int]] = {}
    for n in range(n_nodes):
        x, y = nodes[n, 0], nodes[n, 1]
        dofs = [0, 1, 5]                                        # u, v, drilling
        on_edge = x in (0.0, a) or y in (0.0, b)
        if on_edge:
            dofs.append(2)                                      # w = 0
        fixed[n] = dofs
    free = _free_dofs(n_nodes, fixed)

    lams, _ = linear_buckling(K[np.ix_(free, free)], Kg[np.ix_(free, free)], k=3)
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    sigma_ref = 4.0 * np.pi**2 * D / (b**2 * t)
    assert lams[0] * sigma0 == pytest.approx(sigma_ref, rel=0.05)
