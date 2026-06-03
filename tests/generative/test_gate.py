import math

import numpy as np

from wing_design.generative.gate import section_properties


def test_section_properties_circle():
    area = 4.0e-3
    A, I, J = section_properties(area)
    assert math.isclose(A, area, rel_tol=1e-12)
    # circle: I = A^2 / (4 pi), J = 2 I
    assert math.isclose(I, area**2 / (4.0 * math.pi), rel_tol=1e-12)
    assert math.isclose(J, 2.0 * I, rel_tol=1e-12)


from wing_design.generative.gate import local_beam_stiffness


def test_local_beam_stiffness_symmetric_and_axial():
    E, G, A, I, J, L = 200e9, 80e9, 1e-3, 1e-6, 2e-6, 2.0
    k = local_beam_stiffness(E, G, A, I, J, L)
    assert k.shape == (12, 12)
    # symmetric
    assert np.allclose(k, k.T)
    # axial sub-terms: k[0,0] = EA/L, k[0,6] = -EA/L
    assert math.isclose(k[0, 0], E * A / L, rel_tol=1e-12)
    assert math.isclose(k[0, 6], -E * A / L, rel_tol=1e-12)
    # torsion: k[3,3] = GJ/L
    assert math.isclose(k[3, 3], G * J / L, rel_tol=1e-12)
    # bending: k[1,1] = 12 EI / L^3
    assert math.isclose(k[1, 1], 12.0 * E * I / L**3, rel_tol=1e-12)


from wing_design.generative.gate import beam_transform


def test_beam_transform_length_and_orthonormal():
    T, L = beam_transform((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    assert math.isclose(L, 3.0, rel_tol=1e-12)
    assert T.shape == (12, 12)
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(R[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_beam_transform_handles_vertical_beam():
    T, L = beam_transform((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
    assert math.isclose(L, 5.0, rel_tol=1e-12)
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(R[0], [0.0, 0.0, 1.0], atol=1e-12)


from wing_design.generative.gate import element_global_stiffness


def test_element_global_stiffness_symmetric_and_axisaligned():
    E, G, A, I, J = 200e9, 80e9, 1e-3, 1e-6, 2e-6
    ke = element_global_stiffness((0, 0, 0), (2.0, 0, 0), E, G, A, I, J)
    assert ke.shape == (12, 12)
    assert np.allclose(ke, ke.T, atol=1e-3)
    k_local = local_beam_stiffness(E, G, A, I, J, 2.0)
    assert np.allclose(ke, k_local, atol=1e-6)


from wing_design.generative.gate import solve_displacements

# Shared beam properties for the analytic single-element checks.
_E, _G, _A, _I, _J, _L = 200e9, 80e9, 1e-3, 1e-6, 2e-6, 2.0
_CLAMP = [0, 1, 2, 3, 4, 5]  # all 6 DOF at node i


def test_axial_extension_matches_PL_over_EA():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 0] = P  # axial (Fx) at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 0], P * _L / (_E * _A), rtol=1e-6)


def test_horizontal_cantilever_matches_PL3_over_3EI():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 1] = P  # transverse (Fy) at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 1], P * _L**3 / (3.0 * _E * _I), rtol=1e-6)


def test_vertical_cantilever_matches_PL3_over_3EI():
    ke = element_global_stiffness((0, 0, 0), (0, 0, _L), _E, _G, _A, _I, _J)
    P = 1000.0
    load = np.zeros(12)
    load[6 + 0] = P  # transverse (Fx) at the top node
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 0], P * _L**3 / (3.0 * _E * _I), rtol=1e-6)


def test_torsion_matches_TL_over_GJ():
    ke = element_global_stiffness((0, 0, 0), (_L, 0, 0), _E, _G, _A, _I, _J)
    Tq = 1000.0
    load = np.zeros(12)
    load[6 + 3] = Tq  # torque (Mx) about the beam axis at node j
    u = solve_displacements(ke, load, _CLAMP)
    assert np.isclose(u[6 + 3], Tq * _L / (_G * _J), rtol=1e-6)
